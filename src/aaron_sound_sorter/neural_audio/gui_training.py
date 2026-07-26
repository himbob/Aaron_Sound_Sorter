"""Source-name-blind GUI intake and versioned neural prototype rebuilding."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aaron_sound_sorter.taxonomy_contracts import is_trainable_taxonomy_label

from .cache import EmbeddingCache
from .contracts import EmbeddingRecord
from .curation import TRAINING_USE
from .hashing import decoded_audio_sha256, normalized_audio_sha256, sha256_file
from .prototype_index import PrototypeIndex, PrototypeIndexBuilder
from .providers.base import EmbeddingProvider

INBOX_FIELDS = (
    "candidate_id",
    "source_path",
    "approved_folder",
    "label_source",
    "source_kind",
    "source_group",
    "human_approved",
    "file_sha256",
    "decoded_audio_sha256",
    "normalized_audio_sha256",
    "duplicate_group_id",
    "allowed_use",
    "first_approved_utc",
    "last_approved_utc",
    "confirmation_count",
    "source_manifest",
)
TRAINABLE_IMPORT_STATUSES = frozenset({"staged", "duplicate_existing"})
DEFAULT_INBOX_ROOT = Path("neural_artifacts") / "gui_training_inbox"
DEFAULT_TRAINING_CONFIG = Path("config") / "neural_training.json"
EXACT_TRAINING_POLICY = "content_hash_approved_label_v1"
LOCKED_SEED_OVERRIDE_CONFIRMATIONS = 2


@dataclass(frozen=True)
class NeuralIntakeSummary:
    """Summary of GUI corrections committed to the durable neural inbox.

    Args:
        current_manifest_path: Materialized current training-only examples.
        event_log_path: Append-only approval event log.
        queued_count: New or relabeled content groups accepted.
        reaffirmed_count: Existing content/label pairs approved again.
        superseded_count: Earlier labels replaced by a newer approval.
        errors: Non-fatal rows rejected from neural intake.

    Side Effects:
        None. ``NeuralTrainingInbox`` performs the writes.
    """

    current_manifest_path: Path
    event_log_path: Path
    queued_count: int
    reaffirmed_count: int
    superseded_count: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class NeuralTrainingExample:
    """One content-identified example used to build neural prototypes."""

    audio_path: Path
    label: str
    file_sha256: str
    decoded_audio_sha256: str
    normalized_audio_sha256: str
    duplicate_group_id: str
    source_kind: str


@dataclass(frozen=True)
class CorrectionPrediction:
    """Before/after neural evidence for one GUI-approved training example."""

    file_sha256: str
    approved_label: str
    predicted_before: str
    predicted_after: str
    second_after: str
    top_similarity_after: float
    margin_after: float
    known_distribution_after: bool
    prototype_predicted_after: str = ""


@dataclass(frozen=True)
class PendingTrainingConflict:
    """One GUI relabel awaiting deliberate confirmation.

    A single GUI click must not silently replace a locked human seed for the
    same decoded audio. Repeating the same explicit relabel confirms intent.
    """

    file_sha256: str
    locked_label: str
    proposed_label: str
    confirmation_count: int
    required_confirmation_count: int = LOCKED_SEED_OVERRIDE_CONFIRMATIONS


@dataclass(frozen=True)
class NeuralPrototypeBuildSummary:
    """Result of a versioned incremental prototype rebuild."""

    status: str
    message: str
    index_path: Path | None
    pointer_path: Path | None
    previous_index_path: str
    training_example_count: int
    label_count: int
    correction_predictions: tuple[CorrectionPrediction, ...]
    missing_audio_hashes: tuple[str, ...]
    invalidated_evaluation_hashes: tuple[str, ...]
    pending_training_conflicts: tuple[PendingTrainingConflict, ...]
    report_path: Path | None


class NeuralTrainingInbox:
    """Commit GUI-approved corrections to a durable training-only ledger.

    Args:
        project_root: Repository root containing training and neural artifacts.
        inbox_root: Optional inbox folder. Defaults to
            ``neural_artifacts/gui_training_inbox``.

    Side Effects:
        Appends JSONL approval events and atomically replaces ``current.csv``.

    Important Constraints:
        Content hashes identify audio. The approved taxonomy label is the only
        supervised target. Filenames and source paths are never model evidence.
    """

    def __init__(self, project_root: Path, inbox_root: Path | None = None) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        configured_root = inbox_root or self.project_root / DEFAULT_INBOX_ROOT
        self.inbox_root = Path(configured_root).expanduser().resolve()
        self.current_manifest_path = self.inbox_root / "current.csv"
        self.event_log_path = self.inbox_root / "events.jsonl"

    def queue_import_manifest(self, import_manifest_path: Path) -> NeuralIntakeSummary:
        """Queue trainable rows from one GUI training-import manifest."""
        manifest_path = Path(import_manifest_path).expanduser().resolve()
        existing_rows = {
            row["duplicate_group_id"]: row
            for row in _read_csv(self.current_manifest_path)
            if row.get("duplicate_group_id")
        }
        queued_count = 0
        reaffirmed_count = 0
        superseded_count = 0
        errors: list[str] = []
        events: list[dict[str, Any]] = []
        approved_utc = datetime.now(timezone.utc).isoformat()

        for position, import_row in enumerate(_read_csv(manifest_path), start=1):
            if import_row.get("status") not in TRAINABLE_IMPORT_STATUSES:
                continue
            label = str(import_row.get("approved_folder", "")).strip()
            if not is_trainable_taxonomy_label(label):
                errors.append(f"row {position}: invalid neural training label: {label or '<empty>'}")
                continue
            audio_path = Path(str(import_row.get("staged_path", ""))).expanduser()
            if not audio_path.is_file():
                errors.append(f"row {position}: staged audio is missing")
                continue
            try:
                portable_audio_path = _project_uri(self.project_root, audio_path)
                file_digest, decoded_digest, normalized_digest = _audio_hashes(audio_path)
            except (OSError, RuntimeError, ValueError) as exc:
                errors.append(f"row {position}: audio identity failed: {exc}")
                continue
            duplicate_group_id = _duplicate_group_id(file_digest, decoded_digest, normalized_digest)
            previous = existing_rows.get(duplicate_group_id)
            confirmation_count = int(previous.get("confirmation_count", "0") or 0) + 1 if previous else 1
            first_approved_utc = (
                str(previous.get("first_approved_utc", "")).strip() if previous else approved_utc
            ) or approved_utc
            previous_label = str(previous.get("approved_folder", "")).strip() if previous else ""
            if previous_label == label:
                reaffirmed_count += 1
                event_action = "reaffirmed"
            else:
                queued_count += 1
                event_action = "relabeled" if previous_label else "queued"
                if previous_label:
                    superseded_count += 1
            current_row = {
                "candidate_id": f"gui_{duplicate_group_id}",
                "source_path": portable_audio_path,
                "approved_folder": label,
                "label_source": "explicit GUI approved_folder",
                "source_kind": "recent_gui_correction",
                "source_group": "gui_feedback",
                "human_approved": "1",
                "file_sha256": file_digest,
                "decoded_audio_sha256": decoded_digest,
                "normalized_audio_sha256": normalized_digest,
                "duplicate_group_id": duplicate_group_id,
                "allowed_use": TRAINING_USE,
                "first_approved_utc": first_approved_utc,
                "last_approved_utc": approved_utc,
                "confirmation_count": str(confirmation_count),
                "source_manifest": _project_uri(self.project_root, manifest_path),
            }
            existing_rows[duplicate_group_id] = current_row
            events.append(
                {
                    "schema_version": 1,
                    "event_utc": approved_utc,
                    "action": event_action,
                    "previous_label": previous_label,
                    **current_row,
                }
            )

        self.inbox_root.mkdir(parents=True, exist_ok=True)
        if events:
            with self.event_log_path.open("a", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        _write_csv_atomic(
            self.current_manifest_path,
            [existing_rows[key] for key in sorted(existing_rows)],
            INBOX_FIELDS,
        )
        return NeuralIntakeSummary(
            current_manifest_path=self.current_manifest_path,
            event_log_path=self.event_log_path,
            queued_count=queued_count,
            reaffirmed_count=reaffirmed_count,
            superseded_count=superseded_count,
            errors=tuple(errors),
        )


class NeuralPrototypeTrainer:
    """Rebuild one provider index from frozen curated rows plus GUI intake."""

    def __init__(
        self,
        *,
        project_root: Path,
        provider: EmbeddingProvider,
        cache_root: Path,
        index_root: Path,
        pointer_path: Path,
        base_split_path: Path,
        inbox_path: Path,
        training_root: Path,
        sample_library_root: Path | None = None,
        max_prototypes_per_label: int = 6,
        keep_fraction: float = 0.95,
        production_ownership_enabled: bool = False,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.provider = provider
        self.cache = EmbeddingCache(Path(cache_root).expanduser().resolve())
        self.index_root = Path(index_root).expanduser().resolve()
        self.pointer_path = Path(pointer_path).expanduser().resolve()
        self.base_split_path = Path(base_split_path).expanduser().resolve()
        self.inbox_path = Path(inbox_path).expanduser().resolve()
        self.training_root = Path(training_root).expanduser().resolve()
        self.sample_library_root = (
            Path(sample_library_root).expanduser().resolve() if sample_library_root is not None else None
        )
        self.production_ownership_enabled = bool(production_ownership_enabled)
        self.builder = PrototypeIndexBuilder(
            max_prototypes_per_label=max_prototypes_per_label,
            keep_fraction=keep_fraction,
        )

    def rebuild(self) -> NeuralPrototypeBuildSummary:
        """Build and activate a new version while retaining older versions."""
        base_rows = _read_csv(self.base_split_path)
        inbox_rows = _read_csv(self.inbox_path)
        eligible_inbox_rows, pending_training_conflicts = _partition_confirmed_inbox_rows(
            base_rows,
            inbox_rows,
        )
        wanted_hashes = {
            str(row.get("file_sha256", "")).strip()
            for row in [*base_rows, *eligible_inbox_rows]
            if str(row.get("file_sha256", "")).strip()
        }
        training_hash_paths = _training_paths_by_hash(self.training_root, wanted_hashes)
        base_examples, missing_base = _examples_from_rows(
            base_rows,
            project_root=self.project_root,
            sample_library_root=self.sample_library_root,
            training_hash_paths=training_hash_paths,
            only_training_use=True,
        )
        correction_examples, missing_corrections = _examples_from_rows(
            eligible_inbox_rows,
            project_root=self.project_root,
            sample_library_root=self.sample_library_root,
            training_hash_paths=training_hash_paths,
            only_training_use=False,
        )
        examples = _merge_examples(base_examples, correction_examples)
        if not examples:
            return NeuralPrototypeBuildSummary(
                status="skipped",
                message="No trainable neural examples were available.",
                index_path=None,
                pointer_path=None,
                previous_index_path="",
                training_example_count=0,
                label_count=0,
                correction_predictions=(),
                missing_audio_hashes=tuple(sorted({*missing_base, *missing_corrections})),
                invalidated_evaluation_hashes=(),
                pending_training_conflicts=pending_training_conflicts,
                report_path=None,
            )

        training_signature = _training_signature(
            examples,
            provider_id=self.provider.provider_id,
            model_id=self.provider.model_id,
            max_prototypes_per_label=self.builder.max_prototypes_per_label,
            keep_fraction=self.builder.keep_fraction,
            production_ownership_enabled=self.production_ownership_enabled,
            pending_training_conflicts=pending_training_conflicts,
        )
        with _exclusive_rebuild_lock(self.index_root / ".rebuild.lock"):
            existing = self._matching_active_build(training_signature)
            if existing is not None:
                return existing
            return self._build_version(
                base_rows=base_rows,
                base_examples=base_examples,
                correction_examples=correction_examples,
                examples=examples,
                missing_audio_hashes=tuple(sorted({*missing_base, *missing_corrections})),
                training_signature=training_signature,
                pending_training_conflicts=pending_training_conflicts,
            )

    def _build_version(
        self,
        *,
        base_rows: Sequence[Mapping[str, str]],
        base_examples: Sequence[NeuralTrainingExample],
        correction_examples: Sequence[NeuralTrainingExample],
        examples: Sequence[NeuralTrainingExample],
        missing_audio_hashes: tuple[str, ...],
        training_signature: str,
        pending_training_conflicts: tuple[PendingTrainingConflict, ...],
    ) -> NeuralPrototypeBuildSummary:
        """Build one new version after acquiring the rebuild lock."""
        baseline_index = self._build_index(base_examples) if base_examples else None
        updated_index = self._build_index(examples)
        correction_predictions = self._correction_predictions(
            correction_examples,
            baseline_index=baseline_index,
            updated_index=updated_index,
        )
        invalidated_evaluation_hashes = _evaluation_overlap_hashes(base_rows, correction_examples)

        version = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")
        version_root = self.index_root / version
        index_path = version_root / "index"
        updated_index.save(index_path)
        self.pointer_path.parent.mkdir(parents=True, exist_ok=True)
        previous_index_path = (
            self.pointer_path.read_text(encoding="utf-8").strip() if self.pointer_path.is_file() else ""
        )
        _write_text_atomic(self.pointer_path, _project_relative_value(self.project_root, index_path) + "\n")
        _write_training_manifest(version_root / "training_manifest.csv", examples)
        _write_prediction_manifest(version_root / "correction_verification.csv", correction_predictions)
        report_path = version_root / "build_summary.json"
        report_payload = {
            "schema_version": 1,
            "status": "built",
            "provider_id": self.provider.provider_id,
            "model_id": self.provider.model_id,
            "index_path": _project_relative_value(self.project_root, index_path),
            "previous_index_path": previous_index_path,
            "training_example_count": len(examples),
            "label_count": len({example.label for example in examples}),
            "missing_audio_hashes": list(missing_audio_hashes),
            "training_signature": training_signature,
            "invalidated_evaluation_hashes": list(invalidated_evaluation_hashes),
            "correction_predictions": [asdict(row) for row in correction_predictions],
            "pending_training_conflicts": [asdict(row) for row in pending_training_conflicts],
            "source_name_policy": "audio bytes and explicit human labels only",
            "production_ownership_enabled": self.production_ownership_enabled,
            "exact_training_policy": EXACT_TRAINING_POLICY,
        }
        report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return NeuralPrototypeBuildSummary(
            status="built",
            message=f"Built {len(examples)} examples across {len(updated_index.metadata.label_example_counts)} labels.",
            index_path=index_path,
            pointer_path=self.pointer_path,
            previous_index_path=previous_index_path,
            training_example_count=len(examples),
            label_count=len(updated_index.metadata.label_example_counts),
            correction_predictions=correction_predictions,
            missing_audio_hashes=missing_audio_hashes,
            invalidated_evaluation_hashes=invalidated_evaluation_hashes,
            pending_training_conflicts=pending_training_conflicts,
            report_path=report_path,
        )

    def _matching_active_build(self, training_signature: str) -> NeuralPrototypeBuildSummary | None:
        """Return the active build when its content-only signature is unchanged."""
        if not self.pointer_path.is_file():
            return None
        pointer_value = self.pointer_path.read_text(encoding="utf-8").strip()
        index_path = Path(pointer_value).expanduser()
        if not index_path.is_absolute():
            index_path = self.project_root / index_path
        report_path = index_path.parent / "build_summary.json"
        if not report_path.is_file():
            return None
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            if payload.get("training_signature") != training_signature:
                return None
            predictions = tuple(CorrectionPrediction(**row) for row in payload.get("correction_predictions", []))
            return NeuralPrototypeBuildSummary(
                status="unchanged",
                message="Active neural index already matches the current content and labels.",
                index_path=index_path,
                pointer_path=self.pointer_path,
                previous_index_path=str(payload.get("previous_index_path", "")),
                training_example_count=int(payload.get("training_example_count", 0)),
                label_count=int(payload.get("label_count", 0)),
                correction_predictions=predictions,
                missing_audio_hashes=tuple(payload.get("missing_audio_hashes", [])),
                invalidated_evaluation_hashes=tuple(payload.get("invalidated_evaluation_hashes", [])),
                pending_training_conflicts=tuple(
                    PendingTrainingConflict(**row) for row in payload.get("pending_training_conflicts", [])
                ),
                report_path=report_path,
            )
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def _build_index(self, examples: Sequence[NeuralTrainingExample]) -> PrototypeIndex:
        embedded: dict[str, list[EmbeddingRecord]] = defaultdict(list)
        for example in examples:
            embedded[example.label].append(self.cache.get_or_compute(example.audio_path, self.provider))
        return self.builder.build(embedded)

    def _correction_predictions(
        self,
        correction_examples: Sequence[NeuralTrainingExample],
        *,
        baseline_index: PrototypeIndex | None,
        updated_index: PrototypeIndex,
    ) -> tuple[CorrectionPrediction, ...]:
        rows: list[CorrectionPrediction] = []
        for example in correction_examples:
            record = self.cache.get_or_compute(example.audio_path, self.provider)
            predicted_before = baseline_index.predict(record).predicted_label if baseline_index else ""
            prototype_after = updated_index.predict(record)
            after = updated_index.predict(record, exact_label=example.label)
            rows.append(
                CorrectionPrediction(
                    file_sha256=example.file_sha256,
                    approved_label=example.label,
                    predicted_before=predicted_before,
                    predicted_after=after.predicted_label,
                    second_after=after.second_label,
                    top_similarity_after=after.top_similarity,
                    margin_after=after.margin,
                    known_distribution_after=after.known_distribution,
                    prototype_predicted_after=prototype_after.predicted_label,
                )
            )
        return tuple(rows)


def configured_clap_trainer(project_root: Path) -> NeuralPrototypeTrainer:
    """Build the configured local CLAP trainer.

    Raises:
        FileNotFoundError: If configuration, model, split, or inbox is missing.
        ValueError: If the configured provider is unsupported.
    """
    root = Path(project_root).expanduser().resolve()
    config_path = root / DEFAULT_TRAINING_CONFIG
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not bool(config.get("enabled", False)):
        raise ValueError("incremental neural training is disabled")
    if str(config.get("provider", "")) != "clap":
        raise ValueError("only the configured CLAP lane is eligible for automatic GUI rebuilding")

    from .providers import HuggingFaceClapProvider

    model_path = root / str(config["model_path"])
    if not model_path.is_dir():
        raise FileNotFoundError(f"configured local CLAP model is missing: {model_path}")
    provider = HuggingFaceClapProvider(
        model_name_or_path=str(model_path),
        source_model_id=str(config["source_model_id"]),
        model_revision=str(config.get("model_revision", "")),
        device=str(config.get("device", "auto")),
        allow_network=False,
    )
    sample_library_root = _configured_sample_library_root(root, config)
    return NeuralPrototypeTrainer(
        project_root=root,
        provider=provider,
        cache_root=root / str(config["cache_root"]),
        index_root=root / str(config["index_root"]),
        pointer_path=root / str(config["pointer_path"]),
        base_split_path=root / str(config["base_split_path"]),
        inbox_path=root / str(config["inbox_path"]),
        training_root=root / str(config["training_root"]),
        sample_library_root=sample_library_root,
        max_prototypes_per_label=int(config.get("max_prototypes_per_label", 6)),
        keep_fraction=float(config.get("keep_fraction", 0.95)),
        production_ownership_enabled=bool(config.get("production_ownership_enabled", False)),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _training_signature(
    examples: Sequence[NeuralTrainingExample],
    *,
    provider_id: str,
    model_id: str,
    max_prototypes_per_label: int,
    keep_fraction: float,
    production_ownership_enabled: bool,
    pending_training_conflicts: Sequence[PendingTrainingConflict] = (),
) -> str:
    """Hash content identities, explicit labels, and model/build settings."""
    payload = {
        "provider_id": provider_id,
        "model_id": model_id,
        "max_prototypes_per_label": max_prototypes_per_label,
        "keep_fraction": keep_fraction,
        "production_ownership_enabled": production_ownership_enabled,
        "exact_training_policy": EXACT_TRAINING_POLICY,
        "examples": sorted((example.file_sha256, example.label) for example in examples),
        "pending_training_conflicts": sorted(
            (conflict.file_sha256, conflict.locked_label, conflict.proposed_label, conflict.confirmation_count)
            for conflict in pending_training_conflicts
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@contextmanager
def _exclusive_rebuild_lock(lock_path: Path) -> Iterator[None]:
    """Serialize neural index activation across concurrent GUI jobs."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _audio_hashes(path: Path) -> tuple[str, str, str]:
    file_digest = sha256_file(path)
    try:
        decoded_digest = decoded_audio_sha256(path)
    except (OSError, RuntimeError, ValueError):
        decoded_digest = ""
    try:
        normalized_digest = normalized_audio_sha256(path)
    except (OSError, RuntimeError, ValueError):
        normalized_digest = ""
    return file_digest, decoded_digest, normalized_digest


def _duplicate_group_id(file_digest: str, decoded_digest: str, normalized_digest: str) -> str:
    identity = normalized_digest or decoded_digest or file_digest
    return f"audio_group_{hashlib.sha256(identity.encode()).hexdigest()[:20]}"


def _project_uri(project_root: Path, path: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        relative = resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("neural training intake must reference staged project audio") from exc
    return f"project://{relative.as_posix()}"


def _configured_sample_library_root(project_root: Path, config: Mapping[str, Any]) -> Path | None:
    """Resolve an optional local sample-library root without committing it."""
    environment_name = str(config.get("sample_library_root_env", "")).strip()
    environment_value = os.environ.get(environment_name, "").strip() if environment_name else ""
    candidates: list[str] = []
    if environment_value:
        candidates.append(environment_value)
    pointer_value = str(config.get("sample_library_root_pointer_path", "")).strip()
    if pointer_value:
        pointer_path = Path(pointer_value).expanduser()
        if not pointer_path.is_absolute():
            pointer_path = project_root / pointer_path
        if pointer_path.is_file():
            candidates.append(pointer_path.read_text(encoding="utf-8").strip())
    configured_value = str(config.get("sample_library_root", "")).strip()
    if configured_value:
        candidates.append(configured_value)
    for value in candidates:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _resolve_audio_uri(
    project_root: Path,
    sample_library_root: Path | None,
    value: str,
) -> Path | None:
    if value.startswith("project://"):
        return _safe_uri_path(project_root, value.removeprefix("project://"))
    if value.startswith("sample-library://"):
        if sample_library_root is None:
            return None
        return _safe_uri_path(sample_library_root, value.removeprefix("sample-library://"))
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return None


def _safe_uri_path(root: Path, relative_value: str) -> Path | None:
    """Resolve a manifest URI below its configured operational root."""
    resolved_root = Path(root).expanduser().resolve()
    candidate = (resolved_root / relative_value).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        return None
    return candidate


def _training_paths_by_hash(training_root: Path, wanted_hashes: set[str]) -> dict[str, Path]:
    matches: dict[str, Path] = {}
    if not wanted_hashes or not training_root.is_dir():
        return matches
    for candidate in sorted(training_root.rglob("*")):
        if not candidate.is_file():
            continue
        try:
            digest = sha256_file(candidate)
        except OSError:
            continue
        if digest in wanted_hashes and digest not in matches:
            matches[digest] = candidate.resolve()
            if len(matches) == len(wanted_hashes):
                break
    return matches


def _examples_from_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    project_root: Path,
    sample_library_root: Path | None,
    training_hash_paths: Mapping[str, Path],
    only_training_use: bool,
) -> tuple[list[NeuralTrainingExample], set[str]]:
    examples: list[NeuralTrainingExample] = []
    missing: set[str] = set()
    for row in rows:
        if only_training_use and row.get("allowed_use") != TRAINING_USE:
            continue
        label = str(row.get("approved_folder") or row.get("intended_label") or "").strip()
        if not is_trainable_taxonomy_label(label):
            continue
        file_digest = str(row.get("file_sha256", "")).strip()
        audio_path = _resolve_audio_uri(
            project_root,
            sample_library_root,
            str(row.get("source_path") or row.get("audio_path") or ""),
        )
        if audio_path is None or not audio_path.is_file() or (file_digest and sha256_file(audio_path) != file_digest):
            audio_path = training_hash_paths.get(file_digest)
        if audio_path is None or not audio_path.is_file():
            if file_digest:
                missing.add(file_digest)
            continue
        decoded_digest = str(row.get("decoded_audio_sha256", "")).strip()
        normalized_digest = str(row.get("normalized_audio_sha256", "")).strip()
        duplicate_group_id = str(row.get("duplicate_group_id", "")).strip() or _duplicate_group_id(
            file_digest,
            decoded_digest,
            normalized_digest,
        )
        examples.append(
            NeuralTrainingExample(
                audio_path=audio_path,
                label=label,
                file_sha256=file_digest or sha256_file(audio_path),
                decoded_audio_sha256=decoded_digest,
                normalized_audio_sha256=normalized_digest,
                duplicate_group_id=duplicate_group_id,
                source_kind=str(row.get("source_kind", "")),
            )
        )
    return examples, missing


def _partition_confirmed_inbox_rows(
    base_rows: Sequence[Mapping[str, str]],
    inbox_rows: Sequence[Mapping[str, str]],
) -> tuple[list[Mapping[str, str]], tuple[PendingTrainingConflict, ...]]:
    """Hold a one-click relabel when it conflicts with a locked training seed.

    The comparison uses content identity plus explicit supervised labels. A
    second identical GUI approval confirms that the user intends to replace
    the older locked label. Source names and paths never participate.
    """
    locked_labels_by_identity: dict[str, str] = {}
    for row in base_rows:
        if row.get("allowed_use") != TRAINING_USE:
            continue
        label = str(row.get("intended_label") or row.get("approved_folder") or "").strip()
        if not is_trainable_taxonomy_label(label):
            continue
        for identity in _row_content_identities(row):
            locked_labels_by_identity.setdefault(identity, label)

    eligible: list[Mapping[str, str]] = []
    pending: list[PendingTrainingConflict] = []
    for row in inbox_rows:
        proposed_label = str(row.get("approved_folder") or row.get("intended_label") or "").strip()
        locked_label = next(
            (
                locked_labels_by_identity[identity]
                for identity in _row_content_identities(row)
                if identity in locked_labels_by_identity
            ),
            "",
        )
        try:
            confirmation_count = int(str(row.get("confirmation_count", "0") or "0"))
        except ValueError:
            confirmation_count = 0
        if (
            locked_label
            and proposed_label
            and locked_label != proposed_label
            and confirmation_count < LOCKED_SEED_OVERRIDE_CONFIRMATIONS
        ):
            pending.append(
                PendingTrainingConflict(
                    file_sha256=str(row.get("file_sha256", "")).strip(),
                    locked_label=locked_label,
                    proposed_label=proposed_label,
                    confirmation_count=confirmation_count,
                )
            )
            continue
        eligible.append(row)
    return eligible, tuple(sorted(pending, key=lambda conflict: conflict.file_sha256))


def _row_content_identities(row: Mapping[str, str]) -> tuple[str, ...]:
    """Return strongest-to-weakest content identities for one manifest row."""
    values = (
        str(row.get("normalized_audio_sha256", "")).strip(),
        str(row.get("decoded_audio_sha256", "")).strip(),
        str(row.get("file_sha256", "")).strip(),
        str(row.get("duplicate_group_id", "")).strip(),
    )
    return tuple(dict.fromkeys(value for value in values if value))


def _merge_examples(
    base_examples: Sequence[NeuralTrainingExample],
    corrections: Sequence[NeuralTrainingExample],
) -> list[NeuralTrainingExample]:
    by_content = {_example_content_identity(example): example for example in base_examples}
    for correction in corrections:
        by_content[_example_content_identity(correction)] = correction
    return [by_content[key] for key in sorted(by_content)]


def _example_content_identity(example: NeuralTrainingExample) -> str:
    """Return the strongest available decoded-audio identity for merging."""
    return (
        example.normalized_audio_sha256
        or example.decoded_audio_sha256
        or example.file_sha256
        or example.duplicate_group_id
    )


def _evaluation_overlap_hashes(
    base_rows: Sequence[Mapping[str, str]],
    corrections: Sequence[NeuralTrainingExample],
) -> tuple[str, ...]:
    correction_hashes = {example.file_sha256 for example in corrections}
    overlaps = {
        str(row.get("file_sha256", ""))
        for row in base_rows
        if row.get("allowed_use") != TRAINING_USE and str(row.get("file_sha256", "")) in correction_hashes
    }
    return tuple(sorted(overlaps))


def _project_relative_value(project_root: Path, path: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(project_root))
    except ValueError:
        return str(resolved)


def _write_training_manifest(path: Path, examples: Sequence[NeuralTrainingExample]) -> None:
    fields = (
        "file_sha256",
        "decoded_audio_sha256",
        "normalized_audio_sha256",
        "duplicate_group_id",
        "label",
        "source_kind",
    )
    rows = [
        {
            "file_sha256": example.file_sha256,
            "decoded_audio_sha256": example.decoded_audio_sha256,
            "normalized_audio_sha256": example.normalized_audio_sha256,
            "duplicate_group_id": example.duplicate_group_id,
            "label": example.label,
            "source_kind": example.source_kind,
        }
        for example in examples
    ]
    _write_csv_atomic(path, rows, fields)


def _write_prediction_manifest(path: Path, rows: Sequence[CorrectionPrediction]) -> None:
    fields = tuple(CorrectionPrediction.__dataclass_fields__)
    _write_csv_atomic(path, [asdict(row) for row in rows], fields)
