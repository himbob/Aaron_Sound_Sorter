"""Human-facing cluster review and category-coverage services.

The browser page built on this module intentionally keeps the primary workflow
small: listen to representative audio, choose a category, and save an explicit
decision. Detailed neural evidence remains available as optional context.
"""

from __future__ import annotations

import csv
import hashlib
import json
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aaron_sound_sorter.neural_audio.calibration_rebuild import rebuild_confidence_calibration
from aaron_sound_sorter.neural_audio.cluster_approval import (
    ClusterApprovalImportSummary,
    import_cluster_approval,
)
from aaron_sound_sorter.neural_audio.prototype_index import PrototypeIndex
from aaron_sound_sorter.neural_audio.runtime import run_configured_neural_rebuild
from aaron_sound_sorter.taxonomy_contracts import (
    canonicalize_taxonomy_label,
    is_trainable_taxonomy_label,
)
from aaron_sound_sorter.taxonomy_registry import TaxonomyRegistry

TRAINABLE_CLUSTER_ACTIONS = frozenset({"approve_safe_core", "approve_all"})
REVIEW_ONLY_CLUSTER_ACTIONS = frozenset({"choose_parent", "split", "reject", "mixed"})
VALID_CLUSTER_ACTIONS = TRAINABLE_CLUSTER_ACTIONS | REVIEW_ONLY_CLUSTER_ACTIONS


@dataclass(frozen=True)
class ClusterPackSession:
    """One loaded source-name-blind cluster review pack.

    Args:
        pack_id: Stable identifier used by local browser routes.
        pack_root: Root containing the manifest and hash-named audio.
        manifest_path: Validated cluster manifest location.
        clusters: Source-name-blind manifest rows.
        audio_paths_by_hash: Validated audio locators keyed by content hash.

    Important Constraints:
        Audio paths are used only for playback and copying. Their text never
        influences cluster suggestions or training labels.
    """

    pack_id: str
    pack_root: Path
    manifest_path: Path
    clusters: tuple[dict[str, Any], ...]
    audio_paths_by_hash: dict[str, Path] = field(repr=False, compare=False)


@dataclass
class PrototypeRebuildJob:
    """Human-readable state for one explicit background prototype rebuild."""

    job_id: str
    status: str = "queued"
    message: str = "Preparing the updated neural brain…"
    report_dir: str = ""
    index_path: str = ""
    training_example_count: int = 0
    label_count: int = 0
    error: str = ""


class LearningCenterService:
    """Serve simple review actions over explicit local neural artifacts.

    Args:
        project_root: Repository root containing taxonomy and local artifacts.

    Side Effects:
        Loads local manifests and writes explicit approval/import records only
        after a corresponding user action.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.sessions: dict[str, ClusterPackSession] = {}
        self.rebuild_jobs: dict[str, PrototypeRebuildJob] = {}

    def coverage_payload(self) -> dict[str, Any]:
        """Return all canonical categories with compact readiness evidence."""
        registry = TaxonomyRegistry.load(
            self.project_root / "config" / "canonical_taxonomy.json",
            self.project_root / "config" / "taxonomy_aliases.json",
        )
        index = self._active_prototype_index()
        heldout = self._heldout_metrics()
        calibration = self.calibration_status()
        index_created = index.metadata.created_utc if index is not None else ""
        rows: list[dict[str, Any]] = []
        for category_path in registry.manual_labels():
            category = registry.categories[category_path]
            example_count = int(index.metadata.label_example_counts.get(category_path, 0)) if index else 0
            prototype_count = int(index.metadata.label_prototype_counts.get(category_path, 0)) if index else 0
            metric = heldout.get(category_path, {})
            heldout_count = _safe_int(metric.get("heldout_count"))
            top1_accuracy = _optional_float(metric.get("top1_accuracy"))
            readiness, readiness_reason = category_readiness(
                example_count=example_count,
                prototype_count=prototype_count,
                heldout_count=heldout_count,
                heldout_top1=top1_accuracy,
            )
            rows.append(
                {
                    "path": category_path,
                    "display_name": category.display_name,
                    "manual_selection_enabled": category.manual_selection_enabled,
                    "training_example_count": example_count,
                    "prototype_count": prototype_count,
                    "heldout_example_count": heldout_count,
                    "heldout_top1_accuracy": top1_accuracy,
                    "readiness_tier": readiness,
                    "readiness_reason": readiness_reason,
                    "automatic_classification_status": (
                        "calibrated_candidate"
                        if readiness == "A" and calibration.get("status") == "built"
                        else "manual_or_guarded"
                    ),
                    "calibration_status": calibration.get("status", "not_built"),
                    "last_training_date": index_created,
                    "common_confusions": list(metric.get("common_confusions", [])),
                    "fallback_parent": category.fallback_parent,
                }
            )
        tier_counts = Counter(str(row["readiness_tier"]) for row in rows)
        return {
            "taxonomy_version": registry.taxonomy_version,
            "category_count": len(rows),
            "tier_counts": dict(sorted(tier_counts.items())),
            "calibration": calibration,
            "categories": rows,
        }

    def open_cluster_pack(self, raw_path: str) -> dict[str, Any]:
        """Load a cluster manifest and return simple browser-safe records."""
        requested = Path(str(raw_path)).expanduser().resolve()
        manifest_path = requested / "cluster_manifest.json" if requested.is_dir() else requested
        if not manifest_path.is_file():
            raise FileNotFoundError(f"cluster manifest not found: {manifest_path}")
        payload = _load_json_object(manifest_path)
        raw_clusters = payload.get("clusters", [])
        if not isinstance(raw_clusters, list):
            raise ValueError("cluster manifest must contain a clusters list")
        pack_root = manifest_path.parent.resolve()
        clusters: list[dict[str, Any]] = []
        audio_paths: dict[str, Path] = {}
        for raw_cluster in raw_clusters:
            cluster = _validated_cluster(raw_cluster, pack_root)
            clusters.append(cluster)
            for file_hash, relative_path in cluster["member_paths_by_hash"].items():
                audio_paths[file_hash] = _safe_pack_member(pack_root, relative_path, file_hash)
        pack_id = hashlib.sha256(str(manifest_path).encode("utf-8")).hexdigest()[:16]
        session = ClusterPackSession(pack_id, pack_root, manifest_path, tuple(clusters), audio_paths)
        self.sessions[pack_id] = session
        return {
            "pack_id": pack_id,
            "manifest_path": str(manifest_path),
            "cluster_count": len(clusters),
            "member_count": sum(_safe_int(cluster.get("member_count")) for cluster in clusters),
            "clusters": [cluster_browser_payload(pack_id, cluster) for cluster in clusters],
        }

    def audio_path(self, pack_id: str, file_sha256: str) -> Path:
        """Resolve one hash-addressed review audio file for local playback."""
        session = self.sessions.get(str(pack_id))
        if session is None:
            raise KeyError("unknown cluster review pack")
        audio_path = session.audio_paths_by_hash.get(str(file_sha256))
        if audio_path is None or not audio_path.is_file():
            raise FileNotFoundError("cluster audio is unavailable")
        return audio_path

    def save_cluster_decision(
        self,
        *,
        pack_id: str,
        cluster_id: str,
        action: str,
        approved_category: str,
        reviewed_by: str,
    ) -> dict[str, Any]:
        """Persist one explicit human cluster decision without training."""
        session = self._session(pack_id)
        cluster = _cluster_by_id(session, cluster_id)
        normalized_action = str(action).strip()
        if normalized_action not in VALID_CLUSTER_ACTIONS:
            raise ValueError(f"unsupported cluster action: {normalized_action}")
        category = canonicalize_taxonomy_label(approved_category)
        if normalized_action in TRAINABLE_CLUSTER_ACTIONS and not is_trainable_taxonomy_label(category):
            raise ValueError("safe-core approval requires a complete trainable category")
        if normalized_action == "approve_safe_core":
            approved_hashes = tuple(cluster["safe_core_hashes"])
            approval_scope = "safe_core"
        elif normalized_action == "approve_all":
            approved_hashes = tuple(cluster["member_hashes"])
            approval_scope = "all_members"
        else:
            approved_hashes = ()
            approval_scope = "review_only"
        decision = {
            "schema_version": 1,
            "cluster_id": cluster_id,
            "cluster_manifest_path": str(session.manifest_path),
            "decision_action": normalized_action,
            "approval_scope": approval_scope,
            "allow_non_core": normalized_action == "approve_all",
            "approved_category": category,
            "approved_safe_core_hashes": list(approved_hashes),
            "rejected_hashes": list(cluster["member_hashes"]) if normalized_action in {"reject", "mixed"} else [],
            "boundary_review_required": (
                [] if normalized_action == "approve_all" else list(cluster["boundary_hashes"])
            ),
            "outlier_review_required": ([] if normalized_action == "approve_all" else list(cluster["outlier_hashes"])),
            "reviewed_by": str(reviewed_by).strip() or "Aaron",
            "reviewed_at_utc": _utc_timestamp(),
            "review_version": 1,
            "training_imported": False,
        }
        decision_root = self.project_root / "neural_artifacts" / "approved_cluster_manifests" / session.pack_id
        decision_root.mkdir(parents=True, exist_ok=True)
        decision_path = decision_root / f"{cluster_id}.json"
        _atomic_write_json(decision_path, decision)
        return {
            "status": "saved",
            "message": cluster_decision_message(normalized_action, len(approved_hashes)),
            "decision_path": str(decision_path),
            "training_ready": bool(approved_hashes),
            "training_imported": False,
        }

    def import_saved_decision(self, decision_path: str) -> dict[str, Any]:
        """Import one separately saved approval into the durable inbox."""
        approval_path = Path(str(decision_path)).expanduser().resolve()
        if not approval_path.is_relative_to(self.project_root / "neural_artifacts"):
            raise ValueError("approval must be a local neural artifact")
        run_root = self.project_root / "neural_artifacts" / "approved_cluster_imports"
        output_dir = run_root / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000:06d}"
        summary = import_cluster_approval(self.project_root, approval_path, output_dir)
        self._mark_decision_imported(approval_path, summary)
        return cluster_import_payload(summary)

    def calibration_status(self) -> dict[str, Any]:
        """Return honest local calibration readiness without inventing scores."""
        status_path = self.project_root / "neural_artifacts" / "calibration" / "current" / "calibration_status.json"
        if status_path.is_file():
            payload = _load_json_object(status_path)
            payload["status_path"] = str(status_path)
            return payload
        feedback_path = self.project_root / "neural_artifacts" / "calibration_feedback" / "review_feedback.jsonl"
        unique_hashes: set[str] = set()
        accepted = 0
        rejected = 0
        if feedback_path.is_file():
            latest: dict[str, bool] = {}
            for line in feedback_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                latest[str(row.get("file_sha256", ""))] = bool(row.get("accepted", False))
            unique_hashes = set(latest) - {""}
            accepted = sum(latest[file_hash] for file_hash in unique_hashes)
            rejected = len(unique_hashes) - accepted
        return {
            "status": "insufficient_data",
            "reviewed_unique_hash_count": len(unique_hashes),
            "accepted_count": accepted,
            "rejected_count": rejected,
            "message": "Calibration is not active until reviewed accepted and corrected outcomes are sufficient.",
        }

    def rebuild_calibration(self) -> dict[str, object]:
        """Rebuild calibration from durable GUI reviews, excluding held-out hashes."""
        feedback_path = self.project_root / "neural_artifacts" / "calibration_feedback" / "review_feedback.jsonl"
        output_dir = self.project_root / "neural_artifacts" / "calibration" / "current"
        excluded = self._heldout_hashes()
        return rebuild_confidence_calibration(
            feedback_path,
            output_dir,
            excluded_hashes=excluded,
        )

    def start_prototype_rebuild(self) -> dict[str, Any]:
        """Start one explicit prototype rebuild without blocking the browser."""
        for existing in self.rebuild_jobs.values():
            if existing.status in {"queued", "running"}:
                return prototype_rebuild_payload(existing)
        job_id = uuid.uuid4().hex
        report_dir = (
            self.project_root
            / "neural_artifacts"
            / "learning_center_rebuilds"
            / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{job_id[:6]}"
        )
        job = PrototypeRebuildJob(job_id=job_id, report_dir=str(report_dir))
        self.rebuild_jobs[job_id] = job
        threading.Thread(
            target=self._run_prototype_rebuild,
            args=(job,),
            name=f"neural-rebuild-{job_id[:8]}",
            daemon=True,
        ).start()
        return prototype_rebuild_payload(job)

    def prototype_rebuild_status(self, job_id: str) -> dict[str, Any]:
        """Return compact status for one explicit prototype rebuild."""
        job = self.rebuild_jobs.get(str(job_id))
        if job is None:
            raise KeyError("unknown neural rebuild job")
        return prototype_rebuild_payload(job)

    def _run_prototype_rebuild(self, job: PrototypeRebuildJob) -> None:
        job.status = "running"
        job.message = "Learning from the approved training inbox…"
        try:
            summary = run_configured_neural_rebuild(self.project_root, Path(job.report_dir))
            neural_status = str(summary.get("status", "error"))
            if neural_status not in {"built", "unchanged"}:
                raise RuntimeError(str(summary.get("message", f"neural rebuild returned {neural_status}")))
            job.index_path = str(summary.get("index_path", ""))
            job.training_example_count = _safe_int(summary.get("training_example_count"))
            job.label_count = _safe_int(summary.get("label_count"))
            job.status = "done"
            verb = "Built" if neural_status == "built" else "Checked"
            job.message = (
                f"{verb} the neural brain from {job.training_example_count} approved sounds "
                f"across {job.label_count} categories."
            )
        except Exception as exc:
            job.status = "error"
            job.error = str(exc)
            job.message = "The neural rebuild did not finish. Existing production prototypes were kept."

    def _active_prototype_index(self) -> PrototypeIndex | None:
        pointer = self.project_root / "config" / "runtime" / "neural_clap_index_path.txt"
        if not pointer.is_file():
            return None
        raw_path = Path(pointer.read_text(encoding="utf-8").strip()).expanduser()
        index_path = raw_path if raw_path.is_absolute() else self.project_root / raw_path
        try:
            return PrototypeIndex.load(index_path)
        except (FileNotFoundError, OSError, ValueError):
            return None

    def _heldout_metrics(self) -> dict[str, dict[str, Any]]:
        evaluations = sorted(
            self.project_root.glob("neural_artifacts/**/clap_heldout_evaluation.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not evaluations:
            return {}
        with evaluations[0].open(encoding="utf-8", newline="") as handle:
            return {
                str(row.get("label", "")): dict(row)
                for row in csv.DictReader(handle)
                if str(row.get("label", "")).strip()
            }

    def _heldout_hashes(self) -> set[str]:
        manifests = list(self.project_root.glob("neural_artifacts/**/heldout_eval.csv"))
        manifests.extend(self.project_root.glob("neural_artifacts/**/*heldout_predictions.csv"))
        manifests.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        if not manifests:
            return set()
        heldout_hashes: set[str] = set()
        for manifest in manifests:
            with manifest.open(encoding="utf-8", newline="") as handle:
                heldout_hashes.update(str(row.get("file_sha256", "")).strip() for row in csv.DictReader(handle))
        return heldout_hashes - {""}

    def _session(self, pack_id: str) -> ClusterPackSession:
        session = self.sessions.get(str(pack_id))
        if session is None:
            raise KeyError("unknown cluster review pack")
        return session

    @staticmethod
    def _mark_decision_imported(
        approval_path: Path,
        summary: ClusterApprovalImportSummary,
    ) -> None:
        payload = _load_json_object(approval_path)
        payload["training_imported"] = True
        payload["training_imported_at_utc"] = _utc_timestamp()
        payload["training_import_manifest"] = str(summary.import_manifest_path)
        payload["rollback_record"] = str(summary.rollback_record_path)
        _atomic_write_json(approval_path, payload)


def category_readiness(
    *,
    example_count: int,
    prototype_count: int,
    heldout_count: int,
    heldout_top1: float | None,
) -> tuple[str, str]:
    """Return a concise display-only category readiness tier."""
    if heldout_count >= 3 and heldout_top1 is not None and heldout_top1 >= 0.80 and example_count >= 8:
        return "A", "held-out evidence is promising; calibrated authority may be considered"
    if example_count >= 5 and prototype_count >= 1:
        return "B", "trained, but held-out coverage or calibration is incomplete"
    if example_count >= 2:
        return "C", "prototype suggestions only; add varied approved examples"
    return "D", "manual category; needs at least two distinct approved examples"


def cluster_browser_payload(pack_id: str, cluster: dict[str, Any]) -> dict[str, Any]:
    """Return one cluster record with hash-addressed local playback URLs."""
    representative_hashes = [
        ("Center", str(cluster.get("center_hash", ""))),
        *((f"Typical {position}", str(file_hash)) for position, file_hash in enumerate(cluster["typical_hashes"], 1)),
        ("Boundary", str(cluster.get("boundary_hash", ""))),
        ("Outlier", str(cluster.get("outlier_hash", ""))),
    ]
    representatives = [
        {
            "role": role,
            "file_sha256": file_hash,
            "audio_url": f"/api/learning/audio/{pack_id}/{file_hash}",
        }
        for role, file_hash in representative_hashes
        if file_hash
    ]
    return {key: value for key, value in cluster.items() if key not in {"member_paths_by_hash"}} | {
        "representatives": representatives
    }


def cluster_decision_message(action: str, approved_count: int) -> str:
    """Return plain-English confirmation for one cluster action."""
    messages = {
        "approve_safe_core": f"Saved approval for {approved_count} safe-core sounds. Training has not run yet.",
        "approve_all": f"Saved explicit approval for all {approved_count} sounds. Training has not run yet.",
        "choose_parent": "Saved a parent-only decision; no detailed training will be added.",
        "split": "Marked this cluster for splitting; no training will be added.",
        "reject": "Rejected this cluster; no training will be added.",
        "mixed": "Marked this cluster mixed or unusable; no training will be added.",
    }
    return messages[action]


def cluster_import_payload(summary: ClusterApprovalImportSummary) -> dict[str, Any]:
    """Return a compact browser confirmation for a separate training import."""
    return {
        "status": "imported",
        "cluster_id": summary.cluster_id,
        "approved_category": summary.approved_category,
        "imported_count": summary.imported_count,
        "rollback_record": str(summary.rollback_record_path),
        "message": (
            f"Added {summary.imported_count} approved sounds to the training inbox. "
            "The production prototypes have not been rebuilt yet."
        ),
    }


def prototype_rebuild_payload(job: PrototypeRebuildJob) -> dict[str, Any]:
    """Return browser-safe status without exposing training source paths."""
    return {
        "job_id": job.job_id,
        "status": job.status,
        "message": job.message,
        "report_dir": job.report_dir,
        "index_path": job.index_path,
        "training_example_count": job.training_example_count,
        "label_count": job.label_count,
        "error": job.error,
    }


def _validated_cluster(raw_cluster: object, pack_root: Path) -> dict[str, Any]:
    if not isinstance(raw_cluster, dict):
        raise ValueError("each cluster record must be an object")
    cluster = dict(raw_cluster)
    cluster_id = str(cluster.get("cluster_id", "")).strip()
    if not cluster_id:
        raise ValueError("cluster record is missing cluster_id")
    member_paths = cluster.get("member_paths_by_hash", {})
    if not isinstance(member_paths, dict):
        raise ValueError(f"cluster {cluster_id} has invalid member paths")
    member_hashes = tuple(str(value) for value in cluster.get("member_hashes", []))
    if set(member_hashes) != set(str(key) for key in member_paths):
        raise ValueError(f"cluster {cluster_id} member hashes and paths disagree")
    for file_hash, relative_path in member_paths.items():
        _safe_pack_member(pack_root, relative_path, str(file_hash))
    cluster["member_hashes"] = member_hashes
    cluster["safe_core_hashes"] = tuple(str(value) for value in cluster.get("safe_core_hashes", []))
    cluster["boundary_hashes"] = tuple(str(value) for value in cluster.get("boundary_hashes", []))
    cluster["outlier_hashes"] = tuple(str(value) for value in cluster.get("outlier_hashes", []))
    cluster["typical_hashes"] = tuple(str(value) for value in cluster.get("typical_hashes", []))
    return cluster


def _safe_pack_member(pack_root: Path, relative_path: object, file_sha256: str) -> Path:
    if len(file_sha256) != 64:
        raise ValueError("cluster member hash must be SHA-256")
    member_path = (pack_root / str(relative_path)).resolve()
    if not member_path.is_relative_to(pack_root) or not member_path.is_file():
        raise FileNotFoundError(f"cluster member is unavailable: {file_sha256}")
    return member_path


def _cluster_by_id(session: ClusterPackSession, cluster_id: str) -> dict[str, Any]:
    for cluster in session.clusters:
        if str(cluster.get("cluster_id")) == str(cluster_id):
            return cluster
    raise KeyError(f"cluster is absent from loaded pack: {cluster_id}")


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_int(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if str(value).strip() else None
    except (TypeError, ValueError):
        return None
