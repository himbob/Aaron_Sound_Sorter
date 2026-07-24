#!/usr/bin/env python3
"""Build compact provenance, contamination, readiness, and encoder evidence.

The output is an explicit persistent artifact folder so ``make clean`` can
remove reproducible run data without deleting the compact evidence needed for
the next migration pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.neural_audio.contracts import LaneEvaluation  # noqa: E402
from aaron_sound_sorter.neural_audio.curation import (  # noqa: E402
    CALIBRATION_USE,
    FINAL_HELDOUT_USE,
    TRAINING_USE,
    VALIDATION_USE,
    ProvenanceCandidate,
    assign_leakage_safe_uses,
    readiness_tier,
)
from aaron_sound_sorter.neural_audio.hashing import (  # noqa: E402
    decoded_audio_sha256,
    normalized_audio_sha256,
    sha256_file,
)
from aaron_sound_sorter.neural_audio.lane_selection import write_lane_leaderboard  # noqa: E402
from aaron_sound_sorter.taxonomy_contracts import (  # noqa: E402
    is_trainable_taxonomy_label,
)

AUDIO_SUFFIXES = frozenset({".wav", ".aif", ".aiff", ".flac", ".ogg", ".au", ".mp3", ".m4a"})
STRUCTURE_FOLDER_MAP = {
    "_ONE_SHOTS": "One Shots",
    "_LOOPS": "Loops",
    "_LONG_FX": "Long FX",
}
QUARANTINE_STATUSES = frozenset(
    {
        "duplicate_hash_label_conflict",
        "cross_label_conflict",
        "possible_label_outlier",
    }
)
REVIEW_STATUSES = QUARANTINE_STATUSES | frozenset(
    {
        "ambiguous_cross_label_support",
        "limited_support",
        "prototype_only_singleton",
    }
)
STATUS_PRIORITY = {
    "duplicate_hash_label_conflict": 100,
    "cross_label_conflict": 90,
    "possible_label_outlier": 80,
    "ambiguous_cross_label_support": 70,
    "limited_support": 50,
    "prototype_only_singleton": 40,
    "supported": 0,
}
PERSISTENT_TEXT_SUFFIXES = frozenset({".csv", ".json", ".md"})


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file when it exists."""
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    """Write stable dictionary rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sanitize_persistent_paths(output_dir: Path, project_root: Path, samples_root: Path) -> None:
    """Replace private absolute roots with portable provenance references."""
    replacements = (
        (f"{samples_root.resolve()}/", "sample-library://"),
        (str(samples_root.resolve()), "sample-library://"),
        (f"{project_root.resolve()}/", "project://"),
        (str(project_root.resolve()), "project://"),
    )
    for artifact_path in output_dir.rglob("*"):
        if not artifact_path.is_file() or artifact_path.suffix.lower() not in PERSISTENT_TEXT_SUFFIXES:
            continue
        original = artifact_path.read_text(encoding="utf-8", errors="replace")
        sanitized = original
        for private_root, portable_root in replacements:
            sanitized = sanitized.replace(private_root, portable_root)
        if sanitized != original:
            artifact_path.write_text(sanitized, encoding="utf-8")


def public_label_from_training_path(path: Path, training_root: Path) -> str:
    """Return the explicit taxonomy label represented by a curated slot."""
    parts = list(path.relative_to(training_root).parent.parts)
    if not parts:
        return ""
    structure = STRUCTURE_FOLDER_MAP.get(parts[-1])
    if structure:
        parts[-1] = structure
    label = "/".join(parts)
    return label if is_trainable_taxonomy_label(label) else ""


def audit_rows_by_hash(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Group existing CLAP trainer-audit rows by byte digest."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        digest = str(row.get("file_sha256", ""))
        if digest:
            grouped[digest].append(row)
    return dict(grouped)


def strongest_audit(rows: list[dict[str, str]]) -> tuple[str, str]:
    """Return the most serious contamination status and merged reasons."""
    if not rows:
        return "not_audited", "no real-embedding leave-one-out row is available"
    selected = sorted(
        rows,
        key=lambda row: (
            -STATUS_PRIORITY.get(str(row.get("status", "")), 10),
            str(row.get("label", "")),
        ),
    )[0]
    reasons = sorted({str(row.get("reasons", "")) for row in rows if str(row.get("reasons", ""))})
    return str(selected.get("status", "")), "; ".join(reasons)


def cached_embedding_providers(cache_root: Path) -> dict[str, tuple[str, ...]]:
    """Return cached provider IDs grouped by audio byte digest."""
    providers: dict[str, set[str]] = defaultdict(set)
    if not cache_root.is_dir():
        return {}
    for cache_path in cache_root.rglob("*.npz"):
        try:
            payload = np.load(cache_path, allow_pickle=False)
            metadata = json.loads(str(payload["metadata_json"].item()))
            digest = str(metadata.get("file_sha256", ""))
            provider_id = str(metadata.get("provider_id", ""))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if digest and provider_id:
            providers[digest].add(provider_id)
    return {digest: tuple(sorted(values)) for digest, values in providers.items()}


def audio_hashes(path: Path) -> tuple[str, str, str]:
    """Return byte, decoded, and normalized digests for one audio path."""
    byte_digest = sha256_file(path)
    try:
        decoded_digest = decoded_audio_sha256(path)
    except (OSError, RuntimeError, ValueError):
        decoded_digest = ""
    try:
        normalized_digest = normalized_audio_sha256(path)
    except (OSError, RuntimeError, ValueError):
        normalized_digest = ""
    return byte_digest, decoded_digest, normalized_digest


def source_group_for_path(path: Path, samples_root: Path) -> str:
    """Return display-only pack/panel provenance for a located audio path."""
    try:
        relative = path.resolve().relative_to(samples_root.resolve())
    except ValueError:
        return path.parent.name or "unknown_source"
    return relative.parts[0] if relative.parts else "unknown_source"


def load_legacy_candidates(
    training_root: Path,
    audit_by_hash: dict[str, list[dict[str, str]]],
    embeddings_by_hash: dict[str, tuple[str, ...]],
) -> list[ProvenanceCandidate]:
    """Inventory legacy curated slots without granting automatic trust."""
    candidates: list[ProvenanceCandidate] = []
    for position, path in enumerate(
        sorted(
            candidate
            for candidate in training_root.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in AUDIO_SUFFIXES
        ),
        start=1,
    ):
        label = public_label_from_training_path(path, training_root)
        if not label:
            continue
        byte_digest, decoded_digest, normalized_digest = audio_hashes(path)
        status, warnings = strongest_audit(audit_by_hash.get(byte_digest, []))
        candidates.append(
            ProvenanceCandidate(
                candidate_id=f"legacy_{position:05d}_{byte_digest[:12]}",
                audio_path=str(path.resolve()),
                intended_label=label,
                label_source="legacy locked curated slot; label not automatically trusted",
                source_kind="legacy_curated_candidate",
                source_group="locked_curated_v1_unknown_origin",
                human_approved=False,
                file_sha256=byte_digest,
                decoded_audio_sha256=decoded_digest,
                normalized_audio_sha256=normalized_digest,
                contamination_status=status,
                contamination_warnings=warnings,
                embedding_providers=embeddings_by_hash.get(byte_digest, ()),
            )
        )
    return candidates


def load_seed_candidates(
    manifest_path: Path,
    embeddings_by_hash: dict[str, tuple[str, ...]],
) -> list[ProvenanceCandidate]:
    """Load explicit locked human-seed supervision."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates: list[ProvenanceCandidate] = []
    for position, row in enumerate(payload.get("cases", []), start=1):
        path = Path(str(row.get("audio_path", "")))
        path = path if path.is_absolute() else PROJECT_ROOT / path
        label = str(row.get("training_label", "")).strip()
        if not path.is_file() or not is_trainable_taxonomy_label(label):
            continue
        byte_digest, decoded_digest, normalized_digest = audio_hashes(path)
        candidates.append(
            ProvenanceCandidate(
                candidate_id=f"seed_{position:04d}_{byte_digest[:12]}",
                audio_path=str(path.resolve()),
                intended_label=label,
                label_source=f"explicit label in {manifest_path.name}",
                source_kind="locked_human_seed",
                source_group=str(payload.get("panel_name", manifest_path.stem)),
                human_approved=True,
                file_sha256=byte_digest,
                decoded_audio_sha256=decoded_digest,
                normalized_audio_sha256=normalized_digest,
                contamination_status="explicit_human_seed",
                embedding_providers=embeddings_by_hash.get(byte_digest, ()),
            )
        )
    return candidates


def load_gui_candidates(
    manifest_path: Path,
    embeddings_by_hash: dict[str, tuple[str, ...]],
    samples_root: Path,
) -> list[ProvenanceCandidate]:
    """Load explicit recent GUI corrections as training-only supervision."""
    candidates: list[ProvenanceCandidate] = []
    for position, row in enumerate(read_csv(manifest_path), start=1):
        path = Path(str(row.get("source_path", "")))
        label = str(row.get("approved_folder", "")).strip()
        if not path.is_file() or not is_trainable_taxonomy_label(label):
            continue
        byte_digest, decoded_digest, normalized_digest = audio_hashes(path)
        candidates.append(
            ProvenanceCandidate(
                candidate_id=f"gui_{position:04d}_{byte_digest[:12]}",
                audio_path=str(path.resolve()),
                intended_label=label,
                label_source=f"explicit approved_folder in {manifest_path.parent.name}",
                source_kind="recent_gui_correction",
                source_group=source_group_for_path(path, samples_root),
                human_approved=True,
                file_sha256=byte_digest,
                decoded_audio_sha256=decoded_digest,
                normalized_audio_sha256=normalized_digest,
                contamination_status="recent_human_correction",
                embedding_providers=embeddings_by_hash.get(byte_digest, ()),
            )
        )
    return candidates


def candidate_csv_rows(candidates: list[ProvenanceCandidate]) -> list[dict[str, Any]]:
    """Convert candidate contracts into flat provenance rows."""
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = asdict(candidate)
        row["human_approved"] = int(candidate.human_approved)
        row["duplicate_group_id"] = candidate.duplicate_group_id
        row["embedding_providers"] = "|".join(candidate.embedding_providers)
        rows.append(row)
    return rows


def contamination_action(status: str) -> str:
    """Return a reversible proposal for one trainer status."""
    if status in QUARANTINE_STATUSES:
        return "propose_quarantine_pending_human_verdict"
    if status in REVIEW_STATUSES:
        return "exclude_pending_human_review"
    if status == "supported":
        return "candidate_only_until_provenance_review"
    return "exclude_not_audited"


def is_legacy_brain_warning(status: str) -> bool:
    """Return whether a legacy brain audit status requires human attention."""
    return status not in {"clean", "ok"}


def write_contamination_outputs(
    output_dir: Path,
    trainer_rows: list[dict[str, str]],
    legacy_candidates: list[ProvenanceCandidate],
    brain_rows: list[dict[str, str]],
) -> None:
    """Write exact required trainer contamination and quarantine artifacts."""
    paths_by_hash: dict[str, list[str]] = defaultdict(list)
    for candidate in legacy_candidates:
        paths_by_hash[candidate.file_sha256].append(candidate.audio_path)
    report_rows: list[dict[str, Any]] = []
    for row in trainer_rows:
        enriched: dict[str, Any] = dict(row)
        enriched["audio_paths_json"] = json.dumps(sorted(paths_by_hash.get(row.get("file_sha256", ""), [])))
        enriched["proposed_action"] = contamination_action(str(row.get("status", "")))
        enriched["reversible"] = 1
        report_rows.append(enriched)
    write_csv(output_dir / "trainer_contamination_report.csv", report_rows)

    by_group: dict[str, list[ProvenanceCandidate]] = defaultdict(list)
    for candidate in legacy_candidates:
        by_group[candidate.duplicate_group_id].append(candidate)
    conflict_rows: list[dict[str, Any]] = []
    for group_id, candidates in sorted(by_group.items()):
        labels = sorted({candidate.intended_label for candidate in candidates})
        if len(labels) <= 1:
            continue
        conflict_rows.append(
            {
                "duplicate_group_id": group_id,
                "occurrence_count": len(candidates),
                "assigned_label_count": len(labels),
                "assigned_labels_json": json.dumps(labels),
                "file_hashes_json": json.dumps(sorted({candidate.file_sha256 for candidate in candidates})),
                "audio_paths_json": json.dumps(sorted(candidate.audio_path for candidate in candidates)),
                "proposed_action": "quarantine_all_occurrences_until_one_human_label_is_selected",
            }
        )
    write_csv(output_dir / "trainer_conflict_groups.csv", conflict_rows)

    quarantine_rows = [
        {
            "candidate_id": candidate.candidate_id,
            "audio_path": candidate.audio_path,
            "intended_label": candidate.intended_label,
            "duplicate_group_id": candidate.duplicate_group_id,
            "status": candidate.contamination_status,
            "reasons": candidate.contamination_warnings,
        }
        for candidate in legacy_candidates
        if candidate.contamination_status in QUARANTINE_STATUSES
    ]
    brain_conflicts = [
        {
            "identity_sha256": row.get("identity_sha256", ""),
            "brain_path": row.get("brain_path", ""),
            "lane": row.get("lane", ""),
            "owner_target": row.get("owner_target", ""),
            "approved_label": row.get("approved_label", ""),
            "status": row.get("status", ""),
            "reasons": row.get("reasons", ""),
        }
        for row in brain_rows
        if is_legacy_brain_warning(str(row.get("status", "")))
    ]
    quarantine_payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "apply_automatically": False,
        "reversible": True,
        "legacy_trainer_proposals": quarantine_rows,
        "legacy_brain_warning_occurrences": brain_conflicts,
        "rollback": (
            "No source audio or brain was deleted. Reject this proposal or restore any later curated rebuild "
            "from the retained brain files and original locked_curated_v1 tree."
        ),
    }
    (output_dir / "proposed_training_quarantine.json").write_text(
        json.dumps(quarantine_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    review_rows = [
        {
            "priority": STATUS_PRIORITY.get(candidate.contamination_status, 10),
            "candidate_id": candidate.candidate_id,
            "audio_path": candidate.audio_path,
            "current_label": candidate.intended_label,
            "status": candidate.contamination_status,
            "reasons": candidate.contamination_warnings,
            "duplicate_group_id": candidate.duplicate_group_id,
            "human_verdict": "",
            "approved_label": "",
        }
        for candidate in legacy_candidates
        if candidate.contamination_status in REVIEW_STATUSES
    ]
    review_rows.sort(key=lambda row: (-int(row["priority"]), str(row["current_label"]), str(row["candidate_id"])))
    write_csv(output_dir / "human_review_needed.csv", review_rows)


def load_production_labels(brain_path: Path) -> list[str]:
    """Return trainable production labels from the current full brain."""
    payload = json.loads(brain_path.read_text(encoding="utf-8"))
    return sorted({str(label) for label in payload.get("labels", []) if is_trainable_taxonomy_label(label)})


def write_category_readiness(
    output_dir: Path,
    production_labels: list[str],
    candidates: list[ProvenanceCandidate],
) -> None:
    """Write a complete evidence-conservative production-category matrix."""
    by_label: dict[str, list[ProvenanceCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_label[candidate.intended_label].append(candidate)
    total_trusted_groups = {
        candidate.duplicate_group_id
        for candidate in candidates
        if candidate.human_approved and candidate.allowed_use not in {"excluded_duplicate", "excluded_label_conflict"}
    }
    rows: list[dict[str, Any]] = []
    for label in production_labels:
        label_rows = by_label.get(label, [])
        trusted = [row for row in label_rows if row.human_approved and not row.allowed_use.startswith("excluded_")]
        trusted_groups = {row.duplicate_group_id for row in trusted}
        source_groups = {row.source_group for row in trusted}
        final_count = sum(row.allowed_use == FINAL_HELDOUT_USE for row in trusted)
        tier, reason = readiness_tier(
            trusted_duplicate_groups=len(trusted_groups),
            trusted_source_groups=len(source_groups),
            final_heldout_examples=final_count,
        )
        legacy = [row for row in label_rows if row.source_kind == "legacy_curated_candidate"]
        supported_legacy = [row for row in legacy if row.contamination_status == "supported"]
        structures = {label.rsplit("/", 1)[-1]} if trusted else set()
        should_remain_broad = tier in {"C", "D"}
        rows.append(
            {
                "production_label": label,
                "top_family": label.split("/", 1)[0],
                "readiness_tier": tier,
                "readiness_reason": reason,
                "trusted_example_count": len(trusted),
                "trusted_duplicate_group_count": len(trusted_groups),
                "trusted_source_group_count": len(source_groups),
                "trusted_source_groups_json": json.dumps(sorted(source_groups)),
                "subtype_diversity_count": len(structures),
                "available_general_negative_groups": max(0, len(total_trusted_groups - trusted_groups)),
                "prototype_training_count": sum(row.allowed_use == TRAINING_USE for row in trusted),
                "prototype_validation_count": sum(row.allowed_use == VALIDATION_USE for row in trusted),
                "confidence_calibration_count": sum(row.allowed_use == CALIBRATION_USE for row in trusted),
                "final_heldout_count": final_count,
                "heldout_evaluation_possible": int(tier == "A"),
                "legacy_candidate_count": len(legacy),
                "supported_legacy_candidate_count": len(supported_legacy),
                "should_remain_broad": int(should_remain_broad),
                "hierarchy_note": (
                    "insufficient evidence for this detailed leaf; prefer a validated broader owner"
                    if should_remain_broad
                    else "retain leaf provisionally; production ownership still requires calibrated gates"
                ),
            }
        )
    write_csv(output_dir / "category_readiness_matrix.csv", rows)
    tier_counts = Counter(str(row["readiness_tier"]) for row in rows)
    (output_dir / "category_readiness_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "production_category_count": len(rows),
                "tier_counts": {tier: tier_counts.get(tier, 0) for tier in ("A", "B", "C", "D", "E")},
                "tier_E_policy": (
                    "No category is automatically declared redundant. Tier E requires an explicit human taxonomy decision."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_lane_evaluations(paths: list[Path]) -> list[LaneEvaluation]:
    """Load held-out lane results produced by the real neural lab."""
    evaluations: list[LaneEvaluation] = []
    for path in paths:
        for row in read_csv(path):
            evaluations.append(
                LaneEvaluation(
                    provider_id=str(row["provider_id"]),
                    label=str(row["label"]),
                    heldout_count=int(row["heldout_count"]),
                    correct_count=int(row["correct_count"]),
                    top1_accuracy=float(row["top1_accuracy"]),
                    mean_correct_similarity=float(row["mean_correct_similarity"]),
                    mean_margin=float(row["mean_margin"]),
                    known_distribution_rate=float(row["known_distribution_rate"]),
                )
            )
    return evaluations


def model_id_from_index(index_path: Path) -> str:
    """Read a frozen model identity from a saved prototype index."""
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return str(payload.get("metadata", {}).get("model_id", ""))


def write_model_registry(
    output_dir: Path,
    evaluations: list[LaneEvaluation],
    clap_index: Path,
    mert_index: Path,
    runtime_benchmark: Path | None,
) -> None:
    """Write per-category winners with deployment-license constraints."""
    leaderboard_path = output_dir / "encoder_leaderboard.csv"
    choices = write_lane_leaderboard(evaluations, leaderboard_path)
    runtime_by_provider: dict[str, dict[str, Any]] = {}
    if runtime_benchmark is not None and runtime_benchmark.is_file():
        runtime_payload = json.loads(runtime_benchmark.read_text(encoding="utf-8"))
        runtime_by_provider = {
            str(row.get("provider_id", "")): row for row in runtime_payload.get("results", []) if row.get("provider_id")
        }
        coverage = Counter(evaluation.provider_id for evaluation in evaluations)
        enriched_leaderboard: list[dict[str, Any]] = []
        for row in read_csv(leaderboard_path):
            runtime = runtime_by_provider.get(row["provider_id"], {})
            enriched = dict(row)
            enriched["warm_mean_embed_seconds"] = runtime.get("warm_mean_embed_seconds", "")
            enriched["warm_p95_embed_seconds"] = runtime.get("warm_p95_embed_seconds", "")
            enriched["float32_vector_bytes"] = runtime.get("float32_vector_bytes", "")
            enriched["local_model_bytes"] = runtime.get("local_model_bytes", "")
            enriched["evaluated_category_coverage_count"] = coverage.get(row["provider_id"], 0)
            enriched_leaderboard.append(enriched)
        write_csv(leaderboard_path, enriched_leaderboard)
    choices_by_label = {choice.label: choice for choice in choices}
    by_label: dict[str, list[LaneEvaluation]] = defaultdict(list)
    for evaluation in evaluations:
        by_label[evaluation.label].append(evaluation)

    category_registry: dict[str, dict[str, Any]] = {}
    for label, rows in sorted(by_label.items()):
        winner = choices_by_label[label]
        winner_evaluation = next(row for row in rows if row.provider_id == winner.provider_id)
        deployment_provider = winner.provider_id
        selection_status = "best_heldout_lane"
        if winner.provider_id == "hf_mert":
            clap_row = next((row for row in rows if row.provider_id == "hf_clap"), None)
            deployment_provider = "hf_clap" if clap_row is not None else ""
            selection_status = "mert_won_research_comparison_but_checkpoint_is_noncommercial"
        category_registry[label] = {
            "best_evaluated_provider": winner.provider_id,
            "deployment_provider": deployment_provider,
            "selection_status": selection_status,
            "heldout_count": winner.heldout_count,
            "top1_accuracy": winner_evaluation.top1_accuracy,
            "production_ownership_enabled": False,
            "reason_not_enabled": "calibration, source diversity, and category readiness gates are incomplete",
        }

    registry = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "models": {
            "hf_clap": {
                "source_model": "laion/larger_clap_music_and_speech",
                "revision": "195c3a3e68faebb3e2088b9a79e79b43ddbda76b",
                "model_id": model_id_from_index(clap_index),
                "weight_sha256": "d4e5cf6317c7521ca62c11b524f5646565310e040129fe35719cad661696d745",
                "license": "Apache-2.0",
                "distribution_status": "license-compatible candidate; weights are downloaded separately",
                "runtime_benchmark": runtime_by_provider.get("hf_clap", {}),
            },
            "hf_mert": {
                "source_model": "m-a-p/MERT-v1-95M",
                "revision": "7d1bb4c6894b70c0f958a550c20dc861c83b25c3",
                "model_id": model_id_from_index(mert_index),
                "weight_sha256": "a2b8b747f72c06e0595aeae41ae5473f4364938c6b39b2c58be38c48e6bd3fcd",
                "license": "CC-BY-NC-4.0",
                "distribution_status": "research comparison only; not eligible as distributable production default",
                "runtime_benchmark": runtime_by_provider.get("hf_mert", {}),
            },
        },
        "categories": category_registry,
        "fusion": {
            "enabled": False,
            "reason": "No fused lane was evaluated on the same held-out panel and no fusion gain was established.",
        },
    }
    (output_dir / "neural_model_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def copy_evaluation_evidence(output_dir: Path, named_paths: dict[str, Path]) -> None:
    """Copy compact held-out and disagreement evidence into the artifact."""
    for name, source in named_paths.items():
        if source.is_file():
            shutil.copy2(source, output_dir / name)


def write_active_learning_queue(output_dir: Path, clap_predictions: Path, mert_predictions: Path) -> None:
    """Rank held-out disagreements and uncertainty for human review."""
    clap_by_hash = {row["file_sha256"]: row for row in read_csv(clap_predictions)}
    mert_by_hash = {row["file_sha256"]: row for row in read_csv(mert_predictions)}
    rows: list[dict[str, Any]] = []
    for digest in sorted(set(clap_by_hash) & set(mert_by_hash)):
        clap = clap_by_hash[digest]
        mert = mert_by_hash[digest]
        disagreement = clap["predicted_label"] != mert["predicted_label"]
        clap_margin = float(clap["margin"])
        mert_margin = float(mert["margin"])
        clap_ood = clap["known_distribution"] not in {"1", "true", "True"}
        mert_ood = mert["known_distribution"] not in {"1", "true", "True"}
        reasons: list[str] = []
        priority = 0.0
        if disagreement:
            priority += 4.0
            reasons.append("encoder_disagreement")
        uncertainty = max(0.0, 0.12 - min(clap_margin, mert_margin)) / 0.12
        priority += uncertainty * 2.0
        if uncertainty > 0.5:
            reasons.append("low_top_two_margin")
        if clap_ood or mert_ood:
            priority += 1.0
            reasons.append("one_or_more_encoders_ood")
        expected = clap["expected_label"]
        if clap["predicted_label"] != expected or mert["predicted_label"] != expected:
            priority += 2.0
            reasons.append("heldout_error")
        rows.append(
            {
                "priority_score": f"{priority:.6f}",
                "file_sha256": digest,
                "display_name": clap["display_name"],
                "expected_label": expected,
                "clap_label": clap["predicted_label"],
                "clap_margin": clap["margin"],
                "clap_known_distribution": clap["known_distribution"],
                "mert_label": mert["predicted_label"],
                "mert_margin": mert["margin"],
                "mert_known_distribution": mert["known_distribution"],
                "encoder_disagreement": int(disagreement),
                "priority_reasons": "|".join(reasons),
                "human_verdict": "",
                "approved_label": "",
            }
        )
    rows.sort(key=lambda row: (-float(row["priority_score"]), str(row["file_sha256"])))
    write_csv(output_dir / "active_learning_queue.csv", rows)


def write_vocal_benchmark(
    output_dir: Path,
    prediction_paths: dict[str, Path],
    experiment_manifest: Path | None,
) -> None:
    """Write broad-vocal and difficult-negative metrics for each encoder."""
    subgroup_by_hash: dict[str, str] = {}
    if experiment_manifest is not None:
        subgroup_by_hash = {
            row["file_sha256"]: row.get("subgroup", "")
            for row in read_csv(experiment_manifest)
            if row.get("split") == "heldout_eval"
        }
    metric_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "schema_version": 1,
        "subtype_evaluation_status": (
            "not_supported: only five clean vocal trainers; loop/one-shot are reported as held-out subgroups, "
            "not trained subtype owners"
        ),
    }
    for provider_id, path in prediction_paths.items():
        rows = read_csv(path)
        vocal_rows = [row for row in rows if row["expected_label"] == "Voice/Musical Vocal"]
        negative_rows = [row for row in rows if row["expected_label"] != "Voice/Musical Vocal"]
        vocal_correct = sum(row["predicted_label"] == "Voice/Musical Vocal" for row in vocal_rows)
        vocal_ood = sum(row["known_distribution"] not in {"1", "true", "True"} for row in vocal_rows)
        negative_voice_claims = [row for row in negative_rows if row["predicted_label"] == "Voice/Musical Vocal"]
        in_distribution_false_positives = [
            row for row in negative_voice_claims if row["known_distribution"] in {"1", "true", "True"}
        ]
        summary[provider_id] = {
            "vocal_total": len(vocal_rows),
            "vocal_correct": vocal_correct,
            "vocal_recall": vocal_correct / len(vocal_rows) if vocal_rows else 0.0,
            "vocal_review_ood_count": vocal_ood,
            "vocal_review_ood_rate": vocal_ood / len(vocal_rows) if vocal_rows else 0.0,
            "difficult_negative_total": len(negative_rows),
            "raw_vocal_false_positive_count": len(negative_voice_claims),
            "raw_vocal_false_positive_rate": (
                len(negative_voice_claims) / len(negative_rows) if negative_rows else 0.0
            ),
            "in_distribution_vocal_false_positive_count": len(in_distribution_false_positives),
            "in_distribution_vocal_false_positive_rate": (
                len(in_distribution_false_positives) / len(negative_rows) if negative_rows else 0.0
            ),
        }
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            subgroup = subgroup_by_hash.get(row["file_sha256"], row["expected_label"])
            grouped[(row["expected_label"], subgroup)].append(row)
        for (expected_label, subgroup), group_rows in sorted(grouped.items()):
            correct_count = sum(row["predicted_label"] == expected_label for row in group_rows)
            known_count = sum(row["known_distribution"] in {"1", "true", "True"} for row in group_rows)
            voice_claim_count = sum(row["predicted_label"] == "Voice/Musical Vocal" for row in group_rows)
            metric_rows.append(
                {
                    "provider_id": provider_id,
                    "expected_label": expected_label,
                    "subgroup": subgroup,
                    "heldout_count": len(group_rows),
                    "correct_count": correct_count,
                    "top1_accuracy": f"{correct_count / len(group_rows):.8f}",
                    "known_distribution_count": known_count,
                    "known_distribution_rate": f"{known_count / len(group_rows):.8f}",
                    "voice_claim_count": voice_claim_count,
                    "voice_claim_rate": f"{voice_claim_count / len(group_rows):.8f}",
                }
            )
    write_csv(output_dir / "vocal_benchmark.csv", metric_rows)
    (output_dir / "vocal_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_gui_vocal_regression(
    output_dir: Path,
    current_manifest: Path | None,
    teaching_manifest: Path | None,
) -> None:
    """Write end-to-end evidence that GUI corrections now own placement."""
    if current_manifest is None or teaching_manifest is None:
        return
    current_rows = read_csv(current_manifest)
    teaching_rows = read_csv(teaching_manifest)
    current_by_source = {row.get("source_path", ""): row for row in current_rows}
    taught_rows: list[dict[str, Any]] = []
    for teaching in teaching_rows:
        current = current_by_source.get(teaching.get("source_path", ""), {})
        taught_rows.append(
            {
                "source_path": teaching.get("source_path", ""),
                "approved_folder": teaching.get("approved_folder", ""),
                "actual_folder": current.get("folder_path", ""),
                "final_claim_source": current.get("final_claim_source", ""),
                "matches_approved_folder": int(
                    bool(current) and current.get("folder_path", "") == teaching.get("approved_folder", "")
                ),
                "owned_by_exact_human_teacher": int(
                    current.get("final_claim_source", "") == "exact_human_teacher_owner_claim"
                ),
            }
        )
    write_csv(output_dir / "gui_vocal_correction_replay.csv", taught_rows)
    voice_related_count = sum(
        row.get("folder_path", "").startswith(("Instruments/Voice/", "FX/Human and Voice FX/")) for row in current_rows
    )
    review_count = sum(row.get("folder_path", "").startswith("_TO_REVIEW") for row in current_rows)
    matched_count = sum(int(row["matches_approved_folder"]) for row in taught_rows)
    exact_owner_count = sum(int(row["owned_by_exact_human_teacher"]) for row in taught_rows)
    summary = {
        "schema_version": 1,
        "pack_file_count": len(current_rows),
        "voice_related_placement_count": voice_related_count,
        "voice_related_placement_rate": voice_related_count / len(current_rows) if current_rows else 0.0,
        "review_count": review_count,
        "explicit_gui_correction_count": len(taught_rows),
        "explicit_gui_correction_exact_folder_match_count": matched_count,
        "explicit_gui_correction_exact_owner_claim_count": exact_owner_count,
    }
    (output_dir / "gui_vocal_correction_replay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_calibration_status(output_dir: Path, split_counts: Counter[str]) -> None:
    """Record why a production confidence calibrator was or was not fitted."""
    calibration_count = int(split_counts.get(CALIBRATION_USE, 0))
    status = {
        "schema_version": 1,
        "method": "logistic_regression",
        "minimum_reviewed_outcomes": 20,
        "nonleaking_calibration_examples_available": calibration_count,
        "fitted": False,
        "reason": (
            "insufficient nonleaking reviewed accepted/rejected outcomes; the code path is tested, "
            "but fitting now would create fake confidence"
        ),
    }
    (output_dir / "calibration_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_license_report(output_dir: Path) -> None:
    """Write the dependency and model-license decision used by this run."""
    content = """# Neural model dependency and licensing report

## LAION CLAP

- Model: `laion/larger_clap_music_and_speech`
- Revision: `195c3a3e68faebb3e2088b9a79e79b43ddbda76b`
- Weight SHA-256: `d4e5cf6317c7521ca62c11b524f5646565310e040129fe35719cad661696d745`
- Checkpoint/model-card license: Apache-2.0
- Runtime result: real 512-dimensional normalized embeddings on Mac CPU.
- Distribution decision: eligible as a separately downloaded production candidate, subject to later data and calibration gates.
- Primary source: https://huggingface.co/laion/larger_clap_music_and_speech

## MERT-v1-95M

- Model: `m-a-p/MERT-v1-95M`
- Revision: `7d1bb4c6894b70c0f958a550c20dc861c83b25c3`
- Weight SHA-256: `a2b8b747f72c06e0595aeae41ae5473f4364938c6b39b2c58be38c48e6bd3fcd`
- Checkpoint/model-card license: CC-BY-NC-4.0
- Runtime result: real 768-dimensional layer-6 normalized embeddings on Mac CPU.
- Compatibility: local remote-code snapshot required; newer Transformers also needs the historical HuBERT `conv_pos_batch_norm=False` default.
- Distribution decision: research comparison only. The non-commercial checkpoint cannot be the intended distributable production default.
- Primary sources: https://huggingface.co/m-a-p/MERT-v1-95M and https://github.com/yizhilll/MERT

## Isolated runtime

- Python 3.9.6 in `.venv_neural`
- PyTorch 2.8.0
- Transformers 4.57.6
- scikit-learn 1.6.1
- Production/legacy sorter environment was not modified.

Model weights and embedding caches are excluded from the patch bundle. They remain reproducible local downloads.
"""
    (output_dir / "model_dependency_license_report.md").write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build the curation-system command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, default=Path("training/locked_curated_v1"))
    parser.add_argument(
        "--seed-manifest",
        type=Path,
        default=Path("tests/acceptance/locked_smoke_v1/trusted_training_seed_v1.json"),
    )
    parser.add_argument("--gui-manifest", type=Path, action="append", default=[])
    parser.add_argument("--trainer-audit", type=Path, required=True)
    parser.add_argument("--brain-audit", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--clap-evaluation", type=Path, required=True)
    parser.add_argument("--mert-evaluation", type=Path, required=True)
    parser.add_argument("--clap-index", type=Path, required=True)
    parser.add_argument("--mert-index", type=Path, required=True)
    parser.add_argument("--clap-predictions", type=Path, required=True)
    parser.add_argument("--mert-predictions", type=Path, required=True)
    parser.add_argument("--shadow-report", type=Path)
    parser.add_argument("--experiment-manifest", type=Path)
    parser.add_argument("--current-sort-manifest", type=Path)
    parser.add_argument("--teaching-manifest", type=Path)
    parser.add_argument("--runtime-benchmark", type=Path)
    parser.add_argument("--samples-root", type=Path, default=Path("/Volumes/T9/music_production/samples"))
    return parser


def resolve(root: Path, path: Path) -> Path:
    """Resolve a project-relative argument."""
    return path.expanduser().resolve() if path.is_absolute() else (root / path).resolve()


def main(argv: list[str] | None = None) -> int:
    """Build all compact neural curation artifacts."""
    args = build_parser().parse_args(argv)
    root = args.project_root.expanduser().resolve()
    output_dir = resolve(root, args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer_audit_path = resolve(root, args.trainer_audit)
    brain_audit_path = resolve(root, args.brain_audit)
    cache_root = resolve(root, args.embedding_cache)
    trainer_rows = read_csv(trainer_audit_path)
    brain_rows = read_csv(brain_audit_path)
    audit_by_hash = audit_rows_by_hash(trainer_rows)
    embeddings_by_hash = cached_embedding_providers(cache_root)

    training_root = resolve(root, args.training_root)
    seed_manifest = resolve(root, args.seed_manifest)
    legacy_candidates = load_legacy_candidates(training_root, audit_by_hash, embeddings_by_hash)
    candidates = list(legacy_candidates)
    candidates.extend(load_seed_candidates(seed_manifest, embeddings_by_hash))
    samples_root = args.samples_root.expanduser().resolve()
    for gui_manifest in args.gui_manifest:
        candidates.extend(load_gui_candidates(resolve(root, gui_manifest), embeddings_by_hash, samples_root))
    assigned = list(assign_leakage_safe_uses(candidates))

    all_rows = candidate_csv_rows(assigned)
    write_csv(output_dir / "corpus_provenance.csv", all_rows)
    write_csv(
        output_dir / "trusted_example_inventory.csv",
        [row for row in all_rows if int(row["human_approved"]) == 1],
    )
    write_csv(
        output_dir / "dataset_splits.csv",
        [
            row
            for row in all_rows
            if row["allowed_use"] in {TRAINING_USE, VALIDATION_USE, CALIBRATION_USE, FINAL_HELDOUT_USE}
        ],
    )
    split_counts = Counter(
        candidate.allowed_use
        for candidate in assigned
        if candidate.allowed_use in {TRAINING_USE, VALIDATION_USE, CALIBRATION_USE, FINAL_HELDOUT_USE}
    )
    (output_dir / "dataset_split_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "counts": dict(sorted(split_counts.items())),
                "byte_overlap_allowed": False,
                "decoded_audio_overlap_allowed": False,
                "normalized_audio_overlap_allowed": False,
                "gui_feedback_policy": "training_only",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_calibration_status(output_dir, split_counts)

    write_contamination_outputs(output_dir, trainer_rows, legacy_candidates, brain_rows)
    write_category_readiness(output_dir, load_production_labels(root / "stage4_folder_brain.json"), assigned)

    clap_evaluation = resolve(root, args.clap_evaluation)
    mert_evaluation = resolve(root, args.mert_evaluation)
    clap_index = resolve(root, args.clap_index)
    mert_index = resolve(root, args.mert_index)
    evaluations = load_lane_evaluations([clap_evaluation, mert_evaluation])
    runtime_benchmark = resolve(root, args.runtime_benchmark) if args.runtime_benchmark else None
    write_model_registry(
        output_dir,
        evaluations,
        clap_index / "index.json",
        mert_index / "index.json",
        runtime_benchmark,
    )

    clap_predictions = resolve(root, args.clap_predictions)
    mert_predictions = resolve(root, args.mert_predictions)
    write_active_learning_queue(output_dir, clap_predictions, mert_predictions)
    experiment_manifest = resolve(root, args.experiment_manifest) if args.experiment_manifest else None
    write_vocal_benchmark(
        output_dir,
        {"hf_clap": clap_predictions, "hf_mert": mert_predictions},
        experiment_manifest,
    )
    current_sort_manifest = resolve(root, args.current_sort_manifest) if args.current_sort_manifest else None
    teaching_manifest = resolve(root, args.teaching_manifest) if args.teaching_manifest else None
    write_gui_vocal_regression(output_dir, current_sort_manifest, teaching_manifest)
    compact_evidence = {
        "clap_heldout_evaluation.csv": clap_evaluation,
        "mert_heldout_evaluation.csv": mert_evaluation,
        "clap_heldout_predictions.csv": clap_predictions,
        "mert_heldout_predictions.csv": mert_predictions,
    }
    if args.shadow_report:
        compact_evidence["neural_vs_legacy_disagreement.csv"] = resolve(root, args.shadow_report)
    copy_evaluation_evidence(output_dir, compact_evidence)
    write_license_report(output_dir)
    sanitize_persistent_paths(output_dir, root, samples_root)

    print(f"Output: {output_dir}")
    print(f"Corpus occurrences: {len(assigned)}")
    print(f"Human-approved occurrences: {sum(candidate.human_approved for candidate in assigned)}")
    print(f"Production categories: {len(load_production_labels(root / 'stage4_folder_brain.json'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
