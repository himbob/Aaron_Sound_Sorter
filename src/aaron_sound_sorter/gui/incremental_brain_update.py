"""Incremental brain updates for GUI-approved corrections.

The updater in this module is intentionally conservative.  It does not rebuild
global router heads from a sparse training tree, and it does not infer labels
from filenames.  It only adds measured audio fingerprints from human-approved
GUI corrections to the active brain JSON prototypes.
"""

from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from aaron_audio_intelligence.user_memory_brain import USER_MEMORY_BRAIN_NAME, build_empty_user_memory_brain
from aaron_sound_sorter.brain import (
    compute_category_fact_profiles,
    compute_category_rival_contrast_facts,
    label_reliability_requirements,
)
from aaron_sound_sorter.core import FEATURE_NAMES, FEATURE_WEIGHTS, FP_SIZE, FeatureRow
from aaron_sound_sorter.features import (
    choose_adaptive_label_model,
    deterministic_centroids,
    make_fingerprint_safe,
    select_deterministic_exemplars,
)
from aaron_sound_sorter.infrastructure.brain_repository import BrainRepository
from aaron_sound_sorter.io_utils import source_pack_key_for_path
from aaron_sound_sorter.training_labels import (
    label_default_structure,
    public_label,
    top_for_public_label,
)

DEFAULT_INCREMENTAL_BRAIN_NAMES = (
    "stage4_folder_brain.json",
    "stage4_folder_brain_core_baby.json",
    "stage4_folder_brain_spread_baby.json",
    "stage4_folder_brain_outlier_baby.json",
    USER_MEMORY_BRAIN_NAME,
)
MAX_INCREMENTAL_EXAMPLES_PER_LABEL = 120
DEFAULT_MAX_CENTROIDS = 6
DEFAULT_HUMAN_OVERRIDE_EVIDENCE_WEIGHT = 32
MAX_HUMAN_OVERRIDE_EVIDENCE_WEIGHT = 1200
DYNAMIC_HUMAN_OVERRIDE_COMPETITOR_MULTIPLIER = 1.25
DYNAMIC_HUMAN_OVERRIDE_COMPETITOR_CUSHION = 8


@dataclass(frozen=True)
class IncrementalCorrection:
    """One human-approved training correction to merge into active brains.

    Args:
        label: Public taxonomy label approved by the user.
        audio_path: Existing audio file in the trusted training slot.
        source_path: Original GUI source path for audit only.
        row_id: GUI row id for audit only.
        status: Import status such as ``staged`` or ``duplicate_existing``.
        evidence_weight: Bounded prototype weight for this deliberate human
            override. A value greater than one makes a correction more than a
            single weak teacher without copying audio files.

    Side Effects:
        None.
    """

    label: str
    audio_path: Path
    source_path: Path
    row_id: str
    status: str
    evidence_weight: int = DEFAULT_HUMAN_OVERRIDE_EVIDENCE_WEIGHT


@dataclass(frozen=True)
class MeasuredCorrectionEvidence:
    """Measured audio row plus its human override weight.

    Args:
        row: Feature row computed from audio content.
        evidence_weight: Bounded weight to use for prototype and support math.

    Side Effects:
        None.
    """

    row: FeatureRow
    evidence_weight: int


@dataclass(frozen=True)
class BrainDeltaResult:
    """Summary of one brain file updated by incremental correction evidence.

    Args:
        brain_path: Active brain JSON that was updated.
        backup_path: Backup copy written before mutation.
        label_count_before: Number of labels before the update.
        label_count_after: Number of labels after the update.
        corrections_applied: Number of readable corrections applied.
        effective_evidence_weight: Total effective support applied after dynamic
            competitor-aware weighting.
        labels_touched: Labels whose prototype data changed.
        warnings: Non-fatal warnings for this brain.

    Side Effects:
        None.
    """

    brain_path: Path
    backup_path: Path
    label_count_before: int
    label_count_after: int
    corrections_applied: int
    effective_evidence_weight: int
    labels_touched: list[str]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class IncrementalBrainUpdateSummary:
    """Top-level summary for a GUI incremental brain update run.

    Args:
        report_dir: Existing GUI training report directory.
        manifest_path: CSV manifest with per-brain update rows.
        backup_dir: Folder containing pre-update brain copies.
        updated_brains: Per-brain update summaries.
        correction_count: Number of trainable correction rows requested.
        applied_correction_count: Number of readable corrections merged.
        skipped_correction_count: Number of corrections skipped.
        errors: Non-fatal per-correction or per-brain errors.

    Side Effects:
        None.
    """

    report_dir: Path
    manifest_path: Path
    backup_dir: Path
    updated_brains: list[BrainDeltaResult]
    correction_count: int
    applied_correction_count: int
    skipped_correction_count: int
    errors: list[str]


class IncrementalBrainUpdater:
    """Merge GUI correction evidence into active brain prototype data.

    Args:
        project_root: Repository root containing active brain files.
        brain_names: Brain filenames to update when present.

    Side Effects:
        ``apply`` backs up active brain JSON files, rewrites updated brain JSONs,
        and writes an audit manifest under the GUI training report folder.

    Important Constraints:
        The updater computes fingerprints from audio content only. Paths are
        used for locating files and writing reports, never as classifier
        evidence.
    """

    def __init__(
        self,
        project_root: Path,
        brain_names: tuple[str, ...] = DEFAULT_INCREMENTAL_BRAIN_NAMES,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.brain_names = tuple(brain_names)
        self.repository = BrainRepository()

    def apply(
        self,
        corrections: list[IncrementalCorrection],
        *,
        report_dir: Path,
        backup_dir: Path,
    ) -> IncrementalBrainUpdateSummary:
        """Apply trainable corrections to active brain JSON files.

        Args:
            corrections: Human-approved correction rows with existing audio.
            report_dir: Existing GUI report folder for audit artifacts.
            backup_dir: Destination for pre-update brain backups.

        Returns:
            Incremental update summary.

        Raises:
            No intentional exceptions for individual bad corrections; they are
            recorded in the summary. Filesystem failures for report creation or
            brain writes may propagate.
        """
        report_dir = Path(report_dir).expanduser().resolve()
        backup_dir = Path(backup_dir).expanduser().resolve()
        report_dir.mkdir(parents=True, exist_ok=True)
        backup_dir.mkdir(parents=True, exist_ok=True)
        rows, errors = self._read_correction_rows(corrections)
        if rows:
            self._ensure_user_memory_brain_exists()
        updated_brains: list[BrainDeltaResult] = []
        for brain_path in self._existing_brain_paths():
            try:
                updated_brains.append(self._update_one_brain(brain_path, rows, backup_dir))
            except Exception as exc:
                errors.append(f"{brain_path.name}: {exc}")
        manifest_path = report_dir / "Aaron_GUI_Incremental_Brain_Update.csv"
        write_incremental_update_manifest(manifest_path, updated_brains, errors)
        return IncrementalBrainUpdateSummary(
            report_dir=report_dir,
            manifest_path=manifest_path,
            backup_dir=backup_dir,
            updated_brains=updated_brains,
            correction_count=len(corrections),
            applied_correction_count=len(rows),
            skipped_correction_count=max(0, len(corrections) - len(rows)),
            errors=errors,
        )

    def _existing_brain_paths(self) -> list[Path]:
        return [self.project_root / name for name in self.brain_names if (self.project_root / name).is_file()]

    def _ensure_user_memory_brain_exists(self) -> None:
        """Create the dedicated GUI correction memory brain when configured."""
        if USER_MEMORY_BRAIN_NAME not in self.brain_names:
            return
        memory_path = self.project_root / USER_MEMORY_BRAIN_NAME
        if memory_path.exists():
            return
        full_brain_path = self.project_root / "stage4_folder_brain.json"
        if not full_brain_path.exists():
            return
        base_brain = self.repository.load(full_brain_path)
        self.repository.save(memory_path, build_empty_user_memory_brain(base_brain))

    def _read_correction_rows(
        self,
        corrections: list[IncrementalCorrection],
    ) -> tuple[list[MeasuredCorrectionEvidence], list[str]]:
        rows: list[MeasuredCorrectionEvidence] = []
        errors: list[str] = []
        seen: set[tuple[str, str]] = set()
        for correction in corrections:
            key = (str(correction.audio_path.resolve()), correction.label)
            if key in seen:
                continue
            seen.add(key)
            if not correction.audio_path.is_file():
                errors.append(f"{correction.row_id}: audio file not found: {correction.audio_path}")
                continue
            fingerprint, duration_sec, read_status = make_fingerprint_safe(correction.audio_path)
            if read_status != "ok" or len(fingerprint) != FP_SIZE:
                errors.append(
                    f"{correction.row_id}: unreadable correction audio ({read_status}): {correction.audio_path}"
                )
                continue
            feature_row = FeatureRow(
                path=str(correction.audio_path),
                group_key=correction.label,
                label=correction.label,
                top=top_for_public_label(correction.label),
                structure=label_default_structure(correction.label),
                duration_sec=float(duration_sec),
                fingerprint=np.asarray(fingerprint, dtype=np.float32).astype(float).tolist(),
                read_status="ok",
                source_pack=source_pack_key_for_path(str(correction.audio_path)),
                training_active_status="ACTIVE_INCREMENTAL_GUI_CORRECTION",
                label_source_group_count=1,
                label_clean_available=1,
            )
            rows.append(
                MeasuredCorrectionEvidence(
                    row=feature_row,
                    evidence_weight=bounded_human_override_weight(correction.evidence_weight),
                )
            )
        return rows, errors

    def _update_one_brain(
        self,
        brain_path: Path,
        correction_rows: list[MeasuredCorrectionEvidence],
        backup_dir: Path,
    ) -> BrainDeltaResult:
        brain = self.repository.load(brain_path)
        label_count_before = len(brain.get("labels", []) if isinstance(brain.get("labels", []), list) else [])
        backup_path = backup_dir / brain_path.name
        shutil.copy2(brain_path, backup_path)
        labels_touched: list[str] = []
        warnings: list[str] = []
        weighted_rows: list[MeasuredCorrectionEvidence] = []
        for evidence in correction_rows:
            row = evidence.row
            dynamic_weight = dynamic_human_override_weight(
                brain,
                row.label,
                requested_weight=evidence.evidence_weight,
            )
            weighted_evidence = MeasuredCorrectionEvidence(row=row, evidence_weight=dynamic_weight)
            try:
                update_brain_label_with_row(brain, row, evidence_weight=dynamic_weight)
                labels_touched.append(row.label)
                weighted_rows.append(weighted_evidence)
            except Exception as exc:
                warnings.append(f"{row.label}: {exc}")
        if weighted_rows:
            annotate_incremental_update(brain, weighted_rows, brain_path.name)
            recompute_rival_contrast_facts(brain)
            self.repository.save(brain_path, brain)
        label_count_after = len(brain.get("labels", []) if isinstance(brain.get("labels", []), list) else [])
        return BrainDeltaResult(
            brain_path=brain_path,
            backup_path=backup_path,
            label_count_before=label_count_before,
            label_count_after=label_count_after,
            corrections_applied=len(correction_rows),
            effective_evidence_weight=sum(evidence.evidence_weight for evidence in weighted_rows),
            labels_touched=sorted(set(labels_touched)),
            warnings=warnings,
        )


def update_brain_label_with_row(
    brain: dict[str, Any],
    row: FeatureRow,
    *,
    evidence_weight: int = DEFAULT_HUMAN_OVERRIDE_EVIDENCE_WEIGHT,
) -> None:
    """Merge one measured correction row into one brain label.

    Args:
        brain: Brain dictionary loaded from JSON.
        row: Measured correction row with a human-approved label.
        evidence_weight: Bounded amount of prototype support this deliberate
            human override contributes.

    Returns:
        None.

    Side Effects:
        Mutates ``brain`` in memory by updating label lists, counts, prototypes,
        anchors, examples, reliability, and fact profiles for ``row.label``.

    Important Constraints:
        This uses the approved taxonomy label and measured fingerprint. It never
        reads source names as evidence.
    """
    label = str(row.label)
    labels = [str(value) for value in brain.get("labels", []) if str(value)]
    if label not in labels:
        labels.append(label)
        brain["labels"] = sorted(labels)
    top_by_label = ensure_mapping(brain, "top_by_label")
    structure_by_label = ensure_mapping(brain, "structure_by_label")
    top_by_label[label] = row.top or top_for_public_label(label)
    structure_by_label[label] = row.structure or label_default_structure(label)
    examples = append_training_example(brain, row, evidence_weight=evidence_weight)
    raw_vectors = weighted_example_fingerprints(examples)
    raw_vectors = [vector for vector in raw_vectors if vector.size == FP_SIZE]
    if not raw_vectors:
        raise ValueError("no valid fingerprints available for label after update")
    normalized_vectors = normalized_vectors_for_label(brain, label, raw_vectors)
    update_label_models(brain, label, normalized_vectors)
    update_label_counts_and_reliability(
        brain,
        label,
        row,
        effective_label_count=len(raw_vectors),
        unique_label_count=count_unique_examples(examples),
    )
    update_label_fact_profile(brain, label, examples)


def append_training_example(
    brain: dict[str, Any],
    row: FeatureRow,
    *,
    evidence_weight: int,
) -> list[dict[str, Any]]:
    """Append a correction example unless the same path is already present."""
    examples_by_label = ensure_mapping(brain, "training_examples_detailed_by_label")
    examples = examples_by_label.setdefault(row.label, [])
    if not isinstance(examples, list):
        examples = []
        examples_by_label[row.label] = examples
    source_path = str(row.path)
    existing_example = next(
        (
            example
            for example in examples
            if isinstance(example, dict) and str(example.get("source_path", "")) == source_path
        ),
        None,
    )
    if isinstance(existing_example, dict):
        reinforce_training_example(existing_example, evidence_weight=evidence_weight)
    else:
        examples.append(training_example_from_row(row))
        reinforce_training_example(examples[-1], evidence_weight=evidence_weight)
    if len(examples) > MAX_INCREMENTAL_EXAMPLES_PER_LABEL:
        del examples[:-MAX_INCREMENTAL_EXAMPLES_PER_LABEL]
    examples_by_label[row.label] = examples
    simple_examples = ensure_mapping(brain, "examples_by_label")
    simple_list = simple_examples.setdefault(row.label, [])
    if isinstance(simple_list, list) and source_path not in simple_list:
        simple_list.append(source_path)
        del simple_list[:-5]
    return [example for example in examples if isinstance(example, dict)]


def training_example_from_row(row: FeatureRow) -> dict[str, Any]:
    """Return the stored detailed training example for one correction row."""
    return {
        "source_path": row.path,
        "group_key": row.group_key,
        "top": row.top,
        "structure": row.structure,
        "duration_sec": float(row.duration_sec),
        "source_pack": row.source_pack or source_pack_key_for_path(row.path),
        "physics_tags": "",
        "physics_summary": "",
        "fingerprint": [float(value) for value in row.fingerprint],
        "incremental_gui_correction": True,
        "incremental_added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "human_override_confirmation_count": 0,
        "human_override_evidence_weight": 1,
        "human_override_weight_policy": "bounded_incremental_prototype_support",
    }


def reinforce_training_example(example: dict[str, Any], *, evidence_weight: int) -> None:
    """Record bounded human override support on an example.

    Args:
        example: Stored training example to mutate.
        evidence_weight: Requested weight from the GUI correction.

    Returns:
        None.

    Side Effects:
        Updates audit fields on ``example``.
    """
    requested_weight = bounded_human_override_weight(evidence_weight)
    previous_weight = bounded_human_override_weight(example.get("human_override_evidence_weight", 1))
    confirmation_count = int(example.get("human_override_confirmation_count", 0) or 0) + 1
    example["human_override_confirmation_count"] = confirmation_count
    example["human_override_evidence_weight"] = min(
        MAX_HUMAN_OVERRIDE_EVIDENCE_WEIGHT,
        max(requested_weight, previous_weight + 1 if confirmation_count > 1 else previous_weight),
    )
    example["human_override_weight_policy"] = "bounded_incremental_prototype_support"
    example["last_human_override_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


def update_label_models(brain: dict[str, Any], label: str, normalized_vectors: np.ndarray) -> None:
    """Refresh adaptive prototypes for one label."""
    label_models = ensure_mapping(brain, "label_models_by_label")
    max_centroids = max(
        1,
        int(brain.get("incremental_max_centroids") or brain.get("max_incremental_centroids") or DEFAULT_MAX_CENTROIDS),
    )
    model = choose_adaptive_label_model(label, normalized_vectors, max_centroids=max_centroids)
    centroid_count = int(model.get("centroid_count", 1) or 1)
    if str(model.get("model_mode")) == "exemplar_only":
        centroids = select_deterministic_exemplars(normalized_vectors, centroid_count)
    else:
        centroids = deterministic_centroids(normalized_vectors, max_k=centroid_count)
    exemplar_count = min(int(model.get("exemplar_count", 1) or 1), int(normalized_vectors.shape[0]))
    anchor_count = min(int(brain.get("max_anchors_per_label", 12) or 12), int(normalized_vectors.shape[0]))
    ensure_mapping(brain, "centroids")[label] = centroids.astype(float).tolist()
    ensure_mapping(brain, "exemplars_by_label")[label] = (
        select_deterministic_exemplars(normalized_vectors, exemplar_count).astype(float).tolist()
    )
    ensure_mapping(brain, "anchors_by_label")[label] = (
        select_deterministic_exemplars(normalized_vectors, anchor_count).astype(float).tolist()
    )
    ensure_mapping(brain, "centroid_counts")[label] = int(centroid_count)
    model["exemplar_count"] = int(exemplar_count)
    model["anchor_count"] = int(anchor_count)
    label_models[label] = dict(model)


def update_label_counts_and_reliability(
    brain: dict[str, Any],
    label: str,
    row: FeatureRow,
    effective_label_count: int,
    unique_label_count: int,
) -> None:
    """Update support counters and reliability gates for one label."""
    effective_label_count = max(1, int(effective_label_count))
    unique_label_count = max(1, int(unique_label_count))
    ensure_mapping(brain, "counts")[label] = int(effective_label_count)
    ensure_mapping(brain, "effective_counts")[label] = int(effective_label_count)
    ensure_mapping(brain, "raw_counts")[label] = int(unique_label_count)
    reliability = ensure_mapping(brain, "label_reliability_by_label")
    label_models = ensure_mapping(brain, "label_models_by_label")
    model = label_models.get(label, {}) if isinstance(label_models.get(label), dict) else {}
    tier, req_similarity, req_margin = label_reliability_requirements(
        effective_label_count,
        "ACTIVE_INCREMENTAL_GUI_CORRECTION",
        1,
    )
    reliability[label] = {
        **(reliability.get(label, {}) if isinstance(reliability.get(label), dict) else {}),
        "training_count": int(effective_label_count),
        "effective_training_count": int(effective_label_count),
        "raw_training_count": int(unique_label_count),
        "unique_training_count": int(unique_label_count),
        "human_override_effective_weight": int(effective_label_count),
        "clean_available": int(unique_label_count),
        "source_group_count": 1,
        "active_status": "ACTIVE_INCREMENTAL_GUI_CORRECTION",
        "centroid_count": int(model.get("centroid_count", 1) or 1),
        "model_mode": str(model.get("model_mode", "")),
        "exemplar_count": int(model.get("exemplar_count", 1) or 1),
        "spread_mean": float(model.get("spread_mean", 0.0) or 0.0),
        "spread_p90": float(model.get("spread_p90", 0.0) or 0.0),
        "support_tier": tier,
        "min_similarity_for_auto_place": float(req_similarity),
        "min_margin_for_auto_place": float(req_margin),
        "last_incremental_source_path": row.path,
    }
    balance = ensure_mapping(brain, "effective_training_balance_by_label")
    balance[label] = {
        "label": label,
        "top": row.top,
        "raw_count": int(unique_label_count),
        "eligible_count_after_contradiction_filter": int(effective_label_count),
        "excluded_from_math_count": 0,
        "used_fallback_when_all_rows_excluded": "no",
        "effective_count": int(effective_label_count),
        "cap_applied": "no",
        "effective_cap": int(effective_label_count),
        "selection_method": "incremental_gui_correction",
        "central_selected": int(effective_label_count),
        "diverse_selected": 0,
        "edge_selected": 0,
        "source_pack_count": 1,
        "source_pack_counts_top10": {row.source_pack or source_pack_key_for_path(row.path): int(effective_label_count)},
    }


def update_label_fact_profile(
    brain: dict[str, Any],
    label: str,
    examples: list[dict[str, Any]],
) -> None:
    """Refresh one label's measured fact profile from stored examples."""
    rows: list[FeatureRow] = []
    for example in examples:
        vector = example_fingerprint(example)
        if vector.size != FP_SIZE:
            continue
        for _ in range(example_evidence_weight(example)):
            rows.append(
                FeatureRow(
                    path=str(example.get("source_path", "")),
                    group_key=str(example.get("group_key", label) or label),
                    label=label,
                    top=str(example.get("top", "") or top_for_public_label(label)),
                    structure=str(example.get("structure", "") or label_default_structure(label)),
                    duration_sec=float(example.get("duration_sec", 0.0) or 0.0),
                    fingerprint=vector.astype(float).tolist(),
                    read_status="ok",
                    source_pack=str(example.get("source_pack", "") or ""),
                )
            )
    if not rows:
        return
    profile = compute_category_fact_profiles(rows, list(FEATURE_NAMES)).get(label)
    if isinstance(profile, dict):
        ensure_mapping(brain, "category_fact_profiles")[label] = profile


def recompute_rival_contrast_facts(brain: dict[str, Any]) -> None:
    """Refresh contrast facts after one or more label profiles changed."""
    profiles = brain.get("category_fact_profiles", {})
    centroids = brain.get("centroids", {})
    top_by_label = brain.get("top_by_label", {})
    if not isinstance(profiles, dict) or not isinstance(centroids, dict) or not isinstance(top_by_label, dict):
        return
    weights = np.asarray(brain.get("feature_weights", FEATURE_WEIGHTS), dtype=np.float32)
    brain["category_rival_contrast_facts"] = compute_category_rival_contrast_facts(
        profiles,
        centroids,
        top_by_label,
        weights,
        list(FEATURE_NAMES),
    )


def normalized_vectors_for_label(
    brain: dict[str, Any],
    label: str,
    raw_vectors: list[np.ndarray],
) -> np.ndarray:
    """Return label-structure-normalized vectors for prototype storage."""
    structure = str(ensure_mapping(brain, "structure_by_label").get(label, label_default_structure(label)))
    scalers = brain.get("scalers_by_structure", {})
    scaler = scalers.get(structure) if isinstance(scalers, dict) else None
    if isinstance(scaler, dict):
        mean = vector_from_json(scaler.get("mean"), 0.0)
        std = vector_from_json(scaler.get("std"), 1.0)
    else:
        mean = vector_from_json(brain.get("scaler_mean"), 0.0)
        std = vector_from_json(brain.get("scaler_std"), 1.0)
    std = np.where(np.abs(std) < 1e-6, 1.0, std)
    arr = np.vstack([pad_fingerprint(vector) for vector in raw_vectors]).astype(np.float32)
    return ((arr - mean[None, :]) / std[None, :]).astype(np.float32)


def vector_from_json(values: Any, fill: float) -> np.ndarray:
    """Return a finite FP_SIZE vector from JSON data."""
    vector = np.asarray(values if isinstance(values, list) else [], dtype=np.float32).reshape(-1)
    if vector.size < FP_SIZE:
        vector = np.pad(vector, (0, FP_SIZE - vector.size), mode="constant", constant_values=fill)
    return np.nan_to_num(vector[:FP_SIZE], nan=fill, posinf=fill, neginf=fill).astype(np.float32)


def example_fingerprint(example: dict[str, Any]) -> np.ndarray:
    """Return a finite fingerprint vector from a stored training example."""
    return pad_fingerprint(np.asarray(example.get("fingerprint", []), dtype=np.float32))


def pad_fingerprint(values: np.ndarray) -> np.ndarray:
    """Return a finite fingerprint vector with the current feature width."""
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.size < FP_SIZE:
        vector = np.pad(vector, (0, FP_SIZE - vector.size), mode="constant", constant_values=0.0)
    return np.nan_to_num(vector[:FP_SIZE], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def ensure_mapping(brain: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a dict at ``brain[key]``, replacing malformed values."""
    value = brain.get(key)
    if isinstance(value, dict):
        return value
    brain[key] = {}
    return brain[key]


def annotate_incremental_update(
    brain: dict[str, Any],
    rows: list[MeasuredCorrectionEvidence],
    brain_name: str,
) -> None:
    """Record that a brain has received incremental GUI evidence."""
    history = brain.setdefault("incremental_gui_update_history", [])
    if not isinstance(history, list):
        history = []
        brain["incremental_gui_update_history"] = history
    labels = sorted({evidence.row.label for evidence in rows})
    effective_weight = sum(max(1, int(evidence.evidence_weight)) for evidence in rows)
    history.append(
        {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "brain_name": brain_name,
            "correction_count": len(rows),
            "effective_evidence_weight": int(effective_weight),
            "labels": labels,
            "policy": "weighted_prototype_fact_profile_delta_only",
            "note": (
                "GUI incremental update refreshed label prototypes, anchors, counts, "
                "and fact profiles from measured audio. Human overrides carry bounded "
                "extra prototype support. Global ridge/router heads remain from the "
                "last full rebuild."
            ),
        }
    )
    del history[:-50]
    brain["incremental_gui_update_policy"] = "weighted_prototype_fact_profile_delta_only"
    brain["incremental_human_override_weight"] = DEFAULT_HUMAN_OVERRIDE_EVIDENCE_WEIGHT
    brain["incremental_human_override_max_weight"] = MAX_HUMAN_OVERRIDE_EVIDENCE_WEIGHT
    brain["incremental_global_heads_need_full_rebuild"] = True


def corrections_from_import_manifest(manifest_path: Path) -> list[IncrementalCorrection]:
    """Load trainable GUI corrections from an import manifest.

    Args:
        manifest_path: ``Aaron_GUI_Training_Import.csv`` written by the
            correction importer.

    Returns:
        Corrections for rows whose audio exists in the trusted training slot.

    Side Effects:
        Reads the manifest only.
    """
    corrections: list[IncrementalCorrection] = []
    with Path(manifest_path).expanduser().open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            status = str(row.get("status", ""))
            if status not in {"staged", "duplicate_existing"}:
                continue
            staged_path = Path(str(row.get("staged_path", "")).strip())
            if not str(staged_path):
                continue
            corrections.append(
                IncrementalCorrection(
                    label=normalize_public_label(str(row.get("approved_folder", ""))),
                    audio_path=staged_path.expanduser().resolve(),
                    source_path=Path(str(row.get("source_path", ""))).expanduser(),
                    row_id=str(row.get("row_id", "")),
                    status=status,
                    evidence_weight=bounded_human_override_weight(row.get("evidence_weight", "")),
                )
            )
    return corrections


def normalize_public_label(label: str) -> str:
    """Normalize a public taxonomy label for brain storage."""
    return public_label("/".join(part.strip() for part in str(label).replace("\\", "/").split("/") if part.strip()))


def bounded_human_override_weight(value: object) -> int:
    """Return a safe evidence weight for one GUI correction.

    Args:
        value: User/config supplied weight value.

    Returns:
        Integer weight between 1 and ``MAX_HUMAN_OVERRIDE_EVIDENCE_WEIGHT``.

    Side Effects:
        None.
    """
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_HUMAN_OVERRIDE_EVIDENCE_WEIGHT
    return max(1, min(MAX_HUMAN_OVERRIDE_EVIDENCE_WEIGHT, parsed))


def dynamic_human_override_weight(
    brain: dict[str, Any],
    label: str,
    *,
    requested_weight: object = DEFAULT_HUMAN_OVERRIDE_EVIDENCE_WEIGHT,
) -> int:
    """Return a competitor-aware evidence weight for a GUI correction.

    Args:
        brain: Active brain being updated.
        label: Human-approved public taxonomy label.
        requested_weight: Optional manual/configured lower bound.

    Returns:
        A bounded weight large enough to out-support the strongest competing
        label neighborhood already present in this brain.

    Side Effects:
        None.

    Important Constraints:
        This uses only internal brain label metadata and learned support counts.
        It does not inspect source filenames, folder names, ZIP member names, or
        sample-pack labels.
    """
    requested = bounded_human_override_weight(requested_weight)
    competitor_weight = strongest_competing_label_weight(brain, label)
    if competitor_weight <= 0:
        return requested
    dynamic_target = int(
        round(
            (float(competitor_weight) * DYNAMIC_HUMAN_OVERRIDE_COMPETITOR_MULTIPLIER)
            + DYNAMIC_HUMAN_OVERRIDE_COMPETITOR_CUSHION
        )
    )
    return bounded_human_override_weight(max(requested, dynamic_target))


def strongest_competing_label_weight(brain: dict[str, Any], label: str) -> float:
    """Return the strongest effective count in the label's competition set.

    Args:
        brain: Active brain dictionary.
        label: Human-approved label being reinforced.

    Returns:
        Weighted effective count for the strongest nearby competitor.

    Side Effects:
        None.
    """
    label = normalize_public_label(label)
    label_top = top_for_public_label(label)
    label_structure = label_default_structure(label)
    label_parts = label.split("/")
    counts = brain.get("effective_counts", brain.get("counts", {}))
    if not isinstance(counts, dict):
        return 0.0
    labels = [str(value) for value in brain.get("labels", []) if str(value)]
    top_by_label = brain.get("top_by_label", {})
    structure_by_label = brain.get("structure_by_label", {})
    strongest = 0.0
    for competitor in labels:
        competitor = normalize_public_label(competitor)
        if not competitor or competitor == label:
            continue
        competitor_top = _label_top_from_maps(competitor, top_by_label)
        if competitor_top != label_top:
            continue
        competitor_structure = _label_structure_from_maps(competitor, structure_by_label)
        count = safe_float(counts.get(competitor), 0.0)
        if count <= 0:
            continue
        strongest = max(
            strongest,
            count
            * competition_weight_multiplier(label_parts, competitor.split("/"), label_structure, competitor_structure),
        )
    return strongest


def competition_weight_multiplier(
    label_parts: list[str],
    competitor_parts: list[str],
    label_structure: str,
    competitor_structure: str,
) -> float:
    """Return how directly one label competes with a correction target.

    Args:
        label_parts: Slash-split approved label.
        competitor_parts: Slash-split competing brain label.
        label_structure: Approved label structure.
        competitor_structure: Competing label structure.

    Returns:
        Multiplier applied to the competitor's effective support count.

    Side Effects:
        None.
    """
    shared_depth = common_prefix_depth(label_parts, competitor_parts)
    same_structure = label_structure == competitor_structure
    if same_structure and shared_depth >= 2:
        return 1.0
    if shared_depth >= 2:
        return 0.85
    if shared_depth >= 1:
        return 1.0
    if same_structure:
        return 0.75
    return 0.0


def common_prefix_depth(left: list[str], right: list[str]) -> int:
    """Return the number of leading taxonomy components shared by two labels."""
    depth = 0
    for left_part, right_part in zip(left, right):
        if left_part != right_part:
            break
        depth += 1
    return depth


def _label_top_from_maps(label: str, top_by_label: object) -> str:
    if isinstance(top_by_label, dict):
        value = top_by_label.get(label)
        if value:
            return str(value)
    return top_for_public_label(label)


def _label_structure_from_maps(label: str, structure_by_label: object) -> str:
    if isinstance(structure_by_label, dict):
        value = structure_by_label.get(label)
        if value:
            return str(value)
    return label_default_structure(label)


def safe_float(value: object, default: float = 0.0) -> float:
    """Return a finite float from untrusted JSON-ish values."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(parsed):
        return default
    return parsed


def example_evidence_weight(example: dict[str, Any]) -> int:
    """Return the bounded prototype weight for a stored example."""
    return bounded_human_override_weight(example.get("human_override_evidence_weight", 1))


def weighted_example_fingerprints(examples: list[dict[str, Any]]) -> list[np.ndarray]:
    """Return example fingerprints repeated by bounded evidence weight."""
    vectors: list[np.ndarray] = []
    for example in examples:
        vector = example_fingerprint(example)
        if vector.size != FP_SIZE:
            continue
        vectors.extend(vector.copy() for _ in range(example_evidence_weight(example)))
    return vectors


def count_unique_examples(examples: list[dict[str, Any]]) -> int:
    """Return the count of unique stored example paths."""
    paths = {str(example.get("source_path", "")) for example in examples if isinstance(example, dict)}
    return len({path for path in paths if path})


def write_incremental_update_manifest(
    path: Path,
    results: list[BrainDeltaResult],
    errors: list[str],
) -> None:
    """Write the incremental brain update audit manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "brain_path",
        "backup_path",
        "label_count_before",
        "label_count_after",
        "corrections_applied",
        "effective_evidence_weight",
        "labels_touched",
        "warnings",
        "errors",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        if results:
            for result in results:
                writer.writerow(
                    {
                        "brain_path": str(result.brain_path),
                        "backup_path": str(result.backup_path),
                        "label_count_before": str(result.label_count_before),
                        "label_count_after": str(result.label_count_after),
                        "corrections_applied": str(result.corrections_applied),
                        "effective_evidence_weight": str(result.effective_evidence_weight),
                        "labels_touched": json.dumps(result.labels_touched, sort_keys=True),
                        "warnings": json.dumps(result.warnings, sort_keys=True),
                        "errors": "",
                    }
                )
            return
        writer.writerow(
            {
                "brain_path": "",
                "backup_path": "",
                "label_count_before": "",
                "label_count_after": "",
                "corrections_applied": "0",
                "effective_evidence_weight": "0",
                "labels_touched": "[]",
                "warnings": "[]",
                "errors": json.dumps(errors, sort_keys=True),
            }
        )
