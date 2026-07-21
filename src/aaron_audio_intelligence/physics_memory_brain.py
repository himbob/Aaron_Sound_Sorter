"""Trainable memory brain for PhysicsVoter branch calibration.

This brain is intentionally separate from the folder brain.  It stores
human-approved audio fingerprints as low-level physics targets: top family,
broad branch, and approved label.  Runtime matching uses numeric audio
fingerprints only; source filenames and source folders are not read as evidence.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from aaron_audio_intelligence.shape_memory_brain import (
    PITCH_REGISTER_SENSITIVE_MEMORY_FEATURES,
    example_fingerprint,
    pad_vector,
    safe_int,
    scaled_vector_distance,
    shape_target_for_label,
    weighted_normalized_signature_vector,
    weighted_normalized_vector,
)
from aaron_audio_intelligence.voter_memory_brain import VOTER_MEMORY_KEY
from aaron_sound_sorter.core import FP_SIZE, FeatureRow
from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.training_labels import label_default_structure, public_label, top_for_public_label

PHYSICS_MEMORY_BRAIN_NAME = "stage4_physics_memory_brain.json"
PHYSICS_MEMORY_KEY = "physics_memory"
MAX_PHYSICS_MEMORY_EXAMPLES_PER_TARGET = 320

_METADATA_KEYS = (
    "version",
    "feature_names",
    "feature_size",
    "feature_weights",
    "scaler_mean",
    "scaler_std",
)

_PHYSICS_SIGNATURE_FEATURES = frozenset(
    {
        "log_rolloff85_hz",
        "log_crest",
        "spectral_flux_mean",
        "spectral_flatness_mean",
        "spectral_entropy_mean",
        "log_decay_ratio",
        "stereo_width",
        "mid_side_ratio",
        "centroid_slope_norm",
        "log_transient_count",
        "sub_bass_ratio_lt_150hz",
        "bass_ratio_150_500hz",
        "mid_ratio_500_2000hz",
        "presence_ratio_2000_8000hz",
        "air_ratio_gt_8000hz",
        "zcr_mean",
        "temporal_centroid_ratio",
        "onset_interval_regularity",
        "attack_rise_time_norm",
        "onset_span_ratio",
        "event_rate_hz",
        "tail_energy_ratio",
        "spectral_flux_variance",
        "pitch_confidence",
        "f0_median_hz",
        "f0_voiced_ratio",
        "f0_stability_cents",
        "f0_slope_cents_per_sec",
        "harmonic_to_noise_ratio",
        "harmonic_peak_count",
        "harmonic_energy_ratio",
        "inharmonicity",
        "fundamental_dominance_ratio",
        "overtone_slope",
        "attack_pitch_confidence",
        "body_pitch_confidence",
        "tail_pitch_confidence",
        "attack_flatness",
        "body_flatness",
        "tail_flatness",
        "attack_entropy",
        "body_entropy",
        "tail_entropy",
        "attack_zcr",
        "body_zcr",
        "tail_zcr",
        "attack_high_ratio",
        "body_high_ratio",
        "tail_high_ratio",
        "attack_low_ratio",
        "body_low_ratio",
        "tail_low_ratio",
        "noise_burst_duration_ms",
        "high_band_decay_slope",
        "spectral_centroid_decay_slope",
        "noise_tail_decay_slope",
        "attack_noise_ratio",
        "body_noise_ratio",
        "tail_noise_ratio",
        "low_peak_frequency_hz",
        "sub_decay_time_ms",
        "low_pitch_drop_hz_per_sec",
        "low_end_phase_correlation",
        "mono_low_energy_ratio",
        "low_band_stereo_width",
        "loop_pitched_event_ratio",
        "loop_percussive_event_ratio",
        "loop_noisy_event_ratio",
        "loop_mean_event_low_ratio",
        "loop_mean_event_mid_ratio",
        "loop_mean_event_high_ratio",
        "loop_sustained_tonal_frame_ratio",
        "loop_drumlike_frame_ratio",
        "loop_tonal_to_percussive_balance",
        "loop_non_event_tonal_ratio",
        "loop_true_repetition_score",
        "formant_like_peak_spacing",
    }
)

_PHYSICS_REGISTER_INVARIANT_EXCLUSIONS = frozenset(
    {
        *PITCH_REGISTER_SENSITIVE_MEMORY_FEATURES,
        "low_pitch_drop_hz_per_sec",
    }
)


@dataclass(frozen=True)
class PhysicsMemoryTarget:
    """Low-level physics target derived from a human-approved label.

    Args:
        key: Stable target key used inside the memory payload.
        top_family: Top family such as ``Drums``, ``Instruments``, or ``FX``.
        branch: Broad physics branch inside the top family.

    Side Effects:
        None.
    """

    key: str
    top_family: str
    branch: str


@dataclass(frozen=True)
class PhysicsMemoryMatch:
    """Nearest learned PhysicsVoter branch match for a query fingerprint.

    Args:
        matched: True when the query is accepted by the learned physics memory.
        target_key: Stable top/branch memory target key.
        top_family: Learned target top family.
        branch: Learned target branch.
        label: Human-approved taxonomy label nearest to the query.
        shape: Broad shape associated with the nearest example.
        nearest_distance: Signature distance to learned evidence.
        confidence: Calibration confidence contributed by this memory lane.
        example_count: Number of examples inspected for the winning target.
        effective_weight: Human support weight for the winning target.
        match_kind: Diagnostic match mode.
        threshold: Active maximum accepted distance.

    Side Effects:
        None.
    """

    matched: bool
    target_key: str
    top_family: str
    branch: str
    label: str
    shape: str
    nearest_distance: float
    confidence: float
    example_count: int
    effective_weight: int
    match_kind: str
    threshold: float

    def evidence(self) -> dict[str, Any]:
        """Return flat and nested diagnostics for manifests and voters."""
        nested = {
            "enabled": self.example_count > 0,
            "matched": bool(self.matched),
            "target_key": str(self.target_key),
            "top_family": str(self.top_family),
            "branch": str(self.branch),
            "label": str(self.label),
            "shape": str(self.shape),
            "confidence": round(float(self.confidence), 6),
            "nearest_distance": round(float(self.nearest_distance), 6),
            "threshold": round(float(self.threshold), 6),
            "example_count": int(self.example_count),
            "effective_weight": int(self.effective_weight),
            "match_kind": str(self.match_kind),
            "policy": "teacher_fingerprint_physics_branch_calibration",
        }
        return {
            "learned_physics_memory": nested,
            "learned_physics_memory_enabled": nested["enabled"],
            "learned_physics_memory_matched": nested["matched"],
            "learned_physics_memory_target_key": nested["target_key"],
            "learned_physics_memory_top_family": nested["top_family"],
            "learned_physics_memory_branch": nested["branch"],
            "learned_physics_memory_label": nested["label"],
            "learned_physics_memory_shape": nested["shape"],
            "learned_physics_memory_confidence": nested["confidence"],
            "learned_physics_memory_nearest_distance": nested["nearest_distance"],
            "learned_physics_memory_threshold": nested["threshold"],
            "learned_physics_memory_example_count": nested["example_count"],
            "learned_physics_memory_effective_weight": nested["effective_weight"],
            "learned_physics_memory_match_kind": nested["match_kind"],
            "learned_physics_memory_policy": nested["policy"],
        }


def build_empty_physics_memory_brain(base_brain: dict[str, Any]) -> dict[str, Any]:
    """Return an empty brain JSON ready for PhysicsVoter training.

    Args:
        base_brain: Active full brain used as the source of feature metadata.

    Returns:
        Brain-shaped dictionary with an empty physics-memory payload.

    Side Effects:
        None.
    """
    memory: dict[str, Any] = {key: deepcopy(base_brain[key]) for key in _METADATA_KEYS if key in base_brain}
    memory.update(
        {
            "brain_type": "physics_memory_brain",
            "labels": [],
            PHYSICS_MEMORY_KEY: empty_physics_memory_payload(),
            "physics_memory_policy": "gui_corrections_teach_physics_top_family_and_branch",
        }
    )
    return memory


def empty_physics_memory_payload() -> dict[str, Any]:
    """Return the empty nested physics-memory payload."""
    return {
        "version": "v1",
        "examples_by_target": {},
        "centroids_by_target": {},
        "counts_by_target": {},
        "effective_weights_by_target": {},
        "target_metadata": {},
    }


def is_physics_memory_brain(brain: dict[str, Any] | None) -> bool:
    """Return true when ``brain`` is the dedicated physics-memory lane."""
    return bool(isinstance(brain, dict) and brain.get("brain_type") == "physics_memory_brain")


def merge_physics_memory_into_brain(
    brain: dict[str, Any],
    physics_memory_brain: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge standalone physics-memory evidence into an in-memory brain view.

    Args:
        brain: Loaded full brain used for the current sort run.
        physics_memory_brain: Optional loaded physics-memory brain.

    Returns:
        ``brain`` after in-memory merge.

    Side Effects:
        Mutates only the passed ``brain`` object for this run. No files are
        written.
    """
    loaded_dedicated_physics_memory = False
    if is_physics_memory_brain(physics_memory_brain):
        source_payload = physics_memory_brain.get(PHYSICS_MEMORY_KEY)
        if isinstance(source_payload, dict):
            merge_physics_memory_payload(ensure_physics_memory_payload(brain), source_payload)
            loaded_dedicated_physics_memory = True
    if not loaded_dedicated_physics_memory and not physics_memory_payload_has_examples(brain.get(PHYSICS_MEMORY_KEY)):
        seed_physics_memory_from_voter_memory(brain)
    payload = brain.get(PHYSICS_MEMORY_KEY)
    if isinstance(payload, dict):
        brain["_physics_memory_brain_loaded"] = loaded_dedicated_physics_memory
        brain["_physics_memory_target_count"] = len(payload.get("examples_by_target", {}))
    return brain


def physics_memory_payload_has_examples(payload: Any) -> bool:
    """Return true when a physics-memory payload has target examples.

    Args:
        payload: Optional physics-memory payload from a loaded brain.

    Returns:
        True when at least one target stores at least one example.

    Side Effects:
        None.
    """
    if not isinstance(payload, dict):
        return False
    examples_by_target = payload.get("examples_by_target")
    if not isinstance(examples_by_target, dict):
        return False
    return any(isinstance(examples, list) and bool(examples) for examples in examples_by_target.values())


def update_physics_memory_with_row(
    brain: dict[str, Any],
    row: FeatureRow,
    *,
    evidence_weight: int,
    memory_origin: str = "gui_correction",
) -> str:
    """Store one human correction as learned PhysicsVoter branch evidence.

    Args:
        brain: Brain dictionary to mutate.
        row: Measured correction row with a human-approved taxonomy label.
        evidence_weight: Bounded human-correction support weight.
        memory_origin: Audit source for this teacher example.

    Returns:
        Learned target key touched by the update, or an empty string if no
        physics target was derivable.

    Side Effects:
        Mutates ``brain`` by adding one numeric fingerprint example and
        refreshing that target's centroid summary.
    """
    target = physics_memory_target_for_label(row.label, row.structure)
    if not target.key:
        return ""
    payload = ensure_physics_memory_payload(brain)
    examples_by_target = ensure_payload_map(payload, "examples_by_target")
    examples = examples_by_target.setdefault(target.key, [])
    if not isinstance(examples, list):
        examples = []
        examples_by_target[target.key] = examples
    existing = next(
        (
            example
            for example in examples
            if isinstance(example, dict) and str(example.get("source_path")) == str(row.path)
        ),
        None,
    )
    if isinstance(existing, dict):
        reinforce_physics_memory_example(existing, evidence_weight=evidence_weight)
    else:
        examples.append(
            physics_memory_example_from_row(
                row,
                target,
                evidence_weight=evidence_weight,
                memory_origin=memory_origin,
            )
        )
    del examples[:-MAX_PHYSICS_MEMORY_EXAMPLES_PER_TARGET]
    refresh_physics_memory_target(payload, target.key)
    record_target_metadata(payload, target, row.label)
    if is_physics_memory_brain(brain):
        labels = [str(label) for label in brain.get("labels", []) if str(label)]
        if target.key not in labels:
            labels.append(target.key)
            brain["labels"] = sorted(labels)
    brain["physics_memory_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return target.key


def attach_physics_memory_to_facts(brain: dict[str, Any], facts: SharedAudioFacts) -> PhysicsMemoryMatch:
    """Attach the nearest learned physics-memory match to ``facts.evidence``.

    Args:
        brain: Active in-memory brain with optional merged physics memory.
        facts: Shared facts for the current audio file.

    Returns:
        The nearest physics-memory match.

    Side Effects:
        Updates ``facts.evidence`` with flat and nested diagnostics.
    """
    match = physics_memory_match_for_facts(brain, facts.feature_vector)
    if isinstance(getattr(facts, "evidence", None), dict):
        facts.evidence.update(match.evidence())
    return match


def physics_memory_match_for_facts(
    brain: dict[str, Any],
    feature_vector: tuple[float, ...] | list[float],
) -> PhysicsMemoryMatch:
    """Return the strongest learned physics-memory match for a query vector."""
    payload = brain.get(PHYSICS_MEMORY_KEY)
    if not isinstance(payload, dict):
        return no_physics_memory_match()
    examples_by_target = payload.get("examples_by_target")
    if not isinstance(examples_by_target, dict) or not examples_by_target:
        return no_physics_memory_match()
    query = physics_signature_vector(brain, feature_vector)
    if query.size <= 0:
        return no_physics_memory_match()
    register_invariant_query = physics_register_invariant_signature_vector(brain, feature_vector)

    best_target = ""
    best_top = ""
    best_branch = ""
    best_label = ""
    best_shape = ""
    best_distance = float("inf")
    best_count = 0
    best_weight = 0
    best_match_kind = "teacher_physics_cloud"
    for target_key, examples in examples_by_target.items():
        if not isinstance(examples, list):
            continue
        valid_count = 0
        effective_weight = 0
        nearest = float("inf")
        nearest_kind = "teacher_physics_cloud"
        nearest_example: dict[str, Any] | None = None
        for example in examples:
            if not isinstance(example, dict):
                continue
            vector = example_fingerprint(example)
            if vector.size <= 0:
                continue
            signature = physics_signature_vector(brain, vector)
            distance = scaled_vector_distance(query, signature)
            invariant_signature = physics_register_invariant_signature_vector(brain, vector)
            invariant_distance = scaled_vector_distance(
                register_invariant_query,
                invariant_signature,
                reference_size=query.size,
            )
            if invariant_distance < distance:
                distance = invariant_distance
                candidate_kind = "teacher_physics_register_invariant_cloud"
            else:
                candidate_kind = "teacher_physics_cloud"
            if math.isfinite(distance):
                valid_count += 1
                effective_weight = max(effective_weight, safe_int(example.get("human_override_evidence_weight")))
                if distance < nearest:
                    nearest = distance
                    nearest_kind = candidate_kind
                    nearest_example = example
        if valid_count > 0 and nearest < best_distance:
            best_target = str(target_key)
            best_distance = nearest
            best_count = valid_count
            best_weight = effective_weight
            best_match_kind = nearest_kind
            best_top = str(nearest_example.get("top_family", "")) if isinstance(nearest_example, dict) else ""
            best_branch = str(nearest_example.get("physics_branch", "")) if isinstance(nearest_example, dict) else ""
            best_label = str(nearest_example.get("approved_label", "")) if isinstance(nearest_example, dict) else ""
            best_shape = str(nearest_example.get("target_shape", "")) if isinstance(nearest_example, dict) else ""

    if not best_target:
        return no_physics_memory_match()
    threshold = physics_memory_distance_threshold(best_weight, example_count=best_count)
    matched = bool(best_distance <= threshold)
    match_kind = "none"
    confidence = 0.0
    if matched:
        exact_threshold = min(1.20, max(0.18, 0.18 + math.log1p(max(1, best_weight)) / 12.0))
        match_kind = (
            "fingerprint"
            if best_match_kind == "teacher_physics_cloud" and best_distance <= exact_threshold
            else best_match_kind
        )
        distance_ratio = min(1.0, best_distance / max(threshold, 1e-6))
        confidence = min(0.96, max(0.66, 0.96 - 0.28 * distance_ratio + math.log1p(max(1, best_count)) / 30.0))
    return PhysicsMemoryMatch(
        matched=matched,
        target_key=best_target,
        top_family=best_top,
        branch=best_branch,
        label=best_label,
        shape=best_shape,
        nearest_distance=best_distance,
        confidence=confidence,
        example_count=best_count,
        effective_weight=best_weight,
        match_kind=match_kind,
        threshold=threshold,
    )


def physics_memory_distance_threshold(effective_weight: int, *, example_count: int) -> float:
    """Return accepted signature distance for learned physics-memory recall."""
    weight = max(1, int(effective_weight or 0))
    count = max(1, int(example_count or 0))
    return min(4.85, 1.85 + math.log1p(weight) / 4.4 + math.log1p(count) / 7.5)


def physics_signature_vector(
    brain: dict[str, Any],
    feature_vector: tuple[float, ...] | list[float] | np.ndarray,
) -> np.ndarray:
    """Return a compact normalized vector for physics-memory matching.

    Args:
        brain: Brain metadata containing feature names, scaler, and weights.
        feature_vector: Raw feature vector for the query or teacher example.

    Returns:
        A normalized physics-signature vector. Falls back to the first 32
        weighted features when test brains do not use real feature names.

    Side Effects:
        None.
    """
    weighted = weighted_normalized_vector(brain, feature_vector)
    if weighted.size <= 0:
        return np.asarray([], dtype=np.float32)
    names = [str(name) for name in brain.get("feature_names", [])]
    selected = [index for index, name in enumerate(names[: weighted.size]) if name in _PHYSICS_SIGNATURE_FEATURES]
    if len(selected) < 12:
        return weighted[: min(32, weighted.size)].astype(np.float32)
    return weighted[np.asarray(selected, dtype=np.int64)].astype(np.float32)


def physics_register_invariant_signature_vector(
    brain: dict[str, Any],
    feature_vector: tuple[float, ...] | list[float] | np.ndarray,
) -> np.ndarray:
    """Return a physics signature that can generalize across key/register.

    Args:
        brain: Brain metadata containing feature names, scaler, and weights.
        feature_vector: Raw feature vector for the query or teacher example.

    Returns:
        Weighted physics signature with exact pitch/register coordinates
        removed. Falls back to the existing physics signature for test brains
        without named production features.

    Side Effects:
        None.
    """
    return weighted_normalized_signature_vector(
        brain,
        feature_vector,
        include_feature_names=_PHYSICS_SIGNATURE_FEATURES,
        exclude_feature_names=_PHYSICS_REGISTER_INVARIANT_EXCLUSIONS,
        minimum_feature_count=32,
        fallback_width=32,
    )


def ensure_physics_memory_payload(brain: dict[str, Any]) -> dict[str, Any]:
    """Return a mutable physics-memory payload on ``brain``."""
    payload = brain.get(PHYSICS_MEMORY_KEY)
    if not isinstance(payload, dict):
        payload = empty_physics_memory_payload()
        brain[PHYSICS_MEMORY_KEY] = payload
    for key, value in empty_physics_memory_payload().items():
        if key not in payload:
            payload[key] = deepcopy(value)
    return payload


def merge_physics_memory_payload(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Merge one physics-memory payload into another."""
    target_examples = ensure_payload_map(target, "examples_by_target")
    source_examples = source.get("examples_by_target")
    if not isinstance(source_examples, dict):
        return
    for target_key, examples in source_examples.items():
        if not isinstance(examples, list):
            continue
        current = target_examples.setdefault(str(target_key), [])
        if not isinstance(current, list):
            current = []
            target_examples[str(target_key)] = current
        indexes_by_path = {
            str(example.get("source_path", "")): index
            for index, example in enumerate(current)
            if isinstance(example, dict) and str(example.get("source_path", ""))
        }
        for example in examples:
            if not isinstance(example, dict):
                continue
            example_path = str(example.get("source_path", ""))
            if example_path and example_path in indexes_by_path:
                existing_index = indexes_by_path[example_path]
                existing_weight = safe_int(current[existing_index].get("human_override_evidence_weight"))
                incoming_weight = safe_int(example.get("human_override_evidence_weight"))
                if incoming_weight > existing_weight:
                    current[existing_index] = deepcopy(example)
                continue
            if example_path:
                indexes_by_path[example_path] = len(current)
            current.append(deepcopy(example))
        del current[:-MAX_PHYSICS_MEMORY_EXAMPLES_PER_TARGET]
        refresh_physics_memory_target(target, str(target_key))
    merge_target_metadata(target, source)


def seed_physics_memory_from_voter_memory(brain: dict[str, Any]) -> None:
    """Seed physics memory in-memory from older voter-memory examples.

    This keeps existing GUI corrections useful after the new physics brain is
    introduced. The conversion uses stored approved labels and numeric
    fingerprints from the old memory lane; it is not source-name evidence.
    """
    voter_payload = brain.get(VOTER_MEMORY_KEY)
    if not isinstance(voter_payload, dict):
        return
    examples_by_role = voter_payload.get("examples_by_role")
    if not isinstance(examples_by_role, dict):
        return
    payload = ensure_physics_memory_payload(brain)
    examples_by_target = ensure_payload_map(payload, "examples_by_target")
    for examples in examples_by_role.values():
        if not isinstance(examples, list):
            continue
        for example in examples:
            if not isinstance(example, dict):
                continue
            label = str(example.get("approved_label", ""))
            if not label:
                continue
            target = physics_memory_target_for_label(label, str(example.get("structure", "")))
            if not target.key:
                continue
            current = examples_by_target.setdefault(target.key, [])
            if not isinstance(current, list):
                current = []
                examples_by_target[target.key] = current
            source_path = str(example.get("source_path", ""))
            if source_path and any(
                isinstance(existing, dict) and str(existing.get("source_path", "")) == source_path
                for existing in current
            ):
                continue
            migrated = physics_memory_example_from_memory_example(example, target)
            current.append(migrated)
            del current[:-MAX_PHYSICS_MEMORY_EXAMPLES_PER_TARGET]
            record_target_metadata(payload, target, label)
    for target_key in list(examples_by_target):
        refresh_physics_memory_target(payload, str(target_key))


def ensure_payload_map(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a nested mutable mapping from a physics-memory payload."""
    value = payload.get(key)
    if isinstance(value, dict):
        return value
    payload[key] = {}
    return payload[key]


def physics_memory_example_from_row(
    row: FeatureRow,
    target: PhysicsMemoryTarget,
    *,
    evidence_weight: int,
    memory_origin: str = "gui_correction",
) -> dict[str, Any]:
    """Return one stored physics-memory example."""
    return {
        "source_path": str(row.path),
        "approved_label": str(row.label),
        "target_key": str(target.key),
        "top_family": str(target.top_family),
        "physics_branch": str(target.branch),
        "target_shape": str(shape_target_for_label(row.label, row.structure)),
        "structure": str(row.structure),
        "duration_sec": float(row.duration_sec),
        "fingerprint": finite_feature_vector(row.fingerprint),
        "memory_origin": str(memory_origin or "gui_correction"),
        "incremental_gui_correction": str(memory_origin or "gui_correction") == "gui_correction",
        "physics_memory_added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "human_override_confirmation_count": 1,
        "human_override_evidence_weight": safe_int(evidence_weight),
    }


def physics_memory_example_from_memory_example(
    example: dict[str, Any],
    target: PhysicsMemoryTarget,
) -> dict[str, Any]:
    """Return one physics-memory example migrated from voter memory."""
    return {
        "source_path": str(example.get("source_path", "")),
        "approved_label": str(example.get("approved_label", "")),
        "target_key": str(target.key),
        "top_family": str(target.top_family),
        "physics_branch": str(target.branch),
        "target_shape": str(example.get("target_shape", "")),
        "structure": str(example.get("structure", "")),
        "duration_sec": float(example.get("duration_sec", 0.0) or 0.0),
        "fingerprint": finite_feature_vector(example.get("fingerprint", [])),
        "memory_origin": "voter_memory_migration",
        "incremental_gui_correction": bool(example.get("incremental_gui_correction", False)),
        "physics_memory_added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "human_override_confirmation_count": safe_int(example.get("human_override_confirmation_count")) or 1,
        "human_override_evidence_weight": safe_int(example.get("human_override_evidence_weight")),
    }


def reinforce_physics_memory_example(example: dict[str, Any], *, evidence_weight: int) -> None:
    """Increase human support for an existing physics-memory example."""
    previous = safe_int(example.get("human_override_evidence_weight"))
    confirmation_count = safe_int(example.get("human_override_confirmation_count")) + 1
    example["human_override_confirmation_count"] = confirmation_count
    example["human_override_evidence_weight"] = max(previous, safe_int(evidence_weight))
    example["physics_memory_last_confirmed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


def refresh_physics_memory_target(payload: dict[str, Any], target_key: str) -> None:
    """Refresh centroid/count summaries for one learned physics target."""
    examples = ensure_payload_map(payload, "examples_by_target").get(target_key, [])
    if not isinstance(examples, list):
        examples = []
    vectors: list[np.ndarray] = []
    weight_total = 0
    for example in examples:
        if not isinstance(example, dict):
            continue
        vector = example_fingerprint(example)
        if vector.size != FP_SIZE:
            continue
        weight = max(1, safe_int(example.get("human_override_evidence_weight")))
        vectors.append(vector * float(weight))
        weight_total += weight
    ensure_payload_map(payload, "counts_by_target")[target_key] = len(vectors)
    ensure_payload_map(payload, "effective_weights_by_target")[target_key] = int(weight_total)
    centroids = ensure_payload_map(payload, "centroids_by_target")
    if vectors and weight_total > 0:
        centroids[target_key] = (np.sum(np.vstack(vectors), axis=0) / float(weight_total)).astype(float).tolist()
    else:
        centroids[target_key] = []


def record_target_metadata(payload: dict[str, Any], target: PhysicsMemoryTarget, label: str) -> None:
    """Persist target metadata used by diagnostics."""
    metadata = ensure_payload_map(payload, "target_metadata")
    metadata[target.key] = {
        "top_family": target.top_family,
        "branch": target.branch,
        "last_label": public_label(label),
    }


def merge_target_metadata(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Merge target metadata blocks from one payload into another."""
    source_metadata = source.get("target_metadata")
    if not isinstance(source_metadata, dict):
        return
    target_metadata = ensure_payload_map(target, "target_metadata")
    for key, value in source_metadata.items():
        if isinstance(value, dict):
            target_metadata[str(key)] = deepcopy(value)


def physics_memory_target_for_label(label: str, structure: str | None = None) -> PhysicsMemoryTarget:
    """Return the physics top/branch target implied by a supervised label."""
    normalized = public_label(label)
    top = top_for_public_label(normalized)
    structure_name = str(structure or label_default_structure(normalized) or "").lower()
    parts = [part.strip().lower() for part in normalized.replace("\\", "/").split("/") if part.strip()]
    joined = "/".join(parts)
    if top == "Drums":
        branch = drum_physics_branch_for_label(joined, structure_name)
    elif top == "Instruments":
        branch = instrument_physics_branch_for_label(joined)
    elif top == "FX":
        branch = fx_physics_branch_for_label(joined)
    else:
        branch = ""
    key = f"{top}/{branch}" if top and branch else ""
    return PhysicsMemoryTarget(key=key, top_family=top, branch=branch)


def drum_physics_branch_for_label(joined_label: str, structure_name: str) -> str:
    """Map a drum taxonomy label to a broad measured physics branch."""
    if "drum loops" in joined_label or structure_name == "loop":
        return "DrumLoop"
    if "kick" in joined_label:
        return "Kick"
    if "snare" in joined_label:
        return "Snare"
    if "clap" in joined_label or "snap" in joined_label or "slap" in joined_label:
        return "Clap"
    if "hat" in joined_label:
        return "Hat"
    if any(token in joined_label for token in ("cymbal", "crash", "ride", "splash")):
        return "Cymbal"
    if any(token in joined_label for token in ("tom", "conga", "bongo", "tabla")):
        return "TomOrConga"
    if any(token in joined_label for token in ("rim", "stick", "clave", "wood block")):
        return "RimOrStick"
    if "shaker" in joined_label or "tambourine" in joined_label:
        return "ShakerTambourine"
    if any(token in joined_label for token in ("guiro", "scrape", "rasp")):
        return "ScrapeGuiro"
    if any(token in joined_label for token in ("metallic", "cowbell", "bell")):
        return "MetallicPercussion"
    return "DrumLoop" if structure_name == "loop" else "MetallicPercussion"


def instrument_physics_branch_for_label(joined_label: str) -> str:
    """Map an instrument taxonomy label to a broad measured physics branch."""
    if any(token in joined_label for token in ("voice", "vocal", "choir", "spoken")):
        return "Voice"
    if "bass" in joined_label or "808" in joined_label:
        return "Bass"
    if any(token in joined_label for token in ("synth", "lead", "pad", "arp")):
        return "Synth"
    if any(token in joined_label for token in ("piano", "keys", "rhodes", "wurlitzer", "organ", "clav")):
        return "KeysPiano"
    if any(token in joined_label for token in ("guitar", "koto", "plucked", "harp", "banjo", "mandolin", "sitar")):
        return "PluckedString"
    if any(token in joined_label for token in ("sax", "reed", "woodwind", "flute", "clarinet", "bassoon")):
        return "Woodwinds"
    if any(token in joined_label for token in ("brass", "trumpet", "trombone", "horn")):
        return "Brass"
    if any(token in joined_label for token in ("strings", "violin", "viola", "cello", "bowed")):
        return "Strings"
    if any(token in joined_label for token in ("mallet", "bell", "vibraphone", "marimba", "glockenspiel")):
        return "MalletBell"
    if "mixed musical" in joined_label or "instrument loops" in joined_label:
        return "MixedInstrument"
    return "MixedInstrument"


def fx_physics_branch_for_label(joined_label: str) -> str:
    """Map an FX taxonomy label to a broad measured physics branch."""
    if any(token in joined_label for token in ("human and voice", "spoken", "mouth", "breath", "scream", "crowd")):
        return "HumanCreatureFX"
    if "applause" in joined_label:
        return "HumanCreatureFX"
    if any(token in joined_label for token in ("impact", "boom", "slam", "sub hit")):
        return "ImpactHit"
    if "riser" in joined_label or "build" in joined_label:
        return "RiserBuild"
    if "drop" in joined_label or "downlifter" in joined_label:
        return "DropDownlifter"
    if "whoosh" in joined_label or "sweep" in joined_label:
        return "WhooshSweep"
    if "reverse" in joined_label or "swell" in joined_label:
        return "ReverseSwell"
    if "glitch" in joined_label or "stutter" in joined_label:
        return "GlitchStutter"
    if "blip" in joined_label or "beep" in joined_label:
        return "BlipBeep"
    if "siren" in joined_label or "alarm" in joined_label:
        return "SirenAlarm"
    if "formant" in joined_label:
        return "FormantFX"
    if any(token in joined_label for token in ("radio", "electrical", "electric", "zap")):
        return "RadioElectrical"
    if any(token in joined_label for token in ("machine", "motor", "engine", "mechanical")):
        return "MachineMechanical"
    if any(token in joined_label for token in ("door", "foley", "glass", "metal", "wood", "cloth")):
        return "FoleyMaterial"
    if any(token in joined_label for token in ("coin", "key", "small object")):
        return "SmallObjectCluster"
    if any(token in joined_label for token in ("animal", "bird", "cat", "dog", "cricket", "creature")):
        return "HumanCreatureFX"
    if any(token in joined_label for token in ("texture", "ambience", "water", "rain", "ocean", "wind", "fire")):
        return "TextureAmbience"
    return "DesignedNoiseHybrid"


def no_physics_memory_match() -> PhysicsMemoryMatch:
    """Return the empty learned physics-memory match."""
    return PhysicsMemoryMatch(
        matched=False,
        target_key="",
        top_family="",
        branch="",
        label="",
        shape="",
        nearest_distance=float("inf"),
        confidence=0.0,
        example_count=0,
        effective_weight=0,
        match_kind="none",
        threshold=physics_memory_distance_threshold(0, example_count=0),
    )


def finite_feature_vector(values: Any) -> list[float]:
    """Return a JSON-safe finite fingerprint vector."""
    return pad_vector(np.asarray(values, dtype=np.float32), 0.0).astype(float).tolist()
