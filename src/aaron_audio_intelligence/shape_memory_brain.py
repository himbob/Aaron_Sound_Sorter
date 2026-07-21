"""Trainable shape-memory brain for GUI corrections.

This module stores human-approved correction fingerprints as broad structural
shape evidence.  It is intentionally source-name blind: source paths are kept
only for audit and duplicate handling, while matching uses numeric fingerprints.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from aaron_sound_sorter.core import FEATURE_WEIGHTS, FP_SIZE, FeatureRow
from aaron_sound_sorter.training_labels import label_default_structure, public_label, top_for_public_label

SHAPE_MEMORY_BRAIN_NAME = "stage4_shape_memory_brain.json"
SHAPE_STARTER_MEMORY_BRAIN_NAME = "stage4_shape_memory_starter_brain.json"
SHAPE_MEMORY_KEY = "shape_memory"
MAX_SHAPE_MEMORY_EXAMPLES_PER_SHAPE = 240

_METADATA_KEYS = (
    "version",
    "feature_names",
    "feature_size",
    "feature_weights",
    "scaler_mean",
    "scaler_std",
)

PITCH_REGISTER_SENSITIVE_MEMORY_FEATURES = frozenset(
    {
        "f0_median_hz",
        "low_peak_frequency_hz",
        "top1_peak_frequency_hz",
        "top2_peak_frequency_hz",
        "top3_peak_frequency_hz",
    }
)

SHAPE_ROLE_MEMORY_FEATURES = frozenset(
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
        "zcr_mean",
        "temporal_centroid_ratio",
        "onset_interval_regularity",
        "attack_rise_time_norm",
        "onset_span_ratio",
        "event_rate_hz",
        "tail_energy_ratio",
        "spectral_flux_variance",
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
        "sub_attack_time_ms",
        "sub_decay_time_ms",
        "sub_to_click_offset_ms",
        "sub_sustain_ratio",
        "loop_pitched_event_ratio",
        "loop_percussive_event_ratio",
        "loop_noisy_event_ratio",
        "loop_event_timbre_diversity",
        "loop_sustained_tonal_frame_ratio",
        "loop_drumlike_frame_ratio",
        "loop_tonal_to_percussive_balance",
        "loop_mean_event_pitch_confidence",
        "loop_mean_event_noise_ratio",
        "loop_mean_event_low_ratio",
        "loop_mean_event_high_ratio",
        "loop_non_event_tonal_ratio",
    }
)


@dataclass(frozen=True)
class ShapeMemoryMatch:
    """Nearest learned shape-memory match for a query fingerprint.

    Args:
        matched: True when the query is close enough to learned shape evidence.
        shape: Learned structural shape target.
        nearest_distance: Weighted normalized distance to the nearest stored
            correction fingerprint.
        confidence: Shape confidence contributed by the learned memory lane.
        example_count: Valid correction examples inspected for the winning
            shape.
        effective_weight: Human correction support weight for the winning
            shape.
        match_kind: Diagnostic match mode.

    Side Effects:
        None.
    """

    matched: bool
    shape: str
    nearest_distance: float
    confidence: float
    example_count: int
    effective_weight: int
    match_kind: str
    threshold: float

    def evidence(self) -> dict[str, Any]:
        """Return flat diagnostics suitable for ShapeVoter output."""
        return {
            "learned_shape_memory_enabled": self.example_count > 0,
            "learned_shape_memory_matched": bool(self.matched),
            "learned_shape_memory_shape": str(self.shape),
            "learned_shape_memory_confidence": round(float(self.confidence), 6),
            "learned_shape_memory_nearest_distance": round(float(self.nearest_distance), 6),
            "learned_shape_memory_threshold": round(float(self.threshold), 6),
            "learned_shape_memory_example_count": int(self.example_count),
            "learned_shape_memory_effective_weight": int(self.effective_weight),
            "learned_shape_memory_match_kind": str(self.match_kind),
            "learned_shape_memory_policy": "shape_voter_teacher_fingerprint_memory",
        }


def build_empty_shape_memory_brain(base_brain: dict[str, Any]) -> dict[str, Any]:
    """Return an empty brain JSON ready to store learned shape evidence.

    Args:
        base_brain: Active full brain used as the source of feature metadata.

    Returns:
        Brain-shaped dictionary with no shape examples.

    Side Effects:
        None.
    """
    memory: dict[str, Any] = {key: deepcopy(base_brain[key]) for key in _METADATA_KEYS if key in base_brain}
    memory.update(
        {
            "brain_type": "shape_memory_brain",
            "labels": [],
            SHAPE_MEMORY_KEY: empty_shape_memory_payload(),
            "shape_memory_policy": "gui_correction_teaches_shape_voter_only",
        }
    )
    return memory


def empty_shape_memory_payload() -> dict[str, Any]:
    """Return the empty nested shape-memory payload.

    Returns:
        JSON-safe payload with example, centroid, count, and effective-weight
        maps.

    Side Effects:
        None.
    """
    return {
        "version": "v1",
        "shape_examples_by_shape": {},
        "shape_centroids_by_shape": {},
        "shape_counts_by_shape": {},
        "shape_effective_weights_by_shape": {},
    }


def is_shape_memory_brain(brain: dict[str, Any] | None) -> bool:
    """Return true when a loaded brain is the standalone shape-memory lane.

    Args:
        brain: Optional loaded brain dictionary.

    Returns:
        True when ``brain`` is the dedicated ShapeVoter memory brain.

    Side Effects:
        None.
    """
    return bool(isinstance(brain, dict) and brain.get("brain_type") == "shape_memory_brain")


def merge_shape_memory_into_brain(
    brain: dict[str, Any],
    shape_memory_brain: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge standalone shape-memory evidence into an in-memory brain view.

    Args:
        brain: Loaded full brain used for the current sort run.
        shape_memory_brain: Optional loaded shape-memory brain.

    Returns:
        ``brain`` after in-memory merge.

    Side Effects:
        Mutates only the passed ``brain`` object for this run. No files are
        written.
    """
    if not is_shape_memory_brain(shape_memory_brain):
        return brain
    source_payload = shape_memory_brain.get(SHAPE_MEMORY_KEY)
    if not isinstance(source_payload, dict):
        return brain
    target_payload = ensure_shape_memory_payload(brain)
    merge_shape_memory_payload(target_payload, source_payload)
    brain["_shape_memory_brain_loaded"] = True
    brain["_shape_memory_shape_count"] = len(target_payload.get("shape_examples_by_shape", {}))
    return brain


def update_shape_memory_with_row(
    brain: dict[str, Any],
    row: FeatureRow,
    *,
    evidence_weight: int,
    memory_source: str = "gui_correction",
) -> str:
    """Store one GUI correction as learned shape evidence.

    Args:
        brain: Brain dictionary to mutate.
        row: Measured correction row with a human-approved taxonomy label.
        evidence_weight: Bounded human-correction support weight.
        memory_source: Audit source for this teacher example.

    Returns:
        Learned shape name touched by the update.

    Side Effects:
        Mutates ``brain`` in memory by adding a shape-memory example and
        refreshing that shape's centroid.

    Important Constraints:
        The approved taxonomy label is used only as a supervised target for the
        human correction. Matching later uses numeric audio fingerprints.
    """
    shape = shape_target_for_label(row.label, row.structure)
    if not shape:
        return ""
    payload = ensure_shape_memory_payload(brain)
    examples_by_shape = ensure_payload_map(payload, "shape_examples_by_shape")
    examples = examples_by_shape.setdefault(shape, [])
    if not isinstance(examples, list):
        examples = []
        examples_by_shape[shape] = examples
    source_path = str(row.path)
    existing = next(
        (
            example
            for example in examples
            if isinstance(example, dict) and str(example.get("source_path")) == source_path
        ),
        None,
    )
    if isinstance(existing, dict):
        reinforce_shape_memory_example(existing, evidence_weight=evidence_weight)
    else:
        examples.append(
            shape_memory_example_from_row(row, shape, evidence_weight=evidence_weight, memory_source=memory_source)
        )
    if len(examples) > MAX_SHAPE_MEMORY_EXAMPLES_PER_SHAPE:
        del examples[:-MAX_SHAPE_MEMORY_EXAMPLES_PER_SHAPE]
    refresh_shape_memory_shape(payload, shape)
    if is_shape_memory_brain(brain):
        labels = [str(label) for label in brain.get("labels", []) if str(label)]
        if shape not in labels:
            labels.append(shape)
            brain["labels"] = sorted(labels)
    brain["shape_memory_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    brain["shape_memory_policy"] = shape_memory_policy_for_source(memory_source)
    return shape


def shape_target_for_label(label: str, structure: str | None = None) -> str:
    """Return the broad ShapeVoter target implied by an approved taxonomy label.

    Args:
        label: Human-approved public taxonomy label.
        structure: Optional approved structure such as ``loop`` or
            ``one_shot``.

    Returns:
        Broad structural shape target used by ShapeVoter memory, or an empty
        string when the label cannot teach shape.

    Side Effects:
        None.

    Important Constraints:
        The taxonomy label is a supervised correction target, not source-file
        evidence. Runtime matching uses fingerprints only.
    """
    normalized = public_label(label)
    top = top_for_public_label(normalized)
    structure_name = str(structure or label_default_structure(normalized) or "").lower()
    parts = [part.lower() for part in normalized.split("/") if part]
    joined = "/".join(parts)
    if top == "Drums":
        return "beat_loop" if structure_name == "loop" else "single_hit"
    if top == "Instruments":
        if structure_name == "loop":
            if "mixed musical loops" in joined or "multi instrument" in joined:
                return "mixed_instrument_loop"
            if "instrument loops" in joined:
                return "compound_musical_loop"
            if "bass" in parts:
                return "bass_phrase"
            return "pitched_repetition_phrase"
        if "voice" in parts:
            return "pitched_phrase_shape"
        return "solo_phrase"
    if top == "FX":
        if "riser" in joined or "build" in joined:
            return "transition_riser"
        if "drop" in joined or "downlifter" in joined:
            return "transition_drop"
        if "reverse" in joined:
            return "reverse_swell"
        if "whoosh" in joined or "sweep" in joined:
            return "whoosh_sweep"
        if "glitch" in joined or "stutter" in joined:
            return "designed_motion_fx_loop"
        if "blip" in joined or "beep" in joined:
            return "ui_blip"
        if "designed noise fx" in joined or "hybrid designed fx" in joined or "siren" in joined:
            return "hybrid_fx_motion"
        if "texture" in joined or "ambience" in joined or "noise" in joined or "static" in joined:
            return "texture_bed"
        if "impact" in joined or "boom" in joined or "slam" in joined:
            return "impact_with_tail"
        if "long fx" in joined or structure_name == "long_fx":
            return "hybrid_fx_motion"
        return "foley_action"
    return ""


def shape_memory_match_for_facts(
    brain: dict[str, Any],
    feature_vector: tuple[float, ...] | list[float],
) -> ShapeMemoryMatch:
    """Return the strongest learned shape-memory match for a query vector.

    Args:
        brain: Loaded full brain with any merged shape-memory payload.
        feature_vector: Numeric fingerprint for the current audio.

    Returns:
        Best learned shape match and diagnostics. The returned object is
        unmatched when no memory is close enough.

    Side Effects:
        None.
    """
    payload = brain.get(SHAPE_MEMORY_KEY)
    if not isinstance(payload, dict):
        return no_shape_memory_match()
    examples_by_shape = payload.get("shape_examples_by_shape")
    if not isinstance(examples_by_shape, dict) or not examples_by_shape:
        return no_shape_memory_match()
    query = weighted_normalized_vector(brain, feature_vector)
    if query.size <= 0:
        return no_shape_memory_match()
    shape_signature_query = weighted_normalized_signature_vector(
        brain,
        feature_vector,
        include_feature_names=SHAPE_ROLE_MEMORY_FEATURES,
        exclude_feature_names=PITCH_REGISTER_SENSITIVE_MEMORY_FEATURES,
        minimum_feature_count=24,
    )

    best_shape = ""
    best_distance = float("inf")
    best_count = 0
    best_weight = 0
    best_match_kind = "teacher_shape_cloud"
    for shape, examples in examples_by_shape.items():
        if not isinstance(examples, list):
            continue
        valid_count = 0
        effective_weight = 0
        nearest = float("inf")
        nearest_kind = "teacher_shape_cloud"
        for example in examples:
            if not isinstance(example, dict):
                continue
            vector = example_fingerprint(example)
            if vector.size <= 0:
                continue
            weighted_example = weighted_normalized_vector(brain, vector)
            distance = scaled_vector_distance(query, weighted_example)
            shape_signature_example = weighted_normalized_signature_vector(
                brain,
                vector,
                include_feature_names=SHAPE_ROLE_MEMORY_FEATURES,
                exclude_feature_names=PITCH_REGISTER_SENSITIVE_MEMORY_FEATURES,
                minimum_feature_count=24,
            )
            signature_distance = scaled_vector_distance(
                shape_signature_query,
                shape_signature_example,
                reference_size=query.size,
            )
            if signature_distance < distance:
                distance = signature_distance
                candidate_kind = "teacher_shape_signature_cloud"
            else:
                candidate_kind = "teacher_shape_cloud"
            if math.isfinite(distance):
                valid_count += 1
                if distance < nearest:
                    nearest = distance
                    nearest_kind = candidate_kind
                effective_weight = max(effective_weight, safe_int(example.get("human_override_evidence_weight")))
        if valid_count > 0 and nearest < best_distance:
            best_shape = str(shape)
            best_distance = nearest
            best_count = valid_count
            best_weight = effective_weight
            best_match_kind = nearest_kind

    if not best_shape:
        return no_shape_memory_match()
    threshold = shape_memory_distance_threshold(best_weight, example_count=best_count)
    matched = bool(best_distance <= threshold)
    match_kind = "none"
    confidence = 0.0
    if matched:
        exact_threshold = min(1.35, max(0.24, 0.24 + math.log1p(max(1, best_weight)) / 10.5))
        match_kind = (
            "fingerprint"
            if best_match_kind == "teacher_shape_cloud" and best_distance <= exact_threshold
            else best_match_kind
        )
        distance_ratio = min(1.0, best_distance / max(threshold, 1e-6))
        confidence = min(0.96, max(0.68, 0.96 - 0.24 * distance_ratio + math.log1p(max(1, best_count)) / 30.0))
    return ShapeMemoryMatch(
        matched=matched,
        shape=best_shape,
        nearest_distance=best_distance,
        confidence=confidence,
        example_count=best_count,
        effective_weight=best_weight,
        match_kind=match_kind,
        threshold=threshold,
    )


def shape_memory_distance_threshold(effective_weight: int, *, example_count: int) -> float:
    """Return accepted weighted distance for learned shape-memory recall.

    Args:
        effective_weight: Human correction support weight for a shape.
        example_count: Number of valid examples for that shape.

    Returns:
        Maximum normalized weighted distance allowed for a match.

    Side Effects:
        None.
    """
    weight = max(1, int(effective_weight or 0))
    count = max(1, int(example_count or 0))
    return min(3.10, 1.00 + math.log1p(weight) / 6.8 + math.log1p(count) / 7.0)


def ensure_shape_memory_payload(brain: dict[str, Any]) -> dict[str, Any]:
    """Return a mutable shape-memory payload on ``brain``.

    Args:
        brain: Brain dictionary that may or may not already contain shape
            memory.

    Returns:
        Mutable shape-memory payload attached to ``brain``.

    Side Effects:
        Mutates ``brain`` when the payload is absent or incomplete.
    """
    payload = brain.get(SHAPE_MEMORY_KEY)
    if not isinstance(payload, dict):
        payload = empty_shape_memory_payload()
        brain[SHAPE_MEMORY_KEY] = payload
    for key, value in empty_shape_memory_payload().items():
        if key not in payload:
            payload[key] = deepcopy(value)
    return payload


def merge_shape_memory_payload(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Merge one shape-memory payload into another.

    Args:
        target: Mutable destination shape-memory payload.
        source: Source shape-memory payload to copy from.

    Returns:
        None.

    Side Effects:
        Mutates ``target`` by copying unseen examples and refreshing affected
        centroid summaries.
    """
    target_examples = ensure_payload_map(target, "shape_examples_by_shape")
    source_examples = source.get("shape_examples_by_shape")
    if not isinstance(source_examples, dict):
        return
    for shape, examples in source_examples.items():
        if not isinstance(examples, list):
            continue
        current = target_examples.setdefault(str(shape), [])
        if not isinstance(current, list):
            current = []
            target_examples[str(shape)] = current
        indexes_by_source_path = {
            str(example.get("source_path", "")): index
            for index, example in enumerate(current)
            if isinstance(example, dict) and str(example.get("source_path", ""))
        }
        for example in examples:
            if not isinstance(example, dict):
                continue
            source_path = str(example.get("source_path", ""))
            if source_path and source_path in indexes_by_source_path:
                existing_index = indexes_by_source_path[source_path]
                existing_weight = safe_int(current[existing_index].get("human_override_evidence_weight"))
                incoming_weight = safe_int(example.get("human_override_evidence_weight"))
                if incoming_weight > existing_weight:
                    current[existing_index] = deepcopy(example)
                continue
            if source_path:
                indexes_by_source_path[source_path] = len(current)
            current.append(deepcopy(example))
        del current[:-MAX_SHAPE_MEMORY_EXAMPLES_PER_SHAPE]
        refresh_shape_memory_shape(target, str(shape))


def ensure_payload_map(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a nested mutable mapping from a shape-memory payload.

    Args:
        payload: Shape-memory payload dictionary.
        key: Nested map key to fetch or create.

    Returns:
        Mutable nested mapping.

    Side Effects:
        Mutates ``payload`` when the requested mapping is absent or malformed.
    """
    value = payload.get(key)
    if isinstance(value, dict):
        return value
    payload[key] = {}
    return payload[key]


def shape_memory_example_from_row(
    row: FeatureRow,
    shape: str,
    *,
    evidence_weight: int,
    memory_source: str = "gui_correction",
) -> dict[str, Any]:
    """Return one stored shape-memory example.

    Args:
        row: Measured correction row.
        shape: Broad learned shape target.
        evidence_weight: Human correction support weight.

    Returns:
        JSON-safe example payload with fingerprint and audit fields.

    Side Effects:
        None.
    """
    return {
        "source_path": str(row.path),
        "approved_label": str(row.label),
        "target_shape": str(shape),
        "top": str(row.top),
        "structure": str(row.structure),
        "duration_sec": float(row.duration_sec),
        "fingerprint": [float(value) for value in row.fingerprint],
        "memory_source": str(memory_source or "gui_correction"),
        "incremental_gui_correction": str(memory_source or "gui_correction") == "gui_correction",
        "shape_memory_added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "human_override_confirmation_count": 1,
        "human_override_evidence_weight": safe_int(evidence_weight),
    }


def shape_memory_policy_for_source(memory_source: str) -> str:
    """Return audit policy text for a shape-memory teacher source.

    Args:
        memory_source: Source name stored on teacher examples.

    Returns:
        Policy string written into brain metadata.

    Side Effects:
        None.
    """
    if str(memory_source or "").strip() == "training_tree_starter":
        return "training_tree_teaches_shape_voter_low_trust_starter_memory"
    return "gui_correction_teaches_shape_voter_only"


def reinforce_shape_memory_example(example: dict[str, Any], *, evidence_weight: int) -> None:
    """Increase human support for an existing shape-memory example.

    Args:
        example: Stored shape-memory example to update.
        evidence_weight: Current correction support weight.

    Returns:
        None.

    Side Effects:
        Mutates ``example`` by increasing confirmation metadata.
    """
    previous = safe_int(example.get("human_override_evidence_weight"))
    confirmation_count = safe_int(example.get("human_override_confirmation_count")) + 1
    example["human_override_confirmation_count"] = confirmation_count
    example["human_override_evidence_weight"] = max(previous, safe_int(evidence_weight))
    example["shape_memory_last_confirmed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


def refresh_shape_memory_shape(payload: dict[str, Any], shape: str) -> None:
    """Refresh centroid/count summaries for one learned shape.

    Args:
        payload: Mutable shape-memory payload.
        shape: Shape target whose examples changed.

    Returns:
        None.

    Side Effects:
        Mutates summary maps inside ``payload``.
    """
    examples = ensure_payload_map(payload, "shape_examples_by_shape").get(shape, [])
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
    centroids_by_shape = ensure_payload_map(payload, "shape_centroids_by_shape")
    counts_by_shape = ensure_payload_map(payload, "shape_counts_by_shape")
    weights_by_shape = ensure_payload_map(payload, "shape_effective_weights_by_shape")
    counts_by_shape[shape] = len(vectors)
    weights_by_shape[shape] = weight_total
    if vectors and weight_total > 0:
        centroids_by_shape[shape] = (np.sum(np.vstack(vectors), axis=0) / float(weight_total)).astype(float).tolist()
    else:
        centroids_by_shape[shape] = []


def weighted_normalized_vector(
    brain: dict[str, Any], values: tuple[float, ...] | list[float] | np.ndarray
) -> np.ndarray:
    """Return a finite normalized and feature-weighted vector.

    Args:
        brain: Brain dictionary containing scaler and feature-weight metadata.
        values: Raw feature vector.

    Returns:
        ``FP_SIZE`` vector normalized with the brain scaler and feature weights.

    Side Effects:
        None.
    """
    vector = pad_vector(np.asarray(values, dtype=np.float32), 0.0)
    mean = pad_vector(
        np.asarray(brain.get("scaler_mean", np.zeros((FP_SIZE,), dtype=np.float32)), dtype=np.float32), 0.0
    )
    std = pad_vector(np.asarray(brain.get("scaler_std", np.ones((FP_SIZE,), dtype=np.float32)), dtype=np.float32), 1.0)
    weights = pad_vector(np.asarray(brain.get("feature_weights", FEATURE_WEIGHTS), dtype=np.float32), 1.0)
    safe_std = np.where(np.abs(std) < 1e-6, 1.0, std)
    return ((vector - mean) / safe_std * weights).astype(np.float32)


def weighted_normalized_signature_vector(
    brain: dict[str, Any],
    values: tuple[float, ...] | list[float] | np.ndarray,
    *,
    include_feature_names: frozenset[str] | set[str] | tuple[str, ...],
    exclude_feature_names: frozenset[str] | set[str] | tuple[str, ...] = (),
    include_feature_prefixes: tuple[str, ...] = (),
    minimum_feature_count: int = 8,
    fallback_width: int | None = None,
) -> np.ndarray:
    """Return a named, weighted feature signature for learned-memory matching.

    Args:
        brain: Brain metadata containing feature names, scaler, and weights.
        values: Raw fingerprint values.
        include_feature_names: Exact feature names to keep.
        exclude_feature_names: Exact feature names to remove even if included.
        include_feature_prefixes: Feature-name prefixes to keep, such as MFCC
            families.
        minimum_feature_count: Minimum named features required before using the
            named signature.
        fallback_width: Optional leading weighted-vector width for test brains
            that lack real feature names. ``None`` returns the full vector.

    Returns:
        Weighted normalized signature vector.

    Side Effects:
        None.
    """
    weighted = weighted_normalized_vector(brain, values)
    if weighted.size <= 0:
        return np.asarray([], dtype=np.float32)
    names = [str(name) for name in brain.get("feature_names", [])]
    include_names = {str(name) for name in include_feature_names}
    exclude_names = {str(name) for name in exclude_feature_names}
    selected: list[int] = []
    for index, feature_name in enumerate(names[: weighted.size]):
        exact_match = feature_name in include_names
        prefix_match = any(feature_name.startswith(prefix) for prefix in include_feature_prefixes)
        if (exact_match or prefix_match) and feature_name not in exclude_names:
            selected.append(index)
    if len(selected) < max(1, int(minimum_feature_count)):
        if fallback_width is None:
            return weighted.astype(np.float32)
        return weighted[: min(max(1, int(fallback_width)), weighted.size)].astype(np.float32)
    return weighted[np.asarray(selected, dtype=np.int64)].astype(np.float32)


def scaled_vector_distance(
    query: np.ndarray,
    example: np.ndarray,
    *,
    reference_size: int | None = None,
) -> float:
    """Return a finite Euclidean distance scaled to a comparable width.

    Args:
        query: Weighted query vector.
        example: Weighted stored example vector.
        reference_size: Optional vector width used to scale compact signatures
            into the same rough distance range as full fingerprints.

    Returns:
        Finite distance, or infinity when either vector is empty.

    Side Effects:
        None.
    """
    usable = min(int(query.size), int(example.size))
    if usable <= 0:
        return float("inf")
    distance = float(np.linalg.norm(query[:usable] - example[:usable]))
    if reference_size is not None and reference_size > usable:
        distance *= math.sqrt(float(reference_size) / float(usable))
    return distance


def pad_vector(values: np.ndarray, fill: float) -> np.ndarray:
    """Return a finite ``FP_SIZE`` vector.

    Args:
        values: Input numeric vector.
        fill: Replacement value for missing or invalid entries.

    Returns:
        Finite float32 vector padded or truncated to ``FP_SIZE``.

    Side Effects:
        None.
    """
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.size < FP_SIZE:
        vector = np.pad(vector, (0, FP_SIZE - vector.size), mode="constant", constant_values=fill)
    return np.nan_to_num(vector[:FP_SIZE], nan=fill, posinf=fill, neginf=fill).astype(np.float32)


def example_fingerprint(example: dict[str, Any]) -> np.ndarray:
    """Return a finite fingerprint vector from one shape-memory example.

    Args:
        example: Stored shape-memory example payload.

    Returns:
        Finite ``FP_SIZE`` vector, or an empty vector when invalid.

    Side Effects:
        None.
    """
    try:
        vector = np.asarray(example.get("fingerprint", []), dtype=np.float32).reshape(-1)
    except Exception:
        return np.asarray([], dtype=np.float32)
    if vector.size <= 0 or not bool(np.all(np.isfinite(vector))):
        return np.asarray([], dtype=np.float32)
    return pad_vector(vector, 0.0)


def no_shape_memory_match() -> ShapeMemoryMatch:
    """Return the empty learned-shape match.

    Returns:
        Diagnostic match object with ``matched`` set to False.

    Side Effects:
        None.
    """
    return ShapeMemoryMatch(
        matched=False,
        shape="",
        nearest_distance=float("inf"),
        confidence=0.0,
        example_count=0,
        effective_weight=0,
        match_kind="none",
        threshold=shape_memory_distance_threshold(0, example_count=0),
    )


def safe_int(value: object) -> int:
    """Return a non-negative integer from loose JSON values.

    Args:
        value: Loose JSON scalar.

    Returns:
        Non-negative integer, or zero when coercion fails.

    Side Effects:
        None.
    """
    try:
        return max(0, int(float(value)))
    except Exception:
        return 0
