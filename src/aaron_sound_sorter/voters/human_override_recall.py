"""Human correction recall for voter-level scoring.

This module turns GUI training corrections into measured audio evidence.  It
never reads producer filenames or source folder text; it compares the current
feature vector with fingerprints stored in the active brain by the incremental
training flow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from aaron_audio_intelligence.adaptive_teacher_metric import (
    DEFAULT_BASELINE_SCALE,
    DEFAULT_MAX_STABLE_UPPER_RMS,
    DEFAULT_MAX_STABLE_VIOLATION_FRACTION,
    adaptive_teacher_metric,
    adaptive_teacher_ranking_score,
)
from aaron_audio_intelligence.learned_memory_features import (
    PITCH_REGISTER_SENSITIVE_MEMORY_FEATURES,
    weighted_normalized_signature_vector,
)
from aaron_sound_sorter.engine.learned_memory_contracts import (
    memory_conflicts_with_candidate_top_family,
    normalize_internal_path,
    top_family_from_path,
)


@dataclass(frozen=True)
class HumanOverrideRecallMatch:
    """Nearest GUI correction match for one candidate label.

    Args:
        matched: True when the nearest GUI correction is close enough to act as
            teacher-level recall evidence.
        exact_match: True when the current audio is close enough to a stored
            correction to be treated as the same measured fingerprint region.
        generalized_match: True when the current audio is close to a supported
            human-taught label neighborhood, but not close enough to be called
            the same measured fingerprint region.
        match_kind: Diagnostic match type. Expected values are ``none``,
            ``fingerprint``, ``adaptive_teacher_cloud``, and
            ``adaptive_teacher_neighbor``.
        nearest_distance: Weighted feature distance to the nearest human-taught
            fingerprint.
        effective_weight: Human override support weight stored for the label.
        example_count: Number of GUI correction examples inspected.
        confirmation_count: Total confirmation count across inspected examples.
        ranking_score: Lower-is-better score used when ``matched`` is true.
        threshold: Active maximum distance accepted for this match.
        exact_threshold: Maximum distance for exact fingerprint recall.
        generalized_threshold: Maximum distance for broader teacher recall.
        adaptive_distance: Robust category-specific distance learned from the
            target's own correction cloud.
        adaptive_threshold: Maximum accepted robust learned distance.
        adaptive_stable_violation_fraction: Fraction of low-variance identity
            dimensions that exceeded the safety guardrail.
        adaptive_stable_upper_rms: RMS deviation across the worst-changing
            stable identity dimensions.
        adaptive_stable_feature_count: Number of learned stable dimensions.
        adaptive_variable_feature_count: Number of learned flexible dimensions.
        identity_neighbor_distance: Register-invariant RMS distance to the
            nearest individual human teacher.
        identity_neighbor_threshold: Maximum accepted teacher-neighbor distance.

    Side Effects:
        None.

    Raises:
        No intentional exceptions. Malformed example data is ignored.

    Important Constraints:
        This object represents measured fingerprint recall only.  It must not
        inspect source paths, file names, or folder names.
    """

    matched: bool
    exact_match: bool
    generalized_match: bool
    match_kind: str
    nearest_distance: float
    effective_weight: int
    example_count: int
    confirmation_count: int
    ranking_score: float
    threshold: float
    exact_threshold: float
    generalized_threshold: float
    adaptive_distance: float
    adaptive_threshold: float
    adaptive_stable_violation_fraction: float
    adaptive_stable_upper_rms: float
    adaptive_stable_feature_count: int
    adaptive_variable_feature_count: int
    identity_neighbor_distance: float
    identity_neighbor_threshold: float

    def evidence(self) -> dict[str, Any]:
        """Return flat diagnostics suitable for voter evidence dictionaries."""
        if self.example_count <= 0:
            return {}
        return {
            "human_override_recall_candidate": True,
            "human_override_matched": bool(self.matched),
            "human_override_exact_audio_match": bool(self.exact_match),
            "human_override_generalized_audio_match": bool(self.generalized_match),
            "human_override_match_kind": str(self.match_kind),
            "human_override_nearest_distance": round(float(self.nearest_distance), 6),
            "human_override_distance_threshold": round(float(self.threshold), 6),
            "human_override_exact_distance_threshold": round(float(self.exact_threshold), 6),
            "human_override_generalized_distance_threshold": round(float(self.generalized_threshold), 6),
            "human_override_adaptive_distance": round(float(self.adaptive_distance), 6),
            "human_override_adaptive_threshold": round(float(self.adaptive_threshold), 6),
            "human_override_adaptive_stable_violation_fraction": round(
                float(self.adaptive_stable_violation_fraction), 6
            ),
            "human_override_adaptive_stable_upper_rms": round(float(self.adaptive_stable_upper_rms), 6),
            "human_override_adaptive_stable_feature_count": int(self.adaptive_stable_feature_count),
            "human_override_adaptive_variable_feature_count": int(self.adaptive_variable_feature_count),
            "human_override_identity_neighbor_distance": round(float(self.identity_neighbor_distance), 6),
            "human_override_identity_neighbor_threshold": round(float(self.identity_neighbor_threshold), 6),
            "human_override_effective_weight": int(self.effective_weight),
            "human_override_example_count": int(self.example_count),
            "human_override_confirmation_count": int(self.confirmation_count),
            "human_override_ranking_score": round(float(self.ranking_score), 6),
            "human_override_recall_policy": ("exact_fingerprint_plus_register_invariant_adaptive_teacher_metric"),
        }


def human_override_recall_match(
    brain: dict[str, Any],
    label: str,
    *,
    raw_query_vector: np.ndarray,
    weighted_query_vector: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    feature_weight_vector: np.ndarray,
) -> HumanOverrideRecallMatch:
    """Find teacher-level GUI correction recall for a label.

    Args:
        brain: Active brain dictionary.
        label: Candidate label being scored.
        raw_query_vector: Current unscaled audio fingerprint.
        weighted_query_vector: Current audio fingerprint after the same
            scaler/weight transform used by the voter score.
        scaler_mean: Label-specific or global scaler mean.
        scaler_std: Label-specific or global scaler standard deviation.
        feature_weight_vector: Feature weights used by the active brain.

    Returns:
        ``HumanOverrideRecallMatch`` with ``matched`` true when the current
        audio is close to a stored GUI correction or to a supported correction
        neighborhood for this same label.

    Side Effects:
        None.

    Raises:
        No intentional exceptions. Invalid stored fingerprints are skipped.

    Important Constraints:
        This function is source-name blind.  It uses only numeric fingerprints
        and human override weights already stored in the brain.
    """
    examples = gui_correction_examples(brain, label)
    if not examples:
        return no_human_override_match()

    weighted_query = np.asarray(weighted_query_vector, dtype=np.float32)
    mean = np.asarray(scaler_mean, dtype=np.float32)
    std = np.asarray(scaler_std, dtype=np.float32)
    weights = np.asarray(feature_weight_vector, dtype=np.float32)

    nearest = float("inf")
    effective_weight = label_human_override_weight(brain, label)
    confirmation_count = 0
    valid_count = 0
    identity_query = weighted_normalized_signature_vector(
        brain,
        raw_query_vector,
        include_feature_names=set(str(name) for name in brain.get("feature_names", [])),
        exclude_feature_names=PITCH_REGISTER_SENSITIVE_MEMORY_FEATURES,
        minimum_feature_count=24,
    )
    identity_examples: list[np.ndarray] = []
    for example in examples:
        if not isinstance(example, dict):
            continue
        confirmation_count += safe_int(example.get("human_override_confirmation_count"))
        effective_weight = max(effective_weight, safe_int(example.get("human_override_evidence_weight")))
        example_vector = numeric_fingerprint(example.get("fingerprint"))
        if example_vector.size <= 0:
            continue
        usable = min(example_vector.size, weighted_query.size, mean.size, std.size, weights.size)
        if usable <= 0:
            continue
        safe_std = np.where(np.abs(std[:usable]) < 1e-6, 1.0, std[:usable])
        weighted_example = ((example_vector[:usable] - mean[:usable]) / safe_std) * weights[:usable]
        distance = float(np.linalg.norm(weighted_query[:usable] - weighted_example))
        if math.isfinite(distance):
            valid_count += 1
            nearest = min(nearest, distance)
            identity_example = weighted_normalized_signature_vector(
                brain,
                example_vector,
                include_feature_names=set(str(name) for name in brain.get("feature_names", [])),
                exclude_feature_names=PITCH_REGISTER_SENSITIVE_MEMORY_FEATURES,
                minimum_feature_count=24,
            )
            if identity_example.size == identity_query.size and identity_example.size > 0:
                identity_examples.append(identity_example)

    if valid_count <= 0:
        return no_human_override_match(example_count=len(examples), effective_weight=effective_weight)

    exact_threshold = human_override_distance_threshold(effective_weight)
    adaptive_match = adaptive_teacher_metric(
        identity_query,
        identity_examples,
        effective_weight=effective_weight,
    )
    identity_neighbor_distance = nearest_identity_neighbor_distance(
        identity_query,
        identity_examples,
    )
    identity_neighbor_threshold = human_override_identity_neighbor_threshold(effective_weight)
    adaptive_guardrail_pass = bool(
        not adaptive_match.available
        or (
            adaptive_match.stable_violation_fraction <= DEFAULT_MAX_STABLE_VIOLATION_FRACTION
            and adaptive_match.stable_upper_rms <= DEFAULT_MAX_STABLE_UPPER_RMS
        )
    )

    exact_match = bool(nearest <= exact_threshold)
    adaptive_cloud_match = bool(valid_count >= 2 and adaptive_match.matched)
    identity_neighbor_match = bool(
        identity_neighbor_distance <= identity_neighbor_threshold and adaptive_guardrail_pass
    )
    generalized_match = bool(not exact_match and (adaptive_cloud_match or identity_neighbor_match))
    matched = bool(exact_match or generalized_match)
    if exact_match:
        match_kind = "fingerprint"
        threshold = exact_threshold
        ranking_score = human_override_ranking_score(effective_weight, nearest)
    elif adaptive_cloud_match:
        match_kind = "adaptive_teacher_cloud"
        threshold = adaptive_match.threshold
        ranking_score = adaptive_teacher_ranking_score(
            effective_weight=effective_weight,
            metric=adaptive_match,
        )
    elif identity_neighbor_match:
        match_kind = "adaptive_teacher_neighbor"
        threshold = identity_neighbor_threshold
        ranking_score = human_override_teacher_neighbor_ranking_score(
            effective_weight,
            identity_neighbor_distance,
        )
    else:
        match_kind = "none"
        threshold = exact_threshold
        ranking_score = float("inf")
    return HumanOverrideRecallMatch(
        matched=matched,
        exact_match=exact_match,
        generalized_match=generalized_match,
        match_kind=match_kind,
        nearest_distance=nearest,
        effective_weight=effective_weight,
        example_count=valid_count,
        confirmation_count=confirmation_count,
        ranking_score=ranking_score,
        threshold=threshold,
        exact_threshold=exact_threshold,
        generalized_threshold=max(
            identity_neighbor_threshold,
            adaptive_match.threshold if adaptive_match.available else 0.0,
        ),
        adaptive_distance=adaptive_match.adjusted_distance,
        adaptive_threshold=adaptive_match.threshold,
        adaptive_stable_violation_fraction=adaptive_match.stable_violation_fraction,
        adaptive_stable_upper_rms=adaptive_match.stable_upper_rms,
        adaptive_stable_feature_count=adaptive_match.stable_feature_count,
        adaptive_variable_feature_count=adaptive_match.variable_feature_count,
        identity_neighbor_distance=identity_neighbor_distance,
        identity_neighbor_threshold=identity_neighbor_threshold,
    )


def human_override_recall_allowed_by_memory(
    candidate_folder_path: str,
    facts: Any,
    *,
    confidence_gate: float = 0.86,
) -> bool:
    """Return whether legacy per-label recall may affect this candidate.

    Args:
        candidate_folder_path: Internal candidate folder being scored.
        facts: SharedAudioFacts-like object carrying learned memory evidence.
        confidence_gate: Minimum learned-memory confidence required to suppress
            a conflicting legacy per-label override.

    Returns:
        ``False`` when newer voter/physics memory strongly owns a different top
        family from this candidate; otherwise ``True``.

    Important Constraints:
        This is source-name blind. It compares internal learned-memory targets
        against internal candidate labels and never reads producer filenames.
    """
    candidate_path = normalize_internal_path(candidate_folder_path)
    if not top_family_from_path(candidate_path):
        return True
    return not memory_conflicts_with_candidate_top_family(
        facts,
        candidate_path,
        minimum_confidence=confidence_gate,
    )


def gui_correction_examples(brain: dict[str, Any], label: str) -> list[dict[str, Any]]:
    """Return detailed training examples created by GUI corrections."""
    examples_by_label = brain.get("training_examples_detailed_by_label", {})
    if not isinstance(examples_by_label, dict):
        return []
    examples = examples_by_label.get(label, [])
    if not isinstance(examples, list):
        return []
    return [
        example for example in examples if isinstance(example, dict) and bool(example.get("incremental_gui_correction"))
    ]


def label_human_override_weight(brain: dict[str, Any], label: str) -> int:
    """Return the strongest stored human override weight for a label."""
    reliability_by_label = brain.get("label_reliability_by_label", {})
    if not isinstance(reliability_by_label, dict):
        return 0
    reliability = reliability_by_label.get(label, {})
    if not isinstance(reliability, dict):
        return 0
    return max(
        safe_int(reliability.get("human_override_effective_weight")),
        safe_int(reliability.get("training_count")),
    )


def numeric_fingerprint(value: Any) -> np.ndarray:
    """Convert stored fingerprint data to a 1D float vector."""
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return np.asarray([], dtype=np.float32)
    if not vector.size:
        return np.asarray([], dtype=np.float32)
    finite = np.isfinite(vector)
    if not bool(np.all(finite)):
        return np.asarray([], dtype=np.float32)
    return vector


def human_override_ranking_score(effective_weight: int, nearest_distance: float) -> float:
    """Return a lower-is-better score for a same-fingerprint correction."""
    weight = max(1, int(effective_weight or 0))
    authority = min(2.4, 0.45 + math.log1p(weight) / 3.0)
    distance_cost = min(0.35, max(0.0, float(nearest_distance)) * 0.08)
    return -max(0.25, authority - distance_cost)


def nearest_identity_neighbor_distance(
    query: np.ndarray,
    examples: list[np.ndarray],
    *,
    baseline_scale: float = DEFAULT_BASELINE_SCALE,
) -> float:
    """Return register-invariant RMS distance to the nearest human teacher.

    This is a bounded local-neighborhood fallback for multimodal categories. It
    does not infer category-wide variability; it only says that a new sample is
    near one approved teacher after absolute pitch-register coordinates are
    removed.
    """
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    if q.size <= 0 or not examples:
        return float("inf")
    scale = max(1e-6, float(baseline_scale))
    distances: list[float] = []
    for example in examples:
        row = np.asarray(example, dtype=np.float32).reshape(-1)
        usable = min(int(q.size), int(row.size))
        if usable <= 0:
            continue
        residual = np.abs((q[:usable] - row[:usable]) / scale)
        full_rms = float(np.sqrt(np.mean(np.square(residual))))
        upper_count = min(
            usable,
            max(4, int(math.ceil(0.15 * float(usable)))),
        )
        upper = np.partition(residual, -upper_count)[-upper_count:]
        upper_rms = float(np.sqrt(np.mean(np.square(upper))))
        distance = 0.65 * full_rms + 0.35 * upper_rms
        if math.isfinite(distance):
            distances.append(distance)
    return min(distances) if distances else float("inf")


def human_override_identity_neighbor_threshold(effective_weight: int) -> float:
    """Return a bounded radius around each individual approved teacher."""
    exact = human_override_distance_threshold(effective_weight)
    weight = max(1, int(effective_weight or 0))
    return min(1.45, exact + 0.30 + math.log1p(weight) / 16.0)


def human_override_teacher_neighbor_ranking_score(
    effective_weight: int,
    neighbor_distance: float,
) -> float:
    """Return conservative rank evidence for one nearby approved teacher."""
    threshold = human_override_identity_neighbor_threshold(effective_weight)
    weight = max(1, int(effective_weight or 0))
    authority = min(1.35, 0.28 + math.log1p(weight) / 5.6)
    normalized_distance = max(0.0, float(neighbor_distance)) / max(threshold, 1e-6)
    distance_cost = min(0.75, 0.62 * normalized_distance)
    return -max(0.08, authority - distance_cost)


def human_override_distance_threshold(effective_weight: int) -> float:
    """Return the narrow distance for same-fingerprint correction recall.

    Repeating a correction may increase trust, but it must not turn the exact
    lane into a broad category cloud.  Generalization is handled separately by
    the adaptive teacher metric.
    """
    weight = max(1, int(effective_weight or 0))
    return min(0.42, max(0.16, 0.12 + math.log1p(weight) / 42.0))


def no_human_override_match(
    *,
    example_count: int = 0,
    effective_weight: int = 0,
) -> HumanOverrideRecallMatch:
    """Return the empty match object used when no GUI correction applies."""
    return HumanOverrideRecallMatch(
        matched=False,
        exact_match=False,
        generalized_match=False,
        match_kind="none",
        nearest_distance=float("inf"),
        effective_weight=int(effective_weight),
        example_count=int(example_count),
        confirmation_count=0,
        ranking_score=float("inf"),
        threshold=human_override_distance_threshold(effective_weight),
        exact_threshold=human_override_distance_threshold(effective_weight),
        generalized_threshold=human_override_identity_neighbor_threshold(effective_weight),
        adaptive_distance=float("inf"),
        adaptive_threshold=0.0,
        adaptive_stable_violation_fraction=1.0,
        adaptive_stable_upper_rms=float("inf"),
        adaptive_stable_feature_count=0,
        adaptive_variable_feature_count=0,
        identity_neighbor_distance=float("inf"),
        identity_neighbor_threshold=human_override_identity_neighbor_threshold(effective_weight),
    )


def safe_int(value: Any) -> int:
    """Return a non-negative integer from loose JSON values."""
    try:
        return max(0, int(float(value)))
    except Exception:
        return 0


def safe_float(value: Any, default: float = 0.0) -> float:
    """Return a finite float from loose JSON values."""
    try:
        number = float(value)
    except Exception:
        return default
    return number if math.isfinite(number) else default


def normalized_path(value: str) -> str:
    """Return a normalized internal folder path string."""
    return normalize_internal_path(value)


def top_family(folder_path: str) -> str:
    """Return the top family token from an internal folder path."""
    return top_family_from_path(folder_path)
