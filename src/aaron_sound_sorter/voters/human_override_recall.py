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
        match_kind: Diagnostic match type.  Expected values are ``none``,
            ``fingerprint``, ``teacher_prototype``, and ``teacher_cloud``.
        nearest_distance: Weighted feature distance to the nearest human-taught
            fingerprint.
        effective_weight: Human override support weight stored for the label.
        example_count: Number of GUI correction examples inspected.
        confirmation_count: Total confirmation count across inspected examples.
        ranking_score: Lower-is-better score used when ``matched`` is true.
        threshold: Active maximum distance accepted for this match.
        exact_threshold: Maximum distance for exact fingerprint recall.
        generalized_threshold: Maximum distance for broader teacher recall.

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
            "human_override_effective_weight": int(self.effective_weight),
            "human_override_example_count": int(self.example_count),
            "human_override_confirmation_count": int(self.confirmation_count),
            "human_override_ranking_score": round(float(self.ranking_score), 6),
            "human_override_recall_policy": "weighted_fingerprint_and_teacher_cloud_match",
        }


def human_override_recall_match(
    brain: dict[str, Any],
    label: str,
    *,
    weighted_query_vector: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    feature_weight_vector: np.ndarray,
) -> HumanOverrideRecallMatch:
    """Find teacher-level GUI correction recall for a label.

    Args:
        brain: Active brain dictionary.
        label: Candidate label being scored.
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

    if valid_count <= 0:
        return no_human_override_match(example_count=len(examples), effective_weight=effective_weight)

    exact_threshold = human_override_distance_threshold(effective_weight)
    support_count = max(valid_count, confirmation_count)
    prototype_threshold = human_override_prototype_distance_threshold(effective_weight)
    cloud_threshold = human_override_teacher_cloud_distance_threshold(
        effective_weight,
        example_count=valid_count,
        support_count=support_count,
    )
    generalized_threshold = cloud_threshold if valid_count >= 2 else prototype_threshold

    exact_match = bool(nearest <= exact_threshold)
    cloud_match = bool(valid_count >= 2 and nearest <= cloud_threshold)
    prototype_match = bool(valid_count == 1 and nearest <= prototype_threshold)
    generalized_match = bool(not exact_match and (cloud_match or prototype_match))
    matched = bool(exact_match or generalized_match)
    if exact_match:
        match_kind = "fingerprint"
        threshold = exact_threshold
        ranking_score = human_override_ranking_score(effective_weight, nearest)
    elif cloud_match:
        match_kind = "teacher_cloud"
        threshold = cloud_threshold
        ranking_score = human_override_teacher_cloud_ranking_score(effective_weight, support_count, nearest)
    elif prototype_match:
        match_kind = "teacher_prototype"
        threshold = prototype_threshold
        ranking_score = human_override_teacher_prototype_ranking_score(effective_weight, nearest)
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
        generalized_threshold=generalized_threshold,
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


def human_override_distance_threshold(effective_weight: int) -> float:
    """Return the accepted weighted distance for human correction recall."""
    weight = max(1, int(effective_weight or 0))
    return min(2.25, max(0.35, 0.35 + math.log1p(weight) / 7.0))


def human_override_prototype_distance_threshold(effective_weight: int) -> float:
    """Return the accepted distance for a single human-taught prototype.

    A single GUI correction should help near neighbors, but it should not become
    a broad class override.  This threshold is intentionally wider than exact
    recall and narrower than the multi-example teacher cloud.
    """
    exact = human_override_distance_threshold(effective_weight)
    weight = max(1, int(effective_weight or 0))
    return min(2.85, exact + 0.30 + math.log1p(weight) / 16.0)


def human_override_teacher_cloud_distance_threshold(
    effective_weight: int,
    *,
    example_count: int,
    support_count: int,
) -> float:
    """Return the accepted distance for multi-example human-taught recall."""
    exact = human_override_distance_threshold(effective_weight)
    support = max(2, int(example_count or 0), int(support_count or 0))
    return min(3.65, exact + 0.45 + math.log1p(support) / 3.5)


def human_override_ranking_score(effective_weight: int, nearest_distance: float) -> float:
    """Return a lower-is-better score for a matched GUI correction."""
    weight = max(1, int(effective_weight or 0))
    authority = min(2.4, 0.45 + math.log1p(weight) / 3.0)
    distance_cost = min(0.35, max(0.0, float(nearest_distance)) * 0.08)
    return -max(0.25, authority - distance_cost)


def human_override_teacher_prototype_ranking_score(effective_weight: int, nearest_distance: float) -> float:
    """Return a conservative score for one nearby human-taught prototype."""
    exact = human_override_distance_threshold(effective_weight)
    weight = max(1, int(effective_weight or 0))
    authority = min(1.65, 0.30 + math.log1p(weight) / 5.0)
    distance_cost = min(0.65, max(0.0, float(nearest_distance) - exact) * 0.22)
    return -max(0.10, authority - distance_cost)


def human_override_teacher_cloud_ranking_score(
    effective_weight: int,
    support_count: int,
    nearest_distance: float,
) -> float:
    """Return a lower-is-better score for a supported correction cloud."""
    exact = human_override_distance_threshold(effective_weight)
    weight = max(1, int(effective_weight or 0))
    support = max(2, int(support_count or 0))
    authority = min(1.9, 0.35 + math.log1p(weight) / 4.5 + math.log1p(support) / 8.0)
    distance_cost = min(0.65, max(0.0, float(nearest_distance) - exact) * 0.18)
    return -max(0.15, authority - distance_cost)


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
        generalized_threshold=human_override_prototype_distance_threshold(effective_weight),
    )


def safe_int(value: Any) -> int:
    """Return a non-negative integer from loose JSON values."""
    try:
        return max(0, int(float(value)))
    except Exception:
        return 0
