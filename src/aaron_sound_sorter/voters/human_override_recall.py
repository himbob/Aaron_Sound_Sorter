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
        nearest_distance: Weighted feature distance to the nearest human-taught
            fingerprint.
        effective_weight: Human override support weight stored for the label.
        example_count: Number of GUI correction examples inspected.
        confirmation_count: Total confirmation count across inspected examples.
        ranking_score: Lower-is-better score used when ``matched`` is true.
        threshold: Maximum distance accepted for teacher-level recall.

    Side Effects:
        None.

    Raises:
        No intentional exceptions. Malformed example data is ignored.

    Important Constraints:
        This object represents measured fingerprint recall only.  It must not
        inspect source paths, file names, or folder names.
    """

    matched: bool
    nearest_distance: float
    effective_weight: int
    example_count: int
    confirmation_count: int
    ranking_score: float
    threshold: float

    def evidence(self) -> dict[str, Any]:
        """Return flat diagnostics suitable for voter evidence dictionaries."""
        if self.example_count <= 0:
            return {}
        return {
            "human_override_recall_candidate": True,
            "human_override_exact_audio_match": bool(self.matched),
            "human_override_nearest_distance": round(float(self.nearest_distance), 6),
            "human_override_distance_threshold": round(float(self.threshold), 6),
            "human_override_effective_weight": int(self.effective_weight),
            "human_override_example_count": int(self.example_count),
            "human_override_confirmation_count": int(self.confirmation_count),
            "human_override_ranking_score": round(float(self.ranking_score), 6),
            "human_override_recall_policy": "weighted_fingerprint_teacher_match",
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
        ``HumanOverrideRecallMatch`` with ``matched`` true only when the current
        audio is close to a stored GUI correction for this same label.

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

    threshold = human_override_distance_threshold(effective_weight)
    matched = bool(nearest <= threshold)
    return HumanOverrideRecallMatch(
        matched=matched,
        nearest_distance=nearest,
        effective_weight=effective_weight,
        example_count=valid_count,
        confirmation_count=confirmation_count,
        ranking_score=human_override_ranking_score(effective_weight, nearest),
        threshold=threshold,
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


def human_override_ranking_score(effective_weight: int, nearest_distance: float) -> float:
    """Return a lower-is-better score for a matched GUI correction."""
    weight = max(1, int(effective_weight or 0))
    authority = min(2.4, 0.45 + math.log1p(weight) / 3.0)
    distance_cost = min(0.35, max(0.0, float(nearest_distance)) * 0.08)
    return -max(0.25, authority - distance_cost)


def no_human_override_match(
    *,
    example_count: int = 0,
    effective_weight: int = 0,
) -> HumanOverrideRecallMatch:
    """Return the empty match object used when no GUI correction applies."""
    return HumanOverrideRecallMatch(
        matched=False,
        nearest_distance=float("inf"),
        effective_weight=int(effective_weight),
        example_count=int(example_count),
        confirmation_count=0,
        ranking_score=float("inf"),
        threshold=human_override_distance_threshold(effective_weight),
    )


def safe_int(value: Any) -> int:
    """Return a non-negative integer from loose JSON values."""
    try:
        return max(0, int(float(value)))
    except Exception:
        return 0
