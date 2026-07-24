# SOURCE-NAME BLINDNESS INVARIANT:
# This module may inspect only numeric feature vectors and stored training
# support. It must never inspect producer filenames, source folders, ZIP member
# names, or sample-pack labels.
"""Robust learned tolerance for human-taught audio categories.

The exact-correction lane answers a narrow question: "is this effectively the
same measured audio fingerprint?"  Generalization is different.  It should ask
which feature dimensions are stable inside a trained target and which dimensions
legitimately vary across that target's examples.

This module implements that second question with a small, source-blind diagonal
metric.  It learns one robust scale per feature from the target's teacher cloud,
shrinks small clouds toward a safe prior, and keeps low-variance dimensions as
automatic guardrails.  No instrument names or category-specific rules are used.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

DEFAULT_BASELINE_SCALE = 0.42
DEFAULT_MINIMUM_SCALE = 0.22
DEFAULT_MAXIMUM_SCALE = 2.75
DEFAULT_STABLE_Z_LIMIT = 2.35
DEFAULT_MAX_STABLE_VIOLATION_FRACTION = 0.05
DEFAULT_MAX_STABLE_UPPER_RMS = 1.85


@dataclass(frozen=True)
class AdaptiveTeacherMetricResult:
    """Result of comparing one query with a learned teacher cloud.

    ``distance`` is a robust, dimension-normalized distance.  Values near 1.0
    mean the query is roughly one learned tolerance unit from its nearest real
    teacher.  ``adjusted_distance`` adds a small penalty for very diffuse clouds
    so broad categories do not win merely because they learned large tolerances.
    """

    available: bool
    matched: bool
    distance: float
    adjusted_distance: float
    threshold: float
    stable_violation_fraction: float
    stable_upper_rms: float
    stable_feature_count: int
    variable_feature_count: int
    example_count: int
    nearest_example_index: int
    broadness_penalty: float


def adaptive_teacher_metric(
    query: np.ndarray,
    examples: Iterable[np.ndarray] | np.ndarray,
    *,
    effective_weight: int = 1,
    minimum_examples: int = 2,
    baseline_scale: float = DEFAULT_BASELINE_SCALE,
    minimum_scale: float = DEFAULT_MINIMUM_SCALE,
    maximum_scale: float = DEFAULT_MAXIMUM_SCALE,
    stable_z_limit: float = DEFAULT_STABLE_Z_LIMIT,
    maximum_stable_violation_fraction: float = DEFAULT_MAX_STABLE_VIOLATION_FRACTION,
    maximum_stable_upper_rms: float = DEFAULT_MAX_STABLE_UPPER_RMS,
) -> AdaptiveTeacherMetricResult:
    """Compare ``query`` with a target-specific learned tolerance profile.

    Args:
        query: One normalized feature signature.
        examples: Normalized signatures from human-approved examples for one
            trained target.
        effective_weight: Human correction support for the target.
        minimum_examples: Minimum cloud size required to learn variability.
        baseline_scale: Safe per-feature tolerance prior in normalized space.
        minimum_scale: Lower clamp preserving numerical stability.
        maximum_scale: Upper clamp preventing unbounded category spread.
        stable_z_limit: Maximum normalized deviation on stable dimensions.
        maximum_stable_violation_fraction: Allowed fraction of stable
            dimensions exceeding ``stable_z_limit``.
        maximum_stable_upper_rms: Maximum RMS deviation across the worst five
            percent of dimensions that remained stable in the teacher cloud.
            This catches concentrated shape contradictions that whole-vector
            averages can hide.

    Returns:
        ``AdaptiveTeacherMetricResult``.  ``available`` is false when too few
        valid examples exist.

    Important Constraints:
        This metric does not replace exact fingerprint recall.  It is a
        generalization lane and therefore requires at least two examples.
    """
    q = _finite_vector(query)
    matrix = _finite_matrix(examples, width=int(q.size))
    count = int(matrix.shape[0])
    if q.size <= 0 or count < max(2, int(minimum_examples)):
        return unavailable_metric_result(example_count=count)

    baseline = max(1e-4, float(baseline_scale))
    floor = max(1e-4, min(float(minimum_scale), float(maximum_scale)))
    ceiling = max(floor, float(maximum_scale))

    robust_scale, observed_scale = _shrunk_robust_scale(
        matrix,
        baseline=baseline,
        floor=floor,
        ceiling=ceiling,
    )
    residuals = np.abs(matrix - q[None, :])
    normalized = residuals / robust_scale[None, :]
    full_rms = np.sqrt(np.mean(np.square(normalized), axis=1))
    upper_band_count = min(
        normalized.shape[1],
        max(4, int(math.ceil(0.15 * float(normalized.shape[1])))),
    )
    upper_band = np.partition(normalized, -upper_band_count, axis=1)[:, -upper_band_count:]
    upper_rms = np.sqrt(np.mean(np.square(upper_band), axis=1))
    # Category similarity should follow the whole learned profile.  A small
    # upper-band term keeps localized differences visible, but concentrated
    # contradictions are governed separately by the stable-feature veto below.
    # This avoids punishing sax register/articulation changes, snare tuning and
    # decay changes, or FX width/motion changes merely because they are among
    # the largest residuals in an otherwise compatible category.
    rms_by_example = 0.85 * full_rms + 0.15 * upper_rms
    nearest_index = int(np.argmin(rms_by_example))
    distance = float(rms_by_example[nearest_index])

    stable_mask = _stable_feature_mask(observed_scale, baseline=baseline)
    stable_count = int(np.sum(stable_mask))
    variable_count = int(q.size - stable_count)
    if stable_count > 0:
        nearest_stable = normalized[nearest_index, stable_mask]
        stable_violation_fraction = float(np.mean(nearest_stable > float(stable_z_limit)))
        stable_upper_count = min(
            nearest_stable.size,
            max(3, int(math.ceil(0.05 * float(nearest_stable.size)))),
        )
        stable_upper = np.partition(nearest_stable, -stable_upper_count)[-stable_upper_count:]
        stable_upper_rms = float(np.sqrt(np.mean(np.square(stable_upper))))
    else:
        stable_violation_fraction = 0.0
        stable_upper_rms = 0.0

    broadness_penalty = _broadness_penalty(robust_scale, baseline=baseline)
    adjusted_distance = float(distance + broadness_penalty)
    threshold = adaptive_teacher_threshold(
        effective_weight=effective_weight,
        example_count=count,
    )
    matched = bool(
        adjusted_distance <= threshold
        and stable_violation_fraction <= float(maximum_stable_violation_fraction)
        and stable_upper_rms <= float(maximum_stable_upper_rms)
    )
    return AdaptiveTeacherMetricResult(
        available=True,
        matched=matched,
        distance=distance,
        adjusted_distance=adjusted_distance,
        threshold=threshold,
        stable_violation_fraction=stable_violation_fraction,
        stable_upper_rms=stable_upper_rms,
        stable_feature_count=stable_count,
        variable_feature_count=variable_count,
        example_count=count,
        nearest_example_index=nearest_index,
        broadness_penalty=broadness_penalty,
    )


def adaptive_teacher_threshold(*, effective_weight: int, example_count: int) -> float:
    """Return a bounded threshold for the robust learned metric.

    More examples increase confidence that the observed variation is real.
    Human evidence weight contributes only a small bounded increase; repeating
    one correction cannot create an unlimited category.
    """
    count = max(2, int(example_count or 0))
    weight = max(1, int(effective_weight or 0))
    count_bonus = min(0.30, math.log1p(count - 1) / 6.0)
    weight_bonus = min(0.12, math.log1p(weight) / 50.0)
    return float(min(1.88, 1.42 + count_bonus + weight_bonus))


def adaptive_teacher_ranking_score(
    *,
    effective_weight: int,
    metric: AdaptiveTeacherMetricResult,
) -> float:
    """Return a conservative lower-is-better score for a matched cloud.

    The score is strong enough to let a coherent human-taught category compete,
    but weaker than an exact fingerprint correction.  Broadness and distance
    remain visible in the score instead of being hidden by a fixed override.
    """
    if not metric.available or not metric.matched:
        return float("inf")
    weight = max(1, int(effective_weight or 0))
    support_authority = min(1.52, 0.42 + math.log1p(weight) / 5.8)
    normalized_distance = metric.adjusted_distance / max(metric.threshold, 1e-6)
    distance_cost = min(0.80, 0.62 * normalized_distance)
    guardrail_cost = min(0.25, 1.5 * metric.stable_violation_fraction)
    return -max(0.08, support_authority - distance_cost - guardrail_cost)


def unavailable_metric_result(*, example_count: int = 0) -> AdaptiveTeacherMetricResult:
    """Return the empty result used when variability cannot be learned."""
    return AdaptiveTeacherMetricResult(
        available=False,
        matched=False,
        distance=float("inf"),
        adjusted_distance=float("inf"),
        threshold=0.0,
        stable_violation_fraction=1.0,
        stable_upper_rms=float("inf"),
        stable_feature_count=0,
        variable_feature_count=0,
        example_count=max(0, int(example_count)),
        nearest_example_index=-1,
        broadness_penalty=0.0,
    )


def _shrunk_robust_scale(
    matrix: np.ndarray,
    *,
    baseline: float,
    floor: float,
    ceiling: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return robust per-feature spread with small-sample shrinkage."""
    center = np.median(matrix, axis=0)
    mad = 1.4826 * np.median(np.abs(matrix - center[None, :]), axis=0)
    if matrix.shape[0] >= 4:
        q25 = np.percentile(matrix, 25.0, axis=0)
        q75 = np.percentile(matrix, 75.0, axis=0)
        iqr_scale = (q75 - q25) / 1.349
        observed = np.maximum(mad, iqr_scale)
    else:
        observed = mad

    count = float(matrix.shape[0])
    learned_fraction = min(0.88, max(0.18, (count - 1.0) / (count + 5.0)))
    prior_variance = float(baseline) ** 2
    observed_variance = np.square(np.maximum(observed, 0.0))
    shrunk = np.sqrt(learned_fraction * observed_variance + (1.0 - learned_fraction) * prior_variance)
    return (
        np.clip(shrunk, floor, ceiling).astype(np.float32),
        np.maximum(observed, 0.0).astype(np.float32),
    )


def _stable_feature_mask(observed_scale: np.ndarray, *, baseline: float) -> np.ndarray:
    """Return dimensions that remained stable inside the teacher cloud."""
    if observed_scale.size <= 0:
        return np.zeros((0,), dtype=bool)
    percentile_cut = float(np.percentile(observed_scale, 60.0))
    cutoff = max(0.035, min(float(baseline) * 0.35, percentile_cut))
    mask = observed_scale <= cutoff
    if not bool(np.any(mask)):
        mask[int(np.argmin(observed_scale))] = True
    return mask


def _broadness_penalty(scale: np.ndarray, *, baseline: float) -> float:
    """Penalize diffuse target clouds without forbidding real variation."""
    ratio = np.maximum(scale / max(float(baseline), 1e-6), 1.0)
    return float(min(0.32, 0.10 * np.mean(np.log(ratio))))


def _finite_vector(values: np.ndarray) -> np.ndarray:
    """Return one finite float32 vector."""
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.size <= 0 or not bool(np.all(np.isfinite(vector))):
        return np.asarray([], dtype=np.float32)
    return vector.astype(np.float32, copy=False)


def _finite_matrix(examples: Iterable[np.ndarray] | np.ndarray, *, width: int) -> np.ndarray:
    """Return a finite 2D matrix truncated to a common width."""
    if width <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    if isinstance(examples, np.ndarray):
        raw_rows = examples if examples.ndim > 1 else examples.reshape(1, -1)
    else:
        raw_rows = list(examples)
    rows: list[np.ndarray] = []
    for raw in raw_rows:
        vector = np.asarray(raw, dtype=np.float32).reshape(-1)
        usable = min(width, int(vector.size))
        if usable != width:
            continue
        vector = vector[:width]
        if bool(np.all(np.isfinite(vector))):
            rows.append(vector.astype(np.float32, copy=False))
    if not rows:
        return np.zeros((0, width), dtype=np.float32)
    return np.vstack(rows).astype(np.float32, copy=False)
