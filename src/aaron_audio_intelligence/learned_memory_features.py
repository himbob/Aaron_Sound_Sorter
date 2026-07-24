# SOURCE-NAME BLINDNESS INVARIANT:
# This module operates only on numeric feature vectors and brain metadata.  It
# must never inspect producer filenames, source folders, ZIP members, or sample
# pack names.
"""Shared numeric feature geometry for learned-memory brains.

The shape, voter, physics, and GUI teacher-memory lanes all need the same
normalization and named-feature selection rules.  Keeping those rules here
prevents one memory brain from becoming an accidental utility dependency for
all the others.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from aaron_sound_sorter.core import FEATURE_WEIGHTS, FP_SIZE

PITCH_REGISTER_SENSITIVE_MEMORY_FEATURES = frozenset(
    {
        "f0_median_hz",
        "low_peak_frequency_hz",
        "top1_peak_frequency_hz",
        "top2_peak_frequency_hz",
        "top3_peak_frequency_hz",
    }
)


def pad_vector(values: np.ndarray, fill: float) -> np.ndarray:
    """Return a finite ``FP_SIZE`` vector padded or truncated as needed."""
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.size < FP_SIZE:
        vector = np.pad(vector, (0, FP_SIZE - vector.size), mode="constant", constant_values=fill)
    return np.nan_to_num(vector[:FP_SIZE], nan=fill, posinf=fill, neginf=fill).astype(np.float32)


def weighted_normalized_vector(
    brain: dict[str, Any], values: tuple[float, ...] | list[float] | np.ndarray
) -> np.ndarray:
    """Return a finite vector normalized by the brain scaler and weights."""
    vector = pad_vector(np.asarray(values, dtype=np.float32), 0.0)
    mean = pad_vector(
        np.asarray(brain.get("scaler_mean", np.zeros((FP_SIZE,), dtype=np.float32)), dtype=np.float32),
        0.0,
    )
    std = pad_vector(
        np.asarray(brain.get("scaler_std", np.ones((FP_SIZE,), dtype=np.float32)), dtype=np.float32),
        1.0,
    )
    weights = pad_vector(
        np.asarray(brain.get("feature_weights", FEATURE_WEIGHTS), dtype=np.float32),
        1.0,
    )
    safe_std = np.where(np.abs(std) < 1e-6, 1.0, std)
    return ((vector - mean) / safe_std * weights).astype(np.float32)


def weighted_normalized_signature_vector(
    brain: dict[str, Any],
    values: tuple[float, ...] | list[float] | np.ndarray,
    *,
    include_feature_names: Iterable[str],
    exclude_feature_names: Iterable[str] = (),
    include_feature_prefixes: tuple[str, ...] = (),
    minimum_feature_count: int = 8,
    fallback_width: int | None = None,
) -> np.ndarray:
    """Return a named subset of one normalized, weighted feature vector.

    Named selection keeps learned-memory geometry stable when features are added
    to the full fingerprint.  Test or migration brains without enough real
    feature names fall back deterministically rather than silently returning an
    empty signature.
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
        width = min(max(1, int(fallback_width)), weighted.size)
        return weighted[:width].astype(np.float32)
    return weighted[np.asarray(selected, dtype=np.int64)].astype(np.float32)


def scaled_vector_distance(
    query: np.ndarray,
    example: np.ndarray,
    *,
    reference_size: int | None = None,
) -> float:
    """Return finite Euclidean distance, optionally scaled to a reference width."""
    usable = min(int(query.size), int(example.size))
    if usable <= 0:
        return float("inf")
    distance = float(np.linalg.norm(query[:usable] - example[:usable]))
    if reference_size is not None and reference_size > usable:
        distance *= math.sqrt(float(reference_size) / float(usable))
    return distance
