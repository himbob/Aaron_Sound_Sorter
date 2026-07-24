from __future__ import annotations

import numpy as np

from aaron_audio_intelligence.adaptive_teacher_metric import adaptive_teacher_metric


def _teacher_cloud() -> np.ndarray:
    """Return a cloud with two learned flexible dimensions and stable shape."""
    base = np.zeros((4, 16), dtype=np.float32)
    base[:, 0] = [0.10, 0.35, 0.60, 0.85]
    base[:, 1] = [0.85, 0.60, 0.35, 0.10]
    base[:, 2:] = 0.25
    return base


def test_adaptive_teacher_metric_learns_only_observed_flexible_dimensions() -> None:
    cloud = _teacher_cloud()
    query = cloud[-1].copy()
    query[0] = 1.02
    query[1] = -0.06

    result = adaptive_teacher_metric(query, cloud, effective_weight=160)

    assert result.available is True
    assert result.matched is True
    assert result.variable_feature_count >= 2
    assert result.stable_upper_rms < 0.10


def test_adaptive_teacher_metric_rejects_concentrated_stable_contradiction() -> None:
    cloud = _teacher_cloud()
    query = cloud[1].copy()
    query[4:10] += 0.90

    result = adaptive_teacher_metric(query, cloud, effective_weight=160)

    assert result.available is True
    assert result.matched is False
    assert result.stable_upper_rms > 1.85


def test_adaptive_teacher_metric_needs_multiple_teacher_examples() -> None:
    query = np.zeros((16,), dtype=np.float32)
    result = adaptive_teacher_metric(query, [query], effective_weight=500)

    assert result.available is False
    assert result.matched is False


def test_adaptive_teacher_metric_does_not_expand_from_repeated_weight_alone() -> None:
    cloud = np.zeros((2, 16), dtype=np.float32)
    query = np.ones((16,), dtype=np.float32) * 2.5

    weak = adaptive_teacher_metric(query, cloud, effective_weight=2)
    repeated = adaptive_teacher_metric(query, cloud, effective_weight=100000)

    assert weak.matched is False
    assert repeated.matched is False
    assert repeated.threshold <= 1.88
