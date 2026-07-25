from __future__ import annotations

from pathlib import Path

import numpy as np

from aaron_sound_sorter.neural_audio.active_learning import (
    _priority_score,
    _select_clustered_candidates,
    ActiveLearningCandidate,
)
from aaron_sound_sorter.neural_audio.contracts import LabelPrediction


def _make_prediction(**overrides) -> LabelPrediction:
    defaults = {
        "provider_id": "test-provider",
        "model_id": "test-model",
        "predicted_label": "Test/Label",
        "top_similarity": 0.86,
        "second_label": "Other/Label",
        "second_similarity": 0.55,
        "margin": 0.31,
        "prototype_id": "test-prototype",
        "prototype_radius_p95": 0.6,
        "distance_to_prototype": 0.28,
        "radius_ratio": 0.47,
        "known_distribution": True,
        "confidence_probability": None,
        "evidence": {},
    }
    defaults.update(overrides)
    return LabelPrediction(**defaults)


def test_priority_score_prefers_out_of_distribution() -> None:
    ood_prediction = _make_prediction(known_distribution=False)
    score, reason = _priority_score(ood_prediction, {})

    assert score > 2.0
    assert reason == "outside_learned_radius"


def test_priority_score_prefers_low_margin_when_in_distribution() -> None:
    low_margin_prediction = _make_prediction(margin=0.02)
    score, reason = _priority_score(low_margin_prediction, {})

    assert score > 1.0
    assert reason == "near_decision_boundary"


def test_select_clustered_candidates_limits_duplicates_and_per_label() -> None:
    vectors = [np.array([1.0, 0.0], dtype=np.float32), np.array([0.99, 0.1], dtype=np.float32)]
    candidates = [
        ActiveLearningCandidate(
            path=Path("file_1.wav"),
            file_sha256="1" * 64,
            prediction=_make_prediction(predicted_label="A/Label", margin=0.2, top_similarity=0.85),
            vector=vectors[0],
            priority_score=5.0,
            review_reason="first",
        ),
        ActiveLearningCandidate(
            path=Path("file_2.wav"),
            file_sha256="2" * 64,
            prediction=_make_prediction(predicted_label="A/Label", margin=0.15, top_similarity=0.83),
            vector=vectors[1],
            priority_score=4.5,
            review_reason="second",
        ),
        ActiveLearningCandidate(
            path=Path("file_3.wav"),
            file_sha256="3" * 64,
            prediction=_make_prediction(predicted_label="B/Label", margin=0.05, top_similarity=0.72),
            vector=np.array([0.0, 1.0], dtype=np.float32),
            priority_score=6.0,
            review_reason="third",
        ),
    ]

    selected = _select_clustered_candidates(candidates, max_candidates=3, max_per_label=1, similarity_threshold=0.90)

    assert len(selected) == 2
    assert {candidate.prediction.predicted_label for candidate in selected} == {"A/Label", "B/Label"}
    assert selected[0].priority_score >= selected[1].priority_score
