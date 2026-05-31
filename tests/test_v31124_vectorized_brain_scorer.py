from __future__ import annotations

from pathlib import Path

import numpy as np

from aaron_sound_sorter.domain.models import AudioPhysics, SharedAudioFacts
from aaron_sound_sorter.voters.brain_voter import BrainVoter
from aaron_sound_sorter.voters.scoring_tools import FP_SIZE, centroid_distance, feature_weights, scaler_for_label
from aaron_sound_sorter.voters.vectorized_brain_scorer import VectorizedCentroidFallbackScorer


def _centroid_only_brain() -> dict:
    zero = [0.0] * FP_SIZE
    label_a_second = [0.0] * FP_SIZE
    label_a_second[0] = 1.0
    label_b_first = [0.0] * FP_SIZE
    label_b_first[1] = 2.0
    label_b_second = [0.0] * FP_SIZE
    label_b_second[1] = 3.0
    return {
        "labels": ["label_a", "label_b"],
        "feature_weights": [1.0, 0.5, 2.0] + [1.0] * (FP_SIZE - 3),
        "scaler_mean": [0.0] * FP_SIZE,
        "scaler_std": [1.0] * FP_SIZE,
        "centroids": {
            "label_a": [zero, label_a_second],
            "label_b": [label_b_first, label_b_second],
        },
        "folders": {
            "label_a": "Instruments/Synths/Synth Loops",
            "label_b": "Drums/Drum Loops",
        },
    }


def test_vectorized_fallback_matches_legacy_scalar_centroid_distance() -> None:
    brain = _centroid_only_brain()
    weights = feature_weights(brain)
    scorer = VectorizedCentroidFallbackScorer(brain, weights)
    weighted_vectors = {}
    raw_vector = np.zeros(FP_SIZE, dtype=np.float32)
    raw_vector[:3] = [0.25, 1.5, 0.0]
    for label in brain["labels"]:
        mean, std = scaler_for_label(brain, label)
        weighted_vectors[label] = ((raw_vector - mean) / std) * weights

    vectorized_scores = scorer.score_labels(weighted_vectors)

    for label, weighted_vector in weighted_vectors.items():
        expected = centroid_distance(brain, label, weighted_vector, weights)
        assert vectorized_scores[label].score == expected
        assert vectorized_scores[label].mode == "centroid_fallback"


def test_brain_voter_vectorized_fallback_preserves_scalar_rows() -> None:
    brain = _centroid_only_brain()
    physics = AudioPhysics(
        source_path=Path("sample.wav"),
        fingerprint=np.asarray(([0.25, 1.5, 0.0] + [0.0] * (FP_SIZE - 3)), dtype=np.float32),
        duration_sec=1.0,
        read_status="ok",
    )
    facts = SharedAudioFacts(False, True, False, False, False)
    voter = BrainVoter()

    scalar_rows = voter.score_labels(physics, facts, brain, brain["labels"], use_vectorized_fallback=False)
    vectorized_rows = voter.score_labels(physics, facts, brain, brain["labels"], use_vectorized_fallback=True)

    assert [(row["label"], row["score"]) for row in vectorized_rows] == [
        (row["label"], row["score"]) for row in scalar_rows
    ]
    assert any(row["evidence"]["used_vectorized_centroid_fallback"] for row in vectorized_rows)
