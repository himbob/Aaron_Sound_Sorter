from __future__ import annotations

from aaron_sound_sorter.neural_audio.calibration import CALIBRATION_FEATURES, feedback_features
from aaron_sound_sorter.neural_audio.calibration_features import review_feedback_from_runtime
from aaron_sound_sorter.neural_audio.runtime import (
    NeuralEventSuggestion,
    NeuralPromptSuggestion,
    NeuralRuntimePrediction,
)


def test_foundation_feedback_keeps_separated_lane_features() -> None:
    label = "Instruments/Voice/Vocal Loops/Loops"
    prediction = NeuralRuntimePrediction(
        row_id="00001",
        file_sha256="a" * 64,
        predicted_label=label,
        second_label="Instruments/Woodwinds/Saxophone/Loops",
        top_similarity=0.82,
        second_similarity=0.62,
        margin=0.20,
        radius_ratio=0.70,
        known_distribution=True,
        label_example_count=8,
        semantic_family="human_voice",
        semantic_top_score=0.30,
        semantic_family_scores={"human_voice": 0.30, "woodwind_reed": 0.10},
        prompt_suggestions=(NeuralPromptSuggestion(label, 0.55, 0.20, 0.35, "voice", 0.60, "sax", 0.20),),
        panns_support_score=0.91,
        panns_supporting_events=(NeuralEventSuggestion("Speech", 0.91),),
    )

    row = review_feedback_from_runtime(
        prediction,
        final_approved_label=label,
        structure_agreement=1.0,
        label_prototype_count=2,
        brain_version="run_test",
        taxonomy_version="test",
    )

    assert row.accepted
    assert row.parent_family_accepted
    assert row.prompt_top_similarity == 0.55
    assert row.panns_support_score == 0.91
    assert row.broad_family_agreement == 1.0
    assert row.out_of_distribution == 0.0
    assert feedback_features(row).shape == (len(CALIBRATION_FEATURES),)
