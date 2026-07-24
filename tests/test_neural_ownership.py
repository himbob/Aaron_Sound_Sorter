from __future__ import annotations

from aaron_sound_sorter.neural_audio.contracts import LabelPrediction
from aaron_sound_sorter.neural_audio.ownership import assess_prototype_ownership


def _prediction(*, known: bool, margin: float, examples: int) -> LabelPrediction:
    return LabelPrediction(
        provider_id="test",
        model_id="test-v1",
        predicted_label="Instruments/Winds/Saxophone/Sax Loops",
        top_similarity=0.80,
        second_label="Instruments/Voice/Vocal Loops/Loops",
        second_similarity=0.80 - margin,
        margin=margin,
        prototype_id="sax::p1",
        prototype_radius_p95=0.05,
        distance_to_prototype=0.03,
        radius_ratio=0.60,
        known_distribution=known,
        evidence={"label_example_count": examples},
    )


def test_exact_human_match_is_immediately_ownership_ready() -> None:
    assessment = assess_prototype_ownership(
        _prediction(known=True, margin=0.01, examples=1),
        exact_training_match=True,
        minimum_margin=0.10,
        minimum_label_examples=3,
    )

    assert assessment.ready
    assert assessment.reason == "exact_human_training_match"


def test_unseen_boundary_example_requires_support_and_separation() -> None:
    sparse = assess_prototype_ownership(
        _prediction(known=True, margin=0.30, examples=2),
        exact_training_match=False,
        minimum_margin=0.10,
        minimum_label_examples=3,
    )
    ambiguous = assess_prototype_ownership(
        _prediction(known=True, margin=0.04, examples=6),
        exact_training_match=False,
        minimum_margin=0.10,
        minimum_label_examples=3,
    )
    supported = assess_prototype_ownership(
        _prediction(known=True, margin=0.30, examples=6),
        exact_training_match=False,
        minimum_margin=0.10,
        minimum_label_examples=3,
    )

    assert not sparse.ready
    assert sparse.reason == "insufficient_label_examples"
    assert not ambiguous.ready
    assert ambiguous.reason == "ambiguous_nearest_labels"
    assert supported.ready
    assert supported.reason == "supported_separated_neighborhood"


def test_outside_radius_never_generalizes_to_unseen_audio() -> None:
    assessment = assess_prototype_ownership(
        _prediction(known=False, margin=0.50, examples=20),
        exact_training_match=False,
        minimum_margin=0.10,
        minimum_label_examples=3,
    )

    assert not assessment.ready
    assert assessment.reason == "outside_learned_radius"
