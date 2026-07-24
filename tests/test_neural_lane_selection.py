from __future__ import annotations

from aaron_sound_sorter.neural_audio.contracts import LaneEvaluation
from aaron_sound_sorter.neural_audio.lane_selection import gate_fusion_candidate


def lane(provider: str, accuracy: float, known: float = 0.9, count: int = 20) -> LaneEvaluation:
    return LaneEvaluation(
        provider_id=provider,
        label="Voice",
        heldout_count=count,
        correct_count=round(accuracy * count),
        top1_accuracy=accuracy,
        mean_correct_similarity=0.8,
        mean_margin=0.2,
        known_distribution_rate=known,
    )


def test_fusion_is_rejected_when_it_only_matches_best_lane() -> None:
    decision = gate_fusion_candidate([lane("clap", 0.85), lane("mert", 0.75)], lane("clap+mert", 0.85))
    assert not decision.accepted
    assert decision.best_single_provider_id == "clap"


def test_fusion_is_allowed_only_after_measured_heldout_gain() -> None:
    decision = gate_fusion_candidate([lane("clap", 0.80), lane("mert", 0.75)], lane("clap+mert", 0.90))
    assert decision.accepted
    assert decision.accuracy_gain > 0.02


def test_fusion_requires_enough_heldout_examples() -> None:
    decision = gate_fusion_candidate(
        [lane("clap", 0.50, count=4), lane("mert", 0.50, count=4)],
        lane("clap+mert", 1.0, count=4),
    )
    assert not decision.accepted
