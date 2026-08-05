from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aaron_sound_sorter.domain.models import (
    AudioPhysics,
    ConsensusDecision,
    SharedAudioFacts,
    SortFileResult,
    VoterResult,
)
from aaron_sound_sorter.neural_audio.authority import apply_neural_prediction_to_result
from aaron_sound_sorter.neural_audio.calibration import ConfidenceCalibrationBundle
from aaron_sound_sorter.neural_audio.runtime import NeuralEventSuggestion, NeuralRuntimePrediction


def _result(
    label: str,
    *,
    loop_like: bool = True,
    consensus_status: str = "legacy_test",
    authority_trace: dict[str, object] | None = None,
) -> SortFileResult:
    physics = AudioPhysics(
        source_path=Path("/audio/content.wav"),
        fingerprint=np.zeros(4, dtype=np.float32),
        duration_sec=4.0,
        read_status="ok",
    )
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=loop_like,
        is_single_event_like=not loop_like,
        is_short_hit_like=not loop_like,
        is_long=loop_like,
        evidence={},
    )
    decision = ConsensusDecision(
        final_label=label,
        final_top=label.split("/", 1)[0],
        folder_path=label,
        consensus_status=consensus_status,
        reason="legacy result",
        authority_trace=authority_trace or {},
    )
    empty_votes = VoterResult(voter_name="test", guesses=[])
    return SortFileResult(
        source_path=physics.source_path,
        placed_path=None,
        physics=physics,
        facts=facts,
        brain_votes=empty_votes,
        physics_votes=empty_votes,
        decision=decision,
    )


def _prediction(
    label: str,
    *,
    ready: bool,
    known: bool = True,
    exact: bool = False,
    reason: str = "supported_separated_neighborhood",
    semantic_family: str = "",
    semantic_top_score: float = 0.0,
    semantic_family_scores: dict[str, float] | None = None,
    panns_family_scores: dict[str, float] | None = None,
    panns_support_score: float = 0.0,
    panns_contradiction_score: float = 0.0,
    panns_supporting_events: tuple[NeuralEventSuggestion, ...] = (),
    panns_contradicting_events: tuple[NeuralEventSuggestion, ...] = (),
) -> NeuralRuntimePrediction:
    return NeuralRuntimePrediction(
        row_id="00001",
        file_sha256="a" * 64,
        predicted_label=label,
        second_label="Instruments/Mixed Musical Loops/Multi Instrument/Loops",
        top_similarity=0.82,
        second_similarity=0.62,
        margin=0.20,
        radius_ratio=0.70,
        known_distribution=known,
        label_example_count=4,
        exact_training_match=exact,
        ownership_ready=ready,
        ownership_block_reason=reason,
        semantic_family=semantic_family,
        semantic_top_score=semantic_top_score,
        semantic_family_scores=semantic_family_scores or {},
        panns_family_scores=panns_family_scores or {},
        panns_support_score=panns_support_score,
        panns_contradiction_score=panns_contradiction_score,
        panns_supporting_events=panns_supporting_events,
        panns_contradicting_events=panns_contradicting_events,
    )


class _ConstantCalibrator:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_probability(self, _feedback) -> float:
        return self.probability


def test_cli_neural_exact_training_match_takes_ownership() -> None:
    label = "Instruments/Synths/Synth Pad/Loops"
    updated = apply_neural_prediction_to_result(
        _result("_TO_REVIEW/Measured Owner Conflict"),
        _prediction(label, ready=True, exact=True, reason="exact_human_training_match"),
    )

    assert updated.decision.folder_path == label
    assert updated.decision.consensus_status == "neural_production_owner"
    assert updated.facts.evidence["neural_runtime"]["exact_training_match"] is True


def test_unpromoted_generalization_keeps_compatible_legacy_owner() -> None:
    legacy = "Instruments/Synths/Synth Pad/Loops"
    original = _result(legacy)

    updated = apply_neural_prediction_to_result(
        original,
        _prediction("Instruments/Keys/Electric Piano/Loops", ready=True),
        enabled_groups=frozenset(),
    )

    assert updated.decision == original.decision
    assert updated.facts.evidence["neural_runtime"]["ownership_block_reason"] == ("confidence_calibration_not_promoted")


def test_promoted_group_with_calibration_can_generalize() -> None:
    label = "Instruments/Voice/Vocal One Shots/One Shots"
    calibration = ConfidenceCalibrationBundle(_ConstantCalibrator(0.93))  # type: ignore[arg-type]

    updated = apply_neural_prediction_to_result(
        _result("Instruments/Voice/Voice Phrase One Shots/One Shots", loop_like=False),
        _prediction(label, ready=True),
        calibration=calibration,
        enabled_groups=frozenset({"voice"}),
    )

    assert updated.decision.folder_path == label
    assert updated.decision.consensus_status == "neural_production_owner"


def test_reviewed_calibration_blocks_low_probability_generalization() -> None:
    label = "Instruments/Synths/Synth Pad/Loops"
    calibration = ConfidenceCalibrationBundle(_ConstantCalibrator(0.61))  # type: ignore[arg-type]

    updated = apply_neural_prediction_to_result(
        _result("Instruments/Mixed Musical Loops/Multi Instrument/Loops"),
        _prediction(label, ready=True, exact=False),
        calibration=calibration,
    )

    assert updated.decision.folder_path == "_TO_REVIEW/Measured Role Conflict"
    assert updated.decision.consensus_status == "neural_calibration_below_threshold_review"
    calibration_evidence = updated.facts.evidence["neural_runtime"]["confidence_calibration"]
    assert calibration_evidence["prediction_correct_probability"] == 0.61


def test_exact_human_match_is_not_blocked_by_generalization_calibration() -> None:
    label = "Instruments/Synths/Synth Pad/Loops"
    calibration = ConfidenceCalibrationBundle(_ConstantCalibrator(0.20))  # type: ignore[arg-type]

    updated = apply_neural_prediction_to_result(
        _result("_TO_REVIEW/Measured Owner Conflict"),
        _prediction(label, ready=True, exact=True, reason="exact_human_training_match"),
        calibration=calibration,
    )

    assert updated.decision.folder_path == label
    assert updated.decision.consensus_status == "neural_production_owner"


@pytest.mark.parametrize(
    "approved_label",
    [
        "Drums/Snares/Generic Snare/One Shots",
        "Instruments/Synths/Synth Pad/Loops",
        "Instruments/Voice/Vocal Loops/Loops",
        "FX/Human and Voice FX/Altered Voice/Long FX",
    ],
)
def test_cli_exact_user_training_overrides_advisory_disagreement_for_every_family(
    approved_label: str,
) -> None:
    updated = apply_neural_prediction_to_result(
        _result("FX/Impacts and Hits/Generic Impact/One Shots", loop_like=False),
        _prediction(
            approved_label,
            ready=True,
            exact=True,
            reason="exact_human_training_match",
            semantic_family="fx_impact",
            semantic_top_score=0.24,
            semantic_family_scores={"fx_impact": 0.24, "human_voice": 0.11},
            panns_contradiction_score=0.88,
            panns_contradicting_events=(NeuralEventSuggestion("Saxophone", 0.88),),
        ),
    )

    assert updated.decision.folder_path == approved_label
    assert updated.decision.consensus_status == "neural_production_owner"
    assert updated.facts.evidence["neural_runtime"]["authority_action"] == ("exact_human_training_override")
    assert updated.facts.evidence["neural_runtime"]["ownership_block_reason"] == ("exact_human_training_match")


def test_cli_exact_voice_memory_with_voice_audio_keeps_ownership() -> None:
    label = "Instruments/Voice/Vocal Loops/Loops"
    updated = apply_neural_prediction_to_result(
        _result("_TO_REVIEW/Measured Owner Conflict"),
        _prediction(
            label,
            ready=True,
            exact=False,
            reason="exact_human_training_match",
            semantic_family="human_voice",
            semantic_top_score=0.18,
            semantic_family_scores={"human_voice": 0.18, "bass": 0.10},
        ),
    )

    assert updated.decision.folder_path == label
    assert updated.decision.consensus_status == "neural_production_owner"


def test_cli_panns_voice_support_prevents_weak_clap_veto() -> None:
    label = "Instruments/Voice/Phrase/One Shots"
    updated = apply_neural_prediction_to_result(
        _result("FX/Impacts and Hits/Generic Impact/One Shots", loop_like=False),
        _prediction(
            label,
            ready=True,
            exact=True,
            semantic_family="fx_impact",
            semantic_top_score=0.24,
            semantic_family_scores={"fx_impact": 0.24, "human_voice": 0.11},
            panns_support_score=0.73,
            panns_supporting_events=(NeuralEventSuggestion("Speech", 0.73),),
        ),
    )

    assert updated.decision.folder_path == label
    assert updated.decision.consensus_status == "neural_production_owner"


def test_cli_panns_contradiction_cannot_veto_exact_user_training() -> None:
    updated = apply_neural_prediction_to_result(
        _result("_TO_REVIEW/Measured Owner Conflict"),
        _prediction(
            "Instruments/Voice/Vocal Loops/Loops",
            ready=True,
            exact=True,
            panns_contradiction_score=0.88,
            panns_contradicting_events=(NeuralEventSuggestion("Saxophone", 0.88),),
        ),
    )

    assert updated.decision.folder_path == "Instruments/Voice/Vocal Loops/Loops"
    assert updated.decision.consensus_status == "neural_production_owner"


def test_cli_neural_unready_cross_family_disagreement_forces_review() -> None:
    updated = apply_neural_prediction_to_result(
        _result("Drums/Drum Loops/Full Drum Loops/Loops"),
        _prediction(
            "FX/Human and Voice FX/Altered Voice/One Shots",
            ready=False,
            reason="outside_learned_radius",
        ),
    )

    assert updated.decision.folder_path == "_TO_REVIEW/Measured Role Conflict"
    assert updated.decision.consensus_status == "neural_legacy_owner_conflict_review"
    assert (
        updated.decision.authority_trace["pre_neural_decision"]["folder_path"]
        == "Drums/Drum Loops/Full Drum Loops/Loops"
    )


def test_cli_neural_unready_same_family_keeps_measured_legacy_result() -> None:
    legacy = "Instruments/Keys/Electric Piano/Loops"
    original = _result(legacy)
    updated = apply_neural_prediction_to_result(
        original,
        _prediction(
            "Instruments/Synths/Synth Pad/Loops",
            ready=False,
            reason="ambiguous_nearest_labels",
        ),
    )

    assert updated.decision == original.decision
    assert updated.facts.evidence["neural_runtime"]["ownership_ready"] is False


def test_cli_neural_unknown_sax_vs_keys_disagreement_forces_review() -> None:
    updated = apply_neural_prediction_to_result(
        _result("Instruments/Woodwinds/Saxophone/Loops"),
        _prediction(
            "Instruments/Keys/Piano/Loops",
            ready=False,
            known=False,
            reason="outside_learned_radius",
        ),
    )

    assert updated.decision.folder_path == "_TO_REVIEW/Measured Role Conflict"
    assert updated.decision.consensus_status == "neural_legacy_owner_conflict_review"


def test_cli_neural_subfamily_conflict_defers_to_strong_measured_support() -> None:
    legacy = "Drums/Percussion/Generic Percussion/One Shots"
    original = _result(
        legacy,
        loop_like=False,
        authority_trace={
            "raw_claim": {
                "path": "Drums/Snares/Acoustic Snare/One Shots",
                "strength": 0.875,
            }
        },
    )
    updated = apply_neural_prediction_to_result(
        original,
        _prediction(
            "Drums/Snares/Acoustic Snare/One Shots",
            ready=False,
            reason="ambiguous_nearest_labels",
        ),
    )

    assert updated.decision == original.decision


def test_cli_neural_unready_conflict_defers_to_exact_human_teacher() -> None:
    legacy = "Instruments/Woodwinds/Saxophone/Loops"
    original = _result(
        legacy,
        consensus_status="exact_human_teacher_owner_claim",
    )
    updated = apply_neural_prediction_to_result(
        original,
        _prediction(
            "FX/Human and Voice FX/Altered Voice/Long FX",
            ready=False,
            reason="ambiguous_nearest_labels",
        ),
    )

    assert updated.decision == original.decision
    assert updated.facts.evidence["neural_runtime"]["authority_action"] == "defer_to_exact_human_teacher"


@pytest.mark.parametrize(
    ("label", "family", "legacy"),
    [
        (
            "FX/Human and Voice FX/Altered Voice/Long FX",
            "human_voice",
            "_TO_REVIEW/Measured Role Conflict",
        ),
        (
            "Instruments/Synths/Synth Pad/Loops",
            "synth",
            "Instruments/Keys/Electric Piano/Loops",
        ),
        (
            "Drums/Kick Drums/Generic Kick/One Shots",
            "drum_kick",
            "FX/Impacts and Hits/Generic Impact/One Shots",
        ),
        (
            "FX/Animals and Creatures/Animal Vocalizations/Long FX",
            "animal_creature",
            "FX/Textures/Abstract Texture/Long FX",
        ),
    ],
)
def test_familiar_memory_with_clap_and_panns_family_agreement_takes_ownership(
    label: str,
    family: str,
    legacy: str,
) -> None:
    updated = apply_neural_prediction_to_result(
        _result(legacy, loop_like=label.endswith(("Loops", "Long FX"))),
        _prediction(
            label,
            ready=False,
            known=True,
            reason="ambiguous_nearest_labels",
            semantic_family=family,
            semantic_top_score=0.344,
            semantic_family_scores={family: 0.344, "fx_impact": 0.18},
            panns_family_scores={family: 0.429},
            panns_support_score=0.429,
        ),
    )

    assert updated.decision.folder_path == label
    assert updated.decision.consensus_status == "neural_cross_model_family_consensus_owner"
    evidence = updated.facts.evidence["neural_runtime"]["cross_model_family_consensus"]
    assert evidence["supported"] is True
    assert evidence["confidence"] >= 0.82
    assert updated.facts.evidence["neural_runtime"]["ownership_ready"] is True
    assert updated.facts.evidence["neural_runtime"]["authority_action"] == ("cross_model_family_consensus_owner")


def test_cross_model_consensus_requires_both_independent_models() -> None:
    label = "FX/Human and Voice FX/Altered Voice/Long FX"
    original = _result("_TO_REVIEW/Measured Role Conflict")
    updated = apply_neural_prediction_to_result(
        original,
        _prediction(
            label,
            ready=False,
            known=True,
            reason="ambiguous_nearest_labels",
            semantic_family="human_voice",
            semantic_top_score=0.344,
            semantic_family_scores={"human_voice": 0.344},
            panns_family_scores={},
            panns_support_score=0.0,
        ),
    )

    assert updated.decision.folder_path != label
    assert updated.facts.evidence["neural_runtime"]["cross_model_family_consensus"]["supported"] is False


def test_cross_model_consensus_does_not_promote_out_of_distribution_memory() -> None:
    label = "FX/Human and Voice FX/Altered Voice/Long FX"
    original = _result("_TO_REVIEW/Measured Role Conflict")
    updated = apply_neural_prediction_to_result(
        original,
        _prediction(
            label,
            ready=False,
            known=False,
            reason="outside_learned_radius",
            semantic_family="human_voice",
            semantic_top_score=0.344,
            semantic_family_scores={"human_voice": 0.344},
            panns_family_scores={"human_voice": 0.429},
            panns_support_score=0.429,
        ),
    )

    assert updated.decision.folder_path != label
    assert updated.facts.evidence["neural_runtime"]["cross_model_family_consensus"]["supported"] is False
