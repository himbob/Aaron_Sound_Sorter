from __future__ import annotations

from pathlib import Path

import numpy as np

from aaron_sound_sorter.domain.models import (
    AudioPhysics,
    ConsensusDecision,
    SharedAudioFacts,
    SortFileResult,
    VoterResult,
)
from aaron_sound_sorter.neural_audio.authority import apply_neural_prediction_to_result
from aaron_sound_sorter.neural_audio.runtime import NeuralRuntimePrediction


def _result(
    label: str,
    *,
    loop_like: bool = True,
    consensus_status: str = "legacy_test",
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
    exact: bool = False,
    reason: str = "supported_separated_neighborhood",
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
        known_distribution=True,
        label_example_count=4,
        exact_training_match=exact,
        ownership_ready=ready,
        ownership_block_reason=reason,
    )


def test_cli_neural_exact_training_match_takes_ownership() -> None:
    label = "Instruments/Synths/Synth Pad/Loops"
    updated = apply_neural_prediction_to_result(
        _result("_TO_REVIEW/Measured Owner Conflict"),
        _prediction(label, ready=True, exact=True, reason="exact_human_training_match"),
    )

    assert updated.decision.folder_path == label
    assert updated.decision.consensus_status == "neural_production_owner"
    assert updated.facts.evidence["neural_runtime"]["exact_training_match"] is True


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
