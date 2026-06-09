"""Measured bass-loop claim guard tests.

These are source-name-blind unit tests for the lower claim-producer layer.  A
strong low rhythmic drum-loop body must not be converted into a Bass Loops claim
just because the same low repeated body can resemble a bass phrase.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.claim_producers.measured_instrument_branches import (
    MeasuredInstrumentBranchClaimProducer,
)
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.eligibility_decision import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path


def _raw_claim():
    return claim_from_folder_path(
        folder_path="FX/Textures/Natural Ambience/Ocean/Long FX",
        source="raw_test_candidate",
        reason="test raw candidate",
        shared=[],
        raw_candidate_score=10.0,
        brain_rank=4,
        physics_rank=4,
        shared_winner="FX/Textures/Natural Ambience/Ocean/Long FX",
        can_override=False,
        strength=0.40,
        is_real_candidate=True,
    )


def _context(facts: SharedAudioFacts) -> DecisionContext:
    return DecisionContext(
        raw=_raw_claim(),
        eligibility=EligibilityDecision(role_name="pitched_music_loop", confidence=0.80),
        facts=facts,
    )


def _bass_phrase_facts(*, drum_loop_source_score: float, low_rhythmic_drum_loop: float) -> SharedAudioFacts:
    shape_vote = {
        "primary_shape": "bass_phrase",
        "confidence": 0.96,
        "low_event_ratio": 0.94,
        "mid_event_ratio": 0.04,
        "high_event_ratio": 0.02,
        "pitched_event_ratio": 0.94,
        "sustained_tonal_frame_ratio": 0.94,
        "non_event_tonal_ratio": 0.96,
        "percussive_event_ratio": 0.04,
        "drumlike_frame_ratio": 0.02,
        "onset_count": 16.0,
        "true_repetition_score": 0.72,
        "pitch_confidence": 0.92,
    }
    subpanel_flat = {
        "bass_synth_score": 0.72,
        "bass_sub_score": 0.78,
        "low_end_source_score": 0.74,
        "drum_loop_source_score": drum_loop_source_score,
    }
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": shape_vote,
            "physics_subpanels": {"flat": subpanel_flat},
            "measured_roles": {
                "bass_loop": 0.86,
                "pitched_music_loop": 0.84,
                "low_rhythmic_drum_loop": low_rhythmic_drum_loop,
            },
            "physics_layer_decision": {
                "instrument_branch_selected": "Bass",
                "instrument_branch_selected_confidence": 0.86,
            },
        },
    )


def test_measured_bass_loop_claim_is_blocked_by_low_rhythmic_drum_loop_authority() -> None:
    producer = MeasuredInstrumentBranchClaimProducer()
    facts = _bass_phrase_facts(drum_loop_source_score=0.72, low_rhythmic_drum_loop=1.0)

    claims = producer.produce(_context(facts))

    assert all(claim.source != "final_measured_bass_loop_invariant" for claim in claims)


def test_clean_bass_loop_claim_survives_when_drum_loop_authority_is_weak() -> None:
    producer = MeasuredInstrumentBranchClaimProducer()
    facts = _bass_phrase_facts(drum_loop_source_score=0.10, low_rhythmic_drum_loop=0.0)

    claims = producer.produce(_context(facts))

    assert any(claim.source == "final_measured_bass_loop_invariant" for claim in claims)
