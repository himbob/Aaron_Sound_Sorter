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


from aaron_sound_sorter.engine.claim_producers.final_drum_loop import FinalDrumLoopClaimProducer


def _clean_low_tonal_beat_bass_facts() -> SharedAudioFacts:
    """Facts matching a clean synth-bass loop mis-shaped as a beat loop."""
    shape_vote = {
        "primary_shape": "beat_loop",
        "confidence": 1.0,
        "onset_count": 15.0,
        "true_repetition_score": 0.74,
        "low_event_ratio": 0.92,
        "mid_event_ratio": 0.08,
        "high_event_ratio": 0.001,
        "pitched_event_ratio": 1.0,
        "sustained_tonal_frame_ratio": 1.0,
        "non_event_tonal_ratio": 1.0,
        "percussive_event_ratio": 0.0,
        "drumlike_frame_ratio": 0.0,
        "pitch_confidence": 0.94,
        "spectral_flatness_mean": 0.015,
    }
    subpanel_flat = {
        "bass_synth_score": 0.84,
        "bass_sub_score": 0.69,
        "sustained_bass_score": 0.67,
        "low_end_source_score": 0.69,
        "drum_loop_source_score": 0.06,
        "rhythmic_break_loop_score": 0.41,
        "drum_hit_score": 0.02,
        "drum_kick_source_score": 0.33,
        "onset_percussive_onset_score": 0.05,
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
                "bass_loop": 1.0,
                "pitched_music_loop": 1.0,
                "low_rhythmic_drum_loop": 0.65,
                "drum_loop": 0.0,
                "percussive_drum_loop": 0.0,
                "bright_drum_loop": 0.0,
            },
            "physics_layer_decision": {
                "instrument_branch_selected": "Bass",
                "physics_layer_branch": "Bass",
                "instrument_branch_selected_confidence": 0.80,
            },
        },
    )


def _context_with_drum_loop_candidate(facts: SharedAudioFacts) -> DecisionContext:
    raw = claim_from_folder_path(
        folder_path="Instruments/Bass/Electric Bass/One Shots",
        source="raw_test_candidate",
        reason="test raw candidate",
        shared=[
            {
                "label": "Drums/Drum Loops/Loops",
                "folder_path": "Drums/Drum Loops/Loops",
                "top_family": "Drums",
                "combined_rank_score": 8.0,
                "brain_rank": 8,
                "physics_rank": 8,
            },
            {
                "label": "Instruments/Bass/Electric Bass/One Shots",
                "folder_path": "Instruments/Bass/Electric Bass/One Shots",
                "top_family": "Instruments",
                "combined_rank_score": 5.0,
                "brain_rank": 3,
                "physics_rank": 2,
            },
        ],
        raw_candidate_score=5.0,
        brain_rank=3,
        physics_rank=2,
        shared_winner="Instruments/Bass/Electric Bass/One Shots",
        can_override=False,
        strength=0.68,
        is_real_candidate=True,
    )
    return DecisionContext(
        raw=raw,
        eligibility=EligibilityDecision(role_name="bass_loop", confidence=1.0),
        facts=facts,
    )


def test_clean_low_tonal_beat_loop_bass_does_not_emit_drum_loop_claim() -> None:
    facts = _clean_low_tonal_beat_bass_facts()

    claims = FinalDrumLoopClaimProducer().produce(_context_with_drum_loop_candidate(facts))

    assert all(claim.source != "measured_drum_loop_claim" for claim in claims)


def test_clean_low_tonal_beat_loop_bass_emits_bass_loop_claim() -> None:
    facts = _clean_low_tonal_beat_bass_facts()

    claims = MeasuredInstrumentBranchClaimProducer().produce(_context_with_drum_loop_candidate(facts))

    assert any(claim.source == "final_measured_bass_loop_invariant" for claim in claims)


def _strong_low_drum_loop_facts_even_with_bass_energy() -> SharedAudioFacts:
    """Low loop that has actual drum-loop authority must remain Drums.

    This guards the exact worry Aaron raised: a bass-loop protection must not
    steal real low drum loops just because they contain strong bass energy.
    """
    facts = _clean_low_tonal_beat_bass_facts()
    evidence = dict(facts.evidence)
    shape_vote = dict(evidence["shape_vote"])
    shape_vote.update(
        {
            "primary_shape": "beat_loop",
            "confidence": 0.96,
            "onset_count": 18.0,
            "true_repetition_score": 0.82,
            "low_event_ratio": 0.78,
            "mid_event_ratio": 0.14,
            "high_event_ratio": 0.08,
            "pitched_event_ratio": 0.55,
            "sustained_tonal_frame_ratio": 0.42,
            "non_event_tonal_ratio": 0.38,
            "percussive_event_ratio": 0.22,
            "drumlike_frame_ratio": 0.26,
            "f0_voiced_ratio": 0.30,
            "spectral_flatness_mean": 0.12,
        }
    )
    subpanel_flat = dict(evidence["physics_subpanels"]["flat"])
    subpanel_flat.update(
        {
            "bass_synth_score": 0.66,
            "bass_sub_score": 0.61,
            "low_end_source_score": 0.62,
            "drum_loop_source_score": 0.76,
            "rhythmic_break_loop_score": 0.58,
            "drum_hit_score": 0.44,
            "drum_kick_source_score": 0.50,
            "drum_snare_source_score": 0.38,
            "drum_closed_hat_source_score": 0.30,
            "onset_percussive_onset_score": 0.52,
        }
    )
    roles = dict(evidence["measured_roles"])
    roles.update(
        {
            "bass_loop": 0.62,
            "pitched_music_loop": 0.40,
            "low_rhythmic_drum_loop": 0.84,
            "drum_loop": 0.82,
            "percussive_drum_loop": 0.72,
            "bright_drum_loop": 0.26,
        }
    )
    physics_layer = dict(evidence["physics_layer_decision"])
    physics_layer.update(
        {
            "instrument_branch_selected": "",
            "physics_layer_branch": "DrumLoop",
            "drum_branch_selected": "DrumLoop",
            "drum_branch_selected_confidence": 0.72,
        }
    )
    evidence["shape_vote"] = shape_vote
    evidence["physics_subpanels"] = {"flat": subpanel_flat}
    evidence["measured_roles"] = roles
    evidence["physics_layer_decision"] = physics_layer
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence=evidence,
    )


def test_true_low_drum_loop_still_emits_drum_loop_claim_despite_bass_energy() -> None:
    facts = _strong_low_drum_loop_facts_even_with_bass_energy()

    claims = FinalDrumLoopClaimProducer().produce(_context_with_drum_loop_candidate(facts))

    assert any(claim.source == "measured_drum_loop_claim" for claim in claims)


def test_true_low_drum_loop_blocks_clean_bass_loop_claim_despite_bass_energy() -> None:
    facts = _strong_low_drum_loop_facts_even_with_bass_energy()

    claims = MeasuredInstrumentBranchClaimProducer().produce(_context_with_drum_loop_candidate(facts))

    assert all(claim.source != "final_measured_bass_loop_invariant" for claim in claims)
