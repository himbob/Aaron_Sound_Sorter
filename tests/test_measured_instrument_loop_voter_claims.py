"""Measured instrument-loop voter claim guards.

These tests keep two real FX_Aaron2 failures below final arbitration:
- a low, bass-heavy multi-sample melody loop should become broad Instrument Loops,
  not Shape Conflict review or Bass One Shot.
- a sax/woodwind ensemble loop with mixed branch evidence should emit a
  Brass/Woodwinds loop claim before review arbitration.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.claim_producers.measured_instrument_branches import (
    MeasuredInstrumentBranchClaimProducer,
)
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.eligibility_decision import EligibilityDecision
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path, review_claim


def _claim(path: str, *, source: str = "strong_consensus", score: float = 7.0, strength: float = 0.50):
    return claim_from_folder_path(
        folder_path=path,
        source=source,
        reason="test claim",
        shared=[],
        raw_candidate_score=score,
        brain_rank=2,
        physics_rank=5,
        shared_winner=path,
        can_override=False,
        strength=strength,
        is_real_candidate=True,
    )


def _context(raw_path: str, facts: SharedAudioFacts, *, role_name: str) -> DecisionContext:
    return DecisionContext(
        raw=_claim(raw_path),
        eligibility=EligibilityDecision(role_name=role_name, confidence=0.90),
        facts=facts,
    )


def _low_mixed_melody_loop_facts() -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {
                "primary_shape": "beat_loop",
                "secondary_shape": "pitched_repetition_phrase",
                "confidence": 0.976,
                "onset_count": 14.0,
                "onset_span_ratio": 0.779,
                "low_event_ratio": 0.870,
                "mid_event_ratio": 0.125,
                "high_event_ratio": 0.005,
                "pitched_event_ratio": 1.0,
                "sustained_tonal_frame_ratio": 1.0,
                "non_event_tonal_ratio": 1.0,
                "percussive_event_ratio": 0.0,
                "drumlike_frame_ratio": 0.0,
                "pitch_confidence": 0.845,
                "spectral_flatness_mean": 0.083,
            },
            "measured_roles": {
                "bass_loop": 0.978,
                "pitched_music_loop": 0.997,
                "low_rhythmic_drum_loop": 0.565,
                "percussive_drum_loop": 0.0,
            },
            "physics_subpanels": {
                "flat": {
                    "bass_synth_score": 0.769,
                    "bass_sub_score": 0.707,
                    "bass_electric_score": 0.548,
                    "low_end_source_score": 0.685,
                    "keys_tonal_decay_score": 0.893,
                    "struck_keys_score": 0.563,
                    "synth_tonal_source_score": 0.670,
                    "synth_pad_score": 0.593,
                    "synth_lead_score": 0.603,
                    "drum_loop_source_score": 0.656,
                    "drum_hit_score": 0.0,
                }
            },
            "physics_layer_decision": {
                "instrument_branch_selected": "Bass",
                "physics_layer_branch": "Bass",
                "instrument_branch_selected_confidence": 0.729,
                "compound_music_strength": 0.470,
            },
        },
    )


def _mixed_reed_woodwind_loop_facts() -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {
                "primary_shape": "pitched_repetition_phrase",
                "secondary_shape": "transition_drop",
                "confidence": 0.859,
                "onset_count": 15.0,
                "onset_span_ratio": 0.787,
                "low_event_ratio": 0.197,
                "mid_event_ratio": 0.482,
                "high_event_ratio": 0.321,
                "pitched_event_ratio": 0.933,
                "f0_voiced_ratio": 0.985,
                "sustained_tonal_frame_ratio": 1.0,
                "non_event_tonal_ratio": 1.0,
                "percussive_event_ratio": 0.067,
                "drumlike_frame_ratio": 0.0,
                "spectral_flatness_mean": 0.285,
                "spectral_entropy_mean": 0.606,
            },
            "measured_roles": {
                "pitched_music_loop": 0.80,
                "vocal_music_phrase": 0.0,
                "voiced_one_shot": 0.0,
                "drum_loop": 0.0,
            },
            "physics_subpanels": {
                "flat": {
                    "reed_wind_score": 0.577,
                    "woodwind_sax_score": 0.550,
                    "reed_wind_authority_score": 0.374,
                    "voice_score": 0.613,
                    "human_spoken_voice_score": 0.764,
                    "synth_tonal_source_score": 0.505,
                    "keys_tonal_decay_score": 0.678,
                    "drum_loop_source_score": 0.130,
                }
            },
            "physics_layer_decision": {
                "instrument_branch_selected": "MixedInstrument",
                "physics_layer_branch": "MixedInstrument",
                "instrument_branch_selected_confidence": 0.935,
                "instrument_branch_Woodwinds": 0.604,
                "instrument_branch_MixedInstrument": 0.935,
                "compound_music_strength": 0.703,
                "instrument_Woodwinds_subpanel_selected": "AiryWoodwind",
                "instrument_Woodwinds_subpanel_confidence": 0.702,
            },
        },
    )


def test_low_mixed_melody_loop_emits_broad_instrument_loop_claim_before_arbiter() -> None:
    claims = MeasuredInstrumentBranchClaimProducer().produce(
        _context(
            "Instruments/Bass/Synth Bass/One Shots",
            _low_mixed_melody_loop_facts(),
            role_name="pitched_music_loop",
        )
    )

    assert any(
        claim.source == "mixed_instrument_loop_role_claim" and claim.folder_path == "Instruments/Instrument Loops/Loops"
        for claim in claims
    )


def test_reed_woodwind_ensemble_loop_emits_branch_loop_claim_before_arbiter() -> None:
    claims = MeasuredInstrumentBranchClaimProducer().produce(
        _context(
            "FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX",
            _mixed_reed_woodwind_loop_facts(),
            role_name="pitched_reed_or_instrument_loop",
        )
    )

    assert any(
        claim.source == "final_measured_branch_loop_broad_bucket"
        and claim.folder_path == "Instruments/Brass and Woodwinds/Loops"
        for claim in claims
    )


def test_review_yields_to_measured_mixed_instrument_loop_claim() -> None:
    facts = _low_mixed_melody_loop_facts()
    raw = _claim("Instruments/Bass/Synth Bass/One Shots", source="strong_consensus", score=7.0, strength=0.56)
    measured = claim_from_folder_path(
        folder_path="Instruments/Instrument Loops/Loops",
        source="mixed_instrument_loop_role_claim",
        reason="test measured claim",
        shared=[],
        raw_candidate_score=7.0,
        brain_rank=2,
        physics_rank=5,
        shared_winner="Instruments/Bass/Synth Bass/One Shots",
        can_override=True,
        strength=0.98,
        is_real_candidate=False,
    )
    review = review_claim(
        label="_TO_REVIEW/Shape Conflict",
        source="measured_shape_conflict_review",
        reason="test over-eager review",
        winner=raw,
        strength=1.0,
    )

    winner = FamilyClaimArbiter().pick_winner(raw_claim=raw, claims=[review, measured], facts=facts)

    assert winner.source == "mixed_instrument_loop_role_claim"
    assert winner.folder_path == "Instruments/Instrument Loops/Loops"


def test_review_yields_to_measured_reed_woodwind_loop_claim() -> None:
    facts = _mixed_reed_woodwind_loop_facts()
    raw = _claim(
        "FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX",
        source="strong_consensus",
        score=21.0,
        strength=0.0,
    )
    measured = claim_from_folder_path(
        folder_path="Instruments/Brass and Woodwinds/Loops",
        source="final_measured_branch_loop_broad_bucket",
        reason="test measured branch claim",
        shared=[],
        raw_candidate_score=21.0,
        brain_rank=3,
        physics_rank=18,
        shared_winner=raw.folder_path,
        can_override=True,
        strength=0.97,
        is_real_candidate=False,
    )
    review = review_claim(
        label="_TO_REVIEW/No Strong Voter Consensus",
        source="weak_voter_consensus",
        reason="test weak consensus review",
        winner=raw,
        strength=1.0,
    )

    winner = FamilyClaimArbiter().pick_winner(raw_claim=raw, claims=[review, measured], facts=facts)

    assert winner.source == "final_measured_branch_loop_broad_bucket"
    assert winner.folder_path == "Instruments/Brass and Woodwinds/Loops"
