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
from aaron_sound_sorter.engine.claim_producers.measured_music_structures import (
    MeasuredMusicStructureClaimProducer,
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


def _sax_like_decoy_facts() -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {
                "primary_shape": "pitched_repetition_phrase",
                "confidence": 0.86,
                "pitched_event_ratio": 1.0,
                "f0_voiced_ratio": 1.0,
                "sustained_tonal_frame_ratio": 1.0,
                "non_event_tonal_ratio": 1.0,
                "percussive_event_ratio": 0.0,
                "drumlike_frame_ratio": 0.0,
            },
            "measured_roles": {
                "pitched_music_loop": 1.0,
                "pitched_music_phrase": 0.90,
            },
            "physics_subpanels": {
                "flat": {
                    "woodwind_sax_score": 0.72,
                    "reed_wind_score": 0.72,
                    "reed_wind_authority_score": 0.66,
                    "synth_tonal_source_score": 0.40,
                    "synth_lead_score": 0.38,
                    "synth_pad_score": 0.38,
                    "struck_keys_score": 0.32,
                    "keys_tonal_decay_score": 0.32,
                }
            },
            "physics_layer_decision": {
                "instrument_branch_selected": "Woodwinds",
                "physics_layer_branch": "Woodwinds",
                "instrument_branch_selected_confidence": 0.90,
                "instrument_panel_Woodwinds_Sax": 0.72,
            },
        },
    )


def _designed_low_mixed_loop_facts() -> SharedAudioFacts:
    facts = _low_mixed_melody_loop_facts()
    facts.evidence["shape_vote"].update(
        {
            "primary_shape": "designed_low_fx",
            "secondary_shape": "pitched_repetition_phrase",
            "confidence": 0.83,
            "low_event_ratio": 0.92,
            "mid_event_ratio": 0.06,
            "high_event_ratio": 0.01,
            "pulse_regularity": 0.07,
        }
    )
    facts.evidence["physics_subpanels"]["flat"].update(
        {
            "bass_synth_score": 0.78,
            "bass_sub_score": 0.67,
            "keys_tonal_decay_score": 0.84,
            "synth_tonal_source_score": 0.65,
            "synth_chord_score": 0.59,
            "drum_loop_source_score": 0.04,
            "fx_motion_score": 0.21,
            "fx_transition_authority_score": 0.30,
        }
    )
    return facts


def _designed_tonal_keys_loop_facts() -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {
                "primary_shape": "designed_tonal_fx",
                "secondary_shape": "pitched_phrase_shape",
                "confidence": 0.757,
                "onset_count": 19.0,
                "onset_span_ratio": 0.87,
                "low_event_ratio": 0.32,
                "mid_event_ratio": 0.66,
                "high_event_ratio": 0.019,
                "pitched_event_ratio": 1.0,
                "f0_voiced_ratio": 1.0,
                "sustained_tonal_frame_ratio": 1.0,
                "non_event_tonal_ratio": 1.0,
                "percussive_event_ratio": 0.0,
                "drumlike_frame_ratio": 0.0,
                "spectral_flatness_mean": 0.003,
                "true_repetition_score": 0.80,
            },
            "measured_roles": {"detected_parent_role": "pitched_music_loop", "pitched_music_loop": 0.92},
            "physics_subpanels": {
                "flat": {
                    "struck_keys_score": 0.56,
                    "struck_keys_authority_score": 0.46,
                    "keys_tonal_decay_score": 0.75,
                    "synth_tonal_source_score": 0.64,
                    "synth_chord_score": 0.69,
                    "woodwind_sax_score": 0.72,
                    "voice_score": 0.62,
                    "human_spoken_voice_score": 0.70,
                    "drum_loop_source_score": 0.01,
                    "drum_hit_score": 0.11,
                    "fx_motion_score": 0.22,
                    "fx_transition_authority_score": 0.30,
                }
            },
        },
    )


def _designed_low_synth_loop_facts() -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {
                "primary_shape": "designed_low_fx",
                "secondary_shape": "pitched_repetition_phrase",
                "confidence": 0.725,
                "onset_count": 18.0,
                "onset_span_ratio": 0.90,
                "low_event_ratio": 0.84,
                "mid_event_ratio": 0.11,
                "high_event_ratio": 0.045,
                "pitched_event_ratio": 1.0,
                "sustained_tonal_frame_ratio": 1.0,
                "non_event_tonal_ratio": 1.0,
                "percussive_event_ratio": 0.0,
                "drumlike_frame_ratio": 0.0,
                "pitch_confidence": 0.71,
                "spectral_flatness_mean": 0.30,
                "true_repetition_score": 0.81,
            },
            "measured_roles": {"detected_parent_role": "pitched_music_loop", "pitched_music_loop": 0.88},
            "physics_subpanels": {
                "flat": {
                    "synth_tonal_source_score": 0.604,
                    "synth_pad_score": 0.54,
                    "synth_lead_score": 0.48,
                    "keys_tonal_decay_score": 0.69,
                    "struck_keys_score": 0.44,
                    "woodwind_sax_score": 0.43,
                    "reed_wind_authority_score": 0.50,
                    "voice_score": 0.42,
                    "human_spoken_voice_score": 0.37,
                    "bass_synth_score": 0.53,
                    "drum_loop_source_score": 0.22,
                    "drum_hit_score": 0.28,
                }
            },
        },
    )


def _processed_voice_loop_facts() -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {
                "primary_shape": "designed_low_fx",
                "secondary_shape": "designed_tonal_fx",
                "confidence": 0.876,
                "onset_count": 82.0,
                "onset_span_ratio": 0.97,
                "pitched_event_ratio": 0.94,
                "f0_voiced_ratio": 0.87,
                "percussive_event_ratio": 0.063,
                "drumlike_frame_ratio": 0.063,
                "sustained_tonal_frame_ratio": 0.90,
            },
            "measured_roles": {"detected_parent_role": "vocal_music_phrase", "vocal_music_phrase": 0.88},
            "physics_subpanels": {
                "flat": {
                    "voice_score": 0.80,
                    "human_spoken_voice_score": 0.90,
                    "human_breath_mouth_score": 0.57,
                    "woodwind_sax_score": 0.70,
                    "fx_formant_score": 0.80,
                    "drum_loop_source_score": 0.21,
                    "drum_hit_score": 0.33,
                }
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


def test_designed_low_mixed_loop_emits_broad_instrument_loop_claim_before_synth_leaf() -> None:
    claims = MeasuredInstrumentBranchClaimProducer().produce(
        _context(
            "Instruments/Synths/Synth Lead/One Shots",
            _designed_low_mixed_loop_facts(),
            role_name="pitched_music_loop",
        )
    )

    assert any(
        claim.source == "mixed_instrument_loop_role_claim" and claim.folder_path == "Instruments/Instrument Loops/Loops"
        for claim in claims
    )


def test_designed_tonal_clean_keys_loop_emits_keys_claim_before_broad_loop() -> None:
    claims = MeasuredInstrumentBranchClaimProducer().produce(
        _context(
            "Instruments/Instrument Loops/Loops",
            _designed_tonal_keys_loop_facts(),
            role_name="pitched_music_loop",
        )
    )

    assert any(
        claim.source == "final_clean_keys_loop_invariant"
        and claim.folder_path == "Instruments/Keys/Electric Piano/Loops"
        for claim in claims
    )


def test_designed_low_clean_synth_loop_emits_synth_loop_claim_before_bass_one_shot() -> None:
    claims = MeasuredInstrumentBranchClaimProducer().produce(
        _context(
            "Instruments/Bass/Synth Bass/One Shots",
            _designed_low_synth_loop_facts(),
            role_name="pitched_music_loop",
        )
    )

    assert any(
        claim.source == "final_measured_synth_loop_invariant" and claim.folder_path == "Instruments/Synths/Synth Loops"
        for claim in claims
    )


def test_processed_voice_loop_emits_instrument_voice_claim_before_fx_human_bucket() -> None:
    claims = MeasuredMusicStructureClaimProducer().produce(
        _context(
            "FX/Human and Voice FX/Spoken Voice/Long FX",
            _processed_voice_loop_facts(),
            role_name="vocal_music_phrase",
        )
    )

    assert any(
        claim.source == "final_measured_voice_invariant" and claim.folder_path == "Instruments/Voice/Vocal Loops/Loops"
        for claim in claims
    )


def test_rank_one_concrete_plucked_instrument_blocks_synthetic_sax_claim() -> None:
    raw = claim_from_folder_path(
        folder_path="Instruments/Plucked Strings/Koto/Loops",
        source="strong_consensus",
        reason="test human-corrected consensus",
        shared=[],
        raw_candidate_score=2.0,
        brain_rank=1,
        physics_rank=1,
        shared_winner="Instruments/Plucked Strings/Koto/Loops",
        can_override=False,
        strength=0.90,
        is_real_candidate=True,
    )
    context = DecisionContext(
        raw=raw,
        eligibility=EligibilityDecision(role_name="pitched_music_loop", confidence=0.95),
        facts=_sax_like_decoy_facts(),
    )

    claims = MeasuredInstrumentBranchClaimProducer().produce(context)

    assert not any(claim.source == "final_measured_sax_loop_invariant" for claim in claims)


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
