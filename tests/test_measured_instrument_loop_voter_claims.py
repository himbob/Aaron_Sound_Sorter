"""Measured instrument-loop voter claim guards.

These tests keep two real FX_Aaron2 failures below final arbitration:
- a low, bass-heavy multi-sample melody loop should become broad Instrument Loops,
  not Shape Conflict review or Bass One Shot.
- a sax/woodwind ensemble loop with mixed branch evidence should emit a
  Brass/Woodwinds loop claim before review arbitration.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import CategoryGuess, SharedAudioFacts, VoterResult
from aaron_sound_sorter.engine.claim_producers.measured_instrument_branches import (
    MeasuredInstrumentBranchClaimProducer,
)
from aaron_sound_sorter.engine.claim_producers.measured_music_structures import (
    MeasuredMusicStructureClaimProducer,
)
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility_decision import EligibilityDecision
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path, review_claim
from aaron_sound_sorter.engine.placement_resolver import PlacementResolver
from aaron_sound_sorter.engine.voice_source_contracts import processed_voice_source_owned


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


def _sax_with_competing_synth_loop_facts() -> SharedAudioFacts:
    facts = _sax_like_decoy_facts()
    facts.evidence["shape_vote"].update(
        {
            "primary_shape": "bass_phrase",
            "confidence": 0.988,
            "low_event_ratio": 0.56,
            "mid_event_ratio": 0.34,
            "high_event_ratio": 0.10,
            "pitched_event_ratio": 0.93,
            "sustained_tonal_frame_ratio": 0.88,
            "percussive_event_ratio": 0.03,
            "drumlike_frame_ratio": 0.0,
            "spectral_flatness_mean": 0.18,
            "pitch_confidence": 0.70,
        }
    )
    facts.evidence["measured_roles"].update(
        {
            "pitched_music_loop": 0.92,
            "pitched_music_phrase": 0.88,
        }
    )
    facts.evidence["physics_subpanels"]["flat"].update(
        {
            "woodwind_sax_score": 0.622,
            "reed_wind_score": 0.61,
            "reed_wind_authority_score": 0.60,
            "synth_tonal_source_score": 0.508,
            "synth_lead_score": 0.620,
            "synth_pad_score": 0.500,
            "synth_chord_score": 0.574,
        }
    )
    facts.evidence["physics_vote_result"] = {
        "top_guesses": [
            {
                "label": "Instruments/Woodwinds/Saxophone/Loops",
                "folder_path": "Instruments/Woodwinds/Saxophone/Loops",
                "rank": 1,
                "score": 0.92,
            }
        ]
    }
    return facts


def _synth_pad_with_false_woodwind_branch_facts() -> SharedAudioFacts:
    facts = _sax_with_competing_synth_loop_facts()
    facts.evidence["shape_vote"].update(
        {
            "primary_shape": "bass_phrase",
            "confidence": 0.946,
            "low_event_ratio": 0.554,
            "mid_event_ratio": 0.443,
            "high_event_ratio": 0.002,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
            "non_event_tonal_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        }
    )
    facts.evidence["physics_subpanels"]["flat"].update(
        {
            "woodwind_sax_score": 0.691,
            "reed_wind_score": 0.619,
            "reed_wind_authority_score": 0.494,
            "synth_tonal_source_score": 0.714,
            "synth_lead_score": 0.743,
            "synth_pad_score": 0.785,
            "synth_chord_score": 0.776,
            "struck_keys_score": 0.514,
            "struck_keys_authority_score": 0.364,
            "keys_tonal_decay_score": 0.863,
        }
    )
    return facts


def _raw_instrument_loop_with_sax_and_synth_support():
    return claim_from_folder_path(
        folder_path="Instruments/Instrument Loops/Loops",
        source="strong_consensus",
        reason="test raw broad instrument loop",
        shared=[
            {
                "folder_path": "Instruments/Woodwinds/Saxophone/Loops",
                "top_family": "Instruments",
                "combined_rank_score": 2.0,
                "brain_rank": 1,
                "physics_rank": 1,
            },
            {
                "folder_path": "Instruments/Synths/Synth Loops",
                "top_family": "Instruments",
                "combined_rank_score": 9.0,
                "brain_rank": 7,
                "physics_rank": 5,
            },
        ],
        raw_candidate_score=9.0,
        brain_rank=7,
        physics_rank=5,
        shared_winner="Instruments/Instrument Loops/Loops",
        can_override=False,
        strength=0.50,
        is_real_candidate=True,
    )


def _raw_sax_loop_with_synth_support():
    broad_raw = _raw_instrument_loop_with_sax_and_synth_support()
    return claim_from_folder_path(
        folder_path="Instruments/Woodwinds/Saxophone/Loops",
        source="strong_consensus",
        reason="test false sax consensus",
        shared=broad_raw.shared_candidates,
        raw_candidate_score=2.0,
        brain_rank=1,
        physics_rank=1,
        shared_winner="Instruments/Woodwinds/Saxophone/Loops",
        can_override=False,
        strength=0.875,
        is_real_candidate=True,
    )


def _raw_human_taught_sax_loop_with_synth_support():
    raw = _raw_sax_loop_with_synth_support()
    shared = [dict(row) for row in raw.shared_candidates]
    shared[0]["brain_evidence"] = {"human_override_generalized_audio_match": True}
    return claim_from_folder_path(
        folder_path=raw.folder_path,
        source=raw.source,
        reason="test human-taught false sax consensus",
        shared=shared,
        raw_candidate_score=raw.raw_candidate_score,
        brain_rank=raw.brain_rank,
        physics_rank=raw.physics_rank,
        shared_winner=raw.shared_winner,
        can_override=raw.can_override,
        strength=raw.strength,
        is_real_candidate=raw.is_real_candidate,
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
            "brain_ensemble_vote_result": {
                "top_guesses": [
                    {
                        "label": "Instruments/Voice/Vocal Loops/Loops",
                        "folder_path": "Instruments/Voice/Vocal Loops/Loops",
                        "top_family": "Instruments",
                        "rank": 3,
                        "score": 1.80,
                        "confidence": 0.82,
                    },
                ]
            },
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


def _processed_formant_fx_decoy_facts() -> SharedAudioFacts:
    """Return formant-like FX facts that old Voice invariants over-believed."""
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {
                "primary_shape": "pitched_repetition_phrase",
                "secondary_shape": "designed_tonal_fx",
                "confidence": 0.926,
                "duration_sec": 7.71,
                "onset_count": 18.0,
                "high_event_ratio": 0.12,
                "pitched_event_ratio": 0.84,
                "f0_voiced_ratio": 0.92,
                "percussive_event_ratio": 0.09,
                "drumlike_frame_ratio": 0.09,
                "sustained_tonal_frame_ratio": 0.91,
            },
            "measured_roles": {"detected_parent_role": "pitched_music_loop", "pitched_music_loop": 0.88},
            "brain_ensemble_vote_result": {
                "top_guesses": [
                    {
                        "label": "Instruments/Guitar/Guitar Loops/One Shots",
                        "folder_path": "Instruments/Guitar/Guitar Loops/One Shots",
                        "top_family": "Instruments",
                        "rank": 1,
                        "score": 0.94,
                        "confidence": 0.78,
                    },
                    {
                        "label": "Instruments/Voice/Vocal Loops/Loops",
                        "folder_path": "Instruments/Voice/Vocal Loops/Loops",
                        "top_family": "Instruments",
                        "rank": 3,
                        "score": 1.20,
                        "confidence": 0.70,
                    },
                ]
            },
            "physics_vote_1": "Instruments/Voice/Choir/Loops",
            "physics_subpanels": {
                "flat": {
                    "voice_score": 0.62,
                    "human_spoken_voice_score": 0.78,
                    "human_breath_mouth_score": 0.47,
                    "voice_choir_score": 0.55,
                    "fx_formant_score": 0.65,
                    "fx_glitch_stutter_score": 0.53,
                    "fx_motion_score": 0.07,
                    "fx_transition_authority_score": 0.11,
                    "drum_loop_source_score": 0.29,
                    "drum_hit_score": 0.36,
                }
            },
        },
    )


def _rank_one_voice_decoy_melodic_loop_facts() -> SharedAudioFacts:
    """Return a non-vocal melodic loop where the brain overcalls Voice."""
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {
                "primary_shape": "pitched_repetition_phrase",
                "secondary_shape": "repeated_phrase_loop",
                "confidence": 0.992,
                "duration_sec": 12.0,
                "onset_count": 107.0,
                "pitched_event_ratio": 1.0,
                "f0_voiced_ratio": 1.0,
                "percussive_event_ratio": 0.0,
                "drumlike_frame_ratio": 0.0,
            },
            "measured_roles": {"detected_parent_role": "pitched_music_loop", "pitched_music_loop": 0.90},
            "brain_ensemble_vote_result": {
                "top_guesses": [
                    {
                        "label": "Instruments/Voice/Vocal Loops/Loops",
                        "folder_path": "Instruments/Voice/Vocal Loops/Loops",
                        "top_family": "Instruments",
                        "rank": 1,
                        "score": 0.22,
                        "confidence": 1.0,
                        "support": 4.51,
                    },
                    {
                        "label": "Instruments/Mixed Musical Loops/Multi Instrument/Loops",
                        "folder_path": "Instruments/Mixed Musical Loops/Multi Instrument/Loops",
                        "top_family": "Instruments",
                        "rank": 4,
                        "score": 0.40,
                        "confidence": 1.0,
                        "support": 2.50,
                    },
                ]
            },
            "physics_vote_1": "Instruments/Woodwinds/Saxophone/Loops",
            "physics_layer_decision": {
                "instrument_branch_selected": "Woodwinds",
                "physics_layer_branch": "Woodwinds",
                "instrument_branch_selected_confidence": 0.57,
            },
            "physics_subpanels": {
                "flat": {
                    "voice_score": 0.73,
                    "human_spoken_voice_score": 0.88,
                    "human_breath_mouth_score": 0.51,
                    "fx_formant_score": 0.76,
                    "reed_wind_score": 0.69,
                    "woodwind_sax_score": 0.73,
                    "synth_tonal_source_score": 0.58,
                    "bowed_string_score": 0.79,
                    "drum_loop_source_score": 0.10,
                    "drum_hit_score": 0.18,
                    "fx_motion_score": 0.29,
                    "fx_transition_authority_score": 0.38,
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


def test_processed_formant_fx_decoy_does_not_emit_instrument_voice_claim() -> None:
    claims = MeasuredMusicStructureClaimProducer().produce(
        _context(
            "FX/Human and Voice FX/Spoken Voice/Long FX",
            _processed_formant_fx_decoy_facts(),
            role_name="pitched_music_loop",
        )
    )

    assert not any(claim.source == "final_measured_voice_invariant" for claim in claims)


def test_rank_one_voice_brain_candidate_does_not_own_generic_melodic_loop() -> None:
    facts = _rank_one_voice_decoy_melodic_loop_facts()

    claims = MeasuredMusicStructureClaimProducer().produce(
        _context(
            "Instruments/Woodwinds/Saxophone/Loops",
            facts,
            role_name="pitched_music_loop",
        )
    )

    assert processed_voice_source_owned(facts) is False
    assert not any(claim.source == "final_measured_voice_invariant" for claim in claims)


def test_processed_formant_fx_decoy_cannot_use_final_voice_invariant() -> None:
    raw = _claim("Instruments/Instrument Loops/Loops", score=12.0, strength=0.45)
    voice = claim_from_folder_path(
        folder_path="Instruments/Voice/Vocal Loops/Loops",
        source="final_measured_voice_invariant",
        reason="test overbroad processed voice invariant",
        shared=[],
        raw_candidate_score=6.0,
        brain_rank=3,
        physics_rank=3,
        shared_winner="Instruments/Voice/Vocal Loops/Loops",
        can_override=True,
        strength=0.99,
        is_real_candidate=True,
    )

    winner = FamilyClaimArbiter().pick_winner(raw_claim=raw, claims=[voice], facts=_processed_formant_fx_decoy_facts())

    assert winner.folder_path == "Instruments/Instrument Loops/Loops"
    assert winner.source == "strong_consensus"


def test_rank_one_voice_decoy_cannot_use_final_voice_invariant() -> None:
    raw = _claim("Instruments/Instrument Loops/Loops", score=12.0, strength=0.45)
    voice = claim_from_folder_path(
        folder_path="Instruments/Voice/Vocal Loops/Loops",
        source="final_measured_voice_invariant",
        reason="test overbroad rank-one voice invariant",
        shared=[],
        raw_candidate_score=6.0,
        brain_rank=1,
        physics_rank=12,
        shared_winner="Instruments/Voice/Vocal Loops/Loops",
        can_override=True,
        strength=0.99,
        is_real_candidate=True,
    )

    winner = FamilyClaimArbiter().pick_winner(
        raw_claim=raw,
        claims=[voice],
        facts=_rank_one_voice_decoy_melodic_loop_facts(),
    )

    assert winner.folder_path == "Instruments/Instrument Loops/Loops"
    assert winner.source == "strong_consensus"


def test_processed_voice_candidate_blocks_broad_mixed_loop_claim() -> None:
    raw = _claim("Instruments/Mixed Musical Loops/Multi Instrument/Loops", score=11.0, strength=0.40)
    facts = _processed_voice_loop_facts()
    brain_result = VoterResult(
        voter_name="brain_full",
        guesses=[
            CategoryGuess(
                label="Instruments/Mixed Musical Loops/Multi Instrument/Loops",
                folder_path="Instruments/Mixed Musical Loops/Multi Instrument/Loops",
                top_family="Instruments",
                score=1.0,
                confidence=0.90,
                rank=1,
                reason="synthetic mixed loop candidate",
            ),
            CategoryGuess(
                label="Instruments/Voice/Vocal Loops/Loops",
                folder_path="Instruments/Voice/Vocal Loops/Loops",
                top_family="Instruments",
                score=1.2,
                confidence=0.86,
                rank=2,
                reason="synthetic voice candidate",
            ),
        ],
    )

    claims = DecisionCoreV2().gather_eligibility_claims(
        raw,
        EligibilityDecision(role_name="pitched_music_loop", confidence=0.90),
        facts,
        brain_result=brain_result,
        physics_result=None,
    )

    assert not any(claim.source == "profile_candidate_mixed_musical_loop_claim" for claim in claims)


def test_processed_voice_candidate_beats_clean_keys_loop_claim() -> None:
    facts = _processed_voice_loop_facts()
    raw = _claim("Instruments/Mixed Musical Loops/Multi Instrument/Loops", score=12.0, strength=0.45)
    keys = claim_from_folder_path(
        folder_path="Instruments/Keys/Electric Piano/Loops",
        source="final_clean_keys_loop_invariant",
        reason="test competing keys branch",
        shared=[],
        raw_candidate_score=5.0,
        brain_rank=1,
        physics_rank=3,
        shared_winner="Instruments/Keys/Electric Piano/Loops",
        can_override=True,
        strength=1.0,
        is_real_candidate=True,
    )
    voice = claim_from_folder_path(
        folder_path="Instruments/Voice/Vocal Loops/Loops",
        source="final_measured_voice_invariant",
        reason="test measured processed voice branch",
        shared=[],
        raw_candidate_score=6.0,
        brain_rank=3,
        physics_rank=4,
        shared_winner="Instruments/Voice/Vocal Loops/Loops",
        can_override=True,
        strength=0.99,
        is_real_candidate=True,
    )

    winner = FamilyClaimArbiter().pick_winner(raw_claim=raw, claims=[keys, voice], facts=facts)

    assert winner.source == "final_measured_voice_invariant"
    assert winner.folder_path == "Instruments/Voice/Vocal Loops/Loops"


def test_voice_loop_claim_keeps_specific_voice_loop_path_in_resolver() -> None:
    claim = claim_from_folder_path(
        folder_path="Instruments/Voice/Vocal Loops/Loops",
        source="final_measured_voice_invariant",
        reason="test measured processed voice branch",
        shared=[],
        raw_candidate_score=6.0,
        brain_rank=3,
        physics_rank=4,
        shared_winner="Instruments/Voice/Vocal Loops/Loops",
        can_override=True,
        strength=0.99,
        is_real_candidate=False,
    )

    assert PlacementResolver().resolve(claim) == "Instruments/Voice/Vocal Loops/Loops"


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


def test_synth_loop_claim_stands_down_when_brain_and_physics_support_sax() -> None:
    context = DecisionContext(
        raw=_raw_instrument_loop_with_sax_and_synth_support(),
        eligibility=EligibilityDecision(role_name="pitched_reed_or_instrument_loop", confidence=0.95),
        facts=_sax_with_competing_synth_loop_facts(),
    )

    claims = MeasuredInstrumentBranchClaimProducer().produce(context)

    assert any(
        claim.source == "final_measured_sax_loop_invariant"
        and claim.folder_path == "Instruments/Woodwinds/Saxophone/Loops"
        and claim.is_real_candidate
        for claim in claims
    )
    assert not any(claim.source == "final_measured_synth_loop_invariant" for claim in claims)


def test_strong_synth_pad_body_overrides_false_woodwind_branch() -> None:
    context = DecisionContext(
        raw=_raw_sax_loop_with_synth_support(),
        eligibility=EligibilityDecision(role_name="pitched_music_loop", confidence=0.95),
        facts=_synth_pad_with_false_woodwind_branch_facts(),
    )

    claims = MeasuredInstrumentBranchClaimProducer().produce(context)

    assert any(
        claim.source == "final_measured_synth_loop_invariant" and claim.folder_path == "Instruments/Synths/Pads/Loops"
        for claim in claims
    )


def test_human_taught_woodwind_vs_measured_synth_pad_goes_to_review() -> None:
    context = DecisionContext(
        raw=_raw_human_taught_sax_loop_with_synth_support(),
        eligibility=EligibilityDecision(role_name="pitched_music_loop", confidence=0.95),
        facts=_synth_pad_with_false_woodwind_branch_facts(),
    )

    claims = MeasuredInstrumentBranchClaimProducer().produce(context)

    assert any(
        claim.source == "measured_memory_owner_conflict_review"
        and claim.folder_path == "_TO_REVIEW/Measured Owner Conflict"
        for claim in claims
    )
    winner = FamilyClaimArbiter().pick_winner(raw_claim=context.raw, claims=claims, facts=context.facts)
    assert winner.source == "measured_memory_owner_conflict_review"


def test_legacy_synth_loop_predicate_stands_down_to_supported_sax_loop() -> None:
    raw = _raw_instrument_loop_with_sax_and_synth_support()

    supported = FamilyClaimArbiter()._facts_support_synth_loop(
        _sax_with_competing_synth_loop_facts(),
        raw,
    )

    assert supported is False


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
