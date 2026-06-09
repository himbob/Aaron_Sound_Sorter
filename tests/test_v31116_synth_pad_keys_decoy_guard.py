"""Regression guards for source-safe phrase shape and synth-pad keys decoys."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts, VoterResult
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import infer_parent_eligibility
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path


def _raw_claim(path: str):
    return claim_from_folder_path(
        folder_path=path,
        source="strong_consensus",
        reason="test raw",
        shared=[
            {
                "folder_path": "Instruments/Synths/Pads/Loops",
                "label": "Instruments/Synths/Pads/Loops",
                "top_family": "Instruments",
                "combined_rank_score": 10.0,
                "brain_rank": 4,
                "physics_rank": 3,
            },
            {
                "folder_path": path,
                "label": path,
                "top_family": "Instruments",
                "combined_rank_score": 8.0,
                "brain_rank": 2,
                "physics_rank": 5,
            },
        ],
        raw_candidate_score=8.0,
        brain_rank=2,
        physics_rank=5,
        shared_winner=path,
        can_override=False,
        strength=0.84,
        is_real_candidate=True,
    )


def _facts_with_shape_and_panels(*, synth_pad_score: float) -> SharedAudioFacts:
    evidence = {
        "shape_vote": {
            "primary_shape": "pitched_phrase_shape",
            "confidence": 0.91,
            "pitch_confidence": 0.78,
            "pitched_event_ratio": 0.94,
            "sustained_tonal_frame_ratio": 0.91,
            "f0_voiced_ratio": 0.86,
            "low_event_ratio": 0.52,
            "mid_event_ratio": 0.45,
            "high_event_ratio": 0.018,
            "spectral_flatness_mean": 0.055,
            "spectral_entropy_mean": 0.33,
            "percussive_event_ratio": 0.04,
            "drumlike_frame_ratio": 0.03,
            "onset_count": 5.0,
        },
        "measured_roles": {
            "detected_parent_role": "pitched_music_loop",
            "pitched_music_loop": 0.91,
            "evidence": {
                "pitched_music_loop": 0.91,
                "loop_pitched_event_ratio": 0.94,
                "loop_sustained_tonal_frame_ratio": 0.91,
            },
        },
        "physics_subpanels": {
            "flat": {
                "synth_pad_score": synth_pad_score,
                "synth_chord_score": 0.50,
                "synth_lead_score": 0.45,
                "synth_tonal_source_score": 0.82,
                "woodwind_sax_score": 0.46,
                "reed_wind_score": 0.42,
                "struck_keys_score": 0.38,
                "keys_tonal_decay_score": 0.40,
            }
        },
        "physics_layer_decision": {
            "physics_layer_branch": "Synth",
            "instrument_branch_selected": "Synth",
            "physics_layer_branch_confidence": 0.82,
            "instrument_branch_selected_confidence": 0.82,
            "instrument_branch_Synth": 0.82,
            "instrument_branch_KeysPiano": 0.55,
            "instrument_Synth_subpanel_selected": "SynthPad",
            "instrument_Synth_subpanel_confidence": 0.84,
            "instrument_Synth_subpanel_margin": 0.12,
        },
    }
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence=evidence,
    )


def _decide_with_core_claims(raw, measured: SharedAudioFacts):
    core = DecisionCoreV2()
    claims = core.gather_eligibility_claims(
        raw,
        infer_parent_eligibility(measured),
        measured,
        brain_result=VoterResult(voter_name="brain_full", guesses=[]),
        physics_result=None,
    )
    return core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)


def test_source_safe_phrase_shape_synth_pad_can_correct_keys_decoy() -> None:
    """A source-safe phrase shape must still let strong synth-pad facts beat keys."""
    raw = _raw_claim("Instruments/Keys/Electric Piano/Loops")
    measured = _facts_with_shape_and_panels(synth_pad_score=0.86)

    final = _decide_with_core_claims(raw, measured)

    assert final.folder_path == "Instruments/Synths/Pads/Loops"
    assert final.consensus_status == "final_measured_synth_loop_invariant"


def test_weak_synth_pad_panel_does_not_steal_clean_keys_loop() -> None:
    """The decoy correction is gated by strong synth-pad evidence."""
    raw = _raw_claim("Instruments/Keys/Electric Piano/Loops")
    measured = _facts_with_shape_and_panels(synth_pad_score=0.55)

    final = _decide_with_core_claims(raw, measured)

    assert final.folder_path == "Instruments/Keys/Electric Piano/Loops"
