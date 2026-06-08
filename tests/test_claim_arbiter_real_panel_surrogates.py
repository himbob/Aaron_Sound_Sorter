"""Synthetic locks for real FX_Aaron2 one-file panel failures.

The test names describe the verification case, but the assertions use only
internal voter categories and measured audio facts.  These are regression locks
for the Claim-Arbiter design, not filename evidence.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import CategoryGuess, SharedAudioFacts, VoterResult
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path


def guess(path: str, rank: int) -> CategoryGuess:
    """Build a ranked candidate guess."""
    return CategoryGuess(
        label=path,
        folder_path=path,
        top_family=path.split("/", 1)[0],
        score=float(rank),
        confidence=max(0.0, 1.0 - rank / 30.0),
        rank=rank,
        reason="synthetic candidate",
        evidence={},
    )


def facts(shape: str, confidence: float, roles: dict[str, float]) -> SharedAudioFacts:
    """Build measured facts without source names."""
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {"primary_shape": shape, "confidence": confidence},
            "measured_roles": roles,
            "direct_body_view": {"available": True, "measured_roles": roles},
        },
    )


def raw_claim(path: str, shared: list[dict], score: float = 3.0):
    """Build the raw claim produced before eligibility claims."""
    return claim_from_folder_path(
        folder_path=path,
        source="strong_consensus",
        reason="synthetic raw shared winner",
        shared=shared,
        raw_candidate_score=score,
        brain_rank=1,
        physics_rank=2,
        shared_winner=path,
        can_override=False,
        strength=0.80,
        is_real_candidate=True,
    )


def shared_row(path: str, score: float, roles: dict[str, float] | None = None) -> dict:
    """Build one shared candidate row."""
    role_signature = roles or {}
    return {
        "label": path,
        "folder_path": path,
        "top_family": path.split("/", 1)[0],
        "brain_rank": int(score),
        "physics_rank": int(score),
        "combined_rank_score": float(score),
        "candidate_role_signature": role_signature,
        "brain_evidence": {"candidate_role_signature": role_signature},
        "physics_evidence": {"candidate_role_signature": role_signature},
    }


def eligibility(role: str, broad: str, confidence: float = 0.90) -> EligibilityDecision:
    """Build an explicit parent-eligibility decision."""
    return EligibilityDecision(
        role_name=role,
        confidence=confidence,
        allowed_top_families=("Instruments", "_TO_REVIEW"),
        blocked_path_fragments=(),
        broad_folder_path=broad,
        reason="synthetic eligibility",
    )


def test_nearby_woodwind_profile_claim_does_not_overrule_concrete_fx_without_specific_support() -> None:
    """A weak nearby woodwind row must not become another overcorrection vacuum."""
    shared = [
        shared_row("FX/Designed Noise FX/Alarm/Long FX", 3.0),
        shared_row("Instruments/Woodwinds/Flute/One Shots", 6.0),
        shared_row("Instruments/Instrument Loops/Loops", 11.0),
    ]
    brain_result = VoterResult(
        voter_name="brain_full",
        guesses=[
            guess("FX/Designed Noise FX/Alarm/Long FX", 1),
            guess("Instruments/Instrument Loops/Loops", 3),
            guess("Instruments/Woodwinds/Flute/One Shots", 6),
        ],
    )
    raw = raw_claim("FX/Designed Noise FX/Alarm/Long FX", shared)
    measured = facts("pitched_phrase", 0.91, {"pitched_music_loop": 0.94, "vocal_music_phrase": 0.20})
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("pitched_reed_or_instrument_loop", "Instruments/Woodwinds/Saxophone/Loops", 0.94),
        measured,
        brain_result=brain_result,
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "FX/Designed Noise FX/Alarm/Long FX"
    assert final.consensus_status == "strong_consensus"


def test_stronger_reed_candidate_stays_broad_when_identity_is_not_physics_supported() -> None:
    """A reed-like loop may stay broad instead of claiming a specific sibling branch."""
    shared = [
        shared_row("Instruments/Instrument Loops/Loops", 16.0),
        shared_row("Instruments/Woodwinds/Saxophone/One Shots", 6.0),
    ]
    brain_result = VoterResult(
        voter_name="brain_full",
        guesses=[
            guess("Instruments/Woodwinds/Saxophone/One Shots", 1),
            guess("Instruments/Instrument Loops/Loops", 3),
        ],
    )
    raw = raw_claim("Instruments/Instrument Loops/Loops", shared, score=16.0)
    measured = facts(
        "pitched_phrase",
        0.92,
        {"pitched_reed_or_instrument_loop": 0.92, "pitched_music_loop": 0.88},
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("pitched_reed_or_instrument_loop", "Instruments/Woodwinds/Saxophone/Loops", 0.92),
        measured,
        brain_result=brain_result,
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Instruments/Instrument Loops/Loops"
    assert final.consensus_status in {"strong_consensus", "profile_candidate_ambiguous_reed_like_parent_claim"}


def test_reed_one_shot_leaf_broadens_to_brass_woodwind_loop_bucket() -> None:
    """A sax/woodwind one-shot leaf with loop structure should become broad reed loops."""
    shared = [
        shared_row("Instruments/Woodwinds/Saxophone/One Shots", 4.0),
        shared_row("Instruments/Instrument Loops/Loops", 12.0),
    ]
    raw = raw_claim("Instruments/Woodwinds/Saxophone/One Shots", shared, score=4.0)
    measured = facts(
        "pitched_phrase",
        0.94,
        {"pitched_music_loop": 0.94, "pitched_music_phrase": 0.90},
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("pitched_music_loop", "Instruments/Instrument Loops/Loops", 0.94),
        measured,
        brain_result=VoterResult(voter_name="brain_full", guesses=[]),
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Instruments/Brass and Woodwinds/Loops"
    assert final.consensus_status in {"candidate_true_bucket_rescue", "parent_eligibility_broad_bucket"}


def test_synth_mallet_counter_stops_weak_sax_branch_redeepening() -> None:
    """Weak Sax physics must not undo safe broadening when synth/mallet evidence wins."""
    shared = [
        shared_row("Instruments/Synths/Synth Lead/One Shots", 2.0),
        shared_row("Instruments/Woodwinds/Saxophone/One Shots", 3.0),
        shared_row("Instruments/Instrument Loops/Loops", 12.0),
    ]
    raw = raw_claim("Instruments/Woodwinds/Saxophone/One Shots", shared, score=3.0)
    measured = facts(
        "pitched_repetition_phrase",
        0.93,
        {"pitched_music_loop": 0.80, "pitched_music_phrase": 0.81},
    )
    measured.evidence["shape_vote"].update(
        {
            "pitched_event_ratio": 1.0,
            "f0_voiced_ratio": 0.80,
            "sustained_tonal_frame_ratio": 0.81,
            "non_event_tonal_ratio": 0.61,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
            "onset_count": 34.0,
        }
    )
    measured.evidence.update(
        {
            "physics_layer_decision": {
                "physics_layer_branch": "Woodwinds",
                "physics_layer_branch_confidence": 0.96,
                "instrument_branch_selected": "Woodwinds",
                "instrument_branch_selected_confidence": 0.96,
                "instrument_branch_Woodwinds": 0.96,
                "instrument_branch_Synth": 0.88,
                "instrument_reed_woodwind_source_signal": True,
                "instrument_woodwind_source_signal": True,
                "instrument_Woodwinds_subpanel_selected": "Sax",
                "instrument_Woodwinds_subpanel_confidence": 0.80,
                "instrument_Woodwinds_subpanel_margin": 0.11,
            },
            "woodwind_sax_score": 0.51,
            "reed_wind_score": 0.37,
            "reed_wind_authority_score": 0.20,
            "synth_tonal_source_score": 0.61,
            "synth_chord_score": 0.65,
            "synth_lead_score": 0.74,
            "synth_pad_score": 0.55,
            "pitched_mallet_instrument_score": 0.70,
            "struck_keys_score": 0.47,
            "drum_loop_source_score": 0.10,
        }
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("pitched_music_loop", "Instruments/Instrument Loops/Loops", 0.94),
        measured,
        brain_result=VoterResult(
            voter_name="brain_full", guesses=[guess("Instruments/Synths/Synth Lead/One Shots", 1)]
        ),
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Instruments/Instrument Loops/Loops"


def test_woodwind_subtype_depth_abstains_when_synth_pad_pressure_is_stronger() -> None:
    """Woodwind subtype depth needs reed authority before it can steal a synth-like loop."""
    shared = [
        shared_row("Instruments/Synths/Synth Lead/One Shots", 5.0),
        shared_row("Instruments/Woodwinds/Clarinet/One Shots", 7.0),
        shared_row("Instruments/Instrument Loops/Loops", 13.0),
    ]
    raw = raw_claim("Instruments/Synths/Synth Lead/One Shots", shared, score=5.0)
    measured = facts("pitched_repetition_phrase", 0.97, {"pitched_music_loop": 0.90})
    measured.evidence["shape_vote"].update(
        {
            "pitched_event_ratio": 1.0,
            "f0_voiced_ratio": 0.89,
            "sustained_tonal_frame_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
            "onset_count": 32.0,
        }
    )
    measured.evidence.update(
        {
            "physics_layer_decision": {
                "physics_layer_branch": "Woodwinds",
                "physics_layer_branch_confidence": 0.91,
                "instrument_branch_selected": "Woodwinds",
                "instrument_branch_selected_confidence": 0.91,
                "instrument_branch_Woodwinds": 0.91,
                "instrument_branch_Synth": 0.86,
                "instrument_Woodwinds_subpanel_selected": "Clarinet",
                "instrument_Woodwinds_subpanel_confidence": 0.82,
                "instrument_Woodwinds_subpanel_margin": 0.12,
                "instrument_woodwind_source_signal": True,
            },
            "woodwind_sax_score": 0.72,
            "woodwind_flute_score": 0.58,
            "reed_wind_score": 0.64,
            "reed_wind_authority_score": 0.56,
            "synth_tonal_source_score": 0.58,
            "synth_lead_score": 0.80,
            "synth_pad_score": 0.72,
            "voice_score": 0.62,
            "human_spoken_voice_score": 0.82,
            "bowed_string_score": 0.79,
        }
    )
    core = DecisionCoreV2()

    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=[], facts=measured)

    assert final.folder_path == "Instruments/Synths/Synth Loops"
    assert final.consensus_status == "final_measured_synth_loop_invariant"


def test_fx_texture_noise_candidate_blocks_broad_instrument_loop_release() -> None:
    """FX texture/noise conflicts should review instead of becoming Instrument Loops."""
    shared = [
        shared_row("FX/Textures/Noise and Static/Vinyl Noise/Long FX", 2.0),
        shared_row("Instruments/Instrument Loops/Loops", 14.0),
    ]
    raw = raw_claim("FX/Textures/Noise and Static/Vinyl Noise/Long FX", shared, score=2.0)
    review = claim_from_folder_path(
        folder_path="_TO_REVIEW/Measured Role Conflict",
        source="parent_eligibility_review",
        reason="synthetic measured role conflict",
        shared=shared,
        raw_candidate_score=9999.0,
        brain_rank=1,
        physics_rank=14,
        shared_winner="FX/Textures/Noise and Static/Vinyl Noise/Long FX",
        can_override=True,
        strength=1.0,
        is_real_candidate=False,
    )
    measured = facts(
        "pitched_repetition_phrase",
        0.86,
        {"pitched_music_loop": 0.68, "pitched_music_phrase": 0.60},
    )
    measured.evidence["shape_vote"].update(
        {
            "pitched_event_ratio": 0.88,
            "sustained_tonal_frame_ratio": 0.74,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
            "onset_count": 18.0,
        }
    )
    measured.evidence.update(
        {
            "brain_ensemble_vote_result": {
                "top_guesses": [
                    {
                        "folder_path": "FX/Textures/Noise and Static/Vinyl Noise/Long FX",
                        "label": "FX/Textures/Noise and Static/Vinyl Noise/Long FX",
                        "rank": 1,
                        "score": 0.86,
                    }
                ]
            },
            "physics_vote_result": {
                "top_guesses": [
                    {
                        "folder_path": "Instruments/Instrument Loops/Loops",
                        "label": "Instruments/Instrument Loops/Loops",
                        "rank": 1,
                        "score": 0.80,
                    }
                ]
            },
            "physics_subpanels": {
                "flat": {
                    "texture_bed_score": 0.43,
                    "texture_noise_static_score": 0.13,
                    "texture_water_ocean_score": 0.39,
                    "synth_tonal_source_score": 0.59,
                    "low_end_source_score": 0.54,
                }
            },
        }
    )
    core = DecisionCoreV2()

    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=[review], facts=measured)

    assert final.folder_path in {
        "FX/Textures/Noise and Static/Vinyl Noise/Long FX",
        "_TO_REVIEW/Measured Role Conflict",
    }
    assert not final.folder_path.startswith("Instruments/")


def test_clean_pitched_tail_reviews_instead_of_sticking_to_weak_tom_leaf() -> None:
    """Clean tonal tail hits need real drum-material proof before Tom can stick."""
    shared = [
        shared_row("Drums/Toms/Generic Tom/One Shots", 10.0),
        shared_row("FX/Designed Noise FX/Blip/One Shots", 11.0),
        shared_row("Instruments/Guitar/Acoustic Guitar/One Shots", 12.0),
    ]
    raw = raw_claim("Drums/Toms/Generic Tom/One Shots", shared, score=10.0)
    measured = facts("hit_with_tail", 0.79, {"pitched_music_phrase": 0.60})
    measured.evidence["shape_vote"].update(
        {
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 0.975,
            "non_event_tonal_ratio": 0.973,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.025,
            "onset_count": 2.0,
        }
    )
    measured.evidence.update(
        {
            "physics_subpanels": {
                "flat": {
                    "physics_subpanel_clean_tone": 0.65,
                    "drum_hit_score": 0.56,
                    "drum_tom_conga_source_score": 0.36,
                    "hand_drum_membrane_score": 0.71,
                    "pitched_mallet_instrument_score": 0.89,
                }
            },
        }
    )
    core = DecisionCoreV2()

    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=[], facts=measured)

    assert final.folder_path == "_TO_REVIEW/Measured Role Conflict"
    assert final.consensus_status == "final_clean_tonal_tail_drum_leaf_conflict_review"


def test_human_voice_fx_lane_blocks_weak_synth_leaf_when_spoken_evidence_is_high() -> None:
    """Human/Voice FX lane authority should beat a fragile Synth Lead steal."""
    shared = [
        shared_row("Instruments/Synths/Synth Lead/One Shots", 7.0),
        shared_row("Instruments/Woodwinds/Saxophone/One Shots", 11.0),
        shared_row("FX/Animals and Creatures/Dog/Long FX", 16.0),
    ]
    raw = raw_claim("Instruments/Synths/Synth Lead/One Shots", shared, score=7.0)
    measured = facts("pitched_repetition_phrase", 0.91, {"pitched_music_phrase": 0.62})
    measured.evidence["shape_vote"].update(
        {
            "pitched_event_ratio": 0.74,
            "sustained_tonal_frame_ratio": 0.58,
            "percussive_event_ratio": 0.04,
            "drumlike_frame_ratio": 0.04,
            "onset_count": 10.0,
        }
    )
    measured.evidence.update(
        {
            "human_spoken_voice_score": 0.75,
            "texture_room_crowd_ambience_score": 0.44,
            "voice_score": 0.30,
            "synth_tonal_source_score": 0.56,
            "synth_lead_score": 0.79,
            "brain_ensemble_vote_result": {
                "top_guesses": [
                    {
                        "folder_path": "FX/Human and Voice FX/Crowd/Long FX",
                        "label": "FX/Human and Voice FX/Crowd/Long FX",
                        "rank": 1,
                        "score": 1.23,
                        "confidence": 0.44,
                    },
                    {
                        "folder_path": "Instruments/Synths/Synth Lead/One Shots",
                        "label": "Instruments/Synths/Synth Lead/One Shots",
                        "rank": 3,
                        "score": 1.82,
                        "confidence": 0.29,
                    },
                ]
            },
        }
    )
    core = DecisionCoreV2()

    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=[], facts=measured)

    assert final.folder_path == "FX/Human and Voice FX/Crowd/Long FX"
    assert final.consensus_status == "final_human_voice_fx_lane_authority"


def test_human_voice_fx_lane_does_not_steal_measured_synth_loop_body() -> None:
    """Human/Voice FX authority must stand down when measured synth-loop proof is strong."""
    shared = [
        shared_row("Instruments/Synths/Synth Lead/One Shots", 5.0),
        shared_row("FX/Human and Voice FX/Crowd/One Shots", 6.0),
    ]
    raw = raw_claim("Instruments/Synths/Synth Lead/One Shots", shared, score=5.0)
    measured = facts("pitched_repetition_phrase", 0.97, {"pitched_music_loop": 0.91})
    measured.evidence["shape_vote"].update(
        {
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
            "f0_voiced_ratio": 0.89,
            "low_event_ratio": 0.02,
            "mid_event_ratio": 0.95,
            "high_event_ratio": 0.02,
            "spectral_flatness_mean": 0.17,
            "spectral_entropy_mean": 0.46,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
            "true_repetition_score": 0.85,
            "onset_count": 32.0,
        }
    )
    measured.evidence.update(
        {
            "human_spoken_voice_score": 0.83,
            "voice_score": 0.62,
            "synth_tonal_source_score": 0.58,
            "synth_lead_score": 0.80,
            "synth_pad_score": 0.72,
            "synth_chord_score": 0.63,
            "reed_wind_authority_score": 0.57,
            "physics_subpanels": {
                "flat": {
                    "human_spoken_voice_score": 0.83,
                    "voice_score": 0.62,
                    "synth_tonal_source_score": 0.58,
                    "synth_lead_score": 0.80,
                    "synth_pad_score": 0.72,
                    "synth_chord_score": 0.63,
                    "reed_wind_authority_score": 0.57,
                }
            },
            "brain_ensemble_vote_result": {
                "top_guesses": [
                    {
                        "folder_path": "Instruments/Synths/Synth Lead/One Shots",
                        "label": "Instruments/Synths/Synth Lead/One Shots",
                        "rank": 1,
                        "score": 0.44,
                    },
                    {
                        "folder_path": "FX/Human and Voice FX/Crowd/One Shots",
                        "label": "FX/Human and Voice FX/Crowd/One Shots",
                        "rank": 2,
                        "score": 0.62,
                    },
                ]
            },
        }
    )
    core = DecisionCoreV2()

    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=[], facts=measured)

    assert final.folder_path == "Instruments/Synths/Synth Loops"
    assert final.consensus_status == "final_measured_synth_loop_invariant"


def test_weak_transition_fx_leaf_releases_to_broad_instrument_loop() -> None:
    """Riser/build leaves need measured motion before beating pitched music-loop facts."""
    shared = [
        shared_row("FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX", 2.0),
        shared_row("Instruments/Instrument Loops/Loops", 8.0),
    ]
    raw = raw_claim("FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX", shared, score=2.0)
    measured = facts("hybrid_fx_motion", 0.71, {"pitched_music_loop": 0.86})
    measured.evidence["shape_vote"].update(
        {
            "pitched_event_ratio": 0.42,
            "sustained_tonal_frame_ratio": 0.88,
            "percussive_event_ratio": 0.26,
            "drumlike_frame_ratio": 0.0,
            "true_repetition_score": 0.95,
            "onset_count": 19.0,
        }
    )
    measured.evidence.update(
        {
            "physics_subpanels": {
                "flat": {
                    "fx_transition_authority_score": 0.21,
                    "fx_riser_build_score": 0.21,
                    "fx_motion_score": 0.17,
                    "synth_chord_score": 0.37,
                    "struck_keys_score": 0.40,
                }
            },
            "brain_ensemble_vote_result": {
                "top_guesses": [
                    {
                        "folder_path": "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX",
                        "label": "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX",
                        "rank": 1,
                        "score": 0.46,
                        "lanes": ["full", "core_baby", "spread_baby"],
                    }
                ]
            },
        }
    )
    core = DecisionCoreV2()

    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=[], facts=measured)

    assert final.folder_path == "Instruments/Instrument Loops/Loops"
    assert final.consensus_status == "final_fx_leaf_pitched_loop_broad_instrument_invariant"


def test_bright_high_band_loop_can_escape_fx_review_to_drum_loops() -> None:
    """High-band shaker/cymbal loop evidence should authorize broad Drum Loops."""
    shared = [
        shared_row("FX/Animals and Creatures/Cricket/Long FX", 2.0),
        shared_row("Drums/Hi Hats/Open Hat/One Shots", 9.0),
    ]
    raw = raw_claim("FX/Animals and Creatures/Cricket/Long FX", shared, score=2.0)
    measured = facts("pitched_repetition_phrase", 0.86, {"pitched_music_loop": 0.80})
    measured.evidence["shape_vote"].update(
        {
            "low_event_ratio": 0.0,
            "mid_event_ratio": 0.04,
            "high_event_ratio": 0.96,
            "true_repetition_score": 0.82,
            "onset_count": 24.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
        }
    )
    measured.evidence.update(
        {
            "physics_subpanels": {
                "flat": {
                    "drum_shaker_tambourine_source_score": 0.92,
                    "drum_cymbal_source_score": 0.88,
                    "drum_metallic_percussion_source_score": 0.76,
                    "drum_loop_source_score": 0.60,
                    "synth_tonal_source_score": 0.46,
                    "synth_lead_score": 0.46,
                    "synth_pad_score": 0.42,
                    "synth_chord_score": 0.37,
                }
            },
        }
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("drum_loop", "Drums/Drum Loops/Loops", 0.86),
        measured,
        brain_result=None,
        physics_result=VoterResult("physics", [guess("Drums/Hi Hats/Open Hat/One Shots", 2)]),
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Drums/Drum Loops/Loops"
    assert final.consensus_status == "measured_drum_loop_claim"


def test_synth_bass_one_shot_candidate_broadens_to_synth_loops_not_generic_instruments() -> None:
    """Synth one-shot candidates with measured loop body should keep Synth depth."""
    shared = [
        shared_row("Instruments/Bass/Synth Bass/One Shots", 8.0),
        shared_row("Instruments/Instrument Loops/Loops", 14.0),
    ]
    raw = raw_claim("Instruments/Bass/Synth Bass/One Shots", shared, score=8.0)
    measured = facts("pitched_repetition_phrase", 0.85, {"pitched_music_loop": 0.90, "synth_loop": 0.88})
    measured.evidence["shape_vote"].update(
        {
            "pitched_event_ratio": 1.0,
            "pitch_confidence": 0.70,
            "sustained_tonal_frame_ratio": 1.0,
            "non_event_tonal_ratio": 1.0,
            "low_event_ratio": 0.84,
            "mid_event_ratio": 0.11,
            "high_event_ratio": 0.045,
            "spectral_flatness_mean": 0.30,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
            "true_repetition_score": 0.81,
            "onset_count": 18.0,
        }
    )
    measured.evidence.update(
        {
            "synth_tonal_source_score": 0.60,
            "bass_synth_score": 0.53,
            "voice_score": 0.41,
            "struck_keys_score": 0.44,
            "drum_loop_source_score": 0.22,
        }
    )
    core = DecisionCoreV2()

    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=[], facts=measured)

    assert final.folder_path == "Instruments/Synths/Synth Loops"
    assert final.consensus_status == "final_measured_synth_loop_invariant"


def test_non_reed_one_shot_leaf_still_broadens_to_generic_instrument_loops() -> None:
    """Generic one-shot instrument leaves should stay broad unless a safe subfamily exists."""
    shared = [
        shared_row("Instruments/Guitar/Nylon Guitar/One Shots", 4.0),
        shared_row("Instruments/Instrument Loops/Loops", 12.0),
    ]
    raw = raw_claim("Instruments/Guitar/Nylon Guitar/One Shots", shared, score=4.0)
    measured = facts(
        "pitched_phrase",
        0.94,
        {"pitched_music_loop": 0.94, "pitched_music_phrase": 0.90},
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("pitched_music_loop", "Instruments/Instrument Loops/Loops", 0.94),
        measured,
        brain_result=VoterResult(voter_name="brain_full", guesses=[]),
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Instruments/Instrument Loops/Loops"


def test_weaker_reed_candidate_does_not_deepen_generic_instrument_loop() -> None:
    """A reed-like role alone must not over-narrow broad Instrument Loops."""
    shared = [
        shared_row("Instruments/Instrument Loops/Loops", 8.0),
        shared_row("Instruments/Woodwinds/Saxophone/One Shots", 15.0),
    ]
    brain_result = VoterResult(
        voter_name="brain_full",
        guesses=[
            guess("Instruments/Woodwinds/Saxophone/One Shots", 1),
            guess("Instruments/Instrument Loops/Loops", 3),
        ],
    )
    raw = raw_claim("Instruments/Instrument Loops/Loops", shared, score=8.0)
    measured = facts(
        "pitched_phrase",
        0.92,
        {"pitched_reed_or_instrument_loop": 0.92, "pitched_music_loop": 0.88},
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("pitched_reed_or_instrument_loop", "Instruments/Woodwinds/Saxophone/Loops", 0.92),
        measured,
        brain_result=brain_result,
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Instruments/Instrument Loops/Loops"


def test_bass_phrase_with_drum_loop_role_does_not_steal_drum_loop_to_bass() -> None:
    """A bass-shaped drum loop must stay in Drum Loops when the measured role is drum_loop."""
    shared = [
        shared_row("FX/Impacts and Hits/Generic Impact/Long FX", 3.0),
        shared_row("Drums/Drum Loops/Loops", 6.0),
    ]
    brain_result = VoterResult(
        voter_name="brain_full",
        guesses=[
            guess("FX/Impacts and Hits/Generic Impact/Long FX", 1),
            guess("Instruments/Bass/Synth Bass/One Shots", 2),
        ],
    )
    raw = raw_claim("FX/Impacts and Hits/Generic Impact/Long FX", shared)
    measured = facts("bass_phrase", 0.96, {"drum_loop": 0.78, "low_rhythmic_drum_loop": 0.74})
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("drum_loop", "Drums/Drum Loops/Loops", 0.78),
        measured,
        brain_result=brain_result,
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Drums/Drum Loops/Loops"
    assert "Bass" not in final.folder_path


def test_stable_pitched_phrase_blocks_vocal_shape_without_voice_candidate() -> None:
    """A vocal-shaped pitched phrase with no real voice candidate should not become Human/Voice."""
    shared = [
        shared_row("FX/Designed Noise FX/Alarm/Long FX", 3.0),
        shared_row("Instruments/Instrument Loops/Loops", 18.0),
    ]
    raw = raw_claim("FX/Designed Noise FX/Alarm/Long FX", shared)
    measured = facts(
        "vocal_phrase",
        1.0,
        {"pitched_music_loop": 0.98, "pitched_music_phrase": 0.85, "vocal_music_phrase": 1.0},
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("vocal_music_phrase", "FX/Human and Voice FX", 1.0),
        measured,
        brain_result=VoterResult(voter_name="brain_full", guesses=[]),
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert "Human and Voice" not in final.folder_path


def test_stronger_voice_candidate_can_deepen_generic_instrument_loop() -> None:
    """A real stronger voice candidate may beat generic Instrument Loops."""
    shared = [
        shared_row("Instruments/Instrument Loops/Loops", 25.0),
        shared_row("FX/Human and Voice FX/Spoken Voice/Long FX", 12.0, {"vocal_music_phrase": 0.88}),
    ]
    raw = raw_claim("Instruments/Instrument Loops/Loops", shared, score=25.0)
    measured = facts("vocal_phrase", 0.94, {"vocal_music_phrase": 0.91, "pitched_music_loop": 0.70})
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        EligibilityDecision(
            role_name="vocal_music_phrase",
            confidence=0.91,
            allowed_top_families=("FX", "Instruments", "_TO_REVIEW"),
            blocked_path_fragments=(),
            broad_folder_path="FX/Human and Voice FX",
            reason="synthetic vocal phrase eligibility",
        ),
        measured,
        brain_result=VoterResult(voter_name="brain_full", guesses=[]),
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Instruments/Voice/Phrase/One Shots"
    assert final.consensus_status == "human_voice_true_bucket_rescue"


def test_voice_candidate_does_not_steal_generic_pitched_instrument_loop() -> None:
    """Voice depth needs measured vocal eligibility, not just a vocal-looking row."""
    shared = [
        shared_row("Instruments/Instrument Loops/Loops", 8.0),
        shared_row("FX/Human and Voice FX/Spoken Voice/Long FX", 12.0, {"vocal_music_phrase": 0.88}),
    ]
    raw = raw_claim("Instruments/Instrument Loops/Loops", shared, score=8.0)
    measured = facts("pitched_phrase", 0.94, {"vocal_music_phrase": 0.20, "pitched_music_loop": 0.91})
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        EligibilityDecision(
            role_name="pitched_music_loop",
            confidence=0.91,
            allowed_top_families=("Instruments", "_TO_REVIEW"),
            blocked_path_fragments=(),
            broad_folder_path="Instruments/Instrument Loops/Loops",
            reason="synthetic pitched phrase eligibility",
        ),
        measured,
        brain_result=VoterResult(voter_name="brain_full", guesses=[]),
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Instruments/Instrument Loops/Loops"


def test_measured_percussive_hit_can_escape_fx_glitch_leaf() -> None:
    """A measured percussion one-shot may broaden an FX glitch false positive."""
    shared = [
        shared_row("FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/One Shots", 4.0),
        shared_row("Drums/Rims and Sticks/Rimshot/One Shots", 8.0, {"percussive_one_shot": 0.98}),
    ]
    raw = raw_claim(
        "FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/One Shots",
        shared,
        score=4.0,
    )
    measured = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "duration_sec": 0.25,
            "event_count_estimate": 1.0,
            "shape_vote": {"primary_shape": "single_hit", "confidence": 0.83, "onset_count": 1.0},
            "measured_roles": {"percussive_one_shot": 0.97, "primary_roles": ["percussive_one_shot"]},
        },
        feature_values_by_name={"duration_sec": 0.25, "event_count_estimate": 1.0},
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        EligibilityDecision(
            role_name="protected_percussive_one_shot",
            confidence=0.97,
            allowed_top_families=("Drums", "_TO_REVIEW"),
            blocked_path_fragments=("FX", "Glitch", "Stutter"),
            broad_folder_path="Drums/Percussion/Generic Percussion/One Shots",
            reason="synthetic measured percussion one-shot",
        ),
        measured,
        brain_result=VoterResult(voter_name="brain_full", guesses=[]),
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Drums/Percussion/Generic Percussion/One Shots"


def test_measured_percussive_hit_does_not_steal_specific_bass_one_shot() -> None:
    """The percussion broad bucket must not swallow a specific instrument hit."""
    shared = [
        shared_row("Instruments/Bass/808 Bass/One Shots", 3.0),
        shared_row("Drums/Percussion/Generic Percussion/One Shots", 9.0, {"percussive_one_shot": 0.98}),
    ]
    raw = raw_claim("Instruments/Bass/808 Bass/One Shots", shared, score=3.0)
    measured = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "duration_sec": 0.75,
            "event_count_estimate": 1.0,
            "shape_vote": {"primary_shape": "single_hit", "confidence": 0.82, "onset_count": 1.0},
            "measured_roles": {"percussive_one_shot": 0.90, "primary_roles": ["percussive_one_shot"]},
        },
        feature_values_by_name={"duration_sec": 0.75, "event_count_estimate": 1.0},
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        EligibilityDecision(
            role_name="protected_percussive_one_shot",
            confidence=0.90,
            allowed_top_families=("Drums", "_TO_REVIEW"),
            blocked_path_fragments=("Instruments", "Bass"),
            broad_folder_path="Drums/Percussion/Generic Percussion/One Shots",
            reason="synthetic measured percussion one-shot",
        ),
        measured,
        brain_result=VoterResult(voter_name="brain_full", guesses=[]),
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Instruments/Bass/808 Bass/One Shots"


def test_decisive_struck_membrane_one_shot_stays_under_drums_after_sax_invariant() -> None:
    """Compact struck membrane evidence must beat a late false sax loop invariant."""
    raw = raw_claim(
        "Instruments/Woodwinds/Saxophone/Loops",
        [shared_row("Instruments/Woodwinds/Saxophone/Loops", 3.0)],
        score=3.0,
    )
    measured = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "duration_sec": 0.42,
            "event_count_estimate": 1.0,
            "shape_vote": {
                "primary_shape": "solo_phrase",
                "confidence": 0.78,
                "onset_count": 1.0,
                "true_repetition_score": 0.12,
            },
            "measured_roles": {"percussive_one_shot": 0.88, "primary_roles": ["percussive_one_shot"]},
            "physics_subpanels": {
                "flat": {
                    "role_one_shot_score": 0.96,
                    "compact_struck_tonal_percussion_score": 0.86,
                    "hand_drum_membrane_score": 0.91,
                    "struck_wood_score": 0.86,
                    "pitched_metal_percussion_score": 0.54,
                    "drum_tom_conga_source_score": 0.84,
                    "drum_hit_score": 0.50,
                    "onset_percussive_onset_score": 0.66,
                    "voice_score": 0.10,
                }
            },
        },
        feature_values_by_name={"duration_sec": 0.42, "event_count_estimate": 1.0},
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        EligibilityDecision(
            role_name="protected_percussive_one_shot",
            confidence=0.90,
            allowed_top_families=("Drums", "_TO_REVIEW"),
            blocked_path_fragments=("Instruments", "FX"),
            broad_folder_path="Drums/Percussion/Generic Percussion/One Shots",
            reason="synthetic measured percussion one-shot",
        ),
        measured,
        brain_result=VoterResult(voter_name="brain_full", guesses=[]),
        physics_result=None,
    )
    assert any(claim.source == "final_decisive_struck_percussion_parent_invariant" for claim in claims)
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Drums/Rims and Sticks/Generic Rim or Stick/One Shots"
    assert final.consensus_status == "final_decisive_struck_percussion_parent_invariant"


def test_decisive_struck_metal_scrape_one_shot_stays_under_drums_after_fx_invariant() -> None:
    """Measured metal/scrape percussion should not finish as animal or voice-like FX."""
    raw = raw_claim(
        "FX/Human and Voice FX",
        [shared_row("FX/Human and Voice FX", 3.0)],
        score=3.0,
    )
    measured = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "duration_sec": 0.55,
            "event_count_estimate": 2.0,
            "shape_vote": {
                "primary_shape": "texture_bed",
                "confidence": 0.63,
                "onset_count": 2.0,
                "true_repetition_score": 0.22,
            },
            "measured_roles": {"percussive_one_shot": 0.80, "primary_roles": ["percussive_one_shot"]},
            "physics_subpanels": {
                "flat": {
                    "role_one_shot_score": 0.70,
                    "compact_struck_tonal_percussion_score": 0.64,
                    "pitched_metal_percussion_score": 0.73,
                    "drum_cymbal_source_score": 0.74,
                    "drum_guiro_scrape_source_score": 0.82,
                    "drum_metallic_percussion_source_score": 0.76,
                    "drum_hit_score": 0.70,
                    "onset_percussive_onset_score": 0.65,
                    "voice_score": 0.18,
                    "human_spoken_voice_score": 0.10,
                    "human_breath_mouth_score": 0.10,
                }
            },
        },
        feature_values_by_name={"duration_sec": 0.55, "event_count_estimate": 2.0},
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        EligibilityDecision(
            role_name="protected_percussive_one_shot",
            confidence=0.90,
            allowed_top_families=("Drums", "_TO_REVIEW"),
            blocked_path_fragments=("Instruments", "FX"),
            broad_folder_path="Drums/Percussion/Generic Percussion/One Shots",
            reason="synthetic measured percussion one-shot",
        ),
        measured,
        brain_result=VoterResult(voter_name="brain_full", guesses=[]),
        physics_result=None,
    )
    assert any(claim.source == "final_decisive_struck_percussion_parent_invariant" for claim in claims)
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Drums/Percussion/Guiros Scrapes and Rasps/One Shots"
    assert final.consensus_status == "final_decisive_struck_percussion_parent_invariant"


def test_mixed_percussive_animal_fx_claim_reviews_instead_of_confident_dog() -> None:
    """Moderate percussion-vs-animal conflicts should review, not claim Dog."""
    raw = raw_claim(
        "FX/Animals and Creatures/Dog/One Shots",
        [shared_row("FX/Animals and Creatures/Dog/One Shots", 3.0)],
        score=3.0,
    )
    measured = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "duration_sec": 0.48,
            "event_count_estimate": 2.0,
            "shape_vote": {
                "primary_shape": "solo_phrase",
                "confidence": 0.66,
                "onset_count": 2.0,
                "true_repetition_score": 0.18,
            },
            "measured_roles": {"percussive_one_shot": 0.66, "primary_roles": ["percussive_one_shot"]},
            "physics_subpanels": {
                "flat": {
                    "role_one_shot_score": 0.67,
                    "hand_drum_membrane_score": 0.75,
                    "compact_struck_tonal_percussion_score": 0.54,
                    "onset_percussive_onset_score": 0.55,
                    "animal_dog_score": 0.65,
                    "animal_voice_score": 0.64,
                    "voice_score": 0.36,
                }
            },
        },
        feature_values_by_name={"duration_sec": 0.48, "event_count_estimate": 2.0},
    )
    core = DecisionCoreV2()

    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=[], facts=measured)

    assert final.folder_path == "_TO_REVIEW/Measured Role Conflict"
    assert final.consensus_status == "percussive_voice_or_animal_fx_conflict_review"
