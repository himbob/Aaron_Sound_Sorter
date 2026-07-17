"""Source-blind claim boundary contract tests."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.claim_boundary_policy import ClaimBoundaryPolicy
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path


def _claim(
    path: str,
    *,
    source: str = "final_measured_branch_loop_broad_bucket",
    strength: float = 0.94,
    shared: list[dict] | None = None,
):
    return claim_from_folder_path(
        folder_path=path,
        source=source,
        reason="test claim",
        shared=shared or [],
        raw_candidate_score=5.0,
        brain_rank=2,
        physics_rank=3,
        shared_winner=path,
        can_override=True,
        strength=strength,
        is_real_candidate=False,
    )


def _facts(
    shape: str,
    confidence: float,
    shape_values: dict[str, float],
    panel_values: dict[str, float],
    measured_roles: dict[str, float] | None = None,
):
    evidence = {
        "shape_vote": {"primary_shape": shape, "confidence": confidence, **shape_values},
        "physics_subpanels": {"flat": panel_values},
    }
    if measured_roles is not None:
        evidence["measured_roles"] = measured_roles
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence=evidence,
    )


def test_drum_loop_claim_stands_down_for_clean_bass_loop() -> None:
    raw = _claim("Instruments/Bass/Synth Bass/Loops", source="strong_consensus")
    claim = _claim("Drums/Drum Loops/Loops", source="measured_drum_loop_claim")
    facts = _facts(
        "bass_phrase",
        0.96,
        {
            "low_event_ratio": 0.94,
            "pitched_event_ratio": 0.94,
            "sustained_tonal_frame_ratio": 0.94,
            "percussive_event_ratio": 0.02,
            "drumlike_frame_ratio": 0.02,
        },
        {
            "bass_synth_score": 0.78,
            "bass_sub_score": 0.74,
            "low_end_source_score": 0.70,
            "drum_loop_source_score": 0.18,
            "rhythmic_break_loop_score": 0.22,
        },
    )

    decision = ClaimBoundaryPolicy().evaluate(raw_claim=raw, claim=claim, facts=facts)

    assert not decision.allowed
    assert decision.reason == "drum_loop_claim_conflicts_with_clean_bass_loop"


def test_drum_loop_claim_stands_down_for_clean_rhythmic_bass_phrase() -> None:
    raw = _claim("Instruments/Bass/808 Bass/Loops", source="strong_consensus")
    claim = _claim("Drums/Drum Loops/Loops", source="placement_depth_broad_bucket")
    facts = _facts(
        "bass_phrase",
        0.98,
        {
            "low_event_ratio": 0.99,
            "pitched_event_ratio": 1.0,
            "pitch_confidence": 0.92,
            "non_event_tonal_ratio": 0.63,
            "sustained_tonal_frame_ratio": 0.67,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        },
        {
            "bass_synth_score": 0.78,
            "bass_sub_score": 0.67,
            "low_end_source_score": 0.64,
            "drum_loop_source_score": 0.72,
            "rhythmic_break_loop_score": 0.38,
            "drum_hit_score": 0.19,
            "drum_kick_source_score": 0.42,
            "drum_tom_conga_source_score": 0.32,
            "drum_snare_source_score": 0.18,
        },
    )

    decision = ClaimBoundaryPolicy().evaluate(raw_claim=raw, claim=claim, facts=facts)

    assert not decision.allowed
    assert decision.reason == "drum_loop_claim_conflicts_with_clean_bass_loop"


def test_drum_loop_claim_stands_down_for_clean_synth_loop() -> None:
    raw = _claim("Instruments/Synths/Pads/Loops", source="strong_consensus")
    claim = _claim("Drums/Drum Loops/Loops", source="measured_drum_loop_claim")
    facts = _facts(
        "beat_loop",
        0.92,
        {
            "pitched_event_ratio": 0.98,
            "sustained_tonal_frame_ratio": 0.84,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        },
        {
            "synth_tonal_source_score": 0.64,
            "synth_pad_score": 0.61,
            "drum_hit_score": 0.10,
            "hand_drum_membrane_score": 0.12,
            "rhythmic_break_loop_score": 0.20,
        },
    )

    decision = ClaimBoundaryPolicy().evaluate(raw_claim=raw, claim=claim, facts=facts)

    assert not decision.allowed
    assert decision.reason == "drum_loop_claim_conflicts_with_clean_synth_loop"


def test_woodwind_leaf_claim_stands_down_for_bright_percussive_loop() -> None:
    raw = _claim("Drums/Drum Loops/Loops", source="strong_consensus")
    claim = _claim("Instruments/Woodwinds/Saxophone/Loops", source="final_measured_sax_loop_invariant")
    facts = _facts(
        "pitched_repetition_phrase",
        0.86,
        {
            "high_event_ratio": 0.91,
            "low_event_ratio": 0.02,
            "onset_count": 24.0,
        },
        {
            "drum_shaker_tambourine_source_score": 0.82,
            "drum_cymbal_source_score": 0.77,
            "reed_wind_authority_score": 0.35,
            "woodwind_sax_score": 0.48,
        },
    )

    decision = ClaimBoundaryPolicy().evaluate(raw_claim=raw, claim=claim, facts=facts)

    assert not decision.allowed
    assert decision.reason == "woodwind_leaf_conflicts_with_bright_percussive_loop"


def test_woodwind_subtype_claim_stands_down_when_synth_voice_pressure_is_stronger() -> None:
    raw = _claim("Instruments/Synths/Synth Lead/One Shots", source="strong_consensus")
    claim = _claim("Instruments/Woodwinds/Clarinet/Loops", source="final_measured_branch_loop_broad_bucket")
    facts = _facts(
        "pitched_repetition_phrase",
        0.97,
        {
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
        },
        {
            "woodwind_sax_score": 0.70,
            "woodwind_flute_score": 0.58,
            "reed_wind_score": 0.63,
            "reed_wind_authority_score": 0.52,
            "synth_lead_score": 0.82,
            "synth_pad_score": 0.75,
            "human_spoken_voice_score": 0.80,
        },
    )

    decision = ClaimBoundaryPolicy().evaluate(raw_claim=raw, claim=claim, facts=facts)

    assert not decision.allowed
    assert decision.reason == "woodwind_leaf_lacks_source_margin"


def test_broad_wind_loop_claim_stands_down_for_crowded_pitched_instrument_loop() -> None:
    raw = _claim("Instruments/Guitar/Electric Guitar/One Shots", source="strong_consensus")
    claim = _claim("Instruments/Brass and Woodwinds/Loops", source="final_measured_branch_loop_broad_bucket")
    facts = _facts(
        "pitched_repetition_phrase",
        0.89,
        {
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        },
        {
            "woodwind_sax_score": 0.68,
            "woodwind_flute_score": 0.55,
            "reed_wind_score": 0.62,
            "reed_wind_authority_score": 0.42,
            "voice_score": 0.77,
            "plucked_string_score": 0.62,
            "bowed_string_score": 0.57,
            "string_violin_score": 0.59,
            "pitched_mallet_instrument_score": 0.71,
        },
    )

    decision = ClaimBoundaryPolicy().evaluate(raw_claim=raw, claim=claim, facts=facts)

    assert not decision.allowed
    assert decision.reason == "broad_wind_loop_lacks_source_margin"


def test_exact_sax_loop_claim_is_not_blocked_by_broad_wind_stand_down_rule() -> None:
    raw = _claim("Instruments/Instrument Loops/Loops", source="strong_consensus")
    claim = _claim("Instruments/Woodwinds/Saxophone/Loops", source="final_measured_sax_loop_invariant")
    facts = _facts(
        "pitched_repetition_phrase",
        0.89,
        {
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
        },
        {
            "woodwind_sax_score": 0.68,
            "reed_wind_score": 0.62,
            "reed_wind_authority_score": 0.42,
            "voice_score": 0.77,
            "plucked_string_score": 0.62,
            "bowed_string_score": 0.57,
            "pitched_mallet_instrument_score": 0.71,
        },
    )

    decision = ClaimBoundaryPolicy().evaluate(raw_claim=raw, claim=claim, facts=facts)

    assert decision.allowed


def test_transition_fx_claim_stands_down_for_stable_music_loop_without_motion() -> None:
    raw = _claim("Instruments/Synths/Synth Loops", source="strong_consensus")
    claim = _claim(
        "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX",
        source="final_measured_transition_fx_invariant",
    )
    facts = _facts(
        "pitched_repetition_phrase",
        0.91,
        {
            "pitched_event_ratio": 0.93,
            "sustained_tonal_frame_ratio": 0.88,
        },
        {
            "fx_transition_authority_score": 0.21,
            "fx_motion_score": 0.18,
            "fx_riser_build_score": 0.24,
        },
    )

    decision = ClaimBoundaryPolicy().evaluate(raw_claim=raw, claim=claim, facts=facts)

    assert not decision.allowed
    assert decision.reason == "transition_fx_lacks_motion_authority"


def test_same_family_transition_fx_refinement_is_not_blocked() -> None:
    raw = _claim(
        "FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX",
        source="strong_consensus",
    )
    claim = _claim(
        "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX",
        source="final_measured_transition_fx_invariant",
    )
    facts = _facts(
        "transition_drop",
        0.80,
        {
            "pitched_event_ratio": 0.93,
            "sustained_tonal_frame_ratio": 0.85,
        },
        {
            "fx_transition_authority_score": 0.20,
            "fx_motion_score": 0.18,
        },
    )

    decision = ClaimBoundaryPolicy().evaluate(raw_claim=raw, claim=claim, facts=facts)

    assert decision.allowed


def test_human_voice_fx_claim_stands_down_for_clean_instrument_loop() -> None:
    raw = _claim("Instruments/Synths/Synth Lead/One Shots", source="strong_consensus")
    claim = _claim("FX/Human and Voice FX/Crowd/One Shots", source="final_human_voice_fx_lane_authority")
    facts = _facts(
        "pitched_repetition_phrase",
        0.96,
        {
            "pitched_event_ratio": 0.98,
            "sustained_tonal_frame_ratio": 0.96,
            "percussive_event_ratio": 0.01,
            "drumlike_frame_ratio": 0.0,
        },
        {
            "synth_tonal_source_score": 0.64,
            "voice_score": 0.62,
            "human_spoken_voice_score": 0.80,
        },
    )

    decision = ClaimBoundaryPolicy().evaluate(raw_claim=raw, claim=claim, facts=facts)

    assert not decision.allowed
    assert decision.reason == "human_voice_fx_conflicts_with_clean_instrument_loop"


def test_human_voice_fx_claim_stands_down_for_true_vocal_music_phrase() -> None:
    raw = _claim("FX/Human and Voice FX/Spoken Voice/Long FX", source="strong_consensus")
    claim = _claim("FX/Human and Voice FX/Spoken Voice/Long FX", source="strong_consensus")
    facts = _facts(
        "designed_low_fx",
        0.88,
        {
            "pitched_event_ratio": 0.94,
            "f0_voiced_ratio": 0.86,
            "sustained_tonal_frame_ratio": 0.90,
            "percussive_event_ratio": 0.06,
            "drumlike_frame_ratio": 0.06,
        },
        {
            "voice_score": 0.80,
            "human_spoken_voice_score": 0.90,
            "fx_formant_score": 0.80,
            "fx_motion_score": 0.20,
            "fx_transition_authority_score": 0.22,
            "fx_riser_build_score": 0.18,
        },
        {"vocal_music_phrase": 0.93},
    )

    decision = ClaimBoundaryPolicy().evaluate(raw_claim=raw, claim=claim, facts=facts)

    assert not decision.allowed
    assert decision.reason == "human_voice_fx_conflicts_with_true_voice_instrument"


def test_broad_instrument_loop_claim_is_not_treated_as_source_identity() -> None:
    raw = _claim("FX/Textures/Noise and Static/Hiss/Long FX", source="strong_consensus")
    claim = _claim("Instruments/Instrument Loops/Loops", source="parent_eligibility_broad_bucket")
    facts = _facts(
        "pitched_repetition_phrase",
        0.88,
        {
            "pitched_event_ratio": 0.90,
            "sustained_tonal_frame_ratio": 0.82,
            "percussive_event_ratio": 0.05,
            "drumlike_frame_ratio": 0.02,
        },
        {"synth_tonal_source_score": 0.62},
    )

    decision = ClaimBoundaryPolicy().evaluate(raw_claim=raw, claim=claim, facts=facts)

    assert decision.allowed


def test_raw_woodwind_winner_stands_down_for_clean_synth_pad_loop() -> None:
    raw = _claim("Instruments/Woodwinds/Saxophone/One Shots", source="strong_consensus", strength=0.97)
    facts = _facts(
        "pitched_repetition_phrase",
        0.93,
        {
            "duration_sec": 7.0,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 0.82,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        },
        {
            "synth_tonal_source_score": 0.62,
            "synth_pad_score": 0.66,
            "woodwind_sax_score": 0.51,
            "reed_wind_authority_score": 0.20,
        },
    )

    winner = FamilyClaimArbiter().pick_winner(raw_claim=raw, claims=[], facts=facts)

    assert winner.folder_path == "Instruments/Synths/Pads/Loops"
    assert winner.source == "raw_contract_clean_synth_loop_over_woodwind_leaf"


def test_raw_drum_loop_winner_stands_down_for_clean_synth_loop() -> None:
    raw = _claim("Drums/Drum Loops/Loops", source="strong_consensus", strength=0.96)
    facts = _facts(
        "beat_loop",
        0.92,
        {
            "duration_sec": 7.0,
            "pitched_event_ratio": 0.97,
            "sustained_tonal_frame_ratio": 0.80,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        },
        {
            "synth_tonal_source_score": 0.63,
            "synth_chord_score": 0.61,
            "reed_wind_authority_score": 0.20,
            "drum_loop_source_score": 0.70,
        },
    )

    winner = FamilyClaimArbiter().pick_winner(raw_claim=raw, claims=[], facts=facts)

    assert winner.folder_path == "Instruments/Synths/Synth Loops"
    assert winner.source == "raw_contract_clean_synth_loop_over_drum_loop"


def test_raw_drum_leaf_winner_stands_down_for_clean_tonal_synth_hit() -> None:
    raw = _claim("Drums/Rims and Sticks/Rimshot/One Shots", source="strong_consensus", strength=0.94)
    facts = _facts(
        "single_hit",
        0.84,
        {
            "duration_sec": 0.82,
            "onset_count": 1.0,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 0.92,
            "non_event_tonal_ratio": 0.96,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        },
        {
            "synth_tonal_source_score": 0.61,
            "synth_chord_score": 0.58,
            "drum_hit_score": 0.34,
            "drum_rim_stick_source_score": 0.42,
        },
    )

    winner = FamilyClaimArbiter().pick_winner(raw_claim=raw, claims=[], facts=facts)

    assert winner.folder_path == "Instruments/Synths/Synth One Shots"
    assert winner.source == "raw_contract_clean_synth_hit_over_drum_leaf"


def test_raw_hat_one_shot_winner_broadens_to_drum_loop_when_repetition_is_measured() -> None:
    raw = _claim("Drums/Hi Hats/Open Hat/One Shots", source="strong_consensus", strength=0.95)
    facts = _facts(
        "pitched_repetition_phrase",
        0.86,
        {
            "duration_sec": 2.0,
            "onset_count": 18.0,
            "high_event_ratio": 0.94,
            "true_repetition_score": 0.88,
        },
        {
            "drum_shaker_tambourine_source_score": 0.91,
            "drum_cymbal_source_score": 0.86,
            "synth_tonal_source_score": 0.30,
            "reed_wind_authority_score": 0.20,
        },
    )

    winner = FamilyClaimArbiter().pick_winner(raw_claim=raw, claims=[], facts=facts)

    assert winner.folder_path == "Drums/Drum Loops/Loops"
    assert winner.source == "raw_contract_high_percussion_loop_broadening"


def test_raw_transition_fx_winner_stands_down_for_stable_music_loop_without_motion() -> None:
    raw = _claim(
        "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX",
        source="strong_consensus",
        strength=0.95,
    )
    facts = _facts(
        "hybrid_fx_motion",
        0.72,
        {
            "pitched_event_ratio": 0.95,
            "sustained_tonal_frame_ratio": 0.86,
            "true_repetition_score": 0.90,
            "percussive_event_ratio": 0.02,
            "drumlike_frame_ratio": 0.02,
        },
        {
            "fx_transition_authority_score": 0.21,
            "fx_motion_score": 0.18,
            "fx_riser_build_score": 0.22,
            "synth_tonal_source_score": 0.61,
            "synth_chord_score": 0.64,
            "reed_wind_authority_score": 0.20,
        },
    )

    winner = FamilyClaimArbiter().pick_winner(raw_claim=raw, claims=[], facts=facts)

    assert winner.folder_path == "Instruments/Synths/Synth Loops"
    assert winner.source == "raw_contract_stable_music_loop_over_transition_fx"


def test_measured_voice_claim_can_beat_fx_human_voice_when_role_is_true_vocal() -> None:
    raw = _claim("FX/Human and Voice FX/Spoken Voice/Long FX", source="strong_consensus", strength=0.88)
    claim = _claim(
        "Instruments/Voice/Vocal Loops/Loops",
        source="final_measured_voice_invariant",
        strength=0.98,
    )
    facts = _facts(
        "designed_low_fx",
        0.88,
        {
            "pitched_event_ratio": 0.94,
            "f0_voiced_ratio": 0.86,
            "sustained_tonal_frame_ratio": 0.90,
            "percussive_event_ratio": 0.06,
            "drumlike_frame_ratio": 0.06,
        },
        {
            "voice_score": 0.80,
            "human_spoken_voice_score": 0.90,
            "fx_transition_authority_score": 0.26,
            "fx_motion_score": 0.20,
        },
        {"vocal_music_phrase": 0.92},
    )

    winner = FamilyClaimArbiter().pick_winner(raw_claim=raw, claims=[claim], facts=facts)

    assert winner.folder_path == "Instruments/Voice/Vocal Loops/Loops"
    assert winner.source == "final_measured_voice_invariant"


def test_measured_sax_claim_beats_broad_brass_woodwinds_parent() -> None:
    shared_sax = [
        {
            "folder_path": "Instruments/Woodwinds/Saxophone/Loops",
            "combined_rank_score": 19.0,
            "brain_rank": 7,
            "physics_rank": 12,
        },
    ]
    raw = _claim("Instruments/Mallets and Bells/Vibraphone/Loops", source="strong_consensus", strength=0.88)
    broad = _claim(
        "Instruments/Brass and Woodwinds/Loops",
        source="final_measured_branch_loop_broad_bucket",
        strength=0.97,
        shared=shared_sax,
    )
    sax = _claim(
        "Instruments/Woodwinds/Saxophone/Loops",
        source="final_measured_sax_loop_invariant",
        strength=0.94,
        shared=shared_sax,
    )
    facts = _facts(
        "pitched_phrase",
        1.0,
        {
            "pitched_event_ratio": 1.0,
            "f0_voiced_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
            "non_event_tonal_ratio": 1.0,
            "mid_event_ratio": 0.71,
            "high_event_ratio": 0.13,
            "low_event_ratio": 0.16,
            "spectral_flatness_mean": 0.25,
            "spectral_entropy_mean": 0.40,
            "onset_count": 9.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        },
        {
            "woodwind_sax_score": 0.71,
            "reed_wind_score": 0.62,
            "reed_wind_authority_score": 0.47,
            "voice_score": 0.61,
            "synth_tonal_source_score": 0.61,
            "plucked_string_score": 0.50,
        },
    )

    winner = FamilyClaimArbiter().pick_winner(raw_claim=raw, claims=[broad, sax], facts=facts)

    assert winner.folder_path == "Instruments/Woodwinds/Saxophone/Loops"
    assert winner.source == "final_measured_sax_loop_invariant"


def test_profile_synth_one_shot_claim_deepens_to_loop_when_shape_is_loop() -> None:
    raw = _claim("Instruments/Instrument Loops/Loops", source="strong_consensus", strength=0.12)
    claim = _claim(
        "Instruments/Synths/Synth Lead/One Shots",
        source="profile_candidate_synth_claim",
        strength=0.89,
    )
    facts = _facts(
        "pitched_repetition_phrase",
        0.95,
        {
            "duration_sec": 12.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
            "sustained_tonal_frame_ratio": 0.84,
        },
        {
            "synth_tonal_source_score": 0.57,
            "synth_pad_score": 0.65,
            "synth_chord_score": 0.62,
            "synth_lead_score": 0.51,
        },
    )

    winner = FamilyClaimArbiter().pick_winner(raw_claim=raw, claims=[claim], facts=facts)

    assert winner.folder_path == "Instruments/Synths/Pads/Loops"
    assert winner.source == "profile_synth_loop_depth_contract"
