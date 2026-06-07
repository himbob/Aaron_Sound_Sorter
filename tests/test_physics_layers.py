from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.voters.physics_layer_types import PhysicsLayerDecision
from aaron_sound_sorter.voters.physics_layers import LayeredPhysicsScorer


def facts(
    shape: dict,
    roles: dict | None = None,
    values: dict | None = None,
    *,
    is_loop_like: bool = True,
    is_single_event_like: bool = False,
    is_short_hit_like: bool = False,
    is_long: bool = True,
) -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=is_loop_like,
        is_single_event_like=is_single_event_like,
        is_short_hit_like=is_short_hit_like,
        is_long=is_long,
        evidence={
            "shape_vote": shape,
            "measured_roles": roles or {},
        },
        feature_values_by_name=values or {},
    )


def test_layered_physics_promotes_clean_bass_branch() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "bass_phrase",
            "confidence": 0.98,
            "low_event_ratio": 0.99,
            "pitch_confidence": 0.92,
            "pitched_event_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        },
        roles={"bass_loop": 0.58, "pitched_music_loop": 0.80},
        values={
            "f0_median_hz": 82.0,
            "low_peak_frequency_hz": 72.0,
            "sub_bass_ratio_lt_150hz": 0.54,
            "bass_ratio_150_500hz": 0.28,
            "mid_ratio_500_2000hz": 0.12,
            "presence_ratio_2000_8000hz": 0.03,
            "loop_mean_event_low_ratio": 0.99,
            "loop_pitched_event_ratio": 1.0,
            "loop_percussive_event_ratio": 0.0,
            "loop_drumlike_frame_ratio": 0.0,
        },
    )

    decision = scorer.analyze(sample_facts)
    bass_score, bass_evidence = scorer.apply("Instruments/Bass/Electric Bass/One Shots", 2.25, decision)
    fx_score, fx_evidence = scorer.apply(
        "FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/Long FX", 0.56, decision
    )

    assert decision.top_family == "Instruments"
    assert decision.branch == "Bass"
    assert bass_score < 0.25
    assert fx_score > 1.0
    assert "bass_branch_target:0.160" in bass_evidence["physics_layer_adjustment_reasons"]
    assert "bass_branch_blocks_non_instrument:+0.55" in fx_evidence["physics_layer_adjustment_reasons"]


def test_layered_physics_blocks_dense_vocal_phrase_from_sax_leaf() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "vocal_phrase",
            "confidence": 0.95,
            "onset_count": 82.0,
            "spectral_flatness_mean": 0.43,
            "mid_event_ratio": 0.58,
            "high_event_ratio": 0.19,
            "pitch_confidence": 0.67,
            "pitched_event_ratio": 0.94,
        },
        roles={"vocal_music_phrase": 0.92, "pitched_music_loop": 0.86},
        values={"spectral_flatness_mean": 0.43},
    )

    decision = scorer.analyze(sample_facts)
    sax_score, evidence = scorer.apply("Instruments/Woodwinds/Saxophone/One Shots", 0.18, decision)

    assert decision.top_family == "Instruments"
    assert decision.branch == "Voice"
    assert sax_score > 1.0
    assert "dense_voice_branch_blocks_sax:+0.95" in evidence["physics_layer_adjustment_reasons"]


def test_layered_physics_detects_articulated_rap_voice_texture() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "bass_phrase",
            "confidence": 0.76,
            "onset_count": 33.0,
            "onset_span_ratio": 0.92,
            "pitch_confidence": 0.13,
            "pitched_event_ratio": 0.94,
            "percussive_event_ratio": 0.06,
            "drumlike_frame_ratio": 0.06,
            "low_event_ratio": 0.64,
            "mid_event_ratio": 0.23,
            "high_event_ratio": 0.13,
            "f0_voiced_ratio": 0.88,
            "spectral_flatness_mean": 0.37,
            "spectral_entropy_mean": 0.43,
            "tail_ratio": 0.70,
        },
        roles={"pitched_music_loop": 0.80},
        values={
            "f0_voiced_ratio": 0.88,
            "loop_pitched_event_ratio": 0.94,
            "loop_sustained_tonal_frame_ratio": 0.94,
            "loop_non_event_tonal_ratio": 0.96,
            "spectral_flatness_mean": 0.37,
            "spectral_entropy_mean": 0.43,
            "sub_bass_ratio_lt_150hz": 0.36,
            "bass_ratio_150_500hz": 0.26,
            "mid_ratio_500_2000hz": 0.31,
            "presence_ratio_2000_8000hz": 0.05,
            "air_ratio_gt_8000hz": 0.03,
            "loop_mean_event_high_ratio": 0.13,
            "loop_percussive_event_ratio": 0.06,
            "loop_drumlike_frame_ratio": 0.06,
        },
    )
    sample_facts.evidence["first_arrival_telemetry"] = {
        "status": "ok",
        "formant_center_std_hz": 600.0,
        "stochastic_modulation_coherence": 0.56,
        "first_arrival_presence_contrast_db": 27.0,
    }

    decision = scorer.analyze(sample_facts)

    assert decision.top_family == "Instruments"
    assert decision.branch == "Voice"
    assert decision.evidence["instrument_rap_voice_texture"] >= 0.59


def test_layered_physics_lifts_wet_woodwind_branch_sax_candidate() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "vocal_phrase",
            "confidence": 1.0,
            "onset_count": 20.0,
            "pitch_confidence": 0.82,
            "pitched_event_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        },
        roles={"vocal_music_phrase": 1.0, "pitched_music_loop": 0.98},
        values={
            "harmonic_energy_ratio": 0.63,
            "presence_ratio_2000_8000hz": 0.53,
            "spectral_flatness_mean": 0.27,
        },
    )

    decision = scorer.analyze(sample_facts)
    sax_score, evidence = scorer.apply("Instruments/Woodwinds/Saxophone/One Shots", 0.75, decision)

    assert decision.top_family == "Instruments"
    assert decision.branch == "Woodwinds"
    assert sax_score < 0.40
    assert evidence["physics_layer_leaf_strategy"] == "profile_leaf_with_branch_safeguard"


def test_layered_physics_low_mid_wet_sax_does_not_become_organ_or_synth() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "bass_phrase",
            "confidence": 1.0,
            "onset_count": 14.0,
            "onset_span_ratio": 0.81,
            "pitch_confidence": 0.883,
            "f0_voiced_ratio": 0.996,
            "pitched_event_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
            "low_event_ratio": 0.889,
            "mid_event_ratio": 0.082,
            "high_event_ratio": 0.029,
            "sustained_tonal_frame_ratio": 1.0,
            "non_event_tonal_ratio": 1.0,
            "spectral_flatness_mean": 0.203,
            "spectral_entropy_mean": 0.256,
        },
        roles={
            "pitched_music_loop": 1.0,
            "pitched_music_phrase": 1.0,
            "low_rhythmic_drum_loop": 0.347,
            "evidence": {"formant_light_voice_identity": 0.0},
        },
        values={
            "pitch_confidence": 0.883,
            "f0_voiced_ratio": 0.996,
            "f0_median_hz": 401.0,
            "loop_pitched_event_ratio": 1.0,
            "loop_sustained_tonal_frame_ratio": 1.0,
            "loop_non_event_tonal_ratio": 1.0,
            "loop_percussive_event_ratio": 0.0,
            "loop_drumlike_frame_ratio": 0.0,
            "loop_mean_event_low_ratio": 0.889,
            "loop_mean_event_mid_ratio": 0.082,
            "loop_mean_event_high_ratio": 0.029,
            "spectral_flatness_mean": 0.203,
            "spectral_entropy_mean": 0.256,
            "harmonic_energy_ratio": 0.464,
            "inharmonicity": 0.178,
            "sub_bass_ratio_lt_150hz": 0.0,
            "bass_ratio_150_500hz": 0.873,
            "mid_ratio_500_2000hz": 0.091,
            "presence_ratio_2000_8000hz": 0.036,
            "air_ratio_gt_8000hz": 0.0001,
            "body_noise_ratio": 0.104,
            "tail_noise_ratio": 0.209,
            "body_flatness": 0.141,
            "body_entropy": 0.186,
            "fundamental_dominance_ratio": 0.931,
            "spectral_peak_stability": 0.277,
        },
    )
    sample_facts.evidence["first_arrival_telemetry"] = {
        "status": "ok",
        "cepstral_pitch_period_coherence": 0.894,
        "first_arrival_conical_balance": 0.134,
        "first_arrival_presence_contrast_db": 27.0,
        "formant_center_mean_hz": 1217.0,
        "formant_center_std_hz": 65.0,
        "formant_stability_score": 0.962,
        "stochastic_modulation_coherence": 0.685,
        "strike_flatness": 0.282,
        "settle_flatness": 0.225,
        "strike_inharmonic_energy_ratio": 0.530,
        "settle_inharmonic_energy_ratio": 0.351,
    }

    decision = scorer.analyze(sample_facts)
    sax_score, sax_evidence = scorer.apply("Instruments/Woodwinds/Saxophone/Loops", 0.90, decision)
    synth_score, synth_evidence = scorer.apply("Instruments/Synths/Synth Lead/Loops", 0.90, decision)

    assert decision.top_family == "Instruments"
    assert decision.branch == "Woodwinds"
    assert decision.evidence["instrument_low_mid_wet_sax_signal"] is True
    assert decision.evidence["instrument_branch_Woodwinds"] > decision.evidence["instrument_branch_KeysPiano"]
    assert sax_score < synth_score
    assert any(
        "instrument_Woodwinds_branch_target" in item for item in sax_evidence["physics_layer_adjustment_reasons"]
    )
    assert any("instrument_branch_mismatch" in item for item in synth_evidence["physics_layer_adjustment_reasons"])


def test_layered_physics_clean_tonal_solo_reed_does_not_become_voice() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "solo_phrase",
            "confidence": 0.947451,
            "onset_count": 3.0,
            "pitch_confidence": 0.5879,
            "pitched_event_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
            "low_event_ratio": 0.107901,
            "mid_event_ratio": 0.757198,
            "high_event_ratio": 0.134901,
            "sustained_tonal_frame_ratio": 1.0,
            "non_event_tonal_ratio": 1.0,
            "spectral_flatness_mean": 0.028223,
            "spectral_entropy_mean": 0.342805,
        },
        roles={"voiced_one_shot": 0.908268, "percussive_one_shot": 0.252838},
        values={
            "attack_rise_time_norm": 0.053563,
            "pitch_confidence": 0.5879,
            "f0_voiced_ratio": 1.0,
            "f0_median_hz": 310.563385,
            "loop_pitched_event_ratio": 1.0,
            "loop_sustained_tonal_frame_ratio": 1.0,
            "loop_non_event_tonal_ratio": 1.0,
            "loop_percussive_event_ratio": 0.0,
            "loop_drumlike_frame_ratio": 0.0,
            "loop_mean_event_low_ratio": 0.107901,
            "loop_mean_event_mid_ratio": 0.757198,
            "loop_mean_event_high_ratio": 0.134901,
            "spectral_flatness_mean": 0.028223,
            "spectral_entropy_mean": 0.342805,
            "sub_bass_ratio_lt_150hz": 0.02,
            "bass_ratio_150_500hz": 0.075,
            "mid_ratio_500_2000hz": 0.817563,
            "presence_ratio_2000_8000hz": 0.08,
            "air_ratio_gt_8000hz": 0.0065,
            "body_flatness": 0.028223,
            "body_noise_ratio": 0.028223,
            "tail_noise_ratio": 0.035,
        },
    )
    sample_facts.evidence["measured_roles"]["evidence"] = {
        "formant_light_voice_identity": 0.965213,
    }
    sample_facts.evidence["first_arrival_telemetry"] = {
        "status": "ok",
        "first_arrival_presence_contrast_db": 25.060242,
        "formant_center_mean_hz": 1031.799,
        "formant_center_std_hz": 71.463,
        "formant_stability_score": 0.92,
        "stochastic_modulation_coherence": 0.44,
    }

    decision = scorer.analyze(sample_facts)
    sax_score, sax_evidence = scorer.apply("Instruments/Woodwinds/Saxophone/Loops", 0.90, decision)
    voice_score, voice_evidence = scorer.apply("Instruments/Voice/Choir/One Shots", 0.20, decision)

    assert decision.top_family == "Instruments"
    assert decision.branch == "Woodwinds"
    assert decision.evidence["instrument_clean_tonal_reed_solo_signal"] is True
    assert decision.evidence["instrument_branch_Woodwinds"] > decision.evidence["instrument_branch_Voice"]
    assert sax_score < voice_score
    assert any(
        "instrument_Woodwinds_branch_target" in item for item in sax_evidence["physics_layer_adjustment_reasons"]
    )
    assert any("instrument_branch_mismatch" in item for item in voice_evidence["physics_layer_adjustment_reasons"])


def test_layered_physics_clean_low_mid_solo_phrase_is_not_conga() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "solo_phrase",
            "confidence": 0.99,
            "onset_count": 2.0,
            "onset_span_ratio": 0.02,
            "pitch_confidence": 0.97,
            "pitched_event_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
            "low_event_ratio": 0.83,
            "mid_event_ratio": 0.17,
            "high_event_ratio": 0.002,
            "sustained_tonal_frame_ratio": 1.0,
            "non_event_tonal_ratio": 1.0,
            "spectral_flatness_mean": 0.02,
            "spectral_entropy_mean": 0.25,
        },
        roles={"percussive_one_shot": 0.88, "voiced_one_shot": 0.44},
        values={
            "attack_rise_time_norm": 0.035,
            "temporal_centroid_ratio": 0.11,
            "log_transient_count": 1.0986122886681098,
            "onset_span_ratio": 0.02,
            "pitch_confidence": 0.97,
            "f0_voiced_ratio": 1.0,
            "harmonic_energy_ratio": 0.98,
            "inharmonicity": 0.008,
            "low_peak_frequency_hz": 355.0,
            "sub_bass_ratio_lt_150hz": 0.00004,
            "bass_ratio_150_500hz": 0.84,
            "mid_ratio_500_2000hz": 0.155,
            "presence_ratio_2000_8000hz": 0.001,
            "air_ratio_gt_8000hz": 0.0,
            "loop_pitched_event_ratio": 1.0,
            "loop_percussive_event_ratio": 0.0,
            "loop_drumlike_frame_ratio": 0.0,
            "loop_sustained_tonal_frame_ratio": 1.0,
            "loop_non_event_tonal_ratio": 1.0,
            "loop_mean_event_low_ratio": 0.83,
            "loop_mean_event_mid_ratio": 0.17,
            "loop_mean_event_high_ratio": 0.002,
            "spectral_flatness_mean": 0.02,
            "spectral_entropy_mean": 0.25,
        },
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=False,
        is_long=False,
    )

    decision = scorer.analyze(sample_facts)
    conga_score, conga_evidence = scorer.apply(
        "Drums/World Percussion/Latin Percussion/Conga/One Shots",
        0.20,
        decision,
    )

    assert decision.top_family == "Instruments"
    assert decision.evidence["drum_anchor_clean_tonal_solo_phrase"] is True
    assert decision.evidence["drum_anchor_strength"] <= 0.52
    assert conga_score > 2.0
    assert "top_family_mismatch:+2.25" in conga_evidence["physics_layer_adjustment_reasons"]


def test_layered_physics_sub_kick_exception_stays_drums() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "solo_phrase",
            "confidence": 0.92,
            "onset_count": 1.0,
            "pitch_confidence": 0.98,
            "pitched_event_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
            "low_event_ratio": 1.0,
            "high_event_ratio": 0.0,
            "sustained_tonal_frame_ratio": 1.0,
            "non_event_tonal_ratio": 1.0,
        },
        roles={"percussive_one_shot": 1.0, "voiced_one_shot": 0.24},
        values={
            "attack_rise_time_norm": 0.001,
            "temporal_centroid_ratio": 0.23,
            "log_transient_count": 0.6931471805599453,
            "pitch_confidence": 0.98,
            "f0_voiced_ratio": 0.0,
            "harmonic_energy_ratio": 0.0,
            "inharmonicity": 1.0,
            "low_peak_frequency_hz": 54.0,
            "sub_bass_ratio_lt_150hz": 0.998,
            "bass_ratio_150_500hz": 0.002,
            "mid_ratio_500_2000hz": 0.0,
            "presence_ratio_2000_8000hz": 0.0,
            "air_ratio_gt_8000hz": 0.0,
            "loop_pitched_event_ratio": 1.0,
            "loop_percussive_event_ratio": 0.0,
            "loop_drumlike_frame_ratio": 0.0,
            "loop_sustained_tonal_frame_ratio": 1.0,
            "loop_non_event_tonal_ratio": 1.0,
            "loop_mean_event_low_ratio": 1.0,
            "loop_mean_event_high_ratio": 0.0,
        },
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
    )

    decision = scorer.analyze(sample_facts)

    assert decision.top_family == "Drums"
    assert decision.branch == "Kick"
    assert decision.evidence["drum_anchor_low_sub_kick_exception"] is True


def test_layered_physics_does_not_call_high_register_mixed_loop_clean_bass() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "bass_phrase",
            "confidence": 0.99,
            "onset_count": 12.0,
            "onset_span_ratio": 0.73,
            "pitch_confidence": 0.82,
            "pitched_event_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
            "low_event_ratio": 0.93,
            "mid_event_ratio": 0.06,
            "high_event_ratio": 0.012,
            "sustained_tonal_frame_ratio": 1.0,
            "non_event_tonal_ratio": 1.0,
        },
        roles={"bass_loop": 0.75, "pitched_music_loop": 0.98, "low_rhythmic_drum_loop": 0.74},
        values={
            "pitch_confidence": 0.82,
            "f0_median_hz": 735.0,
            "f0_voiced_ratio": 0.22,
            "low_peak_frequency_hz": 54.0,
            "sub_bass_ratio_lt_150hz": 0.64,
            "bass_ratio_150_500hz": 0.26,
            "mid_ratio_500_2000hz": 0.09,
            "presence_ratio_2000_8000hz": 0.016,
            "air_ratio_gt_8000hz": 0.0,
            "loop_mean_event_low_ratio": 0.93,
            "loop_mean_event_mid_ratio": 0.06,
            "loop_mean_event_high_ratio": 0.012,
            "loop_pitched_event_ratio": 1.0,
            "loop_percussive_event_ratio": 0.0,
            "loop_drumlike_frame_ratio": 0.0,
            "loop_sustained_tonal_frame_ratio": 1.0,
            "loop_non_event_tonal_ratio": 1.0,
        },
    )

    decision = scorer.analyze(sample_facts)

    assert decision.top_family == "Instruments"
    assert decision.branch != "Bass"
    assert decision.evidence["instrument_high_register_low_band_conflict"] is True
    assert decision.evidence["instrument_clean_bass_phrase"] is False


def test_layered_physics_processed_vocal_shot_prefers_voice_over_woodwind() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "vocal_phrase",
            "confidence": 0.99,
            "onset_count": 1.0,
            "pitch_confidence": 0.77,
            "pitched_event_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
            "mid_event_ratio": 0.60,
            "high_event_ratio": 0.29,
            "sustained_tonal_frame_ratio": 1.0,
            "non_event_tonal_ratio": 1.0,
            "spectral_flatness_mean": 0.36,
        },
        roles={"percussive_one_shot": 1.0, "voiced_one_shot": 0.41},
        values={
            "pitch_confidence": 0.77,
            "f0_voiced_ratio": 1.0,
            "formant_like_peak_spacing": 4.0,
            "spectral_flatness_mean": 0.36,
            "body_noise_ratio": 0.34,
            "mid_ratio_500_2000hz": 0.59,
            "presence_ratio_2000_8000hz": 0.29,
            "loop_pitched_event_ratio": 1.0,
            "loop_percussive_event_ratio": 0.0,
            "loop_drumlike_frame_ratio": 0.0,
        },
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
    )
    sample_facts.evidence["measured_roles"]["evidence"] = {
        "formant_light_voice_identity": 0.91,
    }

    decision = scorer.analyze(sample_facts)
    sax_score, sax_evidence = scorer.apply("Instruments/Woodwinds/Saxophone/One Shots", 0.20, decision)

    assert decision.top_family == "Instruments"
    assert decision.branch == "Voice"
    assert decision.evidence["instrument_processed_vocal_shot_signal"] is True
    assert sax_score > 1.0
    assert "dense_voice_branch_blocks_sax:+0.95" in sax_evidence["physics_layer_adjustment_reasons"]


def test_layered_physics_prefers_broad_bucket_for_compound_music_loop() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "pitched_phrase",
            "confidence": 0.96,
            "onset_count": 48.0,
            "onset_span_ratio": 0.86,
            "pitch_confidence": 0.88,
            "pitched_event_ratio": 0.96,
            "percussive_event_ratio": 0.30,
            "drumlike_frame_ratio": 0.22,
            "low_event_ratio": 0.30,
            "mid_event_ratio": 0.48,
            "high_event_ratio": 0.18,
            "sustained_tonal_frame_ratio": 0.72,
            "non_event_tonal_ratio": 0.62,
            "spectral_flatness_mean": 0.24,
            "spectral_entropy_mean": 0.52,
            "tail_ratio": 0.55,
        },
        roles={"pitched_music_loop": 0.90},
        values={
            "pitch_confidence": 0.88,
            "f0_voiced_ratio": 0.82,
            "loop_pitched_event_ratio": 0.96,
            "loop_sustained_tonal_frame_ratio": 0.72,
            "loop_non_event_tonal_ratio": 0.62,
            "harmonic_energy_ratio": 0.58,
            "fundamental_dominance_ratio": 0.42,
            "spectral_flatness_mean": 0.24,
            "spectral_entropy_mean": 0.52,
            "sub_bass_ratio_lt_150hz": 0.18,
            "bass_ratio_150_500hz": 0.23,
            "mid_ratio_500_2000hz": 0.45,
            "presence_ratio_2000_8000hz": 0.12,
            "air_ratio_gt_8000hz": 0.07,
            "loop_mean_event_low_ratio": 0.30,
            "loop_mean_event_mid_ratio": 0.48,
            "loop_mean_event_high_ratio": 0.18,
            "loop_percussive_event_ratio": 0.30,
            "loop_drumlike_frame_ratio": 0.22,
            "body_noise_ratio": 0.32,
            "tail_noise_ratio": 0.35,
            "tail_energy_ratio": 0.55,
            "formant_like_peak_spacing": 0.52,
            "stereo_width": 0.72,
        },
    )

    decision = scorer.analyze(sample_facts)
    broad_score, broad_evidence = scorer.apply("Instruments/Instrument Loops/Loops", 0.82, decision)
    sax_score, sax_evidence = scorer.apply("Instruments/Woodwinds/Saxophone/Loops", 0.18, decision)
    fx_score, fx_evidence = scorer.apply(
        "FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX",
        0.30,
        decision,
    )

    assert decision.top_family == "Instruments"
    assert decision.branch == "MixedInstrument"
    assert decision.evidence["compound_music_prefer_broad_loop"] is True
    assert broad_score < 0.30
    assert sax_score > 0.60
    assert fx_score > 6.0
    assert any(
        "instrument_MixedInstrument_branch_target" in item
        for item in broad_evidence["physics_layer_adjustment_reasons"]
    )
    assert "mixed_compound_branch_blocks_specific_leaf:+0.55" in sax_evidence["physics_layer_adjustment_reasons"]
    assert "mixed_compound_branch_blocks_non_instrument:+6.00" in fx_evidence["physics_layer_adjustment_reasons"]


def test_layered_physics_fx_role_detects_riser_without_filename() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "transition_riser",
            "confidence": 0.90,
            "onset_count": 2.0,
            "onset_span_ratio": 0.40,
            "tail_ratio": 0.55,
            "centroid_slope_norm": 0.24,
        },
        values={
            "centroid_slope_norm": 0.24,
            "tail_energy_ratio": 0.55,
            "spectral_flatness_mean": 0.42,
            "spectral_entropy_mean": 0.72,
            "stereo_width": 0.64,
            "log_transient_count": 1.0986122886681098,
            "onset_span_ratio": 0.40,
            "event_rate_hz": 0.40,
            "attack_rise_time_norm": 0.28,
            "temporal_centroid_ratio": 0.55,
            "body_noise_ratio": 0.42,
            "tail_noise_ratio": 0.48,
            "presence_ratio_2000_8000hz": 0.24,
            "air_ratio_gt_8000hz": 0.09,
        },
    )

    decision = scorer.analyze(sample_facts)
    riser_score, riser_evidence = scorer.apply(
        "FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX",
        0.82,
        decision,
    )
    piano_score, piano_evidence = scorer.apply("Instruments/Keys/Piano/Loops", 0.20, decision)

    assert decision.top_family == "FX"
    assert decision.branch == "RiserBuild"
    assert decision.evidence["fx_role_strength"] >= 0.78
    assert riser_score < 0.32
    assert piano_score > 0.55
    assert any("fx_RiserBuild_branch_target" in item for item in riser_evidence["physics_layer_adjustment_reasons"])
    assert "top_family_mismatch:+0.40" in piano_evidence["physics_layer_adjustment_reasons"]


def test_layered_physics_fx_role_detects_glitch_stutter() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "glitch_stutter",
            "confidence": 0.88,
            "onset_count": 18.0,
            "onset_span_ratio": 0.70,
            "tail_ratio": 0.22,
        },
        values={
            "spectral_flatness_mean": 0.50,
            "spectral_entropy_mean": 0.78,
            "spectral_flux_mean": 0.35,
            "spectral_flux_variance": 0.06,
            "log_transient_count": 2.9444389791664403,
            "event_rate_hz": 6.0,
            "onset_interval_regularity": 0.18,
            "onset_span_ratio": 0.70,
            "tail_energy_ratio": 0.18,
            "attack_rise_time_norm": 0.04,
            "temporal_centroid_ratio": 0.45,
            "zcr_mean": 0.26,
            "presence_ratio_2000_8000hz": 0.16,
            "air_ratio_gt_8000hz": 0.08,
            "body_noise_ratio": 0.52,
        },
    )

    decision = scorer.analyze(sample_facts)
    glitch_score, evidence = scorer.apply(
        "FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/One Shots",
        0.74,
        decision,
    )

    assert decision.top_family == "FX"
    assert decision.branch == "GlitchStutter"
    assert decision.evidence["fx_vs_texture_conflict"] < 0.30
    assert glitch_score < 0.30
    assert any("fx_GlitchStutter_branch_target" in item for item in evidence["physics_layer_adjustment_reasons"])


def test_layered_physics_fx_role_does_not_steal_clean_instrument_loop() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "pitched_phrase",
            "confidence": 0.95,
            "onset_count": 10.0,
            "onset_span_ratio": 0.75,
            "pitch_confidence": 0.90,
            "f0_voiced_ratio": 0.86,
            "pitched_event_ratio": 0.95,
            "sustained_tonal_frame_ratio": 0.88,
            "non_event_tonal_ratio": 0.86,
            "percussive_event_ratio": 0.04,
            "drumlike_frame_ratio": 0.03,
            "spectral_flatness_mean": 0.06,
        },
        roles={"pitched_music_loop": 0.90},
        values={
            "pitch_confidence": 0.90,
            "f0_voiced_ratio": 0.86,
            "loop_pitched_event_ratio": 0.95,
            "loop_sustained_tonal_frame_ratio": 0.88,
            "loop_non_event_tonal_ratio": 0.86,
            "spectral_flatness_mean": 0.06,
            "spectral_entropy_mean": 0.32,
            "harmonic_energy_ratio": 0.70,
            "log_transient_count": 2.3978952727983707,
            "onset_span_ratio": 0.75,
            "event_rate_hz": 1.20,
            "tail_energy_ratio": 0.40,
            "centroid_slope_norm": 0.16,
            "stereo_width": 0.50,
        },
    )

    decision = scorer.analyze(sample_facts)

    assert decision.top_family == "Instruments"
    assert decision.evidence["fx_vs_instrument_conflict"] >= 0.70
    assert decision.evidence["fx_role_strength"] < 0.50


def test_layered_physics_strong_kick_anchor_beats_impact_shape() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "hit_with_tail",
            "confidence": 0.90,
            "onset_count": 1.0,
            "tail_ratio": 0.28,
            "temporal_centroid_ratio": 0.12,
            "attack_rise_time_norm": 0.01,
            "percussive_event_ratio": 0.80,
            "drumlike_frame_ratio": 0.82,
            "low_event_ratio": 0.92,
        },
        roles={"percussive_one_shot": 0.85},
        values={
            "log_transient_count": 0.6931471805599453,
            "attack_rise_time_norm": 0.01,
            "temporal_centroid_ratio": 0.12,
            "tail_energy_ratio": 0.28,
            "log_crest": 2.30,
            "stereo_width": 0.20,
            "spectral_flatness_mean": 0.22,
            "spectral_entropy_mean": 0.46,
            "loop_percussive_event_ratio": 0.80,
            "loop_drumlike_frame_ratio": 0.82,
            "sub_bass_ratio_lt_150hz": 0.78,
            "bass_ratio_150_500hz": 0.14,
            "presence_ratio_2000_8000hz": 0.02,
            "air_ratio_gt_8000hz": 0.01,
            "low_peak_frequency_hz": 72.0,
            "sub_decay_time_ms": 120.0,
            "loop_mean_event_low_ratio": 0.92,
        },
    )

    decision = scorer.analyze(sample_facts)

    assert decision.top_family == "Drums"
    assert decision.branch == "Kick"
    assert decision.evidence["fx_branch_selected"] == "ImpactHit"


def test_layered_physics_low_centered_kick_does_not_become_blip_fx() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "solo_phrase",
            "confidence": 0.91,
            "onset_count": 1.0,
            "tail_ratio": 0.56,
            "temporal_centroid_ratio": 0.38,
            "attack_rise_time_norm": 0.003,
            "pitch_confidence": 0.94,
            "low_event_ratio": 1.0,
            "pitched_event_ratio": 1.0,
        },
        values={
            "log_transient_count": 0.6931471805599453,
            "attack_rise_time_norm": 0.003,
            "temporal_centroid_ratio": 0.38,
            "tail_energy_ratio": 0.56,
            "pitch_confidence": 0.94,
            "spectral_flatness_mean": 0.001,
            "spectral_entropy_mean": 0.17,
            "sub_bass_ratio_lt_150hz": 0.998,
            "bass_ratio_150_500hz": 0.002,
            "mid_ratio_500_2000hz": 0.0,
            "presence_ratio_2000_8000hz": 0.0,
            "air_ratio_gt_8000hz": 0.0,
            "low_peak_frequency_hz": 43.0,
            "stereo_width": 0.0,
            "body_noise_ratio": 0.001,
            "tail_noise_ratio": 0.001,
            "attack_noise_ratio": 0.001,
        },
        roles={"percussive_one_shot": 0.82},
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
    )

    decision = scorer.analyze(sample_facts)

    assert decision.top_family == "Drums"
    assert decision.branch == "Kick"
    assert decision.evidence["fx_low_centered_tonal_hit"] >= 0.90
    assert decision.evidence["fx_branch_selected"] != "BlipBeep"


def test_layered_physics_clean_mid_beep_keeps_blip_branch() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "solo_phrase",
            "confidence": 0.90,
            "onset_count": 2.0,
            "tail_ratio": 0.01,
            "temporal_centroid_ratio": 0.07,
            "attack_rise_time_norm": 0.07,
            "pitch_confidence": 0.89,
            "f0_voiced_ratio": 1.0,
            "pitched_event_ratio": 1.0,
        },
        values={
            "log_transient_count": 1.0986122886681098,
            "attack_rise_time_norm": 0.07,
            "temporal_centroid_ratio": 0.07,
            "tail_energy_ratio": 0.01,
            "pitch_confidence": 0.89,
            "f0_voiced_ratio": 1.0,
            "harmonic_energy_ratio": 0.95,
            "inharmonicity": 0.02,
            "spectral_flatness_mean": 0.07,
            "spectral_entropy_mean": 0.35,
            "sub_bass_ratio_lt_150hz": 0.0,
            "bass_ratio_150_500hz": 0.32,
            "mid_ratio_500_2000hz": 0.68,
            "presence_ratio_2000_8000hz": 0.0,
            "air_ratio_gt_8000hz": 0.0,
            "low_peak_frequency_hz": 388.0,
            "zcr_mean": 0.09,
            "stereo_width": 0.27,
            "body_noise_ratio": 0.15,
            "tail_noise_ratio": 0.17,
            "attack_noise_ratio": 0.14,
        },
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=False,
        is_long=False,
    )

    decision = scorer.analyze(sample_facts)

    assert decision.top_family == "FX"
    assert decision.branch == "BlipBeep"
    assert decision.evidence["fx_clean_mid_blip_tone"] >= 0.58


def test_fx_role_layer_does_not_allow_repeated_drum_loop_as_riser() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "transition_riser",
            "confidence": 0.86,
            "onset_count": 18.0,
            "onset_span_ratio": 0.94,
            "true_repetition_score": 0.78,
            "pitch_confidence": 0.82,
            "pitched_event_ratio": 1.0,
            "percussive_event_ratio": 0.34,
            "drumlike_frame_ratio": 0.28,
            "low_event_ratio": 0.84,
            "high_event_ratio": 0.11,
            "centroid_slope_norm": 0.17,
            "spectral_flatness_mean": 0.24,
            "spectral_entropy_mean": 0.59,
        },
        roles={"low_rhythmic_drum_loop": 0.64, "pitched_music_loop": 0.62},
        values={
            "log_transient_count": 2.9444389791664403,
            "onset_span_ratio": 0.94,
            "loop_true_repetition_score": 0.78,
            "loop_pitched_event_ratio": 1.0,
            "loop_percussive_event_ratio": 0.34,
            "loop_drumlike_frame_ratio": 0.28,
            "loop_mean_event_low_ratio": 0.84,
            "loop_mean_event_high_ratio": 0.11,
            "loop_sustained_tonal_frame_ratio": 0.58,
            "centroid_slope_norm": 0.17,
            "spectral_flux_mean": 0.28,
            "spectral_flatness_mean": 0.24,
            "spectral_entropy_mean": 0.59,
            "sub_bass_ratio_lt_150hz": 0.54,
            "bass_ratio_150_500hz": 0.28,
            "mid_ratio_500_2000hz": 0.07,
            "presence_ratio_2000_8000hz": 0.08,
            "air_ratio_gt_8000hz": 0.03,
            "tail_energy_ratio": 0.44,
        },
    )

    decision = scorer.analyze(sample_facts)
    fx_score, fx_evidence = scorer.apply(
        "FX/Structural and Transitional FX/Risers and Builds/Generic Build/Long FX",
        0.18,
        decision,
    )

    assert decision.top_family != "FX"
    assert decision.evidence["fx_role_allows_fx"] is False
    assert decision.evidence["fx_transition_loop_decoy_guard"] is True
    assert fx_score > 1.0
    assert "fx_role_rejected_by_measurements:+1.25" in fx_evidence["physics_layer_adjustment_reasons"]


def test_layered_physics_penalizes_fx_when_voice_shape_rejects_fx_role() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "vocal_phrase",
            "confidence": 0.88,
            "onset_count": 24.0,
            "onset_span_ratio": 0.90,
            "true_repetition_score": 0.42,
            "low_event_ratio": 0.40,
            "mid_event_ratio": 0.42,
            "high_event_ratio": 0.18,
            "pitch_confidence": 0.72,
            "pitched_event_ratio": 0.78,
            "percussive_event_ratio": 0.18,
            "drumlike_frame_ratio": 0.16,
        },
        roles={"vocal_music_phrase": 0.86, "pitched_music_loop": 0.76},
        values={
            "log_transient_count": 3.2188758248682006,
            "onset_span_ratio": 0.90,
            "pitch_confidence": 0.72,
            "f0_voiced_ratio": 0.66,
            "loop_pitched_event_ratio": 0.78,
            "loop_percussive_event_ratio": 0.18,
            "loop_drumlike_frame_ratio": 0.16,
            "loop_mean_event_low_ratio": 0.40,
            "loop_mean_event_high_ratio": 0.18,
            "loop_sustained_tonal_frame_ratio": 0.81,
            "loop_non_event_tonal_ratio": 0.76,
            "spectral_flatness_mean": 0.42,
            "spectral_entropy_mean": 0.56,
            "body_noise_ratio": 0.39,
            "tail_noise_ratio": 0.44,
            "sub_bass_ratio_lt_150hz": 0.16,
            "bass_ratio_150_500hz": 0.24,
            "mid_ratio_500_2000hz": 0.42,
            "presence_ratio_2000_8000hz": 0.14,
            "air_ratio_gt_8000hz": 0.04,
            "tail_energy_ratio": 0.48,
        },
    )

    decision = scorer.analyze(sample_facts)
    fx_score, fx_evidence = scorer.apply(
        "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX",
        0.20,
        decision,
    )

    assert decision.top_family == "Instruments"
    assert decision.evidence["fx_role_allows_fx"] is False
    assert fx_score > 1.0
    assert "fx_role_rejected_by_measurements:+0.92" in fx_evidence["physics_layer_adjustment_reasons"]


def test_layered_physics_fx_texture_is_fx_role_not_conflict() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "texture_bed",
            "confidence": 0.86,
            "onset_count": 2.0,
            "onset_span_ratio": 0.20,
            "tail_ratio": 0.82,
            "spectral_flatness_mean": 0.56,
            "spectral_entropy_mean": 0.82,
            "pitch_confidence": 0.05,
            "pitched_event_ratio": 0.02,
            "percussive_event_ratio": 0.04,
            "drumlike_frame_ratio": 0.03,
            "shape_scores": [
                ["texture_bed", 0.86],
                ["noise_texture", 0.82],
                ["static_bed", 0.78],
            ],
        },
        values={
            "spectral_flatness_mean": 0.56,
            "spectral_entropy_mean": 0.82,
            "body_noise_ratio": 0.58,
            "tail_noise_ratio": 0.64,
            "tail_energy_ratio": 0.82,
            "event_rate_hz": 0.12,
            "log_transient_count": 1.0986122886681098,
            "onset_span_ratio": 0.20,
            "centroid_slope_norm": 0.02,
            "stereo_width": 0.68,
            "pitch_confidence": 0.05,
            "f0_voiced_ratio": 0.03,
        },
        is_loop_like=False,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
    )

    decision = scorer.analyze(sample_facts)
    texture_score, texture_evidence = scorer.apply("FX/Textures/Noise and Static/Hiss/Long FX", 0.90, decision)
    piano_score, piano_evidence = scorer.apply("Instruments/Keys/Piano/Loops", 0.18, decision)

    assert decision.top_family == "FX"
    assert decision.branch == "TextureAmbience"
    assert decision.evidence["fx_role_allows_fx"] is True
    assert decision.evidence["fx_vs_texture_conflict"] < 0.25
    assert texture_score < 0.36
    assert piano_score > 0.55
    assert any(
        "fx_TextureAmbience_branch_target" in item for item in texture_evidence["physics_layer_adjustment_reasons"]
    )
    assert "top_family_mismatch:+0.40" in piano_evidence["physics_layer_adjustment_reasons"]


def test_human_voice_signal_blocks_animal_fx_leaf_boost() -> None:
    scorer = LayeredPhysicsScorer()
    decision = PhysicsLayerDecision(
        top_family="FX",
        top_confidence=0.74,
        branch="HumanCreatureFX",
        branch_confidence=0.82,
        leaf_strategy="profile_leaf_with_fx_role_safeguard",
        evidence={
            "instrument_subpanel_voice_score": 0.64,
            "instrument_subpanel_human_spoken_voice_score": 0.71,
            "instrument_subpanel_human_breath_mouth_score": 0.82,
            "instrument_human_voice_texture": 0.75,
            "fx_formant_motion": 0.59,
            "physics_category_panel_flat": {},
        },
    )

    animal_score, animal_evidence = scorer.apply("FX/Animals and Creatures/Bird/One Shots", 0.30, decision)
    human_score, human_evidence = scorer.apply("FX/Human and Voice FX/Mouth Sounds/One Shots", 0.80, decision)

    assert animal_score > 0.70
    assert human_score < 0.39
    assert "human_voice_signal_blocks_animal_fx_leaf:+0.42" in animal_evidence["physics_layer_adjustment_reasons"]
    assert any(
        "fx_HumanCreatureFX_branch_target" in reason for reason in human_evidence["physics_layer_adjustment_reasons"]
    )


def test_layered_physics_hybrid_motion_loop_can_be_transition_fx() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "hybrid_fx_motion",
            "secondary_shape": "transition_riser",
            "confidence": 0.82,
            "onset_count": 33.0,
            "onset_density_hz": 4.7,
            "onset_span_ratio": 0.64,
            "true_repetition_score": 0.76,
            "centroid_slope_norm": 0.29,
            "tail_ratio": 0.99,
            "pitch_confidence": 0.85,
            "pitched_event_ratio": 0.47,
            "percussive_event_ratio": 0.47,
            "drumlike_frame_ratio": 0.31,
            "shape_scores": [
                ["hybrid_fx_motion", 0.82],
                ["transition_riser", 0.76],
                ["whoosh_sweep", 0.74],
                ["mixed_instrument_loop", 0.72],
            ],
        },
        values={
            "centroid_slope_norm": 0.29,
            "spectral_flux_mean": 0.44,
            "spectral_flux_variance": 0.038,
            "spectral_flatness_mean": 0.14,
            "spectral_entropy_mean": 0.60,
            "tail_energy_ratio": 0.99,
            "attack_rise_time_norm": 0.43,
            "temporal_centroid_ratio": 0.80,
            "event_rate_hz": 4.7,
            "log_transient_count": 3.5263605246161616,
            "onset_span_ratio": 0.64,
            "loop_true_repetition_score": 0.76,
            "loop_pitched_event_ratio": 0.47,
            "loop_percussive_event_ratio": 0.47,
            "loop_drumlike_frame_ratio": 0.31,
            "loop_sustained_tonal_frame_ratio": 0.60,
            "loop_non_event_tonal_ratio": 0.76,
            "pitch_confidence": 0.85,
            "f0_voiced_ratio": 0.22,
            "body_noise_ratio": 0.32,
            "tail_noise_ratio": 0.40,
            "stereo_width": 0.72,
        },
    )

    decision = scorer.analyze(sample_facts)
    riser_score, evidence = scorer.apply(
        "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX",
        0.80,
        decision,
    )

    assert decision.top_family == "FX"
    assert decision.branch == "RiserBuild"
    assert decision.evidence["fx_designed_motion_loop_exception"] is True
    assert decision.evidence["fx_transition_loop_decoy_guard"] is False
    assert decision.evidence["fx_role_allows_fx"] is True
    assert riser_score < 0.34
    assert any("fx_RiserBuild_branch_target" in item for item in evidence["physics_layer_adjustment_reasons"])


def test_kick_anchored_beat_loop_claims_drum_family_before_mixed_instrument() -> None:
    scorer = LayeredPhysicsScorer()
    sample_facts = facts(
        {
            "primary_shape": "bass_phrase",
            "secondary_shape": "beat_loop",
            "confidence": 0.92,
            "onset_count": 27.0,
            "onset_span_ratio": 0.94,
            "true_repetition_score": 0.86,
            "low_event_ratio": 0.75,
            "mid_event_ratio": 0.14,
            "high_event_ratio": 0.11,
            "pitched_event_ratio": 0.74,
            "percussive_event_ratio": 0.26,
            "drumlike_frame_ratio": 0.12,
            "pitch_confidence": 0.99,
            "shape_scores": [
                ["bass_phrase", 0.92],
                ["beat_loop", 0.87],
                ["repeated_phrase_loop", 0.86],
            ],
        },
        roles={"pitched_music_loop": 0.70, "low_rhythmic_drum_loop": 0.42},
        values={
            "log_transient_count": 3.332204510175204,
            "attack_rise_time_norm": 0.002,
            "temporal_centroid_ratio": 0.18,
            "log_crest": 2.1,
            "onset_span_ratio": 0.94,
            "loop_pitched_event_ratio": 0.74,
            "loop_percussive_event_ratio": 0.26,
            "loop_drumlike_frame_ratio": 0.12,
            "loop_mean_event_low_ratio": 0.75,
            "loop_mean_event_high_ratio": 0.11,
            "loop_sustained_tonal_frame_ratio": 0.87,
            "pitch_confidence": 0.99,
            "f0_voiced_ratio": 0.44,
            "spectral_flatness_mean": 0.21,
            "spectral_entropy_mean": 0.31,
            "sub_bass_ratio_lt_150hz": 0.56,
            "bass_ratio_150_500hz": 0.19,
            "mid_ratio_500_2000hz": 0.14,
            "presence_ratio_2000_8000hz": 0.08,
            "air_ratio_gt_8000hz": 0.03,
            "low_peak_frequency_hz": 72.0,
            "sub_decay_time_ms": 92.0,
        },
    )

    decision = scorer.analyze(sample_facts)
    drum_loop_score, drum_loop_evidence = scorer.apply("Drums/Drum Loops/Loops", 1.20, decision)
    kick_score, kick_evidence = scorer.apply("Drums/Kick Drums/Generic Kick/One Shots", 0.20, decision)

    assert decision.top_family == "Drums"
    assert decision.evidence["physics_top_layer_source"] == "kick_anchored_beat_loop"
    assert drum_loop_score < 0.50
    assert kick_score > 1.0
    assert any(
        "kick_anchored_loop_drum_loop_target" in reason
        for reason in drum_loop_evidence["physics_layer_adjustment_reasons"]
    )
    assert "kick_anchored_loop_blocks_kick_one_shot:+0.86" in kick_evidence["physics_layer_adjustment_reasons"]
