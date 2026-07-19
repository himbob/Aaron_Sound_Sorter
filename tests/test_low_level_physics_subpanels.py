from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

TEMP_ROOT = Path(tempfile.gettempdir())

from aaron_sound_sorter.core import FEATURE_NAMES, FP_SIZE
from aaron_sound_sorter.domain.facts import build_shared_audio_facts
from aaron_sound_sorter.domain.models import AudioPhysics, CategoryGuess, ConsensusDecision, SortFileResult, VoterResult
from aaron_sound_sorter.infrastructure.report_writer import PHYSICS_SUBPANEL_MANIFEST_FIELDS, manifest_row

NAME_TO_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


def feature_vector(**values: float) -> np.ndarray:
    vector = np.zeros(FP_SIZE, dtype=np.float32)
    for name, value in values.items():
        vector[NAME_TO_INDEX[name]] = float(value)
    return vector


def make_facts(**values: float):
    physics = AudioPhysics(
        source_path=TEMP_ROOT / "source.wav",
        fingerprint=feature_vector(**values),
        duration_sec=4.0,
        read_status="ok",
    )
    return build_shared_audio_facts(physics)


def test_shared_facts_expose_requested_low_level_subpanel_columns() -> None:
    facts = make_facts(
        log_transient_count=float(np.log1p(10.0)),
        onset_span_ratio=0.72,
        onset_interval_regularity=0.08,
        event_rate_hz=4.0,
        attack_rise_time_norm=0.035,
        pitch_confidence=0.88,
        f0_voiced_ratio=0.90,
        loop_pitched_event_ratio=0.92,
        loop_mean_event_pitch_confidence=0.86,
        harmonic_energy_ratio=0.72,
        harmonic_peak_count=8.0,
        fundamental_dominance_ratio=0.62,
        spectral_flatness_mean=0.08,
        spectral_entropy_mean=0.34,
        mid_ratio_500_2000hz=0.45,
        presence_ratio_2000_8000hz=0.16,
        attack_high_ratio=0.12,
        attack_zcr=0.10,
        body_pitch_confidence=0.82,
        tail_pitch_confidence=0.70,
        spectral_peak_stability=0.32,
    )

    assert "physics_subpanels" in facts.evidence
    for key in [
        "role_loop_score",
        "role_one_shot_score",
        "role_phrase_score",
        "plucked_string_score",
        "plucked_string_authority_score",
        "reed_wind_score",
        "reed_wind_authority_score",
        "struck_keys_score",
        "struck_keys_authority_score",
        "synth_tonal_source_score",
        "bowed_string_score",
        "voice_score",
        "drum_hit_score",
        "drum_loop_source_score",
        "metallic_noise_score",
        "scrape_rasp_score",
        "fx_motion_score",
        "fx_transition_authority_score",
        "texture_bed_score",
    ]:
        assert key in facts.evidence, key
        assert 0.0 <= float(facts.evidence[key]) <= 1.0


def test_plucked_string_subpanel_beats_reed_and_keys_for_fast_tonal_decay() -> None:
    facts = make_facts(
        log_transient_count=float(np.log1p(8.0)),
        onset_span_ratio=0.68,
        onset_interval_regularity=0.10,
        event_rate_hz=3.2,
        attack_rise_time_norm=0.02,
        temporal_centroid_ratio=0.22,
        pitch_confidence=0.88,
        f0_voiced_ratio=0.86,
        loop_pitched_event_ratio=0.94,
        loop_mean_event_pitch_confidence=0.88,
        loop_sustained_tonal_frame_ratio=0.72,
        loop_non_event_tonal_ratio=0.66,
        harmonic_energy_ratio=0.78,
        harmonic_to_noise_ratio=0.70,
        harmonic_peak_count=9.0,
        fundamental_dominance_ratio=0.58,
        inharmonicity=0.18,
        spectral_flatness_mean=0.08,
        spectral_entropy_mean=0.32,
        mid_ratio_500_2000hz=0.46,
        presence_ratio_2000_8000hz=0.18,
        air_ratio_gt_8000hz=0.02,
        attack_high_ratio=0.18,
        attack_zcr=0.12,
        body_noise_ratio=0.12,
        tail_energy_ratio=0.22,
        spectral_peak_stability=0.38,
    )
    flat = facts.evidence["physics_subpanels"]["flat"]

    assert flat["plucked_string_score"] > flat["reed_wind_score"]
    assert flat["plucked_string_authority_score"] > flat["reed_wind_authority_score"]
    assert flat["plucked_string_score"] >= flat["struck_keys_score"] - 0.08
    assert flat["role_phrase_score"] >= 0.40


def test_reed_wind_subpanel_beats_pluck_for_breathy_legato_formant_source() -> None:
    facts = make_facts(
        log_transient_count=float(np.log1p(5.0)),
        onset_span_ratio=0.52,
        onset_interval_regularity=0.28,
        event_rate_hz=1.9,
        attack_rise_time_norm=0.20,
        pitch_confidence=0.86,
        f0_voiced_ratio=0.94,
        f0_stability_cents=55.0,
        f0_slope_cents_per_sec=80.0,
        loop_pitched_event_ratio=0.90,
        loop_mean_event_pitch_confidence=0.82,
        loop_sustained_tonal_frame_ratio=0.92,
        loop_non_event_tonal_ratio=0.88,
        harmonic_energy_ratio=0.64,
        harmonic_to_noise_ratio=0.52,
        spectral_flatness_mean=0.20,
        spectral_entropy_mean=0.46,
        formant_like_peak_spacing=0.64,
        spectral_peak_count=8.0,
        spectral_peak_stability=0.30,
        peak_bandwidth_mean_hz=500.0,
        mid_ratio_500_2000hz=0.48,
        presence_ratio_2000_8000hz=0.12,
        air_ratio_gt_8000hz=0.04,
        attack_noise_ratio=0.20,
        body_noise_ratio=0.26,
        tail_noise_ratio=0.22,
        attack_zcr=0.16,
    )
    flat = facts.evidence["physics_subpanels"]["flat"]

    assert flat["reed_wind_score"] > flat["plucked_string_score"]
    assert flat["reed_breath_attack_score"] > 0.35
    assert flat["reed_formant_envelope_score"] > 0.35


def test_manifest_writes_first_class_physics_subpanel_columns() -> None:
    facts = make_facts(
        log_transient_count=float(np.log1p(9.0)),
        onset_span_ratio=0.76,
        onset_interval_regularity=0.08,
        event_rate_hz=4.5,
        attack_rise_time_norm=0.03,
        pitch_confidence=0.82,
        loop_pitched_event_ratio=0.90,
        harmonic_energy_ratio=0.68,
        mid_ratio_500_2000hz=0.44,
        presence_ratio_2000_8000hz=0.12,
    )
    result = SortFileResult(
        source_path=TEMP_ROOT / "source.wav",
        placed_path=TEMP_ROOT / "out" / "source.wav",
        physics=AudioPhysics(TEMP_ROOT / "source.wav", feature_vector(), 4.0, "ok"),
        facts=facts,
        brain_votes=VoterResult(
            "brain",
            [
                CategoryGuess(
                    "Instruments/Guitar/Guitar Loops",
                    "Instruments/Guitar/Guitar Loops",
                    "Instruments",
                    0.1,
                    0.9,
                    1,
                    "test",
                )
            ],
        ),
        physics_votes=VoterResult(
            "physics",
            [
                CategoryGuess(
                    "Instruments/Guitar/Guitar Loops",
                    "Instruments/Guitar/Guitar Loops",
                    "Instruments",
                    0.1,
                    0.9,
                    1,
                    "test",
                )
            ],
        ),
        decision=ConsensusDecision(
            "Instruments/Guitar/Guitar Loops", "Instruments", "Instruments/Guitar/Guitar Loops", "test", "test"
        ),
    )
    row = manifest_row(result)

    for field in PHYSICS_SUBPANEL_MANIFEST_FIELDS:
        assert field in row, field
    assert row["physics_subpanels_json"]
    assert float(row["role_loop_score"]) >= 0.0


def test_new_category_coverage_panels_expose_drum_fx_texture_human_animal_scores() -> None:
    facts = make_facts(
        log_transient_count=float(np.log1p(18.0)),
        onset_span_ratio=0.82,
        onset_interval_regularity=0.10,
        event_rate_hz=5.0,
        attack_rise_time_norm=0.030,
        temporal_centroid_ratio=0.26,
        spectral_flatness_mean=0.32,
        spectral_entropy_mean=0.64,
        zcr_mean=0.20,
        mid_ratio_500_2000hz=0.40,
        presence_ratio_2000_8000hz=0.26,
        air_ratio_gt_8000hz=0.08,
        attack_noise_ratio=0.30,
        attack_high_ratio=0.24,
        body_noise_ratio=0.28,
        loop_percussive_event_ratio=0.72,
        loop_drumlike_frame_ratio=0.68,
        loop_noisy_event_ratio=0.34,
        noise_burst_duration_ms=160.0,
    )
    flat = facts.evidence["physics_subpanels"]["flat"]

    for key in [
        "drum_snare_source_score",
        "drum_clap_source_score",
        "drum_closed_hat_source_score",
        "drum_cymbal_source_score",
        "drum_tom_conga_source_score",
        "fx_glitch_stutter_score",
        "fx_impact_score",
        "texture_noise_static_score",
        "human_applause_crowd_score",
        "animal_bird_score",
    ]:
        assert key in flat, key
        assert 0.0 <= float(flat[key]) <= 1.0
    assert flat["drum_snare_source_score"] > 0.20
    assert flat["human_applause_crowd_score"] > 0.20


def test_compact_pitched_metal_percussion_survives_tonal_voice_guard() -> None:
    facts = make_facts(
        log_transient_count=float(np.log1p(1.0)),
        onset_span_ratio=0.10,
        onset_interval_regularity=0.90,
        event_rate_hz=0.4,
        attack_rise_time_norm=0.018,
        temporal_centroid_ratio=0.18,
        tail_energy_ratio=0.20,
        pitch_confidence=0.78,
        attack_pitch_confidence=0.78,
        body_pitch_confidence=0.74,
        f0_voiced_ratio=0.76,
        loop_pitched_event_ratio=0.82,
        loop_mean_event_pitch_confidence=0.78,
        loop_sustained_tonal_frame_ratio=0.12,
        loop_non_event_tonal_ratio=0.10,
        loop_percussive_event_ratio=0.04,
        loop_drumlike_frame_ratio=0.02,
        harmonic_energy_ratio=0.54,
        fundamental_dominance_ratio=0.36,
        inharmonicity=0.52,
        spectral_peak_count=10.0,
        spectral_peak_stability=0.44,
        spectral_flatness_mean=0.18,
        spectral_entropy_mean=0.42,
        mid_ratio_500_2000hz=0.36,
        presence_ratio_2000_8000hz=0.28,
        air_ratio_gt_8000hz=0.08,
        attack_high_ratio=0.24,
        tail_high_ratio=0.20,
        noise_burst_duration_ms=90.0,
    )
    flat = facts.evidence["physics_subpanels"]["flat"]

    assert flat["struck_percussion_guard_exception"] is True
    assert flat["pitched_metal_material_evidence"] is True
    assert flat["pitched_metal_percussion_score"] >= 0.54
    assert flat["drum_metallic_percussion_source_score"] >= 0.48
    assert flat["drum_hit_score"] > 0.34


def test_clean_voiced_solo_hit_still_gets_drum_panels_capped() -> None:
    facts = make_facts(
        log_transient_count=float(np.log1p(1.0)),
        onset_span_ratio=0.08,
        onset_interval_regularity=0.95,
        event_rate_hz=0.3,
        attack_rise_time_norm=0.07,
        temporal_centroid_ratio=0.30,
        tail_energy_ratio=0.62,
        pitch_confidence=0.88,
        attack_pitch_confidence=0.86,
        body_pitch_confidence=0.88,
        f0_voiced_ratio=0.92,
        loop_pitched_event_ratio=0.92,
        loop_mean_event_pitch_confidence=0.86,
        loop_sustained_tonal_frame_ratio=0.90,
        loop_non_event_tonal_ratio=0.88,
        loop_percussive_event_ratio=0.03,
        loop_drumlike_frame_ratio=0.02,
        harmonic_energy_ratio=0.72,
        fundamental_dominance_ratio=0.62,
        formant_like_peak_spacing=0.68,
        spectral_peak_count=5.0,
        spectral_peak_stability=0.28,
        spectral_flatness_mean=0.10,
        spectral_entropy_mean=0.32,
        mid_ratio_500_2000hz=0.36,
        presence_ratio_2000_8000hz=0.26,
        air_ratio_gt_8000hz=0.06,
        body_noise_ratio=0.24,
        tail_noise_ratio=0.18,
        noise_burst_duration_ms=40.0,
    )
    flat = facts.evidence["physics_subpanels"]["flat"]

    assert flat["tonal_voiced_non_drum_hit_guard"] is True
    assert flat["struck_percussion_guard_exception"] is False
    assert flat["pitched_metal_material_evidence"] is False
    assert flat["hand_drum_material_evidence"] is False
    assert flat["struck_wood_material_evidence"] is False
    assert flat["drum_hit_score"] <= 0.34
    assert flat["drum_metallic_percussion_source_score"] <= 0.34
    assert flat["drum_tom_conga_source_score"] <= 0.42


def test_texture_motion_panel_identifies_long_noisy_moving_bed_without_drums() -> None:
    facts = make_facts(
        log_transient_count=float(np.log1p(2.0)),
        onset_span_ratio=0.22,
        onset_interval_regularity=0.88,
        event_rate_hz=0.25,
        attack_rise_time_norm=0.42,
        temporal_centroid_ratio=0.60,
        tail_energy_ratio=0.86,
        spectral_flatness_mean=0.46,
        spectral_entropy_mean=0.78,
        zcr_mean=0.18,
        stereo_width=0.48,
        centroid_slope_norm=0.14,
        presence_ratio_2000_8000hz=0.18,
        air_ratio_gt_8000hz=0.12,
        body_noise_ratio=0.40,
        tail_noise_ratio=0.44,
    )
    flat = facts.evidence["physics_subpanels"]["flat"]

    assert flat["texture_bed_score"] > flat["drum_hit_score"]
    assert flat["fx_whoosh_sweep_score"] > 0.25
    assert flat["texture_wind_score"] > 0.25


def test_source_panels_live_outside_facts_module_boundary() -> None:
    from aaron_sound_sorter.domain.physics_subpanels import build_low_level_physics_subpanels

    panels = build_low_level_physics_subpanels(
        {
            "log_transient_count": float(np.log1p(1.0)),
            "attack_rise_time_norm": 0.02,
            "temporal_centroid_ratio": 0.18,
            "sub_bass_ratio_lt_150hz": 0.74,
            "bass_ratio_150_500hz": 0.16,
            "presence_ratio_2000_8000hz": 0.04,
            "pitch_confidence": 0.20,
            "low_peak_frequency_hz": 72.0,
            "sub_attack_time_ms": 18.0,
            "sub_decay_time_ms": 220.0,
            "kick_pitch_drop_cents": 220.0,
        }
    )
    flat = panels["flat"]

    assert flat["drum_kick_source_score"] >= flat["sustained_bass_score"]
    assert flat["low_kick_score"] > 0.35


def test_v31107_complete_panel_roadmap_exposes_current_brain_family_scores() -> None:
    facts = make_facts(
        log_transient_count=float(np.log1p(6.0)),
        onset_span_ratio=0.62,
        onset_interval_regularity=0.18,
        event_rate_hz=3.8,
        attack_rise_time_norm=0.04,
        temporal_centroid_ratio=0.22,
        tail_energy_ratio=0.34,
        pitch_confidence=0.74,
        f0_voiced_ratio=0.78,
        harmonic_energy_ratio=0.52,
        fundamental_dominance_ratio=0.44,
        spectral_flatness_mean=0.10,
        spectral_entropy_mean=0.36,
        mid_ratio_500_2000hz=0.48,
        presence_ratio_2000_8000hz=0.18,
        air_ratio_gt_8000hz=0.04,
        body_noise_ratio=0.12,
        loop_pitched_event_ratio=0.80,
        loop_sustained_tonal_frame_ratio=0.70,
        loop_non_event_tonal_ratio=0.66,
        loop_percussive_event_ratio=0.10,
        loop_drumlike_frame_ratio=0.08,
        stereo_width=0.24,
    )
    flat = facts.evidence["physics_subpanels"]["flat"]

    required = [
        "animal_cat_score",
        "animal_dog_score",
        "fx_door_foley_score",
        "fx_engine_machine_score",
        "fx_motor_machine_score",
        "fx_coin_object_score",
        "fx_key_object_score",
        "fx_boom_score",
        "fx_slam_score",
        "fx_sub_hit_score",
        "fx_alarm_score",
        "fx_siren_score",
        "bass_808_score",
        "bass_sub_score",
        "bass_synth_score",
        "bass_electric_score",
        "bass_upright_score",
        "guitar_acoustic_score",
        "guitar_electric_score",
        "guitar_nylon_score",
        "synth_lead_score",
        "synth_pad_score",
        "synth_chord_score",
        "brass_trumpet_score",
        "woodwind_flute_score",
        "woodwind_sax_score",
        "string_violin_score",
        "string_cello_score",
        "voice_choir_score",
    ]
    for key in required:
        assert key in flat, key
        assert 0.0 <= float(flat[key]) <= 1.0

    coverage = facts.evidence["physics_subpanels"].get("panel_coverage", {})
    assert coverage.get("coverage_status") == "complete_for_current_stage4_brain_labels"
    assert coverage.get("missing_category_panels") == []
    assert int(coverage.get("implemented_category_panel_count", 0)) == int(
        coverage.get("expected_category_panel_count", -1)
    )


def test_all_current_brain_category_panels_are_present() -> None:
    """Every active Drums/Instruments/FX brain label has a named panel score."""
    from aaron_sound_sorter.domain.physics_category_panels import ALL_CATEGORY_SPECS

    facts = make_facts(
        pitch_confidence=0.70,
        f0_voiced_ratio=0.65,
        harmonic_energy_ratio=0.45,
        spectral_flatness_mean=0.18,
        spectral_entropy_mean=0.42,
        log_transient_count=1.60,
        onset_span_ratio=0.50,
        tail_energy_ratio=0.32,
        presence_ratio_2000_8000hz=0.16,
        air_ratio_gt_8000hz=0.08,
        mid_ratio_500_2000hz=0.40,
        bass_ratio_150_500hz=0.20,
        sub_bass_ratio_lt_150hz=0.12,
    )
    subpanels = facts.evidence["physics_subpanels"]
    category_panels = subpanels["category_panels"]
    flat = category_panels["flat"]
    coverage = category_panels["coverage"]

    assert coverage["expected_category_panel_count"] == len(ALL_CATEGORY_SPECS)
    assert coverage["implemented_category_panel_count"] == len(ALL_CATEGORY_SPECS)
    assert coverage["missing_category_panels"] == []
    assert len(flat) == len(ALL_CATEGORY_SPECS)
    for spec in ALL_CATEGORY_SPECS:
        assert spec.score_key in flat, spec.category_path
        assert 0.0 <= float(flat[spec.score_key]) <= 1.0


def test_struck_wood_one_shot_panel_supports_claves_and_wood_blocks() -> None:
    """Dry wood-click evidence should have a Claves/Wood Blocks leaf witness."""
    facts = make_facts(
        log_transient_count=float(np.log1p(2.0)),
        onset_span_ratio=0.10,
        event_rate_hz=2.0,
        attack_rise_time_norm=0.004,
        temporal_centroid_ratio=0.10,
        tail_energy_ratio=0.12,
        spectral_flatness_mean=0.22,
        spectral_entropy_mean=0.46,
        mid_ratio_500_2000hz=0.42,
        presence_ratio_2000_8000hz=0.26,
        air_ratio_gt_8000hz=0.06,
        sub_bass_ratio_lt_150hz=0.02,
        bass_ratio_150_500hz=0.08,
        pitch_confidence=0.38,
        harmonic_energy_ratio=0.24,
        inharmonicity=0.22,
        log_crest=2.20,
        attack_noise_ratio=0.25,
        attack_high_ratio=0.26,
        noise_burst_duration_ms=80.0,
        zcr_mean=0.18,
        spectral_peak_stability=0.46,
    )
    flat = facts.evidence["physics_subpanels"]["flat"]

    clave_score = flat["drums_rims_and_sticks_claves_and_wood_blocks_one_shots_score"]

    assert clave_score >= flat["drums_rims_and_sticks_rimshot_one_shots_score"] - 0.02
    assert clave_score > flat["drums_rims_and_sticks_claves_and_wood_blocks_loops_score"]
    assert clave_score > flat["drums_hi_hats_closed_hat_one_shots_score"]


def test_pitched_repetition_subpanel_suppresses_weak_drum_loop_source() -> None:
    """Tonal repeated phrases should not become drum loops from repetition alone."""
    facts = make_facts(
        log_transient_count=float(np.log1p(14.0)),
        onset_span_ratio=0.88,
        onset_interval_regularity=0.20,
        event_rate_hz=2.0,
        attack_rise_time_norm=0.05,
        temporal_centroid_ratio=0.44,
        tail_energy_ratio=0.35,
        pitch_confidence=0.78,
        f0_voiced_ratio=0.70,
        body_pitch_confidence=0.76,
        loop_pitched_event_ratio=0.82,
        loop_mean_event_pitch_confidence=0.78,
        loop_sustained_tonal_frame_ratio=0.70,
        loop_non_event_tonal_ratio=0.66,
        loop_tonal_to_percussive_balance=0.72,
        loop_percussive_event_ratio=0.18,
        loop_drumlike_frame_ratio=0.08,
        loop_mean_event_low_ratio=0.18,
        loop_mean_event_high_ratio=0.20,
        loop_mean_event_noise_ratio=0.12,
        harmonic_energy_ratio=0.70,
        harmonic_to_noise_ratio=0.62,
        harmonic_peak_count=7.0,
        spectral_peak_stability=0.38,
        fundamental_dominance_ratio=0.52,
        spectral_flatness_mean=0.07,
        spectral_entropy_mean=0.28,
        mid_ratio_500_2000hz=0.42,
        presence_ratio_2000_8000hz=0.20,
        sub_bass_ratio_lt_150hz=0.08,
        bass_ratio_150_500hz=0.12,
        attack_high_ratio=0.18,
        attack_zcr=0.10,
    )
    flat = facts.evidence["physics_subpanels"]["flat"]

    assert flat["pitched_repetition_drum_decoy"] is True
    assert flat["pitched_repetition_phrase_score"] >= 0.70
    assert flat["onset_pitched_onset_score"] > flat["onset_percussive_onset_score"]
    assert flat["drum_loop_source_score"] <= 0.22


def test_real_drum_loop_subpanel_survives_pitched_repetition_guard() -> None:
    """Percussive distributed loops should remain strong drum-loop candidates."""
    facts = make_facts(
        log_transient_count=float(np.log1p(20.0)),
        onset_span_ratio=0.90,
        onset_interval_regularity=0.16,
        event_rate_hz=3.0,
        attack_rise_time_norm=0.025,
        temporal_centroid_ratio=0.50,
        tail_energy_ratio=0.28,
        pitch_confidence=0.20,
        f0_voiced_ratio=0.12,
        loop_pitched_event_ratio=0.12,
        loop_mean_event_pitch_confidence=0.10,
        loop_sustained_tonal_frame_ratio=0.10,
        loop_non_event_tonal_ratio=0.08,
        loop_tonal_to_percussive_balance=0.08,
        loop_percussive_event_ratio=0.72,
        loop_drumlike_frame_ratio=0.66,
        loop_mean_event_low_ratio=0.40,
        loop_mean_event_high_ratio=0.34,
        loop_mean_event_noise_ratio=0.50,
        harmonic_energy_ratio=0.18,
        harmonic_to_noise_ratio=0.15,
        spectral_flatness_mean=0.42,
        spectral_entropy_mean=0.68,
        body_noise_ratio=0.50,
        tail_noise_ratio=0.36,
        mid_ratio_500_2000hz=0.24,
        presence_ratio_2000_8000hz=0.32,
        sub_bass_ratio_lt_150hz=0.16,
        bass_ratio_150_500hz=0.20,
        attack_high_ratio=0.34,
        attack_zcr=0.28,
    )
    flat = facts.evidence["physics_subpanels"]["flat"]

    assert flat["pitched_repetition_drum_decoy"] is False
    assert flat["drum_loop_source_score"] >= 0.65
    assert flat["rhythmic_break_loop_score"] >= 0.75
