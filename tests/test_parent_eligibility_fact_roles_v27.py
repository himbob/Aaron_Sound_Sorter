"""Unit tests for parent role inference from measured facts, independent of brain."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.eligibility import infer_parent_eligibility


def facts(
    duration: float,
    *,
    shape: str = "",
    shape_conf: float = 1.0,
    roles: dict[str, float] | None = None,
    **features: float,
) -> SharedAudioFacts:
    evidence = {
        "duration_sec": duration,
        "event_count_estimate": features.pop("event_count_estimate", 0.0),
        "event_rate_hz": features.pop("event_rate_hz", 0.0),
        "onset_span_ratio": features.pop("onset_span_ratio", 0.0),
        "temporal_centroid_ratio": features.pop("temporal_centroid_ratio", 0.0),
        "attack_rise_time_norm": features.pop("attack_rise_time_norm", 0.0),
        "shape_vote": {"primary_shape": shape, "confidence": shape_conf},
        "measured_roles": {
            **(roles or {}),
            "evidence": {"formant_light_voice_identity": features.pop("formant_light_voice_identity", 0.0)},
        },
        "feature_values_by_name": dict(features),
    }
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=duration >= 2.0,
        is_single_event_like=features.get("event_count_estimate", 0.0) <= 1,
        is_short_hit_like=duration <= 1.25,
        is_long=duration >= 4.0,
        evidence=evidence,
        feature_values_by_name=evidence["feature_values_by_name"],
    )


def test_low_body_drum_loop_with_bass_tail_is_drum_loop_not_bass_loop() -> None:
    role = infer_parent_eligibility(
        facts(
            6.02,
            shape="bass_phrase",
            roles={"bass_loop": 0.79},
            event_count_estimate=15,
            event_rate_hz=2.52,
            onset_span_ratio=0.82,
            attack_rise_time_norm=0.0004,
            temporal_centroid_ratio=0.42,
            pitch_confidence=0.94,
            f0_voiced_ratio=0.36,
            formant_like_peak_spacing=0.0,
            sub_bass_ratio_lt_150hz=0.83,
            bass_ratio_150_500hz=0.12,
            presence_ratio_2000_8000hz=0.016,
            air_ratio_gt_8000hz=0.001,
            mid_ratio_500_2000hz=0.033,
            loop_percussive_event_ratio=0.267,
            loop_drumlike_frame_ratio=0.063,
            loop_pitched_event_ratio=0.733,
            loop_sustained_tonal_frame_ratio=0.708,
            loop_non_event_tonal_ratio=0.722,
            spectral_flatness_mean=0.281,
            log_crest=1.818,
        )
    )
    assert role.role_name == "drum_loop"
    assert role.broad_folder_path == "Drums/Drum Loops/Loops"


def test_noisy_texture_bed_is_fx_texture_not_bright_drum_loop() -> None:
    role = infer_parent_eligibility(
        facts(
            2.8,
            shape="texture_bed",
            shape_conf=0.92,
            roles={"bright_drum_loop": 0.82, "percussive_drum_loop": 0.80},
            event_count_estimate=15,
            event_rate_hz=5.4,
            onset_span_ratio=0.48,
            attack_rise_time_norm=0.21,
            temporal_centroid_ratio=0.32,
            pitch_confidence=0.10,
            f0_voiced_ratio=0.18,
            formant_like_peak_spacing=0.0,
            sub_bass_ratio_lt_150hz=0.02,
            bass_ratio_150_500hz=0.08,
            mid_ratio_500_2000hz=0.54,
            presence_ratio_2000_8000hz=0.24,
            air_ratio_gt_8000hz=0.08,
            loop_percussive_event_ratio=1.0,
            loop_drumlike_frame_ratio=0.82,
            loop_pitched_event_ratio=0.0,
            loop_sustained_tonal_frame_ratio=0.10,
            loop_non_event_tonal_ratio=0.18,
            spectral_flatness_mean=0.60,
            spectral_entropy_mean=0.78,
            body_noise_ratio=0.62,
            tail_noise_ratio=0.58,
            loop_noisy_event_ratio=0.66,
            log_crest=1.85,
        )
    )
    assert role.role_name == "fx_texture_bed"
    assert role.allowed_top_families == ("FX", "_TO_REVIEW")
    assert role.broad_folder_path.startswith("FX/Textures/")


def test_voice_shout_with_formant_light_identity_is_vocal_not_percussion() -> None:
    role = infer_parent_eligibility(
        facts(
            0.77,
            shape="hit_with_tail",
            event_count_estimate=1,
            event_rate_hz=1.44,
            onset_span_ratio=0.0,
            attack_rise_time_norm=0.135,
            temporal_centroid_ratio=0.144,
            pitch_confidence=0.42,
            f0_voiced_ratio=0.91,
            formant_like_peak_spacing=0.0,
            formant_light_voice_identity=0.94,
            sub_bass_ratio_lt_150hz=0.0002,
            bass_ratio_150_500hz=0.019,
            mid_ratio_500_2000hz=0.453,
            presence_ratio_2000_8000hz=0.524,
            air_ratio_gt_8000hz=0.004,
            loop_percussive_event_ratio=0.0,
            loop_drumlike_frame_ratio=0.0,
            loop_pitched_event_ratio=1.0,
            loop_sustained_tonal_frame_ratio=1.0,
            loop_non_event_tonal_ratio=1.0,
            spectral_flatness_mean=0.416,
            log_crest=2.27,
        )
    )
    assert role.role_name == "vocal_one_shot"
    assert role.broad_folder_path == "FX/Human and Voice FX"


def test_clean_tonal_stab_with_light_formant_proxy_is_not_vocal() -> None:
    role = infer_parent_eligibility(
        facts(
            1.53,
            shape="hit_with_tail",
            shape_conf=0.87,
            roles={"voiced_one_shot": 0.91, "pitched_music_phrase": 0.67},
            event_count_estimate=2,
            event_rate_hz=1.37,
            onset_span_ratio=0.03,
            attack_rise_time_norm=0.036,
            temporal_centroid_ratio=0.09,
            pitch_confidence=0.96,
            f0_voiced_ratio=1.0,
            formant_like_peak_spacing=0.0,
            formant_light_voice_identity=1.0,
            sub_bass_ratio_lt_150hz=0.0,
            bass_ratio_150_500hz=0.021,
            mid_ratio_500_2000hz=0.98,
            presence_ratio_2000_8000hz=0.0,
            air_ratio_gt_8000hz=0.0,
            loop_percussive_event_ratio=0.0,
            loop_drumlike_frame_ratio=0.0,
            loop_pitched_event_ratio=1.0,
            loop_sustained_tonal_frame_ratio=1.0,
            loop_non_event_tonal_ratio=1.0,
            spectral_flatness_mean=0.167,
            log_crest=1.86,
            voice_score=0.27,
            human_spoken_voice_score=0.52,
            human_breath_mouth_score=0.29,
            fx_formant_score=0.32,
        )
    )
    assert role.role_name == "short_pitched_instrument_hit"
    assert role.broad_folder_path == "_TO_REVIEW/Measured Role Conflict"
    assert "Voice" in role.blocked_path_fragments


def test_short_clean_tonal_blip_is_fx_one_shot_not_generic_instrument_review() -> None:
    role = infer_parent_eligibility(
        facts(
            0.48,
            shape="solo_phrase",
            shape_conf=0.88,
            roles={"voiced_one_shot": 0.24, "pitched_music_phrase": 0.76},
            event_count_estimate=2,
            event_rate_hz=5.07,
            onset_span_ratio=0.38,
            attack_rise_time_norm=0.004,
            temporal_centroid_ratio=0.14,
            pitch_confidence=0.83,
            f0_voiced_ratio=1.0,
            formant_like_peak_spacing=0.0,
            sub_bass_ratio_lt_150hz=0.0,
            bass_ratio_150_500hz=0.33,
            mid_ratio_500_2000hz=0.66,
            presence_ratio_2000_8000hz=0.006,
            air_ratio_gt_8000hz=0.0,
            loop_percussive_event_ratio=0.0,
            loop_drumlike_frame_ratio=0.0,
            loop_pitched_event_ratio=1.0,
            loop_sustained_tonal_frame_ratio=1.0,
            loop_non_event_tonal_ratio=1.0,
            spectral_flatness_mean=0.047,
            log_crest=2.10,
            fx_blip_beep_score=0.86,
            clean_tone_score=0.94,
            onset_pitched_onset_score=0.93,
            onset_percussive_onset_score=0.57,
            drum_hit_score=0.34,
            drum_kick_source_score=0.42,
            drum_snare_source_score=0.38,
            drum_clap_source_score=0.37,
            drum_tom_conga_source_score=0.39,
            drum_rim_stick_source_score=0.43,
            drum_cymbal_source_score=0.26,
            drum_metallic_percussion_source_score=0.31,
        )
    )
    assert role.role_name == "short_designed_tonal_fx_hit"
    assert role.allowed_top_families == ("FX", "_TO_REVIEW")
    assert role.broad_folder_path == "FX/Designed Noise FX/Beep/One Shots"


def test_processed_reverb_voice_shot_with_human_articulation_stays_vocal() -> None:
    role = infer_parent_eligibility(
        facts(
            5.47,
            shape="echo_tail_hit",
            shape_conf=0.74,
            roles={"voiced_one_shot": 0.82, "pitched_music_phrase": 0.87},
            event_count_estimate=8,
            event_rate_hz=1.49,
            onset_span_ratio=0.15,
            attack_rise_time_norm=0.056,
            temporal_centroid_ratio=0.12,
            pitch_confidence=0.86,
            f0_voiced_ratio=0.94,
            formant_like_peak_spacing=0.0,
            formant_light_voice_identity=0.71,
            sub_bass_ratio_lt_150hz=0.02,
            bass_ratio_150_500hz=0.22,
            mid_ratio_500_2000hz=0.31,
            presence_ratio_2000_8000hz=0.42,
            air_ratio_gt_8000hz=0.03,
            loop_percussive_event_ratio=0.0,
            loop_drumlike_frame_ratio=0.0,
            loop_pitched_event_ratio=1.0,
            loop_sustained_tonal_frame_ratio=0.88,
            loop_non_event_tonal_ratio=0.86,
            spectral_flatness_mean=0.114,
            log_crest=2.75,
            voice_score=0.53,
            human_spoken_voice_score=0.69,
            human_breath_mouth_score=0.54,
            fx_formant_score=0.45,
        )
    )
    assert role.role_name == "vocal_one_shot"
    assert role.broad_folder_path == "FX/Human and Voice FX"


def test_sax_reed_loop_is_reed_instrument_not_voice_or_drum() -> None:
    role = infer_parent_eligibility(
        facts(
            12.0,
            shape="vocal_phrase",
            event_count_estimate=42,
            event_rate_hz=3.53,
            onset_span_ratio=0.98,
            attack_rise_time_norm=0.89,
            temporal_centroid_ratio=0.60,
            pitch_confidence=0.98,
            f0_voiced_ratio=1.0,
            formant_like_peak_spacing=4.0,
            sub_bass_ratio_lt_150hz=0.0002,
            bass_ratio_150_500hz=0.656,
            mid_ratio_500_2000hz=0.302,
            presence_ratio_2000_8000hz=0.041,
            air_ratio_gt_8000hz=0.0002,
            loop_percussive_event_ratio=0.0,
            loop_drumlike_frame_ratio=0.0,
            loop_pitched_event_ratio=1.0,
            loop_sustained_tonal_frame_ratio=1.0,
            loop_non_event_tonal_ratio=1.0,
            spectral_flatness_mean=0.061,
            log_crest=2.14,
        )
    )
    assert role.role_name in {"pitched_reed_or_instrument_loop", "pitched_reed_or_instrument_phrase"}
    assert role.broad_folder_path == "Instruments/Instrument Loops/Loops"


def test_bright_airy_reed_loop_keeps_sax_depth_despite_voice_like_formants() -> None:
    role = infer_parent_eligibility(
        facts(
            10.6,
            shape="pitched_repetition_phrase",
            shape_conf=0.90,
            roles={"pitched_music_loop": 0.78},
            event_count_estimate=20,
            event_rate_hz=1.89,
            onset_span_ratio=0.91,
            attack_rise_time_norm=0.0014,
            temporal_centroid_ratio=0.46,
            pitch_confidence=0.82,
            f0_voiced_ratio=0.99,
            formant_like_peak_spacing=1.2,
            formant_light_voice_identity=0.72,
            sub_bass_ratio_lt_150hz=0.02,
            bass_ratio_150_500hz=0.12,
            mid_ratio_500_2000hz=0.25,
            presence_ratio_2000_8000hz=0.52,
            air_ratio_gt_8000hz=0.08,
            loop_percussive_event_ratio=0.0,
            loop_drumlike_frame_ratio=0.0,
            loop_pitched_event_ratio=1.0,
            loop_sustained_tonal_frame_ratio=1.0,
            loop_non_event_tonal_ratio=1.0,
            loop_mean_event_low_ratio=0.15,
            loop_mean_event_mid_ratio=0.25,
            loop_mean_event_high_ratio=0.61,
            spectral_flatness_mean=0.27,
            spectral_entropy_mean=0.60,
            reed_reed_noise_score=0.83,
            reed_formant_envelope_score=0.82,
            reed_breath_attack_score=0.70,
            onset_pitched_onset_score=0.79,
        )
    )
    assert role.role_name == "pitched_reed_or_instrument_loop"
    assert role.broad_folder_path == "Instruments/Woodwinds/Saxophone/Loops"


def test_clean_mid_band_keys_loop_keeps_keys_depth() -> None:
    role = infer_parent_eligibility(
        facts(
            11.9,
            shape="pitched_repetition_phrase",
            shape_conf=0.98,
            roles={"pitched_music_loop": 0.88},
            event_count_estimate=20,
            event_rate_hz=1.68,
            onset_span_ratio=0.99,
            attack_rise_time_norm=0.086,
            temporal_centroid_ratio=0.49,
            pitch_confidence=0.97,
            f0_voiced_ratio=1.0,
            sub_bass_ratio_lt_150hz=0.01,
            bass_ratio_150_500hz=0.06,
            mid_ratio_500_2000hz=0.64,
            presence_ratio_2000_8000hz=0.13,
            air_ratio_gt_8000hz=0.01,
            loop_percussive_event_ratio=0.0,
            loop_drumlike_frame_ratio=0.0,
            loop_pitched_event_ratio=1.0,
            loop_sustained_tonal_frame_ratio=1.0,
            loop_non_event_tonal_ratio=1.0,
            loop_mean_event_low_ratio=0.07,
            loop_mean_event_mid_ratio=0.79,
            loop_mean_event_high_ratio=0.14,
            spectral_flatness_mean=0.046,
            struck_keys_score=0.58,
            keys_tonal_decay_score=0.80,
            keys_chord_density_score=0.54,
        )
    )
    assert role.role_name == "keys_loop"
    assert role.broad_folder_path == "Instruments/Keys/Electric Piano/Loops"


def test_low_heavy_synthetic_loop_keeps_synth_depth() -> None:
    role = infer_parent_eligibility(
        facts(
            9.7,
            shape="pitched_repetition_phrase",
            shape_conf=0.85,
            roles={"pitched_music_loop": 0.82},
            event_count_estimate=18,
            event_rate_hz=1.87,
            onset_span_ratio=0.90,
            attack_rise_time_norm=0.096,
            temporal_centroid_ratio=0.40,
            pitch_confidence=0.71,
            f0_voiced_ratio=0.15,
            sub_bass_ratio_lt_150hz=0.42,
            bass_ratio_150_500hz=0.38,
            mid_ratio_500_2000hz=0.10,
            presence_ratio_2000_8000hz=0.04,
            air_ratio_gt_8000hz=0.01,
            loop_percussive_event_ratio=0.0,
            loop_drumlike_frame_ratio=0.0,
            loop_pitched_event_ratio=1.0,
            loop_sustained_tonal_frame_ratio=1.0,
            loop_non_event_tonal_ratio=1.0,
            loop_mean_event_low_ratio=0.84,
            loop_mean_event_mid_ratio=0.11,
            loop_mean_event_high_ratio=0.05,
            spectral_flatness_mean=0.30,
            synth_tonal_source_score=0.60,
            struck_keys_score=0.44,
            keys_tonal_decay_score=0.69,
        )
    )
    assert role.role_name == "synth_loop"
    assert role.broad_folder_path == "Instruments/Synths/Synth Loops"
