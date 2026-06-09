from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.voters.physics_instrument_layer import PhysicsInstrumentLayer
from aaron_sound_sorter.voters.third_party_instrument_support import third_party_instrument_support


def facts(evidence: dict[str, object], feature_values: dict[str, float] | None = None) -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence=evidence,
        feature_values_by_name=feature_values or {},
    )


def test_third_party_support_separates_clean_synth_from_reed_or_voice() -> None:
    f = facts(
        {
            "librosa_tonal_confidence": 0.86,
            "librosa_harmonic_energy_ratio": 0.84,
            "librosa_percussive_confidence": 0.10,
            "librosa_loop_confidence": 0.66,
            "librosa_onset_event_count": 7.0,
            "librosa_onset_density_hz": 2.0,
            "librosa_chroma_peak_mean": 0.48,
            "librosa_chroma_entropy_norm": 0.42,
            "librosa_chroma_std": 0.24,
            "librosa_chroma_frame_change_mean": 0.025,
            "librosa_spectral_contrast_mean": 16.0,
            "librosa_spectral_contrast_std": 1.4,
            "librosa_spectral_centroid_mean": 1600.0,
            "librosa_spectral_centroid_std": 120.0,
            "librosa_spectral_rolloff85_mean": 3600.0,
            "librosa_spectral_flatness_mean": 0.025,
            "librosa_zero_crossing_rate_mean": 0.035,
            "librosa_spectral_bandwidth_mean": 1800.0,
            "librosa_mfcc_delta_std": 3.0,
            "librosa_mfcc_delta2_std": 3.0,
            "librosa_yin_voiced_ratio": 0.82,
            "librosa_yin_f0_median_hz": 440.0,
            "librosa_yin_f0_stability": 0.86,
            "librosa_yin_f0_motion_cents": 8.0,
        }
    )

    support = third_party_instrument_support(f)

    assert support["api_synth_support"] >= 0.50
    assert support["api_synth_support"] > support["api_voice_support"]
    assert support["api_synth_support"] >= support["api_reed_wind_support"]


def test_third_party_support_identifies_breathy_moving_voice_texture() -> None:
    f = facts(
        {
            "librosa_tonal_confidence": 0.68,
            "librosa_harmonic_energy_ratio": 0.58,
            "librosa_percussive_confidence": 0.14,
            "librosa_loop_confidence": 0.48,
            "librosa_onset_event_count": 6.0,
            "librosa_onset_density_hz": 1.5,
            "librosa_chroma_peak_mean": 0.30,
            "librosa_chroma_entropy_norm": 0.62,
            "librosa_chroma_std": 0.18,
            "librosa_chroma_frame_change_mean": 0.12,
            "librosa_spectral_contrast_mean": 20.0,
            "librosa_spectral_contrast_std": 4.0,
            "librosa_spectral_centroid_mean": 2100.0,
            "librosa_spectral_centroid_std": 900.0,
            "librosa_spectral_rolloff85_mean": 5200.0,
            "librosa_spectral_flatness_mean": 0.18,
            "librosa_zero_crossing_rate_mean": 0.12,
            "librosa_spectral_bandwidth_mean": 2800.0,
            "librosa_mfcc_delta_std": 18.0,
            "librosa_mfcc_delta2_std": 16.0,
            "librosa_yin_voiced_ratio": 0.78,
            "librosa_yin_f0_median_hz": 240.0,
            "librosa_yin_f0_stability": 0.45,
            "librosa_yin_f0_motion_cents": 140.0,
        }
    )

    support = third_party_instrument_support(f)

    assert support["api_voice_support"] >= 0.48
    assert support["api_voice_support"] > support["api_synth_support"]


def test_instrument_layer_exports_third_party_branch_witnesses() -> None:
    f = facts(
        {
            "shape_vote": {"primary_shape": "sustained_pad", "confidence": 0.82},
            "measured_roles": {"pitched_music_loop": 0.70, "primary_roles": ["pitched_music_loop"]},
            "librosa_tonal_confidence": 0.84,
            "librosa_harmonic_energy_ratio": 0.80,
            "librosa_percussive_confidence": 0.08,
            "librosa_loop_confidence": 0.64,
            "librosa_onset_event_count": 5.0,
            "librosa_chroma_peak_mean": 0.46,
            "librosa_chroma_entropy_norm": 0.42,
            "librosa_chroma_std": 0.22,
            "librosa_chroma_frame_change_mean": 0.02,
            "librosa_spectral_contrast_mean": 15.0,
            "librosa_spectral_contrast_std": 1.2,
            "librosa_spectral_centroid_mean": 1500.0,
            "librosa_spectral_centroid_std": 100.0,
            "librosa_spectral_rolloff85_mean": 3300.0,
            "librosa_spectral_flatness_mean": 0.025,
            "librosa_zero_crossing_rate_mean": 0.030,
            "librosa_spectral_bandwidth_mean": 1600.0,
            "librosa_mfcc_delta_std": 3.0,
            "librosa_mfcc_delta2_std": 2.5,
            "librosa_yin_voiced_ratio": 0.82,
            "librosa_yin_f0_median_hz": 440.0,
            "librosa_yin_f0_stability": 0.82,
            "librosa_yin_f0_motion_cents": 8.0,
        },
        {
            "pitch_confidence": 0.74,
            "f0_voiced_ratio": 0.74,
            "f0_median_hz": 440.0,
            "loop_pitched_event_ratio": 0.82,
            "loop_sustained_tonal_frame_ratio": 0.82,
            "loop_non_event_tonal_ratio": 0.78,
            "loop_percussive_event_ratio": 0.04,
            "loop_drumlike_frame_ratio": 0.04,
            "harmonic_energy_ratio": 0.72,
            "fundamental_dominance_ratio": 0.60,
            "spectral_flatness_mean": 0.03,
            "spectral_entropy_mean": 0.24,
            "attack_rise_time_norm": 0.18,
            "tail_energy_ratio": 0.60,
            "onset_span_ratio": 0.66,
            "log_transient_count": 1.95,
            "mid_ratio_500_2000hz": 0.56,
            "presence_ratio_2000_8000hz": 0.08,
            "air_ratio_gt_8000hz": 0.03,
            "sub_bass_ratio_lt_150hz": 0.02,
            "bass_ratio_150_500hz": 0.14,
        },
    )

    branch, confidence, evidence = PhysicsInstrumentLayer().decide(f)

    assert evidence["third_party_api_synth_support"] >= 0.50
    assert branch in {"Synth", "KeysPiano", "MixedInstrument"}
    assert confidence > 0.0


def test_third_party_support_separates_bowed_string_from_reed_and_synth() -> None:
    f = facts(
        {
            "duration_sec": 2.8,
            "librosa_tonal_confidence": 0.74,
            "librosa_harmonic_energy_ratio": 0.70,
            "librosa_percussive_confidence": 0.06,
            "librosa_loop_confidence": 0.30,
            "librosa_onset_event_count": 1.0,
            "librosa_onset_density_hz": 0.35,
            "librosa_chroma_peak_mean": 0.38,
            "librosa_chroma_entropy_norm": 0.54,
            "librosa_chroma_std": 0.16,
            "librosa_chroma_frame_change_mean": 0.055,
            "librosa_spectral_contrast_mean": 18.0,
            "librosa_spectral_contrast_std": 3.0,
            "librosa_spectral_centroid_mean": 2300.0,
            "librosa_spectral_centroid_std": 850.0,
            "librosa_spectral_centroid_slope_norm": 0.04,
            "librosa_spectral_rolloff85_mean": 5200.0,
            "librosa_spectral_flatness_mean": 0.10,
            "librosa_zero_crossing_rate_mean": 0.07,
            "librosa_zero_crossing_rate_std": 0.03,
            "librosa_spectral_bandwidth_mean": 2500.0,
            "librosa_spectral_bandwidth_std": 900.0,
            "librosa_rms_std": 0.12,
            "librosa_mfcc_delta_std": 12.0,
            "librosa_mfcc_delta2_std": 10.0,
            "librosa_yin_voiced_ratio": 0.86,
            "librosa_yin_f0_median_hz": 660.0,
            "librosa_yin_f0_stability": 0.64,
            "librosa_yin_f0_motion_cents": 80.0,
        }
    )

    support = third_party_instrument_support(f)

    assert support["api_bowed_string_support"] >= 0.48
    assert support["api_bowed_string_support"] > support["api_reed_wind_support"]
    assert support["api_bowed_string_support"] > support["api_synth_support"]
    assert support["api_smooth_bow_texture"] >= 0.45


def test_third_party_support_separates_reed_wind_from_bowed_string() -> None:
    f = facts(
        {
            "duration_sec": 2.2,
            "librosa_tonal_confidence": 0.70,
            "librosa_harmonic_energy_ratio": 0.62,
            "librosa_percussive_confidence": 0.08,
            "librosa_loop_confidence": 0.46,
            "librosa_onset_event_count": 4.0,
            "librosa_onset_density_hz": 1.3,
            "librosa_chroma_peak_mean": 0.34,
            "librosa_chroma_entropy_norm": 0.58,
            "librosa_chroma_std": 0.18,
            "librosa_chroma_frame_change_mean": 0.095,
            "librosa_spectral_contrast_mean": 28.0,
            "librosa_spectral_contrast_std": 6.5,
            "librosa_spectral_centroid_mean": 1850.0,
            "librosa_spectral_centroid_std": 1200.0,
            "librosa_spectral_centroid_slope_norm": 0.07,
            "librosa_spectral_rolloff85_mean": 4600.0,
            "librosa_spectral_flatness_mean": 0.16,
            "librosa_zero_crossing_rate_mean": 0.11,
            "librosa_zero_crossing_rate_std": 0.08,
            "librosa_spectral_bandwidth_mean": 2900.0,
            "librosa_spectral_bandwidth_std": 1100.0,
            "librosa_rms_std": 0.16,
            "librosa_mfcc_delta_std": 19.0,
            "librosa_mfcc_delta2_std": 14.0,
            "librosa_yin_voiced_ratio": 0.78,
            "librosa_yin_f0_median_hz": 330.0,
            "librosa_yin_f0_stability": 0.52,
            "librosa_yin_f0_motion_cents": 115.0,
        }
    )

    support = third_party_instrument_support(f)

    assert support["api_reed_wind_support"] >= 0.48
    assert support["api_reed_wind_support"] > support["api_bowed_string_support"]
    assert support["api_reed_wind_support"] > support["api_synth_support"]
    assert support["api_breath_noise_texture"] >= 0.45


def test_third_party_support_keeps_short_clean_beep_out_of_synth_identity() -> None:
    f = facts(
        {
            "duration_sec": 0.22,
            "librosa_tonal_confidence": 0.92,
            "librosa_harmonic_energy_ratio": 0.90,
            "librosa_percussive_confidence": 0.08,
            "librosa_loop_confidence": 0.02,
            "librosa_onset_event_count": 1.0,
            "librosa_onset_density_hz": 4.5,
            "librosa_chroma_peak_mean": 0.52,
            "librosa_chroma_entropy_norm": 0.36,
            "librosa_chroma_std": 0.20,
            "librosa_chroma_frame_change_mean": 0.008,
            "librosa_spectral_contrast_mean": 20.0,
            "librosa_spectral_contrast_std": 1.0,
            "librosa_spectral_centroid_mean": 1800.0,
            "librosa_spectral_centroid_std": 80.0,
            "librosa_spectral_centroid_slope_norm": 0.0,
            "librosa_spectral_rolloff85_mean": 3600.0,
            "librosa_spectral_flatness_mean": 0.018,
            "librosa_zero_crossing_rate_mean": 0.025,
            "librosa_zero_crossing_rate_std": 0.004,
            "librosa_spectral_bandwidth_mean": 1200.0,
            "librosa_spectral_bandwidth_std": 90.0,
            "librosa_rms_std": 0.06,
            "librosa_mfcc_delta_std": 2.0,
            "librosa_mfcc_delta2_std": 1.5,
            "librosa_yin_voiced_ratio": 0.90,
            "librosa_yin_f0_median_hz": 880.0,
            "librosa_yin_f0_stability": 0.92,
            "librosa_yin_f0_motion_cents": 3.0,
        }
    )

    support = third_party_instrument_support(f)

    assert support["api_short_clean_tonal_event"] >= 0.55
    assert support["api_synth_support"] < 0.58
    assert support["api_clean_tonal_source"] >= 0.70


def test_third_party_support_plucked_guitar_beats_bowed_string_for_eventful_pluck() -> None:
    f = facts(
        {
            "duration_sec": 1.8,
            "librosa_tonal_confidence": 0.76,
            "librosa_harmonic_energy_ratio": 0.72,
            "librosa_percussive_confidence": 0.34,
            "librosa_loop_confidence": 0.52,
            "librosa_onset_event_count": 7.0,
            "librosa_onset_density_hz": 3.8,
            "librosa_chroma_peak_mean": 0.36,
            "librosa_chroma_entropy_norm": 0.56,
            "librosa_chroma_std": 0.22,
            "librosa_chroma_frame_change_mean": 0.11,
            "librosa_spectral_contrast_mean": 24.0,
            "librosa_spectral_contrast_std": 4.0,
            "librosa_spectral_centroid_mean": 2200.0,
            "librosa_spectral_centroid_std": 650.0,
            "librosa_spectral_centroid_slope_norm": 0.03,
            "librosa_spectral_rolloff85_mean": 5100.0,
            "librosa_spectral_flatness_mean": 0.075,
            "librosa_zero_crossing_rate_mean": 0.065,
            "librosa_zero_crossing_rate_std": 0.025,
            "librosa_spectral_bandwidth_mean": 2300.0,
            "librosa_spectral_bandwidth_std": 700.0,
            "librosa_rms_std": 0.10,
            "librosa_mfcc_delta_std": 15.0,
            "librosa_mfcc_delta2_std": 10.0,
            "librosa_yin_voiced_ratio": 0.42,
            "librosa_yin_f0_median_hz": 196.0,
            "librosa_yin_f0_stability": 0.38,
            "librosa_yin_f0_motion_cents": 55.0,
        }
    )

    support = third_party_instrument_support(f)

    assert support["api_plucked_string_support"] >= 0.42
    assert support["api_plucked_string_support"] > support["api_bowed_string_support"]
    assert support["api_plucked_string_support"] > support["api_voice_support"]
