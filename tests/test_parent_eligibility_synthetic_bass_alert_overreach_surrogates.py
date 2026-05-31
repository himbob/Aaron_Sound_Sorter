"""Synthetic surrogate regressions for bass-loop and alert/siren overreach.

These tests encode measured physics from real problem files without storing WAVs:

* Rhodes/keys-like low-mid tonal loop should not become a Bass Loop just because
  most energy is in 150-500 Hz.
* Bright synth lead should not become FX/Alarm just because it is repeated,
  pitched, and formant-like.
* Real sub-foundation bass loops must still keep a strong bass_loop role.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.domain.roles import measured_roles_from_features
from aaron_sound_sorter.engine.eligibility import infer_parent_eligibility


def facts(
    duration: float, *, shape: str = "bass_phrase", shape_confidence: float = 1.0, **features: float
) -> SharedAudioFacts:
    roles = measured_roles_from_features(
        is_loop_like=duration >= 2.0,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=duration >= 4.0,
        feature_values=features,
    ).to_evidence()
    evidence = {
        "duration_sec": duration,
        "event_count_estimate": features.get("event_count_estimate", 0.0),
        "event_rate_hz": features.get("event_rate_hz", 0.0),
        "onset_span_ratio": features.get("onset_span_ratio", 0.0),
        "attack_rise_time_norm": features.get("attack_rise_time_norm", 0.0),
        "temporal_centroid_ratio": features.get("temporal_centroid_ratio", 0.0),
        "shape_vote": {"primary_shape": shape, "confidence": shape_confidence},
        "measured_roles": roles,
        "feature_values_by_name": dict(features),
    }
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=duration >= 2.0,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=duration >= 4.0,
        evidence=evidence,
        feature_values_by_name=evidence["feature_values_by_name"],
    )


def loop_base(**overrides: float) -> dict[str, float]:
    data = {
        "event_count_estimate": 16.0,
        "event_rate_hz": 1.50,
        "onset_span_ratio": 0.90,
        "attack_rise_time_norm": 0.02,
        "temporal_centroid_ratio": 0.40,
        "tail_energy_ratio": 0.10,
        "log_transient_count": 4.0,
        "loop_pitched_event_ratio": 1.0,
        "loop_sustained_tonal_frame_ratio": 1.0,
        "loop_non_event_tonal_ratio": 1.0,
        "loop_percussive_event_ratio": 0.0,
        "loop_drumlike_frame_ratio": 0.0,
        "loop_tonal_to_percussive_balance": 1.0,
        "log_crest": 1.80,
        "air_ratio_gt_8000hz": 0.0,
    }
    data.update(overrides)
    return data


def test_rhodes_low_mid_loop_is_pitched_instrument_not_bass_loop() -> None:
    """Regression surrogate for VINTAGEDREAMS_KATO_rhodes_loop_amber_haze_86_C.wav."""
    features = loop_base(
        sub_bass_ratio_lt_150hz=0.254874,
        bass_ratio_150_500hz=0.736061,
        mid_ratio_500_2000hz=0.009062,
        presence_ratio_2000_8000hz=0.000003,
        pitch_confidence=0.666989,
        f0_voiced_ratio=0.126923,
        formant_like_peak_spacing=4.0,
        spectral_flatness_mean=0.002730,
    )
    roles = measured_roles_from_features(
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        feature_values=features,
    )
    assert roles.bass_loop < 0.62
    assert roles.pitched_music_loop >= 0.58

    role = infer_parent_eligibility(facts(11.90, shape="bass_phrase", shape_confidence=0.93, **features))
    assert role.role_name == "pitched_music_loop"
    assert role.broad_folder_path == "Instruments/Instrument Loops/Loops"


def test_sub_foundation_bass_loop_still_keeps_bass_role() -> None:
    """Control surrogate from real bass-loop physics with strong sub foundation."""
    features = loop_base(
        sub_bass_ratio_lt_150hz=0.725213,
        bass_ratio_150_500hz=0.272149,
        mid_ratio_500_2000hz=0.002623,
        presence_ratio_2000_8000hz=0.000015,
        pitch_confidence=0.921887,
        f0_voiced_ratio=0.009615,
        formant_like_peak_spacing=0.111111,
        spectral_flatness_mean=0.002074,
    )
    roles = measured_roles_from_features(
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        feature_values=features,
    )
    assert roles.bass_loop >= 0.90


def test_bright_synth_lead_surrogate_does_not_become_fx_alarm() -> None:
    """Regression surrogate for a synth lead that was broadened to FX/Alarm."""
    features = loop_base(
        event_count_estimate=23.0,
        event_rate_hz=4.80,
        sub_bass_ratio_lt_150hz=0.000008,
        bass_ratio_150_500hz=0.000015,
        mid_ratio_500_2000hz=0.756844,
        presence_ratio_2000_8000hz=0.241518,
        pitch_confidence=0.951837,
        f0_voiced_ratio=1.0,
        formant_like_peak_spacing=4.0,
        spectral_flatness_mean=0.111635,
    )
    role = infer_parent_eligibility(facts(11.99, shape="vocal_phrase", shape_confidence=1.0, **features))
    assert role.role_name == "pitched_music_loop"
    assert role.allowed_top_families == ("Instruments", "_TO_REVIEW")
    assert "Alarm" not in role.broad_folder_path
    assert not role.broad_folder_path.startswith("FX/")


def test_short_pitched_percussion_surrogate_stays_drum_not_synth() -> None:
    """Regression for a one-shot percussion sample that was stolen by synth lead logic."""
    features = {
        "event_count_estimate": 1.0,
        "event_rate_hz": 1.595052,
        "onset_span_ratio": 0.0,
        "attack_rise_time_norm": 0.008222,
        "temporal_centroid_ratio": 0.073902,
        "tail_energy_ratio": 0.018075,
        "log_transient_count": 0.693147,
        "loop_pitched_event_ratio": 1.0,
        "loop_sustained_tonal_frame_ratio": 1.0,
        "loop_non_event_tonal_ratio": 1.0,
        "loop_percussive_event_ratio": 0.0,
        "loop_drumlike_frame_ratio": 0.0,
        "loop_tonal_to_percussive_balance": 1.0,
        "log_crest": 2.312557,
        "sub_bass_ratio_lt_150hz": 0.000003,
        "bass_ratio_150_500hz": 0.001671,
        "mid_ratio_500_2000hz": 0.911368,
        "presence_ratio_2000_8000hz": 0.086650,
        "air_ratio_gt_8000hz": 0.000308,
        "pitch_confidence": 0.490113,
        "f0_voiced_ratio": 1.0,
        "formant_like_peak_spacing": 1.384615,
        "spectral_flatness_mean": 0.064630,
    }
    role = infer_parent_eligibility(facts(0.717098, shape="single_hit", shape_confidence=0.833333, **features))
    assert role.role_name == "protected_percussive_one_shot"
    assert role.allowed_top_families == ("Drums", "_TO_REVIEW")
    assert role.broad_folder_path == "Drums/Percussion/Generic Percussion/One Shots"


def test_real_low_metallic_pitched_percussion_loop_prefers_drums() -> None:
    """Regression surrogate for confirmed real West African/metal percussion loops."""
    features = loop_base(
        event_count_estimate=46.0,
        event_rate_hz=3.92,
        onset_span_ratio=0.96,
        attack_rise_time_norm=0.02,
        temporal_centroid_ratio=0.51,
        log_crest=1.73,
        sub_bass_ratio_lt_150hz=0.001,
        bass_ratio_150_500hz=0.956,
        mid_ratio_500_2000hz=0.040,
        presence_ratio_2000_8000hz=0.003,
        pitch_confidence=0.228,
        f0_voiced_ratio=0.899,
        loop_pitched_event_ratio=1.0,
        loop_sustained_tonal_frame_ratio=0.938,
        loop_non_event_tonal_ratio=0.889,
        loop_drumlike_frame_ratio=0.042,
        loop_percussive_event_ratio=0.0,
        spectral_flatness_mean=0.227,
    )
    role = infer_parent_eligibility(facts(11.84, shape="bass_phrase", shape_confidence=0.83, **features))
    assert role.role_name == "pitched_percussion_loop"
    assert role.allowed_top_families == ("Drums", "_TO_REVIEW")
    assert role.broad_folder_path == "Drums/Drum Loops/Loops"


def test_reverb_keys_loop_does_not_trigger_pitched_percussion_loop() -> None:
    """Control: slow warm keys/pluck loop has low-mid body but is not percussion."""
    features = loop_base(
        event_count_estimate=16.0,
        event_rate_hz=1.34,
        onset_span_ratio=0.90,
        attack_rise_time_norm=0.02,
        temporal_centroid_ratio=0.40,
        log_crest=1.80,
        sub_bass_ratio_lt_150hz=0.352,
        bass_ratio_150_500hz=0.558,
        mid_ratio_500_2000hz=0.082,
        presence_ratio_2000_8000hz=0.008,
        pitch_confidence=0.924,
        f0_voiced_ratio=0.246,
        loop_pitched_event_ratio=1.0,
        loop_sustained_tonal_frame_ratio=1.0,
        loop_non_event_tonal_ratio=1.0,
        loop_drumlike_frame_ratio=0.0,
        loop_percussive_event_ratio=0.0,
        spectral_flatness_mean=0.069,
    )
    role = infer_parent_eligibility(facts(12.0, shape="bass_phrase", shape_confidence=1.0, **features))
    assert role.role_name != "pitched_percussion_loop"
    assert role.role_name == "pitched_music_loop"
