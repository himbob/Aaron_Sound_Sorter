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
