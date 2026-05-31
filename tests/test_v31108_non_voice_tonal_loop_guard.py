from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.voters.physics_instrument_layer import PhysicsInstrumentLayer


def _bell_like_false_voice_facts() -> SharedAudioFacts:
    features = {
        "pitch_confidence": 0.88,
        "f0_voiced_ratio": 0.93,
        "f0_median_hz": 520.0,
        "loop_pitched_event_ratio": 0.98,
        "loop_sustained_tonal_frame_ratio": 0.94,
        "loop_non_event_tonal_ratio": 0.92,
        "loop_mean_event_low_ratio": 0.10,
        "loop_mean_event_mid_ratio": 0.68,
        "loop_mean_event_high_ratio": 0.18,
        "loop_percussive_event_ratio": 0.02,
        "loop_drumlike_frame_ratio": 0.01,
        "spectral_flatness_mean": 0.08,
        "spectral_entropy_mean": 0.42,
        "inharmonicity": 0.46,
        "onset_count": 28.0,
        "event_count_estimate": 28.0,
        "mid_ratio_500_2000hz": 0.58,
        "presence_ratio_2000_8000hz": 0.24,
        "air_ratio_gt_8000hz": 0.05,
        "bass_ratio_150_500hz": 0.18,
        "sub_bass_ratio_lt_150hz": 0.04,
    }
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        feature_values_by_name=features,
        evidence={
            "shape_vote": {
                "primary_shape": "vocal_phrase",
                "confidence": 0.95,
                **features,
            },
            "measured_roles": {
                "pitched_music_loop": 0.88,
                "vocal_music_phrase": 0.72,
                "voiced_one_shot": 0.0,
                "evidence": features,
            },
            "physics_subpanels": {
                "flat": {
                    "voice_score": 0.74,
                    "human_spoken_voice_score": 0.84,
                    "human_breath_mouth_score": 0.52,
                    "fx_formant_score": 0.72,
                    "synth_tonal_source_score": 0.60,
                    "metallic_noise_score": 0.66,
                    "struck_keys_score": 0.60,
                    "struck_keys_authority_score": 0.49,
                    "reed_wind_score": 0.60,
                    "reed_wind_authority_score": 0.40,
                    "plucked_string_score": 0.62,
                    "plucked_string_authority_score": 0.53,
                    "role_loop_score": 0.50,
                    "role_phrase_score": 0.75,
                    "drum_hit_score": 0.34,
                    "drum_loop_source_score": 0.11,
                }
            },
        },
    )


def test_bell_like_tonal_loop_does_not_promote_to_voice_branch() -> None:
    branch, strength, evidence = PhysicsInstrumentLayer().decide(_bell_like_false_voice_facts())

    assert evidence["instrument_non_voice_tonal_loop_voice_decoy"] is True
    assert evidence["instrument_human_voice_phrase_signal"] is False
    assert branch != "Voice"
    assert evidence["instrument_branch_score_Voice"] < max(
        evidence["instrument_branch_score_Synth"],
        evidence["instrument_branch_score_MalletBell"],
        evidence["instrument_branch_score_MixedInstrument"],
    )
    assert strength >= 0.40
