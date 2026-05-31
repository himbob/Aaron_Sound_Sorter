from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.voters.physics_instrument_layer import PhysicsInstrumentLayer


def _facts(*, reed_authority: float = 0.43) -> SharedAudioFacts:
    features = {
        "pitch_confidence": 0.84,
        "f0_voiced_ratio": 0.82,
        "f0_median_hz": 240.0,
        "low_event_ratio": 0.08,
        "mid_event_ratio": 0.78,
        "high_event_ratio": 0.05,
        "spectral_flatness_mean": 0.31,
        "spectral_entropy_mean": 0.52,
        "percussive_event_ratio": 0.02,
        "drumlike_frame_ratio": 0.02,
        "sustained_tonal_frame_ratio": 0.88,
        "non_event_tonal_ratio": 0.75,
        "onset_count": 18.0,
        "event_count_estimate": 18.0,
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
                "primary_shape": "pitched_phrase",
                "confidence": 0.91,
                **features,
            },
            "measured_roles": {
                "pitched_music_phrase": 0.86,
                "vocal_music_phrase": 0.0,
                "evidence": features,
            },
            "physics_subpanels": {
                "flat": {
                    "voice_score": 0.64,
                    "human_spoken_voice_score": 0.72,
                    "human_breath_mouth_score": 0.30,
                    "fx_formant_score": 0.69,
                    "reed_wind_score": 0.50,
                    "reed_wind_authority_score": reed_authority,
                    "plucked_string_authority_score": 0.34,
                    "drum_hit_score": 0.05,
                    "drum_loop_source_score": 0.03,
                }
            },
        },
    )


def test_instrument_layer_promotes_measured_human_voice_texture_phrase() -> None:
    branch, strength, evidence = PhysicsInstrumentLayer().decide(_facts())

    assert branch == "Voice"
    assert strength >= 0.50
    assert evidence["instrument_human_voice_phrase_signal"] is True
    assert evidence["instrument_branch_Voice"] > evidence["instrument_branch_MixedInstrument"]
    assert evidence["instrument_branch_Voice"] > evidence["instrument_branch_Woodwinds"]


def test_reed_authority_blocks_false_voice_texture_promotion() -> None:
    branch, _strength, evidence = PhysicsInstrumentLayer().decide(_facts(reed_authority=0.78))

    assert evidence["instrument_human_voice_phrase_signal"] is False
    assert branch != "Voice"
