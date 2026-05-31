"""Synthetic surrogate regressions for instrument phrases mistaken for voice.

These tests do not store Aaron's WAV files.  They encode only the measured
physics pattern that caused the bad placement:

* long sustained tonal loop/phrase
* high voiced/pitched frame ratio
* no drumlike or percussive loop evidence
* no positive formant-light voice identity

That pattern may be sax, strings, keys, or another pitched instrument.  It must
not be broadened to FX/Human and Voice FX just because the generic shape voter
called the time-shape ``vocal_phrase``.
"""

from __future__ import annotations

from typing import Any

from aaron_sound_sorter.domain.models import ConsensusDecision, SharedAudioFacts
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import infer_parent_eligibility


def facts(
    duration: float,
    *,
    shape: str = "vocal_phrase",
    shape_confidence: float = 0.80,
    roles: dict[str, float] | None = None,
    voice_identity: float = 0.0,
    **features: float,
) -> SharedAudioFacts:
    roles_payload: dict[str, Any] = {**(roles or {})}
    roles_payload.setdefault("evidence", {})
    roles_payload["evidence"] = {
        **roles_payload["evidence"],
        "formant_light_voice_identity": voice_identity,
        "pitch_confidence": features.get("pitch_confidence", 0.0),
        "f0_voiced_ratio": features.get("f0_voiced_ratio", 0.0),
        "loop_pitched_event_ratio": features.get("loop_pitched_event_ratio", 0.0),
        "loop_sustained_tonal_frame_ratio": features.get("loop_sustained_tonal_frame_ratio", 0.0),
        "loop_non_event_tonal_ratio": features.get("loop_non_event_tonal_ratio", 0.0),
        "loop_drumlike_frame_ratio": features.get("loop_drumlike_frame_ratio", 0.0),
        "loop_percussive_event_ratio": features.get("loop_percussive_event_ratio", 0.0),
    }
    evidence = {
        "duration_sec": duration,
        "event_count_estimate": features.get("event_count_estimate", 0.0),
        "event_rate_hz": features.get("event_rate_hz", 0.0),
        "onset_span_ratio": features.get("onset_span_ratio", 0.0),
        "attack_rise_time_norm": features.get("attack_rise_time_norm", 0.0),
        "temporal_centroid_ratio": features.get("temporal_centroid_ratio", 0.0),
        "shape_vote": {"primary_shape": shape, "confidence": shape_confidence},
        "measured_roles": roles_payload,
        "feature_values_by_name": dict(features),
    }
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=duration >= 2.0,
        is_single_event_like=features.get("event_count_estimate", 0.0) <= 1.0,
        is_short_hit_like=duration <= 1.25,
        is_long=duration >= 4.0,
        evidence=evidence,
        feature_values_by_name=evidence["feature_values_by_name"],
    )


def raw_generic_instrument_loop() -> ConsensusDecision:
    return ConsensusDecision(
        final_label="Instruments/Instrument Loops/Loops",
        final_top="Instruments",
        folder_path="Instruments/Instrument Loops/Loops",
        consensus_status="test_raw_generic_instrument_loop",
        reason="synthetic raw consensus",
        shared_winner=None,
        brain_rank=7,
        physics_rank=13,
        combined_rank_score=20.0,
        shared_candidates=[],
    )


def test_strings_surrogate_vocal_shape_is_still_pitched_instrument_loop() -> None:
    """Regression for 03.strings_77bpm_Ebm.wav without storing the WAV."""
    role = infer_parent_eligibility(
        facts(
            11.999955,
            shape="vocal_phrase",
            shape_confidence=1.0,
            roles={
                "pitched_music_loop": 1.0,
                "pitched_music_phrase": 0.921716,
                "vocal_music_phrase": 0.911973,
                "voiced_one_shot": 0.0,
            },
            voice_identity=0.0,
            event_count_estimate=91.999996,
            event_rate_hz=7.723410,
            onset_span_ratio=0.996094,
            attack_rise_time_norm=0.020488,
            temporal_centroid_ratio=0.464491,
            pitch_confidence=0.922763,
            f0_voiced_ratio=1.0,
            formant_like_peak_spacing=4.0,
            sub_bass_ratio_lt_150hz=0.02,
            bass_ratio_150_500hz=0.242897,
            mid_ratio_500_2000hz=0.488809,
            presence_ratio_2000_8000hz=0.249094,
            air_ratio_gt_8000hz=0.0,
            loop_percussive_event_ratio=0.0,
            loop_drumlike_frame_ratio=0.0,
            loop_pitched_event_ratio=1.0,
            loop_sustained_tonal_frame_ratio=1.0,
            loop_non_event_tonal_ratio=1.0,
            spectral_flatness_mean=0.320105,
            log_crest=1.817465,
        )
    )

    assert role.role_name == "pitched_music_loop"
    assert role.allowed_top_families == ("Instruments", "_TO_REVIEW")
    assert role.broad_folder_path.startswith("Instruments/")
    assert "Human and Voice" not in role.broad_folder_path

    final = DecisionCoreV2().apply_eligibility(raw_generic_instrument_loop(), role)
    assert final.final_top == "Instruments"
    assert "Human and Voice" not in final.folder_path


def test_sax_ensemble_surrogate_goes_to_reed_bucket_not_voice_fx() -> None:
    """Regression for RHSH_Saxophone_Ensemble_02_keyC_85bpm.wav without storing the WAV."""
    role = infer_parent_eligibility(
        facts(
            11.999864,
            shape="vocal_phrase",
            shape_confidence=0.746508,
            roles={
                "pitched_music_loop": 0.80,
                "vocal_music_phrase": 0.0,
                "voiced_one_shot": 0.0,
            },
            voice_identity=0.0,
            event_count_estimate=19.000001,
            event_rate_hz=1.595052,
            onset_span_ratio=0.947266,
            attack_rise_time_norm=0.007872,
            temporal_centroid_ratio=0.474072,
            pitch_confidence=0.351320,
            f0_voiced_ratio=0.996154,
            formant_like_peak_spacing=1.888889,
            sub_bass_ratio_lt_150hz=0.03,
            bass_ratio_150_500hz=0.266788,
            mid_ratio_500_2000hz=0.410428,
            presence_ratio_2000_8000hz=0.292784,
            air_ratio_gt_8000hz=0.0,
            loop_percussive_event_ratio=0.0,
            loop_drumlike_frame_ratio=0.0,
            loop_pitched_event_ratio=1.0,
            loop_sustained_tonal_frame_ratio=1.0,
            loop_non_event_tonal_ratio=1.0,
            spectral_flatness_mean=0.289985,
            log_crest=2.243860,
        )
    )

    assert role.role_name in {"pitched_reed_or_instrument_loop", "pitched_music_loop"}
    assert role.allowed_top_families == ("Instruments", "_TO_REVIEW")
    assert role.broad_folder_path.startswith("Instruments/")
    assert "Human" not in role.broad_folder_path
    assert "Voice" not in role.broad_folder_path

    final = DecisionCoreV2().apply_eligibility(raw_generic_instrument_loop(), role)
    assert final.final_top == "Instruments"
    assert "Human" not in final.folder_path
    assert "Voice" not in final.folder_path


def test_real_vocal_surrogate_still_gets_voice_bucket_when_voice_identity_is_positive() -> None:
    """The sax/string fix must not disable true vocal routing."""
    role = infer_parent_eligibility(
        facts(
            0.77,
            shape="hit_with_tail",
            shape_confidence=0.82,
            roles={"voiced_one_shot": 0.80},
            voice_identity=0.94,
            event_count_estimate=1.0,
            event_rate_hz=1.44,
            onset_span_ratio=0.0,
            attack_rise_time_norm=0.135,
            temporal_centroid_ratio=0.144,
            pitch_confidence=0.42,
            f0_voiced_ratio=0.91,
            formant_like_peak_spacing=0.0,
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
