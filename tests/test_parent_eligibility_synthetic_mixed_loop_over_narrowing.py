"""Synthetic regressions for over-narrowing generic musical loops.

These tests protect the failure seen in the full-run logs:

* a mixed melody loop with bells/chimes/layers was moved from generic
  Instrument Loops into Brass and Woodwinds only because the measured role had
  reed-like sustained-pitched traits.
* a vocal loop was also narrowed to Brass and Woodwinds when broad generic
  Instrument Loops or a true voice bucket would have been safer.

The parent-eligibility seam may protect these sounds from Drums/FX/animal/foley,
but it must not invent a Brass/Woodwind identity when the raw voters did not.
"""

from __future__ import annotations

from typing import Any

from aaron_sound_sorter.domain.models import ConsensusDecision, SharedAudioFacts
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import infer_parent_eligibility


def facts(
    duration: float,
    *,
    shape: str = "pitched_phrase",
    shape_confidence: float = 0.86,
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
        is_single_event_like=False,
        is_short_hit_like=False,
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
        brain_rank=5,
        physics_rank=9,
        combined_rank_score=14.0,
        shared_candidates=[],
    )


def raw_brass_woodwind_loop() -> ConsensusDecision:
    return ConsensusDecision(
        final_label="Instruments/Brass and Woodwinds/Loops",
        final_top="Instruments",
        folder_path="Instruments/Brass and Woodwinds/Loops",
        consensus_status="test_raw_brass_loop",
        reason="synthetic raw consensus",
        shared_winner=None,
        brain_rank=1,
        physics_rank=2,
        combined_rank_score=3.0,
        shared_candidates=[],
    )


def layered_melody_loop_features(**overrides: float) -> dict[str, float]:
    data = {
        "event_count_estimate": 20.0,
        "event_rate_hz": 2.1,
        "onset_span_ratio": 0.92,
        "attack_rise_time_norm": 0.08,
        "temporal_centroid_ratio": 0.46,
        "tail_energy_ratio": 0.16,
        "log_transient_count": 3.1,
        "loop_pitched_event_ratio": 0.96,
        "loop_sustained_tonal_frame_ratio": 0.82,
        "loop_non_event_tonal_ratio": 0.80,
        "loop_percussive_event_ratio": 0.05,
        "loop_drumlike_frame_ratio": 0.05,
        "loop_tonal_to_percussive_balance": 0.91,
        "sub_bass_ratio_lt_150hz": 0.03,
        "bass_ratio_150_500hz": 0.18,
        "mid_ratio_500_2000hz": 0.50,
        "presence_ratio_2000_8000hz": 0.25,
        "air_ratio_gt_8000hz": 0.04,
        "pitch_confidence": 0.58,
        "f0_voiced_ratio": 0.92,
        "formant_like_peak_spacing": 1.20,
        "spectral_flatness_mean": 0.16,
        "log_crest": 1.35,
    }
    data.update(overrides)
    return data


def test_mixed_layered_melody_loop_does_not_get_promoted_to_brass_woodwinds() -> None:
    """Regression for WS_KIT_3_West_Coast_Melody_Loop_Gnarly_Dm_100BPM.wav."""
    role = infer_parent_eligibility(
        facts(
            11.5,
            shape="pitched_phrase",
            shape_confidence=0.88,
            roles={"pitched_music_loop": 0.82, "voiced_one_shot": 0.0},
            **layered_melody_loop_features(),
        )
    )

    assert role.role_name == "pitched_reed_or_instrument_loop"
    assert role.allowed_top_families == ("Instruments", "_TO_REVIEW")
    assert role.broad_folder_path == "Instruments/Instrument Loops/Loops"

    final = DecisionCoreV2().apply_eligibility(raw_generic_instrument_loop(), role)
    assert final.folder_path == "Instruments/Instrument Loops/Loops"
    assert "Brass and Woodwinds" not in final.folder_path


def test_weak_voice_loop_does_not_get_promoted_to_brass_woodwinds() -> None:
    """Regression for vocal-loop material that was narrowed to Brass/Woodwinds."""
    role = infer_parent_eligibility(
        facts(
            9.0,
            shape="vocal_phrase",
            shape_confidence=0.72,
            roles={"pitched_music_loop": 0.76, "vocal_music_phrase": 0.35, "voiced_one_shot": 0.0},
            voice_identity=0.05,
            **layered_melody_loop_features(
                formant_like_peak_spacing=1.18,
                spectral_flatness_mean=0.17,
            ),
        )
    )

    assert role.role_name == "pitched_reed_or_instrument_loop"
    assert role.broad_folder_path == "Instruments/Instrument Loops/Loops"

    final = DecisionCoreV2().apply_eligibility(raw_generic_instrument_loop(), role)
    assert final.final_top == "Instruments"
    assert "Brass and Woodwinds" not in final.folder_path


def test_real_brass_or_woodwind_winner_is_not_blocked_by_broad_loop_guard() -> None:
    """A true voter-selected Brass/Woodwind loop should still be allowed."""
    role = infer_parent_eligibility(
        facts(
            8.0,
            shape="pitched_phrase",
            shape_confidence=0.90,
            roles={"pitched_music_loop": 0.86, "voiced_one_shot": 0.0},
            **layered_melody_loop_features(
                formant_like_peak_spacing=1.25,
                spectral_flatness_mean=0.18,
            ),
        )
    )

    final = DecisionCoreV2().apply_eligibility(raw_brass_woodwind_loop(), role)
    assert final.folder_path == "Instruments/Brass and Woodwinds/Loops"
