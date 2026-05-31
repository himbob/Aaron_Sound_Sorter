"""Synthetic parent-eligibility regression tests for instrument/voice/loop safety.

These tests use generated fact vectors, not WAV fixtures.  They lock broad
physics behavior so a future code change cannot route clean strings/sax-like
instrument phrases into voice/FX, or clean drum loops into bass/instruments.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import ConsensusDecision, SharedAudioFacts
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import infer_parent_eligibility


def make_facts(
    duration: float,
    *,
    shape: str = "",
    shape_confidence: float = 0.90,
    roles: dict[str, float] | None = None,
    role_evidence: dict[str, float] | None = None,
    **features: float,
) -> SharedAudioFacts:
    """Build a minimal synthetic SharedAudioFacts object."""
    event_count = float(features.pop("event_count_estimate", 0.0))
    evidence = {
        "duration_sec": duration,
        "event_count_estimate": event_count,
        "event_rate_hz": features.pop("event_rate_hz", 0.0),
        "onset_span_ratio": features.pop("onset_span_ratio", 0.0),
        "temporal_centroid_ratio": features.pop("temporal_centroid_ratio", 0.0),
        "attack_rise_time_norm": features.pop("attack_rise_time_norm", 0.0),
        "shape_vote": {"primary_shape": shape, "confidence": shape_confidence},
        "measured_roles": {**(roles or {}), "evidence": dict(role_evidence or {})},
        "feature_values_by_name": dict(features),
    }
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=duration >= 1.5,
        is_single_event_like=event_count <= 1.0,
        is_short_hit_like=duration <= 1.25,
        is_long=duration >= 4.0,
        evidence=evidence,
        feature_values_by_name=evidence["feature_values_by_name"],
    )


def raw_decision(path: str, top: str, *, score: float = 10.0) -> ConsensusDecision:
    """Build a minimal raw consensus decision."""
    return ConsensusDecision(
        final_label=path,
        final_top=top,
        folder_path=path,
        consensus_status="strong_consensus",
        reason="synthetic raw decision",
        combined_rank_score=score,
        shared_candidates=[{"folder_path": path, "combined_rank_score": score}],
    )


def test_clean_sustained_string_like_phrase_blocks_voice_and_fx_bucket() -> None:
    """A clean, stable, low-flatness tonal phrase should not become voice/FX."""
    facts = make_facts(
        3.4,
        shape="vocal_phrase",
        roles={"pitched_music_loop": 0.68, "vocal_music_phrase": 0.38, "voiced_one_shot": 0.64},
        role_evidence={"formant_light_voice_identity": 0.12},
        event_count_estimate=3,
        event_rate_hz=0.9,
        onset_span_ratio=0.42,
        attack_rise_time_norm=0.34,
        temporal_centroid_ratio=0.47,
        pitch_confidence=0.62,
        f0_voiced_ratio=0.88,
        formant_like_peak_spacing=0.72,
        sub_bass_ratio_lt_150hz=0.03,
        bass_ratio_150_500hz=0.18,
        mid_ratio_500_2000hz=0.52,
        presence_ratio_2000_8000hz=0.20,
        air_ratio_gt_8000hz=0.01,
        loop_percussive_event_ratio=0.02,
        loop_drumlike_frame_ratio=0.02,
        loop_pitched_event_ratio=0.96,
        loop_sustained_tonal_frame_ratio=0.93,
        loop_non_event_tonal_ratio=0.92,
        spectral_flatness_mean=0.055,
        log_crest=0.88,
    )
    eligibility = infer_parent_eligibility(facts)
    assert eligibility.allowed_top_families == ("Instruments", "_TO_REVIEW")
    assert eligibility.broad_folder_path.startswith("Instruments/")
    assert "Voice" not in eligibility.broad_folder_path

    final = DecisionCoreV2().apply_eligibility(
        raw_decision("FX/Human and Voice FX/Spoken Voice/Long FX", "FX"),
        eligibility,
    )
    assert final.folder_path.startswith("Instruments/")
    assert "Voice" not in final.folder_path
    assert not final.folder_path.startswith("FX/")


def test_sax_like_sustained_phrase_stays_reed_branch_not_voice() -> None:
    """A wet/formant sax-like phrase is an instrument parent, not a voice bucket."""
    facts = make_facts(
        5.8,
        shape="vocal_phrase",
        roles={"pitched_music_loop": 0.74, "vocal_music_phrase": 0.40, "voiced_one_shot": 0.50},
        role_evidence={"formant_light_voice_identity": 0.20},
        event_count_estimate=9,
        event_rate_hz=1.55,
        onset_span_ratio=0.78,
        attack_rise_time_norm=0.62,
        temporal_centroid_ratio=0.50,
        pitch_confidence=0.70,
        f0_voiced_ratio=0.86,
        formant_like_peak_spacing=1.55,
        sub_bass_ratio_lt_150hz=0.02,
        bass_ratio_150_500hz=0.18,
        mid_ratio_500_2000hz=0.56,
        presence_ratio_2000_8000hz=0.18,
        air_ratio_gt_8000hz=0.01,
        loop_percussive_event_ratio=0.04,
        loop_drumlike_frame_ratio=0.05,
        loop_pitched_event_ratio=0.90,
        loop_sustained_tonal_frame_ratio=0.78,
        loop_non_event_tonal_ratio=0.73,
        spectral_flatness_mean=0.13,
        log_crest=1.00,
    )
    eligibility = infer_parent_eligibility(facts)
    assert eligibility.allowed_top_families == ("Instruments", "_TO_REVIEW")
    assert eligibility.broad_folder_path.startswith("Instruments/")
    assert "Voice" not in eligibility.broad_folder_path

    final = DecisionCoreV2().apply_eligibility(
        raw_decision("FX/Human and Voice FX/Spoken Voice/Long FX", "FX"),
        eligibility,
    )
    assert final.folder_path.startswith("Instruments/")
    assert "Voice" not in final.folder_path
    assert not final.folder_path.startswith("FX/")


def test_real_vocal_stab_still_goes_to_voice_bucket() -> None:
    """The new tonal-instrument guard must not silence true vocal evidence."""
    facts = make_facts(
        0.78,
        shape="vocal_one_shot",
        roles={"voiced_one_shot": 0.84, "vocal_music_phrase": 0.66},
        role_evidence={"formant_light_voice_identity": 0.88},
        event_count_estimate=1,
        event_rate_hz=1.2,
        onset_span_ratio=0.08,
        attack_rise_time_norm=0.16,
        temporal_centroid_ratio=0.24,
        pitch_confidence=0.52,
        f0_voiced_ratio=0.91,
        formant_like_peak_spacing=1.65,
        sub_bass_ratio_lt_150hz=0.005,
        bass_ratio_150_500hz=0.03,
        mid_ratio_500_2000hz=0.46,
        presence_ratio_2000_8000hz=0.36,
        air_ratio_gt_8000hz=0.03,
        loop_percussive_event_ratio=0.0,
        loop_drumlike_frame_ratio=0.0,
        loop_pitched_event_ratio=1.0,
        loop_sustained_tonal_frame_ratio=0.96,
        loop_non_event_tonal_ratio=0.94,
        spectral_flatness_mean=0.32,
        log_crest=1.4,
    )
    eligibility = infer_parent_eligibility(facts)
    assert eligibility.role_name in {"vocal_phrase", "vocal_one_shot"}
    assert eligibility.broad_folder_path == "FX/Human and Voice FX"


def test_clean_tonal_bass_phrase_is_not_forced_to_drum_loop() -> None:
    """A sustained tonal bass-like loop is not automatically a drum/percussion loop."""
    facts = make_facts(
        4.2,
        shape="bass_phrase",
        roles={"bass_loop": 0.82, "pitched_music_loop": 0.72, "low_rhythmic_drum_loop": 0.20},
        event_count_estimate=5,
        event_rate_hz=1.1,
        onset_span_ratio=0.50,
        attack_rise_time_norm=0.30,
        temporal_centroid_ratio=0.46,
        pitch_confidence=0.88,
        f0_voiced_ratio=0.80,
        formant_like_peak_spacing=0.20,
        sub_bass_ratio_lt_150hz=0.48,
        bass_ratio_150_500hz=0.20,
        mid_ratio_500_2000hz=0.22,
        presence_ratio_2000_8000hz=0.03,
        air_ratio_gt_8000hz=0.002,
        loop_percussive_event_ratio=0.02,
        loop_drumlike_frame_ratio=0.02,
        loop_pitched_event_ratio=0.96,
        loop_sustained_tonal_frame_ratio=0.96,
        loop_non_event_tonal_ratio=0.96,
        spectral_flatness_mean=0.04,
        log_crest=0.90,
    )
    eligibility = infer_parent_eligibility(facts)
    assert eligibility.role_name in {"bass_loop", "pitched_music_loop", "clean_sustained_tonal_instrument_phrase"}
    assert eligibility.allowed_top_families == ("Instruments", "_TO_REVIEW")
