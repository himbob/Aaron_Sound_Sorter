from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from aaron_sound_sorter import api as mod


def fp(**values):
    idx = {name: i for i, name in enumerate(mod.FEATURE_NAMES)}
    x = np.zeros(mod.FP_SIZE, dtype=np.float32)
    for name, value in values.items():
        assert name in idx, name
        x[idx[name]] = float(value)
    return x


def stat(median: float, p10: float, p90: float, mad: float = 0.05, reliability: float = 0.75) -> dict:
    return {
        "median": median,
        "p10": p10,
        "p90": p90,
        "p25": median - mad,
        "p75": median + mad,
        "mad": mad,
        "reliability": reliability,
        "valid_count": 24,
    }


def test_short_single_event_proves_one_shot_before_loop_labels_compete() -> None:
    sample = fp(
        log_transient_count=np.log1p(1.0),
        temporal_centroid_ratio=0.14,
        onset_interval_regularity=1.0,
        attack_rise_time_norm=0.13,
        onset_span_ratio=0.0,
        event_rate_hz=1.4,
    )
    contract = mod._measured_structure_contract({}, sample, 0.76)
    assert contract["observed_structure"] == "one_shot"
    assert contract["confidence"] == "hard"
    assert "short_single_event_one_shot_proof" in contract["reason"]

    loop_penalty = mod._measured_structure_adjustment_for_label(
        {"structure_by_label": {"Any/Loop/Loops": "loop"}},
        "Any/Loop/Loops",
        contract,
    )
    one_shot_adjustment = mod._measured_structure_adjustment_for_label(
        {"structure_by_label": {"Any/Hit/One Shots": "one_shot"}},
        "Any/Hit/One Shots",
        contract,
    )
    assert loop_penalty >= 5.0
    assert one_shot_adjustment < 0.0


def test_coherent_feature_group_outlier_penalty_uses_learned_profile_not_category_names() -> None:
    sample = fp(
        pitch_confidence=0.70,
        attack_pitch_confidence=0.72,
        body_pitch_confidence=0.69,
        f0_voiced_ratio=0.40,
        harmonic_energy_ratio=0.22,
    )
    profile = {
        "eligible_count": 24,
        "feature_stats": {
            "pitch_confidence": stat(0.20, 0.10, 0.35),
            "attack_pitch_confidence": stat(0.24, 0.12, 0.38),
            "body_pitch_confidence": stat(0.22, 0.11, 0.36),
            "f0_voiced_ratio": stat(0.30, 0.05, 0.55),
            "harmonic_energy_ratio": stat(0.12, 0.04, 0.28),
        },
    }
    penalty, reason = mod._coherent_feature_group_outlier_penalty(
        {"coherent_group_outlier_penalty_enabled": True},
        sample,
        "Any/Learned/Label",
        profile,
        list(mod.FEATURE_NAMES),
    )
    assert penalty > 0.75
    assert "pitch_harmonic:high" in reason


def test_coherent_feature_group_outlier_does_not_fire_for_single_feature_spike() -> None:
    sample = fp(
        pitch_confidence=0.70,
        attack_pitch_confidence=0.24,
        body_pitch_confidence=0.22,
    )
    profile = {
        "eligible_count": 24,
        "feature_stats": {
            "pitch_confidence": stat(0.20, 0.10, 0.35),
            "attack_pitch_confidence": stat(0.24, 0.12, 0.38),
            "body_pitch_confidence": stat(0.22, 0.11, 0.36),
        },
    }
    penalty, reason = mod._coherent_feature_group_outlier_penalty(
        {}, sample, "Any/Learned/Label", profile, list(mod.FEATURE_NAMES)
    )
    assert penalty == 0.0


def test_severe_coherent_group_penalty_blocks_auto_place_not_just_reported() -> None:
    sample = fp(
        pitch_confidence=0.61,
        body_pitch_confidence=0.79,
        tail_pitch_confidence=0.84,
        spectral_entropy_mean=0.36,
        spectral_flatness_mean=0.29,
        log_transient_count=np.log1p(1.0),
        temporal_centroid_ratio=0.12,
    )
    label = "Drums/Cymbals/Crash Cymbal/One Shots"
    brain = {
        "labels": [label],
        "top_by_label": {label: "Drums"},
        "severe_coherent_group_review_enabled": True,
        "severe_coherent_group_review_threshold": 2.0,
        "_last_fact_meta": {
            "coherent_group_penalty": 2.25,
            "coherent_group_reason": "pitch_harmonic:high:4/8:penalty=1.25 | spectral_noise_shape:low:3/9:penalty=1.25",
        },
    }
    final_top, final_label, status, reason = mod.apply_learned_conflict_gates(
        brain=brain,
        fingerprint=sample,
        pred_label=label,
        pred_top="Drums",
        final_top="Drums",
        final_label=label,
        confidence_status="auto_place",
        review_reason="AUDIT ambiguous_pick: test",
        similarity=0.53,
        margin=0.0,
        duration_sec=0.58,
        top5=[(label, 0.1)],
    )
    assert final_top == "_TO_REVIEW"
    assert final_label == label
    assert status == "review"
    assert "severe_coherent_group_outlier_pick" in reason
    assert "safety_policy_review_instead_of_audited_auto_place" in reason


def test_syncopated_distributed_music_or_drum_loop_gets_hard_loop_structure() -> None:
    sample = fp(
        log_transient_count=np.log1p(17.0),
        onset_interval_regularity=0.82,
        temporal_centroid_ratio=0.47,
        onset_span_ratio=0.90,
        event_rate_hz=1.75,
        attack_rise_time_norm=0.001,
    )
    is_loop, reason = mod.fingerprint_loop_evidence_detail(sample, 9.8)
    assert is_loop
    assert "distributed_multi_event_pattern" in reason or "long_loose_distributed_pattern" in reason
    contract = mod._measured_structure_contract({}, sample, 9.8)
    assert contract["observed_structure"] == "loop"
    assert contract["confidence"] == "hard"


def test_reverby_musical_loop_in_fx_can_switch_to_learned_instrument_loop_without_filename_rules() -> None:
    sample = fp(
        log_transient_count=np.log1p(15.0),
        spectral_flatness_mean=0.31,
        spectral_entropy_mean=0.64,
        centroid_slope_norm=-0.03,
        temporal_centroid_ratio=0.45,
        onset_span_ratio=0.97,
        event_rate_hz=1.26,
        pitch_confidence=0.32,
        mid_ratio_500_2000hz=0.38,
        presence_ratio_2000_8000hz=0.28,
        air_ratio_gt_8000hz=0.004,
        tail_energy_ratio=0.58,
    )
    label = "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Loops"
    inst_loop = "Instruments/Instrument Loops/Loops"
    brain = {
        "labels": [label, inst_loop],
        "top_by_label": {label: "FX", inst_loop: "Instruments"},
        "structure_by_label": {label: "loop", inst_loop: "loop"},
    }
    new_label, new_top, action, reason = mod.physical_family_guard_decision(
        brain=brain,
        fingerprint=sample,
        final_label=label,
        final_top="FX",
        predicted_top="FX",
        top5=[(label, 0.1)],
        duration_sec=12.0,
    )
    assert action == "switch"
    assert new_top == "Instruments"
    assert new_label == inst_loop
    assert "tonal_loop_to_instruments" in reason


def test_soft_audit_warnings_do_not_force_review_when_no_hard_physics_conflict() -> None:
    label = "Drums/Drum Loops/Loops"
    sample = fp(
        log_transient_count=np.log1p(17.0),
        temporal_centroid_ratio=0.47,
        onset_span_ratio=0.90,
        event_rate_hz=1.75,
        onset_interval_regularity=0.82,
    )
    brain = {
        "labels": [label],
        "top_by_label": {label: "Drums"},
        "structure_by_label": {label: "loop"},
        "label_reliability_by_label": {
            label: {
                "training_count": 96,
                "support_tier": "strong",
                "active_status": "ACTIVE_TRUSTED",
                "min_similarity_for_auto_place": 0.80,
                "min_margin_for_auto_place": 1.20,
            }
        },
        "counts": {label: 96},
    }
    final_top, final_label, status, reason = mod.apply_learned_conflict_gates(
        brain=brain,
        fingerprint=sample,
        pred_label=label,
        pred_top="Drums",
        final_top="Drums",
        final_label=label,
        confidence_status="auto_place",
        review_reason="AUDIT ambiguous_pick: test",
        similarity=0.55,
        margin=0.0,
        duration_sec=9.8,
        top5=[(label, 0.1)],
    )
    assert final_top == "Drums"
    assert final_label == label
    assert status == "auto_place"
    assert "low_label_support_pick" in reason
    assert "safety_policy_review_instead_of_audited_auto_place" not in reason
