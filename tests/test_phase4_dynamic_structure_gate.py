#!/usr/bin/env python3
"""Fast regression tests for dynamic structure gating.

These tests deliberately use arbitrary folder names in several places. The goal
is to prove the sorter uses learned structure + measured audio features, not
filename or category-name rescue rules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
    if Path(__file__).resolve().parent.name == "tests"
    else Path(__file__).resolve().parent
)
MODULE_PATH = PROJECT_ROOT / "Aaron_Sound_Sorter.py"
if not MODULE_PATH.exists():
    MODULE_PATH = Path(__file__).resolve().parent / "Aaron_Sound_Sorter.py"


def load_module():
    from aaron_sound_sorter import api

    return api


def fp(m, **features):
    arr = np.zeros(m.FP_SIZE, dtype=np.float32)
    for name, value in features.items():
        arr[m.FEATURE_NAMES.index(name)] = value
    return arr.tolist()


def row(m, path: str, label: str, structure: str, features: dict[str, float], duration: float = 1.0):
    top = label.split("/", 1)[0]
    return m.FeatureRow(
        path=path,
        group_key=label,
        label=m.label_with_terminal_structure(label, structure),
        top=top,
        structure=structure,
        duration_sec=duration,
        fingerprint=fp(m, **features),
        read_status="ok",
    )


def build_structure_brain(m, base_a="Alpha", base_b="Beta"):
    rows: list[object] = []
    # Strong one-shot leaf: front loaded, single event, low span/rate.
    for i in range(24):
        rows.append(
            row(
                m,
                f"/train/hit_{i}.wav",
                f"Drums/{base_a}",
                "one_shot",
                {
                    "log_transient_count": np.log1p(1.0),
                    "temporal_centroid_ratio": 0.08 + i * 0.0005,
                    "onset_interval_regularity": 1.0,
                    "onset_span_ratio": 0.05,
                    "event_rate_hz": 0.5,
                    "tail_energy_ratio": 0.12,
                    "attack_rise_time_norm": 0.01,
                    "sub_bass_ratio_lt_150hz": 0.45,
                },
                duration=0.5,
            )
        )
    # Learned loop destination in same top/family: distributed repeated events.
    for i in range(24):
        rows.append(
            row(
                m,
                f"/train/loop_{i}.wav",
                f"Drums/{base_a}",
                "loop",
                {
                    "log_transient_count": np.log1p(9.0),
                    "temporal_centroid_ratio": 0.44 + i * 0.0005,
                    "onset_interval_regularity": 0.35,
                    "onset_span_ratio": 0.78,
                    "event_rate_hz": 2.2,
                    "tail_energy_ratio": 0.42,
                    "attack_rise_time_norm": 0.05,
                    "sub_bass_ratio_lt_150hz": 0.40,
                },
                duration=5.0,
            )
        )
    # A second one-shot sibling so adaptive/rival systems have a distractor.
    for i in range(24):
        rows.append(
            row(
                m,
                f"/train/other_{i}.wav",
                f"Drums/{base_b}",
                "one_shot",
                {
                    "log_transient_count": np.log1p(1.0),
                    "temporal_centroid_ratio": 0.10,
                    "onset_interval_regularity": 1.0,
                    "onset_span_ratio": 0.08,
                    "event_rate_hz": 0.4,
                    "tail_energy_ratio": 0.10,
                    "attack_rise_time_norm": 0.02,
                    "air_ratio_gt_8000hz": 0.35,
                },
                duration=0.45,
            )
        )
    brain = m.build_brain(rows, max_centroids=3)
    return brain


def test_dynamic_structure_gate_reroutes_loop_like_audio_out_of_one_shots():
    m = load_module()
    brain = build_structure_brain(m)
    one_shot_label = "Drums/Alpha/One Shots"
    sample = fp(
        m,
        log_transient_count=np.log1p(10.0),
        temporal_centroid_ratio=0.45,
        onset_interval_regularity=0.35,
        onset_span_ratio=0.82,
        event_rate_hz=2.4,
        tail_energy_ratio=0.45,
        attack_rise_time_norm=0.05,
        sub_bass_ratio_lt_150hz=0.42,
    )
    dest, reason = m.dynamic_structure_gate_destination(brain, sample, one_shot_label, 5.5)
    assert dest
    assert dest.endswith("/Loops")
    assert "One Shots" not in dest
    assert "dynamic_structure_gate" in reason
    assert "filename" not in reason.lower()


def test_dynamic_structure_gate_does_not_move_front_loaded_one_shot():
    m = load_module()
    brain = build_structure_brain(m)
    one_shot_label = "Drums/Alpha/One Shots"
    sample = fp(
        m,
        log_transient_count=np.log1p(1.0),
        temporal_centroid_ratio=0.08,
        onset_interval_regularity=1.0,
        onset_span_ratio=0.05,
        event_rate_hz=0.5,
        tail_energy_ratio=0.10,
        attack_rise_time_norm=0.01,
        sub_bass_ratio_lt_150hz=0.44,
    )
    dest, reason = m.dynamic_structure_gate_destination(brain, sample, one_shot_label, 0.55)
    assert dest == ""
    assert reason == ""


def test_dynamic_structure_gate_works_with_arbitrary_label_names_not_category_names():
    m = load_module()
    brain = build_structure_brain(m, base_a="Folder A", base_b="Folder B")
    one_shot_label = "Drums/Folder A/One Shots"
    sample = fp(
        m,
        log_transient_count=np.log1p(11.0),
        temporal_centroid_ratio=0.46,
        onset_interval_regularity=0.30,
        onset_span_ratio=0.86,
        event_rate_hz=2.8,
        tail_energy_ratio=0.40,
        attack_rise_time_norm=0.04,
    )
    dest, reason = m.dynamic_structure_gate_destination(brain, sample, one_shot_label, 6.0)
    assert dest == "Drums/Folder A/Loops"
    assert "kick" not in reason.lower()
    assert "tom" not in reason.lower()
    assert "snare" not in reason.lower()


def test_apply_conflict_gates_changes_final_label_not_just_audit():
    m = load_module()
    brain = build_structure_brain(m)
    pred_label = "Drums/Alpha/One Shots"
    sample = fp(
        m,
        log_transient_count=np.log1p(10.0),
        temporal_centroid_ratio=0.45,
        onset_interval_regularity=0.35,
        onset_span_ratio=0.82,
        event_rate_hz=2.4,
        tail_energy_ratio=0.45,
        attack_rise_time_norm=0.05,
    )
    final_top, final_label, status, reason = m.apply_learned_conflict_gates(
        brain,
        sample,
        pred_label,
        "Drums",
        "Drums",
        pred_label,
        "auto_place",
        "strong_pick",
        0.80,
        2.0,
        5.5,
        [(pred_label, 0.1), ("Drums/Alpha/Loops", 0.2)],
    )
    assert final_top == "Drums"
    assert final_label.endswith("/Loops")
    assert status == "auto_place"
    assert "dynamic_structure_gate" in reason


def test_dynamic_structure_gate_preserves_identity_when_same_top_has_no_loop_label():
    m = load_module()
    rows = []
    # Predicted top has only one-shot labels.
    for i in range(24):
        rows.append(
            row(
                m,
                f"/train/oneshot_{i}.wav",
                "Drums/Alpha",
                "one_shot",
                {
                    "log_transient_count": np.log1p(1.0),
                    "temporal_centroid_ratio": 0.08,
                    "onset_interval_regularity": 1.0,
                    "onset_span_ratio": 0.05,
                    "event_rate_hz": 0.5,
                    "tail_energy_ratio": 0.12,
                    "attack_rise_time_norm": 0.01,
                },
                duration=0.5,
            )
        )
    # Different learned top has the only loop structure destination.
    for i in range(24):
        rows.append(
            row(
                m,
                f"/train/loop_{i}.wav",
                "Instruments/Omega",
                "loop",
                {
                    "log_transient_count": np.log1p(10.0),
                    "temporal_centroid_ratio": 0.46,
                    "onset_interval_regularity": 0.34,
                    "onset_span_ratio": 0.84,
                    "event_rate_hz": 2.6,
                    "tail_energy_ratio": 0.44,
                    "attack_rise_time_norm": 0.04,
                },
                duration=5.0,
            )
        )
    brain = m.build_brain(rows, max_centroids=3)
    sample = fp(
        m,
        log_transient_count=np.log1p(10.0),
        temporal_centroid_ratio=0.46,
        onset_interval_regularity=0.34,
        onset_span_ratio=0.84,
        event_rate_hz=2.6,
        tail_energy_ratio=0.44,
        attack_rise_time_norm=0.04,
    )
    dest, reason = m.dynamic_structure_gate_destination(brain, sample, "Drums/Alpha/One Shots", 5.5)
    assert dest == "Drums/Alpha/Loops"
    assert "preserved_predicted_identity_with_loop_terminal" in reason
    assert "Instruments/Omega" not in dest


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main(["-q", __file__]))


def test_weak_profile_rival_override_can_switch_tiny_winner_to_reliable_rival():
    m = load_module()
    rows = []
    # Tiny weak current winner: only two examples and therefore weak profile.
    for i in range(2):
        rows.append(
            row(
                m,
                f"/train/tiny_{i}.wav",
                "Instruments/Tiny Low",
                "one_shot",
                {
                    "sub_bass_ratio_lt_150hz": 0.82,
                    "bass_ratio_150_500hz": 0.06,
                    "mid_ratio_500_2000hz": 0.04,
                    "presence_ratio_2000_8000hz": 0.02,
                    "air_ratio_gt_8000hz": 0.01,
                    "pitch_confidence": 0.95,
                    "spectral_flatness_mean": 0.02,
                },
                duration=0.7,
            )
        )
    # Reliable rival with many examples and a stable fact profile.
    for i in range(24):
        rows.append(
            row(
                m,
                f"/train/reliable_{i}.wav",
                "FX/Noise Stable",
                "one_shot",
                {
                    "sub_bass_ratio_lt_150hz": 0.80,
                    "bass_ratio_150_500hz": 0.05,
                    "mid_ratio_500_2000hz": 0.08,
                    "presence_ratio_2000_8000hz": 0.04,
                    "air_ratio_gt_8000hz": 0.03,
                    "pitch_confidence": 0.88,
                    "spectral_flatness_mean": 0.18,
                },
                duration=0.7,
            )
        )
    brain = m.build_brain(rows, max_centroids=3)
    sample = fp(
        m,
        sub_bass_ratio_lt_150hz=0.80,
        bass_ratio_150_500hz=0.05,
        mid_ratio_500_2000hz=0.08,
        presence_ratio_2000_8000hz=0.04,
        air_ratio_gt_8000hz=0.03,
        pitch_confidence=0.88,
        spectral_flatness_mean=0.18,
    )
    # Force the contest condition that exposed the production bug: a tiny weak
    # label wins narrowly, but a reliable rival has stronger measured facts.
    label, _meta = m.weak_profile_rival_override_choose_label(
        brain=brain,
        rows=[
            {"label": "Instruments/Tiny Low/One Shots", "current_score": 1.00, "fact_score": 0.00},
            {"label": "FX/Noise Stable/One Shots", "current_score": 1.55, "fact_score": 0.70},
        ],
        current_label="Instruments/Tiny Low/One Shots",
        raw_x=sample,
        feature_names_list=list(m.FEATURE_NAMES),
        fact_profiles=brain.get("category_fact_profiles", {}),
    )
    assert label == "FX/Noise Stable/One Shots"
