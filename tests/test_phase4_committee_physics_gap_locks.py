#!/usr/bin/env python3
"""Phase 4 committee physics gap tests.

These tests lock the architecture Aaron asked for:
- MFCCs can remain out of the human-readable membership violation list.
- MFCCs still influence fact-profile scoring/ranking.
- Membership penalties are weighted by learned rival discriminators, so a
  non-discriminating feature violation is weaker than a key separator violation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "Aaron_Sound_Sorter.py"


def load_module():
    from aaron_sound_sorter import api

    return api


def feature_index(m, name: str) -> int:
    return list(m.FEATURE_NAMES).index(name)


def make_row(m, label: str, top: str, path: str, values: dict[str, float]):
    fp = np.zeros(m.FP_SIZE, dtype=np.float32)
    for key, value in values.items():
        fp[feature_index(m, key)] = float(value)
    return m.FeatureRow(
        path=path,
        group_key=label.rsplit("/", 1)[0],
        label=label,
        top=top,
        structure="one_shot",
        duration_sec=0.25,
        fingerprint=fp.tolist(),
        read_status="ok",
    )


def test_mfcc_ignored_by_membership_but_used_by_fact_profile_scoring():
    m = load_module()
    label_a = "Drums/Test A/One Shots"
    label_b = "Drums/Test B/One Shots"
    rows = []
    for i in range(8):
        rows.append(
            make_row(
                m,
                label_a,
                "Drums",
                f"/tmp/a_{i}.wav",
                {
                    "mfcc_mu_1": -20.0,
                    "sub_bass_ratio_lt_150hz": 0.40,
                    "spectral_flatness_mean": 0.20,
                    "attack_rise_time_norm": 0.10,
                },
            )
        )
        rows.append(
            make_row(
                m,
                label_b,
                "Drums",
                f"/tmp/b_{i}.wav",
                {
                    "mfcc_mu_1": 20.0,
                    "sub_bass_ratio_lt_150hz": 0.40,
                    "spectral_flatness_mean": 0.20,
                    "attack_rise_time_norm": 0.10,
                },
            )
        )
    brain = m.build_brain(rows, max_centroids=2)

    sample = np.zeros(m.FP_SIZE, dtype=np.float32)
    sample[feature_index(m, "mfcc_mu_1")] = 20.0
    sample[feature_index(m, "sub_bass_ratio_lt_150hz")] = 0.40
    sample[feature_index(m, "spectral_flatness_mean")] = 0.20
    sample[feature_index(m, "attack_rise_time_norm")] = 0.10

    membership_a = m.folder_membership_evidence(brain, sample.tolist(), label_a)
    severe_text = " ".join(membership_a.get("severe_features", []) + membership_a.get("violation_features", []))
    assert "mfcc_mu_1" not in severe_text

    fact_a = m.score_sample_against_label_facts(
        sample.tolist(), brain["category_fact_profiles"][label_a], brain["feature_names"]
    )
    fact_b = m.score_sample_against_label_facts(
        sample.tolist(), brain["category_fact_profiles"][label_b], brain["feature_names"]
    )
    assert fact_b > fact_a


def test_membership_severity_uses_learned_rival_discriminators():
    m = load_module()
    label_a = "Drums/Kickish/One Shots"
    label_b = "Drums/Tomish/One Shots"
    rows = []
    for i in range(12):
        rows.append(
            make_row(
                m,
                label_a,
                "Drums",
                f"/tmp/a_{i}.wav",
                {
                    "sub_bass_ratio_lt_150hz": 0.80,
                    "spectral_flatness_mean": 0.20,
                    "stereo_width": 0.10,
                },
            )
        )
        rows.append(
            make_row(
                m,
                label_b,
                "Drums",
                f"/tmp/b_{i}.wav",
                {
                    "sub_bass_ratio_lt_150hz": 0.10,
                    "spectral_flatness_mean": 0.20,
                    "stereo_width": 0.10,
                },
            )
        )
    brain = m.build_brain(rows, max_centroids=2)

    non_key_violation = np.zeros(m.FP_SIZE, dtype=np.float32)
    non_key_violation[feature_index(m, "sub_bass_ratio_lt_150hz")] = 0.80
    non_key_violation[feature_index(m, "spectral_flatness_mean")] = 0.20
    non_key_violation[feature_index(m, "stereo_width")] = 0.95

    key_violation = np.zeros(m.FP_SIZE, dtype=np.float32)
    key_violation[feature_index(m, "sub_bass_ratio_lt_150hz")] = 0.10
    key_violation[feature_index(m, "spectral_flatness_mean")] = 0.20
    key_violation[feature_index(m, "stereo_width")] = 0.10

    ev_non_key = m.folder_membership_evidence(brain, non_key_violation.tolist(), label_a)
    ev_key = m.folder_membership_evidence(brain, key_violation.tolist(), label_a)

    assert ev_non_key["enabled"] and ev_key["enabled"]
    assert float(ev_key["weighted_severity"]) > float(ev_non_key["weighted_severity"])
    assert any(
        "sub_bass_ratio_lt_150hz" in x and "disc=" in x
        for x in ev_key["severe_features"] + ev_key["violation_features"]
    )
    assert any(
        "stereo_width" in x and "disc=0.00" in x
        for x in ev_non_key["severe_features"] + ev_non_key["violation_features"]
    )


def test_committee_vote_breakdown_includes_rival_weighted_membership_signal():
    m = load_module()
    label_good = "Drums/Good Physics/One Shots"
    label_bad = "Drums/Bad Physics/One Shots"
    rows = []
    for i in range(12):
        rows.append(
            make_row(
                m,
                label_good,
                "Drums",
                f"/tmp/g_{i}.wav",
                {
                    "sub_bass_ratio_lt_150hz": 0.78,
                    "attack_rise_time_norm": 0.10,
                    "spectral_flatness_mean": 0.18,
                },
            )
        )
        rows.append(
            make_row(
                m,
                label_bad,
                "Drums",
                f"/tmp/b_{i}.wav",
                {
                    "sub_bass_ratio_lt_150hz": 0.08,
                    "attack_rise_time_norm": 0.70,
                    "spectral_flatness_mean": 0.75,
                },
            )
        )
    brain = m.build_brain(rows, max_centroids=2)
    sample = np.zeros(m.FP_SIZE, dtype=np.float32)
    sample[feature_index(m, "sub_bass_ratio_lt_150hz")] = 0.78
    sample[feature_index(m, "attack_rise_time_norm")] = 0.10
    sample[feature_index(m, "spectral_flatness_mean")] = 0.18

    contest_rows = [
        {"label": label_bad, "current_score": 1.00, "raw_distance": 1.00, "fact_score": -0.5},
        {"label": label_good, "current_score": 1.06, "raw_distance": 1.06, "fact_score": 0.8},
    ]
    chosen, meta = m.committee_agreement_choose_label(
        brain=brain,
        fingerprint=sample.tolist(),
        rows=contest_rows,
        current_label=label_bad,
        feature_names_list=list(brain["feature_names"]),
        top_scores=[("Drums", 0.9)],
        structure_scores=[("one_shot", 0.9)],
        rival_contrast_facts=brain["category_rival_contrast_facts"],
    )
    assert chosen == label_good
    assert meta["switched"] is True
    assert "chosen_votes" in meta
    assert float(meta["chosen_votes"]["membership"]) > 0.5
