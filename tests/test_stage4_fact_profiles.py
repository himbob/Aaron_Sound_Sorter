#!/usr/bin/env python3
"""Fast architecture tests for v0.4.80 category fact profiles.

These tests do not read audio. They build tiny synthetic FeatureRow pools so the
fact-profile and rival-contrast layers can be checked in under a second.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "Aaron_Sound_Sorter.py"


def load_module():
    from aaron_sound_sorter import api

    return api


def _rows_for_two_close_labels(m):
    rows = []
    base = np.zeros(m.FP_SIZE, dtype=np.float32)
    for idx in range(6):
        kick = base.copy()
        kick[36] = 0.74  # sub_bass_ratio_lt_150hz
        kick[37] = 0.48  # bass_ratio_150_500hz
        kick[40] = 0.09  # air_ratio_gt_8000hz
        rows.append(
            m.FeatureRow(
                f"/tmp/kick_{idx}.wav",
                "Drums/Kick Drums/Generic Kick/_ONE_SHOTS",
                "Drums/Kick Drums/Generic Kick/One Shots",
                "Drums",
                "one_shot",
                0.18,
                kick.tolist(),
                "ok",
            )
        )
        tom = base.copy()
        tom[36] = 0.22
        tom[37] = 0.68
        tom[40] = 0.05
        rows.append(
            m.FeatureRow(
                f"/tmp/tom_{idx}.wav",
                "Drums/Toms/Generic Tom/_ONE_SHOTS",
                "Drums/Toms/Generic Tom/One Shots",
                "Drums",
                "one_shot",
                0.24,
                tom.tolist(),
                "ok",
            )
        )
    return rows


def test_category_fact_profiles_and_rival_contrast_are_written_to_brain():
    m = load_module()
    brain = m.build_brain(_rows_for_two_close_labels(m), max_centroids=3)

    profiles = brain.get("category_fact_profiles", {})
    rivals = brain.get("category_rival_contrast_facts", {})

    assert brain.get("fact_scoring_enabled") is True
    assert brain.get("rival_contrast_enabled") is True
    assert "Drums/Kick Drums/Generic Kick/One Shots" in profiles
    assert "Drums/Toms/Generic Tom/One Shots" in profiles
    assert profiles["Drums/Kick Drums/Generic Kick/One Shots"]["effective_count"] == 6
    assert (
        profiles["Drums/Kick Drums/Generic Kick/One Shots"]["feature_stats"]["sub_bass_ratio_lt_150hz"]["median"]
        == 0.74
    )
    assert "Drums/Toms/Generic Tom/One Shots" in rivals["Drums/Kick Drums/Generic Kick/One Shots"]
    assert rivals["Drums/Kick Drums/Generic Kick/One Shots"]["Drums/Toms/Generic Tom/One Shots"]["best_discriminators"]


def test_predict_stores_fact_and_rival_evidence_for_manifest_writers():
    m = load_module()
    brain = m.build_brain(_rows_for_two_close_labels(m), max_centroids=3)
    sample = np.zeros(m.FP_SIZE, dtype=np.float32)
    sample[36] = 0.74
    sample[37] = 0.48
    sample[40] = 0.09

    label, top, sim, margin, top5 = m.predict(brain, sample.tolist(), "")
    meta = brain.get("_last_fact_meta", {})

    assert label == "Drums/Kick Drums/Generic Kick/One Shots"
    assert top == "Drums"
    assert sim > 0.0
    assert top5
    assert isinstance(meta.get("fact_score"), float)
    assert "Tom" in meta.get("top_rival_label", "")
    assert meta.get("fact_profile_strength") in {"tentative", "ok", "good", "strong"}
    assert "matched_key_facts" in meta
    assert "failed_key_facts" in meta


def test_sort_manifests_include_fact_audit_columns_in_source():
    source = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (PROJECT_ROOT / "src" / "aaron_sound_sorter").glob("*.py")
        if not p.name.startswith(".")
    )
    required = [
        "fact_score",
        "fact_penalty",
        "rival_contrast_score",
        "top_rival_internal_label",
        "top_rival_label",
        "matched_key_facts",
        "failed_key_facts",
        "fact_profile_strength",
    ]
    for column in required:
        assert column in source


if __name__ == "__main__":
    test_category_fact_profiles_and_rival_contrast_are_written_to_brain()
    test_predict_stores_fact_and_rival_evidence_for_manifest_writers()
    test_sort_manifests_include_fact_audit_columns_in_source()
    print("PASS stage4 fact profiles")


def test_rival_contrast_second_pass_can_swap_without_filename_or_category_names():
    m = load_module()
    rows = []
    base = np.zeros(m.FP_SIZE, dtype=np.float32)
    for idx in range(24):
        a = base.copy()
        a[36] = 0.10
        a[37] = 0.20
        b = base.copy()
        b[36] = 0.85
        b[37] = 0.75
        rows.append(
            m.FeatureRow(
                f"/tmp/source_a_{idx}.wav",
                "Top/A/_ONE_SHOTS",
                "Top/A/One Shots",
                "Top",
                "one_shot",
                0.20,
                a.tolist(),
                "ok",
            )
        )
        rows.append(
            m.FeatureRow(
                f"/tmp/source_b_{idx}.wav",
                "Top/B/_ONE_SHOTS",
                "Top/B/One Shots",
                "Top",
                "one_shot",
                0.20,
                b.tolist(),
                "ok",
            )
        )
    brain = m.build_brain(rows, max_centroids=2)
    sample = base.copy()
    sample[36] = 0.85
    sample[37] = 0.75
    contest_rows = [
        {"label": "Top/A/One Shots", "current_score": 1.00, "fact_score": -0.10},
        {"label": "Top/B/One Shots", "current_score": 1.08, "fact_score": 0.90},
    ]
    chosen, meta = m.rival_contrast_second_pass_choose_label(
        brain=brain,
        rows=contest_rows,
        current_label="Top/A/One Shots",
        raw_x=sample,
        feature_names_list=list(brain.get("feature_names", m.FEATURE_NAMES)),
        fact_profiles=brain["category_fact_profiles"],
        rival_contrast_facts=brain["category_rival_contrast_facts"],
    )
    assert chosen == "Top/B/One Shots"
    assert meta["swapped"] is True
    assert meta["from_label"] == "Top/A/One Shots"
    assert meta["to_label"] == "Top/B/One Shots"
    assert float(meta["challenger_contrast_score"]) > 0.3
