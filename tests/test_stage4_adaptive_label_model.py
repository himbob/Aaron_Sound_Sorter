#!/usr/bin/env python3
"""Regression checks for adaptive per-folder model selection.

The folder path is the output truth.  Physics spread chooses only the internal
model shape: uniform folders get one centroid, messy folders get exemplars plus
multiple centroids.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "Aaron_Sound_Sorter.py"


def load_module():
    from aaron_sound_sorter import api

    return api


def test_uniform_and_messy_folders_choose_different_models():
    m = load_module()
    rows = []
    base = np.zeros(m.FP_SIZE, dtype=np.float32)

    for index in range(8):
        vector = base.copy()
        vector[0] = 0.01 * index
        rows.append(
            m.FeatureRow(
                f"/tmp/uniform_{index}.wav",
                "Drums/Test Uniform",
                "Drums/Test Uniform/One Shots",
                "Drums",
                "one_shot",
                0.1,
                vector.tolist(),
                "ok",
            )
        )

    for index, offset in enumerate([0, 5, 10, 15, 20, 25, 30, 35]):
        vector = base.copy()
        vector[0] = offset
        vector[30] = offset / 10.0
        rows.append(
            m.FeatureRow(
                f"/tmp/messy_{index}.wav",
                "FX/Test Messy",
                "FX/Test Messy/One Shots",
                "FX",
                "one_shot",
                0.5,
                vector.tolist(),
                "ok",
            )
        )

    brain = m.build_brain(rows, max_centroids=7)
    models = brain["label_models_by_label"]

    assert models["Drums/Test Uniform/One Shots"]["model_mode"] == "single_centroid"
    assert models["FX/Test Messy/One Shots"]["model_mode"] == "multi_centroid_exemplar"
    assert "exemplars_by_label" in brain
    assert len(brain["exemplars_by_label"]["FX/Test Messy/One Shots"]) >= 2


def test_train_all_keeps_every_row_and_marks_eval_as_reused_diagnostic():
    m = load_module()
    rows = []
    for index in range(4):
        rows.append(
            {
                "source_path": f"/tmp/guiro_{index}.wav",
                "group_key": "Drums/Percussion/Guiros Scrapes and Rasps/_ONE_SHOTS",
                "outlier_status": "CLEAN_CANDIDATE",
                "duration_sec": "0.18",
                "source": "training_tree_scan",
            }
        )
    train, eval_rows, skipped = m.select_rows(
        rows,
        {"Drums", "Instruments", "FX", "Textures"},
        max_train_per_group=0,
        max_eval_per_label=4,
        max_eval_total=10,
    )
    assert len(skipped) == 0
    assert len(train) == 4
    assert len(eval_rows) == 4
    assert all(row["_eval_reused_training_source"] == "1" for row in eval_rows)
    assert all("diagnostic_training_recall" in row["_eval_selection_method"] for row in eval_rows)


def test_teacher_anchor_preserves_training_recall_for_tiny_label():
    m = load_module()
    base = np.zeros(m.FP_SIZE, dtype=np.float32)
    guiro = base.copy()
    guiro[27] = 3.0
    guiro[29] = 0.65
    guiro[35] = np.log1p(1.0)
    clap = base.copy()
    clap[27] = 3.0
    clap[29] = 0.65
    clap[35] = np.log1p(1.0)
    clap[39] = 0.8
    rows = [
        m.FeatureRow(
            "/tmp/guiro.wav",
            "Drums/Percussion/Guiros Scrapes and Rasps",
            "Drums/Percussion/Guiros Scrapes and Rasps/One Shots",
            "Drums",
            "one_shot",
            0.17,
            guiro.tolist(),
            "ok",
            label_source_group_count=1,
            label_clean_available=1,
        ),
        m.FeatureRow(
            "/tmp/clap.wav",
            "Drums/Claps Snaps Slaps/Generic Clap",
            "Drums/Claps Snaps Slaps/Generic Clap/One Shots",
            "Drums",
            "one_shot",
            0.17,
            clap.tolist(),
            "ok",
            label_source_group_count=1,
            label_clean_available=1,
        ),
    ]
    brain = m.build_brain(rows, max_centroids=5)
    pred_label, _pred_top, sim, _margin, _top5 = m.predict(brain, guiro.tolist(), "")
    assert pred_label == "Drums/Percussion/Guiros Scrapes and Rasps/One Shots"
    assert sim >= 0.99


def test_support_balance_is_count_neutral_for_helper_ranking():
    m = load_module()
    big = np.zeros(m.FP_SIZE, dtype=np.float32)
    small = np.zeros(m.FP_SIZE, dtype=np.float32)
    big[0] = 0.05
    small[0] = 0.10
    x = np.zeros(m.FP_SIZE, dtype=np.float32)

    brain = {
        "labels": ["Top/Large Learned Folder", "Top/Small Learned Folder"],
        "counts": {"Top/Large Learned Folder": 100, "Top/Small Learned Folder": 5},
        "max_label_training_count": 100,
        "feature_weights": np.ones(m.FP_SIZE, dtype=float).tolist(),
        "scaler_mean": np.zeros(m.FP_SIZE, dtype=float).tolist(),
        "scaler_std": np.ones(m.FP_SIZE, dtype=float).tolist(),
        "top_by_label": {"Top/Large Learned Folder": "Top", "Top/Small Learned Folder": "Top"},
        "structure_by_label": {"Top/Large Learned Folder": "one_shot", "Top/Small Learned Folder": "one_shot"},
        "centroids": {
            "Top/Large Learned Folder": [big.tolist()],
            "Top/Small Learned Folder": [small.tolist()],
        },
        "label_models_by_label": {
            "Top/Large Learned Folder": {"model_mode": "single_centroid", "spread_mean": 0.2, "training_count": 100},
            "Top/Small Learned Folder": {"model_mode": "single_centroid", "spread_mean": 0.2, "training_count": 5},
        },
        "exemplars_by_label": {},
        "anchors_by_label": {
            "Top/Large Learned Folder": [big.tolist()],
            "Top/Small Learned Folder": [small.tolist()],
        },
        "typical_dist_mean_by_top": {"Top": 1.0},
        # This flag is intentionally ignored by the refactored helper path.
        # Counts must not change committee/helper ranking; normalized physics does.
        "support_balance_enabled": True,
        "support_balance_threshold_ratio": 0.35,
        "support_balance_max_distance_bonus": 0.16,
    }

    count_neutral_label, *_ = m.predict(brain, x.tolist(), "")

    assert count_neutral_label == "Top/Large Learned Folder"
    assert m.support_balance_bonus_for_label(brain, "Top/Small Learned Folder") == 0.0
    assert m.support_balance_bonus_for_label(brain, "Top/Large Learned Folder") == 0.0


if __name__ == "__main__":
    test_uniform_and_messy_folders_choose_different_models()
    test_train_all_keeps_every_row_and_marks_eval_as_reused_diagnostic()
    test_teacher_anchor_preserves_training_recall_for_tiny_label()
    test_support_balance_is_count_neutral_for_helper_ranking()
    print("PASS stage4 adaptive label model")
