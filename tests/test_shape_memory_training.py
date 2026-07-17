from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aaron_audio_intelligence.shape_memory_brain import SHAPE_MEMORY_KEY
from aaron_sound_sorter import shape_memory_training
from aaron_sound_sorter.commands import build_argument_parser
from aaron_sound_sorter.core import FP_SIZE, FeatureRow


def _base_brain() -> dict[str, object]:
    return {
        "version": "test",
        "labels": [],
        "feature_size": FP_SIZE,
        "feature_names": [f"f{i}" for i in range(FP_SIZE)],
        "feature_weights": [1.0 for _ in range(FP_SIZE)],
        "scaler_mean": [0.0 for _ in range(FP_SIZE)],
        "scaler_std": [1.0 for _ in range(FP_SIZE)],
    }


def _feature_row(source_path: Path) -> FeatureRow:
    return FeatureRow(
        path=str(source_path),
        group_key="Instruments/Synths/Synth Loops/_LOOPS",
        label="Instruments/Synths/Synth Loops/Loops",
        top="Instruments",
        structure="loop",
        duration_sec=2.5,
        fingerprint=np.full((FP_SIZE,), 0.4, dtype=np.float32).astype(float).tolist(),
        read_status="ok",
    )


def test_training_tree_builds_low_trust_starter_shape_memory(tmp_path: Path, monkeypatch) -> None:
    training_root = tmp_path / "training"
    training_root.mkdir()
    source_path = training_root / "teacher.wav"
    source_path.write_bytes(b"fake")
    base_brain_path = tmp_path / "stage4_folder_brain.json"
    starter_path = tmp_path / "stage4_shape_memory_starter_brain.json"
    base_brain_path.write_text(json.dumps(_base_brain()), encoding="utf-8")

    monkeypatch.setattr(
        shape_memory_training,
        "scan_training_tree_rows",
        lambda *_args, **_kwargs: [{"source_path": str(source_path), "group_key": "Instruments/Synths/Synth Loops"}],
    )
    monkeypatch.setattr(
        shape_memory_training,
        "select_rows",
        lambda rows, *_args, **_kwargs: (rows, [], []),
    )
    monkeypatch.setattr(
        shape_memory_training,
        "extract_feature_rows",
        lambda *_args, **_kwargs: [_feature_row(source_path)],
    )

    summary = shape_memory_training.train_shape_memory_starter_from_tree(
        training_root=training_root,
        base_brain_path=base_brain_path,
        output_brain_path=starter_path,
        report_dir=tmp_path / "reports",
        allowed_top="Drums,Instruments,FX",
        evidence_weight=8,
        apply=True,
    )
    brain = json.loads(starter_path.read_text(encoding="utf-8"))
    examples = brain[SHAPE_MEMORY_KEY]["shape_examples_by_shape"]["pitched_repetition_phrase"]

    assert summary.applied
    assert summary.applied_row_count == 1
    assert summary.shape_count == 1
    assert examples[0]["memory_source"] == "training_tree_starter"
    assert examples[0]["incremental_gui_correction"] is False
    assert examples[0]["human_override_evidence_weight"] == 8
    assert "APPLIED" in summary.manifest_path.read_text(encoding="utf-8")


def test_training_tree_dry_run_writes_reports_without_brain(tmp_path: Path, monkeypatch) -> None:
    training_root = tmp_path / "training"
    training_root.mkdir()
    source_path = training_root / "teacher.wav"
    source_path.write_bytes(b"fake")
    base_brain_path = tmp_path / "stage4_folder_brain.json"
    starter_path = tmp_path / "stage4_shape_memory_starter_brain.json"
    base_brain_path.write_text(json.dumps(_base_brain()), encoding="utf-8")

    monkeypatch.setattr(
        shape_memory_training,
        "scan_training_tree_rows",
        lambda *_args, **_kwargs: [{"source_path": str(source_path), "group_key": "Drums/Drum Loops"}],
    )
    monkeypatch.setattr(shape_memory_training, "select_rows", lambda rows, *_args, **_kwargs: (rows, [], []))
    monkeypatch.setattr(
        shape_memory_training, "extract_feature_rows", lambda *_args, **_kwargs: [_feature_row(source_path)]
    )

    summary = shape_memory_training.train_shape_memory_starter_from_tree(
        training_root=training_root,
        base_brain_path=base_brain_path,
        output_brain_path=starter_path,
        report_dir=tmp_path / "reports",
        allowed_top="Drums,Instruments,FX",
        apply=False,
    )

    assert not summary.applied
    assert not starter_path.exists()
    assert "WOULD_APPLY" in summary.manifest_path.read_text(encoding="utf-8")


def test_training_commands_enable_shape_memory_starter_by_default() -> None:
    parser = build_argument_parser()

    train_args = parser.parse_args(["train-brain", "/tmp/training"])
    family_args = parser.parse_args(["train-brain-family", "/tmp/training"])
    disabled_args = parser.parse_args(["train-brain", "/tmp/training", "--no-train-shape-memory"])

    assert train_args.train_shape_memory is True
    assert family_args.train_shape_memory is True
    assert disabled_args.train_shape_memory is False
    assert train_args.shape_memory_train_per_label == 3
    assert train_args.shape_memory_evidence_weight == 8
