from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aaron_audio_intelligence.shape_memory_brain import SHAPE_MEMORY_KEY
from aaron_sound_sorter.core import FP_SIZE
from tools import shape_memory_seed


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


def _write_panel(path: Path, source_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "panel_name": "test",
                "cases": [
                    {
                        "id": "loop",
                        "source_path": str(source_path),
                        "approved_label": "Drums/Drum Loops/Loops",
                        "structure": "loop",
                        "expected_shape": "beat_loop",
                        "evidence_weight": 44,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_shape_memory_seed_dry_run_does_not_write_brain(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path
    audio_path = tmp_path / "seed.wav"
    panel_path = tmp_path / "panel.json"
    base_brain_path = tmp_path / "stage4_folder_brain.json"
    shape_brain_path = tmp_path / "stage4_shape_memory_brain.json"
    report_dir = tmp_path / "report"
    audio_path.write_bytes(b"fake")
    base_brain_path.write_text(json.dumps(_base_brain()), encoding="utf-8")
    _write_panel(panel_path, audio_path)

    monkeypatch.setattr(
        shape_memory_seed,
        "make_fingerprint_safe",
        lambda _path: (np.full((FP_SIZE,), 0.25, dtype=np.float32), 1.0, "ok"),
    )

    summary = shape_memory_seed.run_shape_memory_seed(
        project_root=project_root,
        panel_path=panel_path,
        base_brain_path=base_brain_path,
        shape_memory_path=shape_brain_path,
        report_dir=report_dir,
        apply=False,
        default_weight=96,
    )

    assert summary.applied is False
    assert summary.readable_count == 1
    assert summary.error_count == 0
    assert not shape_brain_path.exists()
    assert "would_apply" in summary.manifest_path.read_text(encoding="utf-8")


def test_shape_memory_seed_apply_writes_shape_memory_brain(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path
    audio_path = tmp_path / "seed.wav"
    panel_path = tmp_path / "panel.json"
    base_brain_path = tmp_path / "stage4_folder_brain.json"
    shape_brain_path = tmp_path / "stage4_shape_memory_brain.json"
    report_dir = tmp_path / "report"
    audio_path.write_bytes(b"fake")
    base_brain_path.write_text(json.dumps(_base_brain()), encoding="utf-8")
    _write_panel(panel_path, audio_path)

    monkeypatch.setattr(
        shape_memory_seed,
        "make_fingerprint_safe",
        lambda _path: (np.full((FP_SIZE,), 0.5, dtype=np.float32), 1.25, "ok"),
    )

    summary = shape_memory_seed.run_shape_memory_seed(
        project_root=project_root,
        panel_path=panel_path,
        base_brain_path=base_brain_path,
        shape_memory_path=shape_brain_path,
        report_dir=report_dir,
        apply=True,
        default_weight=96,
    )
    brain = json.loads(shape_brain_path.read_text(encoding="utf-8"))

    assert summary.applied is True
    assert summary.readable_count == 1
    assert brain["brain_type"] == "shape_memory_brain"
    assert "beat_loop" in brain["labels"]
    assert len(brain[SHAPE_MEMORY_KEY]["shape_examples_by_shape"]["beat_loop"]) == 1
    assert "applied" in summary.manifest_path.read_text(encoding="utf-8")


def test_shape_memory_seed_reports_missing_audio(tmp_path: Path) -> None:
    project_root = tmp_path
    panel_path = tmp_path / "panel.json"
    base_brain_path = tmp_path / "stage4_folder_brain.json"
    shape_brain_path = tmp_path / "stage4_shape_memory_brain.json"
    report_dir = tmp_path / "report"
    base_brain_path.write_text(json.dumps(_base_brain()), encoding="utf-8")
    _write_panel(panel_path, tmp_path / "missing.wav")

    summary = shape_memory_seed.run_shape_memory_seed(
        project_root=project_root,
        panel_path=panel_path,
        base_brain_path=base_brain_path,
        shape_memory_path=shape_brain_path,
        report_dir=report_dir,
        apply=True,
        default_weight=96,
    )

    assert summary.error_count == 1
    assert not shape_brain_path.exists()
    assert "ERROR_MISSING_AUDIO" in summary.manifest_path.read_text(encoding="utf-8")
