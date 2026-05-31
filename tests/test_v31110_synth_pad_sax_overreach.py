from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

csv.field_size_limit(sys.maxsize)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SORTER = PROJECT_ROOT / "Aaron_Sound_Sorter.py"
AUDIO_DIR = PROJECT_ROOT / "tests" / "regression_audio"
OUT_ROOT = PROJECT_ROOT / "_pytest_regression_outputs" / "v31110_synth_pad_sax_overreach"


def _run_neutralized_audio(sample_name: str) -> dict[str, str]:
    sample = AUDIO_DIR / sample_name
    if not sample.exists():
        pytest.skip(f"missing regression audio: {sample}")
    case_root = OUT_ROOT / sample.stem.replace(" ", "_")
    input_dir = case_root / "input"
    out_dir = case_root / "output"
    if case_root.exists():
        shutil.rmtree(case_root, ignore_errors=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    neutral = input_dir / "sample_0001.wav"
    shutil.copy2(sample, neutral)
    cmd = [
        sys.executable,
        str(SORTER),
        "sort",
        str(neutral),
        str(out_dir),
        "--no-zip",
        "--workers",
        "1",
    ]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=75,
    )
    assert result.returncode == 0, result.stdout
    manifest = out_dir / "Aaron_Sorted_Sounds_manifest.csv"
    assert manifest.exists(), result.stdout
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    assert len(rows) == 1, result.stdout
    return rows[0]


def test_uploaded_pad_loop_is_not_stolen_by_sax_invariant() -> None:
    row = _run_neutralized_audio("CS_NJ2_135bpm_Pad_Aster_Am.wav")
    label = (row.get("final_label") or row.get("new_relative_path") or "").replace("\\", "/")
    low = label.lower()

    assert label.startswith("Instruments/Synths/"), label
    assert "pad" in low, label
    assert "sax" not in low and "woodwind" not in low, label
    assert row.get("consensus_status") == "final_measured_synth_loop_invariant"
