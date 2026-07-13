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
OUT_ROOT = PROJECT_ROOT / "_pytest_regression_outputs" / "v31108_uploaded_audio"


def _run_one(sample_name: str) -> str:
    sample = AUDIO_DIR / sample_name
    if not sample.exists():
        pytest.skip(f"missing regression audio: {sample}")
    out_dir = OUT_ROOT / sample.stem.replace(" ", "_")
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SORTER),
        "sort",
        str(sample),
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
    row = list(csv.DictReader(manifest.open(encoding="utf-8")))[0]
    return (row.get("final_label") or row.get("new_relative_path") or "").replace("\\", "/")


def test_uploaded_synth_bells_loop_is_not_voice() -> None:
    label = _run_one("01_WCS_No_Safety_BPM92_D#min__Bells.wav")
    low = label.lower()
    assert "voice" not in low and "vocal" not in low, label
    assert label.startswith("Instruments/") or label.startswith("FX/"), label


def test_uploaded_belize_loop_is_not_voice() -> None:
    label = _run_one("belize87bpm_8bars_UNKWN (Bbm).wav")
    low = label.lower()
    assert "voice" not in low and "vocal" not in low, label
    assert label.startswith("Instruments/") or label.startswith("FX/"), label


def test_uploaded_av5_hit_escapes_review_and_allows_trained_brass_hit() -> None:
    label = _run_one("AV5_5_94bpm_Hit 2.wav")
    assert not label.startswith("_TO_REVIEW/Measured Role Conflict"), label
    assert (
        label.startswith("Instruments/Brass/Brass Section")
        or label.startswith("FX/")
        or label.startswith("Instruments/Instrument Loops")
    ), label
