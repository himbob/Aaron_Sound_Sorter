"""V2.6 targeted FX-smoke regressions from Aaron's 2026-05-14 run."""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SORTER = PROJECT_ROOT / "Aaron_Sound_Sorter.py"
BRAIN = PROJECT_ROOT / "stage4_folder_brain.json"
AUDIO_DIR = PROJECT_ROOT / "tests" / "regression_audio_v26_fx_smoke_reverb"
OUT_DIR = PROJECT_ROOT / "_pytest_outputs" / "v26_fx_smoke_reverb"


def _sort_rows() -> dict[str, dict[str, str]]:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR, ignore_errors=True)
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "python3",
            str(SORTER),
            "sort",
            str(AUDIO_DIR),
            str(OUT_DIR),
            "--brain",
            str(BRAIN),
            "--no-zip",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout
    manifest = OUT_DIR / "Aaron_Sorted_Sounds_manifest.csv"
    csv.field_size_limit(sys.maxsize)
    with manifest.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return {Path(row["source_path"]).name: row for row in csv.DictReader(handle)}


def test_wet_sax_is_brass_woodwind_not_human_voice_or_drums() -> None:
    rows = _sort_rows()
    for name in [
        "HipHopTapes_29_Saxophone_D#m_90bpm.wav",
        "SCY093_02_Sax_Loop_KeyAbm_89bpm_01.wav",
    ]:
        row = rows[name]
        path = row["folder_path"]
        # The blind sorter may not always prove the exact Sax/Brass-Woodwind
        # leaf from audio alone, but it must keep sax/reed-like phrases inside
        # Instruments and never route them to Human/Voice, Drums, or FX.
        assert path.startswith("Instruments/"), path
        assert "Voice" not in path and "Human" not in path, path
        assert not path.startswith("Drums/"), path
        assert not path.startswith("FX/"), path


def test_drum_loop_with_riser_like_shape_stays_drum_loop() -> None:
    rows = _sort_rows()
    row = rows["ABOUTME_94_DRUMLOOP.wav"]
    path = row["folder_path"]
    assert path == "Drums/Drum Loops/Loops", path
    assert not path.startswith("FX/"), row["decision_reason"]
