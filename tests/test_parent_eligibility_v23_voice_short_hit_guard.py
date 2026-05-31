"""Regression tests for V2.3 vocal and short-hit structure guards.

These are real uploaded failures from the current thread.  The tests lock broad
family safety only: vocal one-shots must not become machines/percussion, and
short pitched one-shots must not be sent to loop folders.
"""

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
BRAIN = PROJECT_ROOT / "stage4_folder_brain.json"
AUDIO_DIR = PROJECT_ROOT / "tests" / "regression_audio_v23_voice_short_hit_guard"
OUT_DIR = PROJECT_ROOT / "_pytest_regression_outputs" / "parent_eligibility_v23_voice_short_hit_guard"


@pytest.fixture(scope="module")
def sorted_rows() -> dict[str, dict[str, str]]:
    """Sort this focused fixture folder once and return rows by filename."""
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
        timeout=180,
    )
    assert result.returncode == 0, result.stdout
    rows = list(csv.DictReader((OUT_DIR / "Aaron_Sorted_Sounds_manifest.csv").open()))
    return {Path(row.get("source_path", "")).name: row for row in rows}


def final_label(row: dict[str, str]) -> str:
    """Return normalized final label."""
    return (row.get("final_label") or row.get("folder_path") or row.get("placed_path") or "").replace("\\", "/")


def assert_not_in(label: str, *fragments: str) -> None:
    """Assert none of the fragments occur in a label."""
    low = label.lower()
    bad = [fragment for fragment in fragments if fragment.lower() in low]
    assert not bad, f"Unexpected {bad} in {label}"


def test_formant_rich_female_vocal_shot_does_not_go_to_machine_or_percussion(
    sorted_rows: dict[str, dict[str, str]],
) -> None:
    """A true vocal shot/Fx should stay in broad Human/Voice, not machines or percussion."""
    label = final_label(sorted_rows["DOJO_CGNB_Female_Vocal_Shot_01_D.wav"])
    assert (
        label.startswith("FX/Human and Voice FX")
        or label.startswith("Instruments/Voice")
        or label.startswith("_TO_REVIEW/")
    ), label
    assert_not_in(label, "Machine", "Motor", "Engine", "Drums", "Percussion", "Dog", "Cat", "Bird")


def test_short_low_pitched_hit_does_not_go_to_loop_folder(sorted_rows: dict[str, dict[str, str]]) -> None:
    """A short pitched hit may be a bass/instrument one-shot, but not a loop."""
    label = final_label(sorted_rows["29793.wav"])
    assert label.startswith("Instruments/") or label.startswith("_TO_REVIEW/"), label
    assert_not_in(label, "Loops", "Long FX", "Drum Loops")


def test_short_percussive_hit_still_stays_out_of_keys_after_vocal_fix(sorted_rows: dict[str, dict[str, str]]) -> None:
    """The vocal fix must not reopen the old Keys/Instrument theft bug."""
    label = final_label(sorted_rows["16791.wav"])
    assert label.startswith("Drums/") or label.startswith("_TO_REVIEW/"), label
    assert_not_in(label, "Keys", "Instruments", "FX", "Human", "Voice")
