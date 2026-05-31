"""Regression tests for V2.2 percussion protected-zone behavior.

These uploaded-failure tests are not leaf-truth tests. They lock the
architecture rule that short percussive hits must not be stolen by FX
animal/ocean/human leaves or instrument guitar/keys/bass leaves.

When Aaron's private regression WAV folder is not present in the checked-in
bundle, the test creates conservative synthetic percussion stand-ins under
_reports. The synthetic fallback tests the same invariant without pretending the
numeric fixture names are available.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests import synthetic_audio_fixtures as fixtures

csv.field_size_limit(sys.maxsize)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SORTER = PROJECT_ROOT / "Aaron_Sound_Sorter.py"
BRAIN = PROJECT_ROOT / "stage4_folder_brain.json"
REAL_AUDIO_DIR = PROJECT_ROOT / "tests" / "regression_audio_percussion_fx_guard"
SYNTHETIC_AUDIO_DIR = PROJECT_ROOT / "_reports" / "generated_regression_audio_percussion_fx_guard"
OUT_DIR = PROJECT_ROOT / "_reports" / "pytest_regression_outputs" / "parent_eligibility_fx_instrument_steal_guard"

PERCUSSIVE_HIT_SAMPLES = ["125591.wav", "13144.wav", "33157.wav", "50728.wav"]
KICK_LIKE_SAMPLE = "16291.wav"
ALL_SAMPLES = [*PERCUSSIVE_HIT_SAMPLES, KICK_LIKE_SAMPLE]


def _real_audio_fixture_complete() -> bool:
    """Return True only when the private real-audio regression folder exists."""
    return REAL_AUDIO_DIR.exists() and all((REAL_AUDIO_DIR / name).exists() for name in ALL_SAMPLES)


def _write_synthetic_percussion_fixture() -> Path:
    """Create physical percussion stand-ins when private WAVs are absent."""
    if SYNTHETIC_AUDIO_DIR.exists():
        shutil.rmtree(SYNTHETIC_AUDIO_DIR, ignore_errors=True)
    SYNTHETIC_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for index, sample_name in enumerate(PERCUSSIVE_HIT_SAMPLES):
        duration = 0.22 + (index * 0.03)
        fixtures.write_wav(
            SYNTHETIC_AUDIO_DIR / sample_name,
            fixtures.synth_percussion_hit(dur=duration),
        )
    fixtures.write_wav(SYNTHETIC_AUDIO_DIR / KICK_LIKE_SAMPLE, fixtures.synth_kick(dur=0.34))
    return SYNTHETIC_AUDIO_DIR


def _audio_input_dir() -> Path:
    """Use real private fixtures when available, otherwise generated stand-ins."""
    if _real_audio_fixture_complete():
        return REAL_AUDIO_DIR
    return _write_synthetic_percussion_fixture()


@pytest.fixture(scope="module")
def sorted_rows() -> dict[str, dict[str, str]]:
    """Sort the percussion-failure fixture folder once."""
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR, ignore_errors=True)
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = _audio_input_dir()
    result = subprocess.run(
        [
            sys.executable,
            str(SORTER),
            "sort",
            str(audio_dir),
            str(OUT_DIR),
            "--brain",
            str(BRAIN),
            "--no-zip",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=220,
    )
    assert result.returncode == 0, result.stdout
    rows = list(csv.DictReader((OUT_DIR / "Aaron_Sorted_Sounds_manifest.csv").open()))
    return {Path(row.get("source_path", "")).name: row for row in rows}


def final_label(row: dict[str, str]) -> str:
    """Return normalized final folder path."""
    return (row.get("final_label") or row.get("folder_path") or row.get("placed_path") or "").replace("\\", "/")


@pytest.mark.parametrize("sample_name", PERCUSSIVE_HIT_SAMPLES)
def test_short_percussion_hits_do_not_go_to_fx_or_instruments(
    sorted_rows: dict[str, dict[str, str]], sample_name: str
) -> None:
    """Short percussive hits must stay in Drums or review, not FX/Instruments."""
    label = final_label(sorted_rows[sample_name])
    assert label.startswith("Drums/") or label.startswith("_TO_REVIEW/"), label
    forbidden = [
        "FX",
        "Instruments",
        "Human",
        "Voice",
        "Animals",
        "Dog",
        "Cat",
        "Bird",
        "Ocean",
        "Water",
        "Guitar",
        "Keys",
        "Bass",
        "Glitch",
        "Stutter",
        "Loops",
    ]
    low = label.lower()
    bad = [fragment for fragment in forbidden if fragment.lower() in low]
    assert not bad, f"Unexpected {bad} in {label}"


def test_short_sub_heavy_kick_hit_does_not_go_to_bass_or_loops(sorted_rows: dict[str, dict[str, str]]) -> None:
    """Sub-heavy short hit may be broad Generic Kick, never bass or loop."""
    label = final_label(sorted_rows[KICK_LIKE_SAMPLE])
    assert label.startswith("Drums/") or label.startswith("_TO_REVIEW/"), label
    low = label.lower()
    assert "bass" not in low and "loop" not in low, label
