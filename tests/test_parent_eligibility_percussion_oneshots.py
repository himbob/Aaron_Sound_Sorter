"""Regression tests for one-shot percussion parent eligibility.

Private real WAV fixtures are used when present. When they are absent from a
handoff bundle, the test creates conservative synthetic one-shot percussion
stand-ins under _reports. The invariant is broad: these sounds must not become
voice, animal, coin, glitch, or long-FX leaves. They should land in Drums or
review until leaf identity is trustworthy.
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
REAL_AUDIO_DIR = PROJECT_ROOT / "tests" / "regression_audio_percussion_probe"
SYNTHETIC_AUDIO_DIR = PROJECT_ROOT / "_reports" / "generated_regression_audio_percussion_probe"
OUT_DIR = PROJECT_ROOT / "_reports" / "pytest_regression_outputs" / "parent_eligibility_percussion_oneshots"

SAMPLE_NAMES = [
    "bongo_455505.wav",
    "clap_455632.wav",
    "cymbal_455634.wav",
    "hat_411348.wav",
    "kick_414683.wav",
    "snare_414679.wav",
    "tambourine_219259.wav",
]


def _real_audio_fixture_complete() -> bool:
    """Return True only when all private real-audio probes are available."""
    return REAL_AUDIO_DIR.exists() and all((REAL_AUDIO_DIR / name).exists() for name in SAMPLE_NAMES)


def _samples_for_name(sample_name: str):
    """Return a simple physical stand-in for the named percussion role."""
    name = sample_name.lower()
    if "kick" in name:
        return fixtures.synth_kick(dur=0.36)
    if "snare" in name:
        return fixtures.synth_snare(dur=0.24)
    if "hat" in name or "cymbal" in name:
        return fixtures.synth_hat(dur=0.18)
    if "clap" in name:
        return fixtures.synth_percussion_hit(dur=0.22)
    return fixtures.synth_percussion_hit(dur=0.30)


def _write_synthetic_percussion_probe() -> Path:
    """Create synthetic one-shot percussion probes if private WAVs are missing."""
    if SYNTHETIC_AUDIO_DIR.exists():
        shutil.rmtree(SYNTHETIC_AUDIO_DIR, ignore_errors=True)
    SYNTHETIC_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for sample_name in SAMPLE_NAMES:
        fixtures.write_wav(SYNTHETIC_AUDIO_DIR / sample_name, _samples_for_name(sample_name))
    return SYNTHETIC_AUDIO_DIR


def _audio_input_dir() -> Path:
    """Use the private real fixture folder when present, otherwise generated probes."""
    if _real_audio_fixture_complete():
        return REAL_AUDIO_DIR
    return _write_synthetic_percussion_probe()


@pytest.fixture(scope="module")
def sorted_rows() -> dict[str, dict[str, str]]:
    """Sort the percussion probe fixture folder once and return rows by filename."""
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR, ignore_errors=True)
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            str(SORTER),
            "sort",
            str(_audio_input_dir()),
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
    manifest = OUT_DIR / "Aaron_Sorted_Sounds_manifest.csv"
    rows = list(csv.DictReader(manifest.open()))
    return {Path(row.get("source_path", "")).name: row for row in rows}


def final_label(row: dict[str, str]) -> str:
    """Return the normalized final label/folder path."""
    return (row.get("final_label") or row.get("folder_path") or row.get("placed_path") or "").replace("\\", "/")


@pytest.mark.parametrize("sample_name", SAMPLE_NAMES)
def test_real_percussion_one_shots_stay_in_drums_or_review(
    sorted_rows: dict[str, dict[str, str]], sample_name: str
) -> None:
    """One-shot percussion must not become voice, animal, coin, glitch, or long-FX leaves."""
    assert sample_name in sorted_rows, f"Missing row for {sample_name}"
    label = final_label(sorted_rows[sample_name])
    assert label.startswith("Drums/") or label.startswith("_TO_REVIEW/"), label
    forbidden = [
        "Human and Voice",
        "Voice",
        "Animals",
        "Dog",
        "Bird",
        "Cat",
        "Keys Coins",
        "Coins",
        "Glitch",
        "Long FX",
    ]
    low = label.lower()
    bad = [fragment for fragment in forbidden if fragment.lower() in low]
    assert not bad, f"Unexpected {bad} in {label}"
