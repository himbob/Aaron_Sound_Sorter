"""V2.5 transition-FX and reverb-sax broadening regressions."""

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
AUDIO_DIR = PROJECT_ROOT / "tests" / "regression_audio_v25_transition_reverb"
OUT_DIR = PROJECT_ROOT / "_pytest_regression_outputs" / "parent_eligibility_v25_transition_reverb"


@pytest.fixture(scope="module")
def sorted_rows() -> dict[str, dict[str, str]]:
    """Sort the V2.5 transition/reverb fixture folder once."""
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
    rows = list(csv.DictReader((OUT_DIR / "Aaron_Sorted_Sounds_manifest.csv").open()))
    return {Path(row.get("source_path", "")).name: row for row in rows}


def label_for(rows: dict[str, dict[str, str]], name: str) -> str:
    """Return normalized final folder path for a fixture."""
    assert name in rows, f"Missing {name}; got {sorted(rows)}"
    row = rows[name]
    return (row.get("final_label") or row.get("folder_path") or row.get("placed_path") or "").replace("\\", "/")


def test_reverb_sax_phrase_broadens_to_instruments_instead_of_review_or_drums(
    sorted_rows: dict[str, dict[str, str]],
) -> None:
    """A heavy-reverb sax/reed phrase may broaden, but not to drums/human/FX animals."""
    label = label_for(sorted_rows, "GrimyHipHop_Saxophone_26_Fm_Melody_Dark_Dusty_Warm_Loop_84bpm.wav")
    assert label.startswith("Instruments/") or label.startswith("_TO_REVIEW/"), label
    forbidden = ["Drums", "Human", "Voice", "Dog", "Cat", "Bird", "Percussion"]
    bad = [fragment for fragment in forbidden if fragment.lower() in label.lower()]
    assert not bad, f"Unexpected {bad} in {label}"


def test_real_short_riser_stays_fx_transition_not_drum_loop(sorted_rows: dict[str, dict[str, str]]) -> None:
    """A real riser can be noisy/percussive but must remain FX transition."""
    label = label_for(sorted_rows, "Riser Short Effect.wav")
    assert label.startswith("FX/Structural and Transitional FX/Risers and Builds"), label
    assert not label.startswith("Drums/"), label


def test_vocal_phrase_from_fx_pack_stays_human_voice_not_generic_instrument(
    sorted_rows: dict[str, dict[str, str]],
) -> None:
    """A voiced vocal phrase may be broad, but not generic synth/instrument loop."""
    label = label_for(sorted_rows, "US_CHV2_Vocal_female_shouts_processed_13.wav")
    assert (
        label.startswith("FX/Human and Voice FX")
        or label.startswith("Instruments/Voice")
        or label.startswith("_TO_REVIEW/")
    ), label
    forbidden = ["Drums", "Percussion", "Machines", "Motor", "Dog", "Cat", "Bird", "Instrument Loops", "Synths"]
    bad = [fragment for fragment in forbidden if fragment.lower() in label.lower()]
    assert not bad, f"Unexpected {bad} in {label}"


def test_piano_loop_does_not_get_stolen_by_vocal_phrase_guard(sorted_rows: dict[str, dict[str, str]]) -> None:
    """The vocal phrase guard must not steal clean piano/key loops."""
    label = label_for(sorted_rows, "Piano 1 - 80 Bpm - Key C.wav")
    assert label.startswith("Instruments/") or label.startswith("_TO_REVIEW/"), label
    assert "Human and Voice" not in label and "Voice" not in label, label
