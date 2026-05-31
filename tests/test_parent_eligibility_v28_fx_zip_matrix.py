"""V2.8 regression matrix from one-by-one FX_Aaron2 audit.

These fixtures were selected by file name from FX_Aaron2 only to build a
regression matrix. The sorter itself must not use filenames for placement.
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
AUDIO_DIR = PROJECT_ROOT / "tests" / "regression_audio_v28_fx_zip_matrix"
OUT_DIR = PROJECT_ROOT / "_reports" / "pytest_outputs" / "v28_fx_zip_matrix"
SYNTHETIC_FIXTURE_MODE = AUDIO_DIR.is_symlink()


def _case_list(full: list[str], synthetic: list[str]) -> list[str]:
    """Keep generated stand-in mode small while preserving full private fixtures."""
    return synthetic if SYNTHETIC_FIXTURE_MODE else full


class LazySortedLabels:
    """Sort fixture WAVs one at a time so slow audio cannot time out the file."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    def __getitem__(self, name: str) -> str:
        if name in self._cache:
            return self._cache[name]
        source = AUDIO_DIR / name
        assert source.exists(), f"Missing fixture file: {source}"
        case_out = OUT_DIR / source.stem
        if case_out.exists():
            shutil.rmtree(case_out, ignore_errors=True)
        case_out.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [sys.executable, str(SORTER), "sort", str(source), str(case_out), "--brain", str(BRAIN), "--no-zip"],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout
        rows = list(csv.DictReader((case_out / "Aaron_Sorted_Sounds_manifest.csv").open(errors="replace", newline="")))
        assert rows, f"No manifest rows for {source}"
        self._cache[name] = rows[0]["folder_path"].replace("\\", "/")
        return self._cache[name]


@pytest.fixture(scope="module")
def sorted_labels() -> LazySortedLabels:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR, ignore_errors=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return LazySortedLabels()


def assert_no(label: str, fragments: list[str]) -> None:
    low = label.lower()
    bad = [fragment for fragment in fragments if fragment.lower() in low]
    assert not bad, f"Unexpected {bad} in {label}"


@pytest.mark.parametrize(
    "sample_name",
    _case_list(
        [
            "AA_JBL_78bpm_Cm_Sax_Loop_1.wav",
            "AMV_VRNB1_102_brass_saxophone_loop_cranesinthesky_Am.wav",
        ],
        ["AA_JBL_78bpm_Cm_Sax_Loop_1.wav"],
    ),
)
def test_fx_zip_sax_stays_in_instruments_not_fx_voice_or_drums(sorted_labels: dict[str, str], sample_name: str) -> None:
    label = sorted_labels[sample_name]
    assert label.startswith("Instruments/"), label
    assert_no(label, ["Drums", "FX", "Human", "Voice", "Dog", "Cat", "Bird", "Water", "Machine"])


@pytest.mark.parametrize(
    "sample_name",
    _case_list(
        [
            "EWS_Keys_resampled_HipHop_RnB_G_Major_88BPM.wav",
            "WS2_KIT_1_Electric_Piano_Chords_Fm_101BPM.wav",
            "GS_Synth_Gangsta_Lead_G#min_97bpm.wav",
            "MS_TLV1_03_The Way It Is_Synth Lead 1_Eminor_98bpm_Wet.wav",
        ],
        ["EWS_Keys_resampled_HipHop_RnB_G_Major_88BPM.wav"],
    ),
)
def test_fx_zip_keys_synth_do_not_get_mislabeled_as_brass_woodwind_or_drums(
    sorted_labels: dict[str, str], sample_name: str
) -> None:
    label = sorted_labels[sample_name]
    assert label.startswith("Instruments/"), label
    assert_no(label, ["Brass and Woodwinds", "Drums", "FX", "Human", "Voice", "Animals"])


@pytest.mark.parametrize(
    "sample_name",
    _case_list(
        [
            "Money_vocals_female_rap_110bpm.wav",
            "Phonk_Rap_Vocals_26_keyCmin_151bpm.wav",
            "Vocal Phrase We Up 140bpm.wav",
            "Stab 3.wav",
        ],
        ["Money_vocals_female_rap_110bpm.wav"],
    ),
)
def test_fx_zip_voice_and_vocal_stabs_stay_voice_human(sorted_labels: dict[str, str], sample_name: str) -> None:
    label = sorted_labels[sample_name]
    assert (
        label.startswith("FX/Human and Voice FX")
        or label.startswith("Instruments/Voice")
        or label.startswith("_TO_REVIEW/")
    ), label
    assert_no(label, ["Drums", "Percussion", "Machines", "Motor", "Dog", "Cat", "Bird", "Coins"])


def test_fx_zip_tonal_police_fx_stays_fx_not_generic_instrument(sorted_labels: dict[str, str]) -> None:
    label = sorted_labels["MS_O_01_Outlaw_Fx Police_D#minor_91bpm_Wet.wav"]
    assert label.startswith("FX/"), label
    assert_no(label, ["Drums", "Instrument Loops", "Bass Loops", "Human", "Voice", "Animals"])


@pytest.mark.parametrize(
    "sample_name",
    _case_list(["2.Drum Loop_1_100bpm.wav", "ABOUTME_94_DRUMLOOP.wav"], ["ABOUTME_94_DRUMLOOP.wav"]),
)
def test_fx_zip_drum_loops_stay_drum_loops(sorted_labels: dict[str, str], sample_name: str) -> None:
    label = sorted_labels[sample_name]
    assert label.startswith("Drums/Drum Loops"), label
    assert_no(label, ["Instruments", "FX", "Bass Loops", "Human", "Voice", "Riser"])
