"""Regression tests for V2.4 drum-loop protection.

These fixtures come from FX_Aaron2 by filename/path selection only for test
coverage.  The sorter itself must not use filename truth.  The tests lock the
parent-safety rule that repeated drum/percussion loops with bass body cannot be
stolen by Instruments/Bass or generic Instrument Loops.
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
GUARD_AUDIO_DIR = PROJECT_ROOT / "tests" / "regression_audio_drum_loop_steal_guard"
FALLBACK_AUDIO_DIR = PROJECT_ROOT / "tests" / "regression_audio"
AUDIO_DIRS = [path for path in (FALLBACK_AUDIO_DIR, GUARD_AUDIO_DIR) if path.exists()]
AUDIO_DIR = AUDIO_DIRS[0] if AUDIO_DIRS else FALLBACK_AUDIO_DIR
OUT_DIR = PROJECT_ROOT / "_reports" / "pytest_outputs" / "parent_eligibility_v24_drum_loop_steal_guard"


def _fixture_path(name: str) -> Path:
    """Return the first available source-name-blind fixture copy for a case."""
    for audio_dir in AUDIO_DIRS:
        candidate = audio_dir / name
        if candidate.exists():
            return candidate
    return AUDIO_DIR / name


def _case_list(full: list[str], synthetic: list[str]) -> list[str]:
    """Run every requested real fixture that exists, including fallback regression audio."""
    seen: set[str] = set()
    selected: list[str] = []
    for name in [*full, *synthetic]:
        if name not in seen and _fixture_path(name).exists():
            selected.append(name)
            seen.add(name)
    return selected


class LazySortedRows:
    """Sort fixture WAVs one at a time so pytest does not time out."""

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, str]] = {}

    def __contains__(self, name: str) -> bool:
        return _fixture_path(name).exists() or name in self._cache

    def __getitem__(self, name: str) -> dict[str, str]:
        if name in self._cache:
            return self._cache[name]
        source = _fixture_path(name)
        assert source.exists(), f"Missing fixture file: {source}"
        case_out = OUT_DIR / source.stem
        if case_out.exists():
            shutil.rmtree(case_out, ignore_errors=True)
        case_out.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                sys.executable,
                str(SORTER),
                "sort",
                str(source),
                str(case_out),
                "--brain",
                str(BRAIN),
                "--no-zip",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout
        manifest = case_out / "Aaron_Sorted_Sounds_manifest.csv"
        csv.field_size_limit(sys.maxsize)
        rows = list(csv.DictReader(manifest.open("r", encoding="utf-8", errors="replace", newline="")))
        assert rows, f"No manifest rows for {source}"
        self._cache[name] = rows[0]
        return rows[0]

    def keys(self) -> list[str]:
        names: set[str] = set()
        for audio_dir in AUDIO_DIRS:
            names.update(p.name for p in audio_dir.glob("*.wav"))
        return sorted(names)


@pytest.fixture(scope="module")
def sorted_rows() -> LazySortedRows:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR, ignore_errors=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return LazySortedRows()


def label_for(rows: dict[str, dict[str, str]], name: str) -> str:
    """Return normalized final folder path for a fixture."""
    assert name in rows, f"Missing {name}; got {sorted(rows)}"
    row = rows[name]
    return (row.get("final_label") or row.get("folder_path") or row.get("placed_path") or "").replace("\\", "/")


@pytest.mark.parametrize(
    "sample_name",
    _case_list(
        [
            "Full Drum Loop 03 - 78BPM.wav",
            "JL_TDBL_Drum Full_act up_138bpm.wav",
            "SCY097_03_Drums_Full_Loop_90bpm_03.wav",
            "US_JF_Drum_140_Dapocket_FULL.wav",
            "2.Drum Loop_1_100bpm.wav",
            "A1_Kick_Clap_Loop_99bpm.wav",
            "MKS_98_Beat1.wav",
            "QUp_DzU_DrumLp_03_100bpm.wav",
            "WS2_KIT_1_WestCoast_Drum_&_Perc_Loop_101BPM.wav",
        ],
        ["Full Drum Loop 03 - 78BPM.wav", "2.Drum Loop_1_100bpm.wav"],
    ),
)
def test_low_body_drum_loops_do_not_become_bass_or_instrument_loops(
    sorted_rows: dict[str, dict[str, str]], sample_name: str
) -> None:
    """Drum loops with bass body must remain Drum Loops or review."""
    label = label_for(sorted_rows, sample_name)
    assert label.startswith("Drums/Drum Loops") or label.startswith("_TO_REVIEW/"), label
    low = label.lower()
    forbidden = ["instruments", "bass loops", "instrument loops", "fx", "voice", "human"]
    bad = [fragment for fragment in forbidden if fragment in low]
    assert not bad, f"Unexpected {bad} in {label}"


@pytest.mark.parametrize(
    "sample_name",
    _case_list(
        [
            "03_bass_Emn_178bpm.wav",
            "04.bass_92bpm_Em.wav",
            "FL_TR_Kit02_96_Bass_Loop_Synth_Gm.wav",
        ],
        ["03_bass_Emn_178bpm.wav"],
    ),
)
def test_clean_bass_loops_are_not_forced_to_drum_loops(
    sorted_rows: dict[str, dict[str, str]], sample_name: str
) -> None:
    """Clean tonal bass loops should not be caught by drum-loop protection."""
    label = label_for(sorted_rows, sample_name)
    assert label.startswith("Instruments/") or label.startswith("_TO_REVIEW/"), label
    assert not label.startswith("Drums/Drum Loops"), label


def test_sax_loop_stays_in_instruments_or_review(sorted_rows: dict[str, dict[str, str]]) -> None:
    """Sax loop is not solved to leaf identity yet, but must remain safe."""
    label = label_for(sorted_rows, "AA_JBL_78bpm_Cm_Sax_Loop_1.wav")
    assert label.startswith("Instruments/") or label.startswith("_TO_REVIEW/"), label
    forbidden = ["Drums", "Human", "Voice", "Dog", "Cat", "Bird", "FX"]
    bad = [fragment for fragment in forbidden if fragment.lower() in label.lower()]
    assert not bad, f"Unexpected {bad} in {label}"


@pytest.mark.parametrize(
    "sample_name",
    _case_list(
        ["DOJO_FBP_Female_Vocal_Shout.wav", "DOJO_CGNB_Female_Vocal_Shot_01_D.wav"],
        ["DOJO_CGNB_Female_Vocal_Shot_01_D.wav"],
    ),
)
def test_vocal_oneshots_stay_voice_or_review(sorted_rows: dict[str, dict[str, str]], sample_name: str) -> None:
    """Vocal one-shots must not become percussion/machine/animal leaves."""
    label = label_for(sorted_rows, sample_name)
    assert (
        label.startswith("FX/Human and Voice FX")
        or label.startswith("Instruments/Voice")
        or label.startswith("_TO_REVIEW/")
    ), label
    forbidden = ["Drums", "Percussion", "Machines", "Motor", "Dog", "Cat", "Bird"]
    bad = [fragment for fragment in forbidden if fragment.lower() in label.lower()]
    assert not bad, f"Unexpected {bad} in {label}"
