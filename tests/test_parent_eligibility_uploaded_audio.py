"""Regression tests for parent-eligibility failures from uploaded audio.

These tests assert broad family safety, not exact leaf identity. They should not
be fixed with filename rules. Private uploaded WAV fixtures are used when the
folder is present; handoff bundles generate conservative synthetic stand-ins
under _reports so pytest does not fail just because private audio was not
included.
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
REAL_AUDIO_DIR = PROJECT_ROOT / "tests" / "regression_audio_uploaded_current"
SYNTHETIC_AUDIO_DIR = PROJECT_ROOT / "_reports" / "generated_regression_audio_uploaded_current"
LOCKED_SMOKE_AUDIO_DIR = PROJECT_ROOT / "tests" / "acceptance" / "locked_smoke_v1" / "samples"
OUT_DIR = PROJECT_ROOT / "_reports" / "pytest_regression_outputs" / "parent_eligibility_uploaded"

SAX_AND_PITCHED_LOOP_SAMPLES = [
    "GrimyHipHop_Saxophone_26_Fm_Melody_Dark_Dusty_Warm_Loop_84bpm.wav",
    "AA_JBL_78bpm_Cm_Sax_Loop_1.wav",
]
DRUM_LOOP_SAMPLES = [
    "NDAEV4_FULL_DRUM_LOOP_01_AFTERLIFE_74BPM.wav",
    "THO_ck4_drums top_130 bpm.wav",
    "DRUMS_LOOP_125BPM.wav",
]
VOCAL_STAB_SAMPLE = "Stab 3.wav"
CHOPPY_MUSIC_LOOP_SAMPLE = "AV5_5_94bpm_Hit 2.wav"
ALL_SAMPLES = [
    *SAX_AND_PITCHED_LOOP_SAMPLES,
    *DRUM_LOOP_SAMPLES,
    VOCAL_STAB_SAMPLE,
    CHOPPY_MUSIC_LOOP_SAMPLE,
]


def _real_audio_fixture_complete() -> bool:
    """Return True only when all private uploaded-audio fixtures are present."""
    return REAL_AUDIO_DIR.exists() and all((REAL_AUDIO_DIR / name).exists() for name in ALL_SAMPLES)


def _stand_in_for_name(sample_name: str):
    """Return measured-audio stand-ins matching each broad invariant role."""
    name = sample_name.lower()
    if "sax" in name:
        return fixtures.synth_harmonic_phrase(360.0, dur=2.6)
    if "drum" in name:
        return fixtures.synth_drum_loop(sample_name, dur=2.4)
    if "stab" in name:
        return fixtures.synth_voice_like(240.0, dur=0.65)
    return fixtures.synth_drum_loop(sample_name, dur=2.0)


def _locked_smoke_source_for_name(sample_name: str) -> Path | None:
    """Return a bundled real sample with the same broad role when one exists."""
    mapping = {
        "AA_JBL_78bpm_Cm_Sax_Loop_1.wav": "AA_JBL_78bpm_Cm_Sax_Loop_1.wav",
        "GrimyHipHop_Saxophone_26_Fm_Melody_Dark_Dusty_Warm_Loop_84bpm.wav": (
            "GrimyHipHop_Saxophone_15_Fm_Melody_Dark_Dusty_Warm_Loop_84bpm.wav"
        ),
        "Stab 3.wav": "DOJO_CGNB_Female_Vocal_Shot_01_D.wav",
        "AV5_5_94bpm_Hit 2.wav": "AV5_5_94bpm_Hit 2.wav",
    }
    candidate_name = mapping.get(sample_name)
    if not candidate_name:
        return None
    candidate = LOCKED_SMOKE_AUDIO_DIR / candidate_name
    if candidate.exists():
        return candidate
    return None


def _write_synthetic_uploaded_fixture() -> Path:
    """Create bundled-real or synthetic stand-ins for private uploaded fixtures."""
    if SYNTHETIC_AUDIO_DIR.exists():
        shutil.rmtree(SYNTHETIC_AUDIO_DIR, ignore_errors=True)
    SYNTHETIC_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for sample_name in ALL_SAMPLES:
        target = SYNTHETIC_AUDIO_DIR / sample_name
        locked_source = _locked_smoke_source_for_name(sample_name)
        if locked_source is not None:
            shutil.copy2(locked_source, target)
        else:
            fixtures.write_wav(target, _stand_in_for_name(sample_name))
    return SYNTHETIC_AUDIO_DIR


def _audio_input_dir() -> Path:
    """Use real private fixtures when complete, otherwise generated stand-ins."""
    if _real_audio_fixture_complete():
        return REAL_AUDIO_DIR
    return _write_synthetic_uploaded_fixture()


@pytest.fixture(scope="module")
def sorted_rows() -> dict[str, dict[str, str]]:
    """Sort the uploaded fixture folder once and return manifest rows by filename."""
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR, ignore_errors=True)
    OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SORTER),
        "sort",
        str(_audio_input_dir()),
        str(OUT_DIR),
        "--brain",
        str(BRAIN),
        "--no-zip",
    ]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout
    manifest = OUT_DIR / "Aaron_Sorted_Sounds_manifest.csv"
    assert manifest.exists(), result.stdout
    rows = list(csv.DictReader(manifest.open()))
    by_name: dict[str, dict[str, str]] = {}
    for row in rows:
        by_name[Path(row.get("source_path", "")).name] = row
    return by_name


def label_for(sorted_rows: dict[str, dict[str, str]], name: str) -> str:
    """Return normalized final label for a fixture."""
    assert name in sorted_rows, f"Missing manifest row for {name}. Got {list(sorted_rows)}"
    row = sorted_rows[name]
    label = row.get("final_label") or row.get("folder_path") or row.get("placed_path") or ""
    assert label, row
    return label.replace("\\", "/")


def assert_not_in(label: str, *fragments: str) -> None:
    """Assert label avoids unsafe family or leaf fragments."""
    low = label.lower()
    bad = [fragment for fragment in fragments if fragment.lower() in low]
    assert not bad, f"Unexpected {bad} in {label}"


@pytest.mark.parametrize("sample_name", SAX_AND_PITCHED_LOOP_SAMPLES)
def test_sax_and_pitched_loops_do_not_land_in_drums_or_fx_foley(
    sorted_rows: dict[str, dict[str, str]], sample_name: str
) -> None:
    label = label_for(sorted_rows, sample_name)
    assert label.startswith("Instruments/") or label.startswith("_TO_REVIEW/"), label
    assert_not_in(label, "Drums", "Dog", "Bird", "Coins", "Breath", "Spoken Voice", "Water", "Drops")


@pytest.mark.parametrize("sample_name", DRUM_LOOP_SAMPLES)
def test_uploaded_drum_loops_do_not_land_in_fx_voice_or_long_fx(
    sorted_rows: dict[str, dict[str, str]], sample_name: str
) -> None:
    label = label_for(sorted_rows, sample_name)
    assert label.startswith("Drums/Drum Loops") or label.startswith("_TO_REVIEW/"), label
    assert_not_in(label, "FX", "Breath", "Spoken Voice", "Drops", "Water", "Animals")


def test_uploaded_vocal_stab_does_not_land_in_dog_or_bird(sorted_rows: dict[str, dict[str, str]]) -> None:
    label = label_for(sorted_rows, VOCAL_STAB_SAMPLE)
    assert (
        label.startswith("FX/Human and Voice FX")
        or label.startswith("Instruments/Voice")
        or label.startswith("_TO_REVIEW/")
    ), label
    assert_not_in(label, "Dog", "Bird", "Cat", "Animals", "Drums")


def test_choppy_music_loop_does_not_land_in_coin_foley(sorted_rows: dict[str, dict[str, str]]) -> None:
    label = label_for(sorted_rows, CHOPPY_MUSIC_LOOP_SAMPLE)
    assert label.startswith("Instruments/") or label.startswith("Drums/") or label.startswith("_TO_REVIEW/"), label
    assert_not_in(label, "Coins", "Keys Coins", "Dog", "Bird", "Water", "Natural Ambience")
