"""One-file-at-a-time regression matrix for broad audio routing invariants.

The original version of this matrix used WAV files Aaron uploaded during the
thread. Some handoff bundles do not include that real-audio folder, so the test
creates conservative synthetic stand-ins when a fixture is missing. The point of
this file is still the same: one broken sample should identify one failing
routing invariant instead of hiding inside a broad smoke run.
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
REAL_AUDIO_DIR = PROJECT_ROOT / "tests" / "regression_audio_thread_matrix"
SYNTHETIC_AUDIO_DIR = PROJECT_ROOT / "_reports" / "generated_regression_audio_thread_matrix"
LOCKED_SMOKE_AUDIO_DIR = PROJECT_ROOT / "tests" / "acceptance" / "locked_smoke_v1" / "samples"
OUT_ROOT = PROJECT_ROOT / "_reports" / "pytest_regression_outputs" / "full_thread_matrix_one_by_one"

SAX_REED_SAMPLES = [
    "AA_JBL_74bpm_Am_Sax_Loop_18.wav",
    "AA_JBL_78bpm_Cm_Sax_Loop_1.wav",
    "AA_JBL_86bpm_Cm_Sax_Loop_12.wav",
    "AA_JBL_90bpm_Em_Sax_Loop_3.wav",
    "AMV_VRNB1_102_brass_saxophone_loop_cranesinthesky_Am.wav",
    "GrimyHipHop_Saxophone_26_Fm_Melody_Dark_Dusty_Warm_Loop_84bpm.wav",
    "HipHopTapes_28_Saxophone_D#m_90bpm.wav",
    "HipHopTapes_29_Saxophone_D#m_90bpm.wav",
    "SCY093_02_Sax_Loop_KeyAbm_89bpm_01.wav",
]
DRUM_LOOP_SAMPLES = [
    "ABOUTME_94_DRUMLOOP.wav",
    "DRUMS_LOOP_125BPM.wav",
    "Full Drum Loop 03 - 78BPM.wav",
    "NDAEV4_FULL_DRUM_LOOP_01_AFTERLIFE_74BPM.wav",
    "THO_ck4_drums top_130 bpm.wav",
]
VOICE_SAMPLES = [
    "DOJO_CGNB_Female_Vocal_Shot_01_D.wav",
    "DOJO_FBP_Female_Vocal_Shout.wav",
    "Money_vocals_female_rap_110bpm.wav",
    "Stab 3.wav",
]
PERCUSSIVE_HIT_SAMPLES = ["125591.wav", "13144.wav", "16791.wav", "33157.wav", "50728.wav"]
KICK_LIKE_HIT_SAMPLES = ["13115.wav", "16291.wav"]
MIXED_CHOPPY_SAMPLES = ["AV5_5_94bpm_Hit 2.wav"]

ALL_THREAD_MATRIX_SAMPLES = (
    SAX_REED_SAMPLES
    + DRUM_LOOP_SAMPLES
    + VOICE_SAMPLES
    + PERCUSSIVE_HIT_SAMPLES
    + KICK_LIKE_HIT_SAMPLES
    + MIXED_CHOPPY_SAMPLES
)
REAL_AUDIO_MATRIX_PRESENT = all((REAL_AUDIO_DIR / sample_name).exists() for sample_name in ALL_THREAD_MATRIX_SAMPLES)


def representative_samples(sample_names: list[str]) -> list[str]:
    """Use the whole real matrix when present, otherwise one useful sample."""
    if REAL_AUDIO_MATRIX_PRESENT:
        return sample_names
    for sample_name in sample_names:
        if (LOCKED_SMOKE_AUDIO_DIR / sample_name).exists():
            return [sample_name]
    return sample_names[:1]


ACTIVE_SAX_REED_SAMPLES = representative_samples(SAX_REED_SAMPLES)
ACTIVE_DRUM_LOOP_SAMPLES = representative_samples(DRUM_LOOP_SAMPLES)
ACTIVE_VOICE_SAMPLES = representative_samples(VOICE_SAMPLES)
ACTIVE_PERCUSSIVE_HIT_SAMPLES = representative_samples(PERCUSSIVE_HIT_SAMPLES)
ACTIVE_KICK_LIKE_HIT_SAMPLES = representative_samples(KICK_LIKE_HIT_SAMPLES)


def ensure_regression_audio_sample(sample_name: str) -> Path:
    """Create a synthetic stand-in when the real uploaded WAV is absent.

    Args:
        sample_name: Fixture filename used as a stable test identifier.

    Returns:
        Path to an existing WAV fixture.

    Side Effects:
        May use a locked-smoke real WAV or write one small synthetic WAV under ``_reports/generated_regression_audio_thread_matrix``.

    Raises:
        No intentional exceptions beyond normal file-write errors.
    """
    thread_matrix_source = REAL_AUDIO_DIR / sample_name
    if thread_matrix_source.exists():
        return thread_matrix_source
    locked_smoke_source = LOCKED_SMOKE_AUDIO_DIR / sample_name
    if locked_smoke_source.exists():
        return locked_smoke_source
    source = SYNTHETIC_AUDIO_DIR / sample_name
    if source.exists():
        return source
    if sample_name in SAX_REED_SAMPLES:
        samples = fixtures.synth_harmonic_phrase(330.0, dur=2.4)
    elif sample_name in DRUM_LOOP_SAMPLES:
        samples = fixtures.synth_drum_loop(sample_name, dur=2.4)
    elif sample_name in VOICE_SAMPLES:
        samples = fixtures.synth_voice_like(220.0, dur=1.6)
    elif sample_name in KICK_LIKE_HIT_SAMPLES:
        samples = fixtures.synth_kick(dur=0.45)
    elif sample_name in PERCUSSIVE_HIT_SAMPLES:
        samples = fixtures.synth_percussion_hit(dur=0.32)
    elif sample_name in MIXED_CHOPPY_SAMPLES:
        samples = fixtures.synth_drum_loop(sample_name, dur=1.6)
    else:
        samples = fixtures.synth_tone(440.0, dur=1.8)
    return fixtures.write_wav(source, samples)


def sort_one(sample_name: str) -> str:
    """Sort one sample through the CLI and return the chosen final label."""
    source = ensure_regression_audio_sample(sample_name)
    out_dir = OUT_ROOT / sample_name.replace("/", "_").replace(" ", "_")[:90]
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, str(SORTER), "sort", str(source), str(out_dir), "--brain", str(BRAIN), "--no-zip"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout
    rows = list(csv.DictReader((out_dir / "Aaron_Sorted_Sounds_manifest.csv").open(errors="replace", newline="")))
    assert len(rows) == 1
    row = rows[0]
    label = row.get("final_label") or row.get("folder_path") or row.get("placed_path") or ""
    assert label, row
    return label.replace("\\", "/")


def assert_no_fragments(label: str, fragments: list[str]) -> None:
    """Fail when a forbidden folder fragment appears in a final label."""
    low = label.lower()
    bad = [fragment for fragment in fragments if fragment.lower() in low]
    assert not bad, f"Unexpected {bad} in {label}"


@pytest.mark.parametrize("sample_name", ACTIVE_SAX_REED_SAMPLES)
def test_uploaded_sax_reed_material_stays_in_instruments_or_review(sample_name: str) -> None:
    label = sort_one(sample_name)
    assert label.startswith("Instruments/") or label.startswith("_TO_REVIEW/"), label
    assert_no_fragments(label, ["Drums", "FX", "Human", "Voice", "Dog", "Cat", "Bird", "Coins", "Water", "Machine"])


@pytest.mark.parametrize("sample_name", ACTIVE_DRUM_LOOP_SAMPLES)
def test_uploaded_drum_loops_stay_drum_loops_or_review(sample_name: str) -> None:
    label = sort_one(sample_name)
    assert label.startswith("Drums/Drum Loops") or label.startswith("_TO_REVIEW/"), label
    assert_no_fragments(label, ["Instruments", "Bass Loops", "FX", "Human", "Voice", "Riser", "Water"])


@pytest.mark.parametrize("sample_name", ACTIVE_VOICE_SAMPLES)
def test_uploaded_voice_material_stays_voice_human_or_review(sample_name: str) -> None:
    label = sort_one(sample_name)
    assert (
        label.startswith("FX/Human and Voice FX")
        or label.startswith("Instruments/Voice")
        or label.startswith("_TO_REVIEW/")
    ), label
    assert_no_fragments(label, ["Drums", "Percussion", "Machines", "Motor", "Dog", "Cat", "Bird", "Coins"])


@pytest.mark.parametrize("sample_name", ACTIVE_PERCUSSIVE_HIT_SAMPLES)
def test_uploaded_percussive_hits_stay_drums_or_review(sample_name: str) -> None:
    label = sort_one(sample_name)
    assert label.startswith("Drums/") or label.startswith("_TO_REVIEW/"), label
    assert_no_fragments(label, ["FX", "Instruments", "Human", "Voice", "Animal", "Cat", "Dog", "Ocean", "Water"])


@pytest.mark.parametrize("sample_name", ACTIVE_KICK_LIKE_HIT_SAMPLES)
def test_uploaded_kick_like_hits_stay_kicks_or_drums(sample_name: str) -> None:
    label = sort_one(sample_name)
    assert label.startswith("Drums/Kick") or label.startswith("Drums/Percussion") or label.startswith("_TO_REVIEW/"), (
        label
    )
    assert_no_fragments(label, ["Bass Loops", "Instrument Loops", "FX", "Human", "Voice", "Water"])


def test_choppy_music_loop_does_not_become_coin_or_animal() -> None:
    label = sort_one("AV5_5_94bpm_Hit 2.wav")
    assert label.startswith("Instruments/") or label.startswith("Drums/") or label.startswith("_TO_REVIEW/"), label
    assert_no_fragments(label, ["Coins", "Dog", "Cat", "Bird", "Water", "Human and Voice"])
