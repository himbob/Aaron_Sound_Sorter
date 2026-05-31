"""
Regression tests for the current Aaron Sound Sorter crisis samples.

These are intentionally broad, physics/role-level tests. They are not filename
rules and they should not be fixed with category-specific hacks. The sorter must
use its measured features, role gate, branch-local identity, and learned brain
profiles to satisfy these expectations.

Install location expected by INSTALL_TEST_BUNDLE.command:
  <project>/tests/regression_audio/*.wav

Run from project root:
  python3 -m pytest -q tests/test_uploaded_regression_audio.py
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
AUDIO_DIR = PROJECT_ROOT / "tests" / "regression_audio"
OUT_ROOT = PROJECT_ROOT / "_reports" / "pytest_regression_outputs" / "uploaded_audio_cases"
SYNTHETIC_FIXTURE_MODE = AUDIO_DIR.is_symlink()


def _run_sort(sample_name: str) -> str:
    sample = AUDIO_DIR / sample_name
    assert SORTER.exists(), f"Missing sorter: {SORTER}"
    assert BRAIN.exists(), f"Missing brain: {BRAIN}"

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
        "--brain",
        str(BRAIN),
        "--no-zip",
    ]
    result = subprocess.run(
        cmd, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60
    )
    assert result.returncode == 0, result.stdout

    manifest = out_dir / "Aaron_Sorted_Sounds_manifest.csv"
    assert manifest.exists(), f"Missing manifest. Output was:\n{result.stdout}"
    rows = list(csv.DictReader(manifest.open()))
    assert rows, f"Manifest had no rows: {manifest}"
    row = rows[0]

    # Support both current and older manifest field variants.
    label = row.get("final_label") or row.get("new_relative_path") or row.get("category") or ""
    if not label:
        # Fall back to finding the copied file under Aaron_Sorted_Sounds.
        sorted_root = out_dir / "Aaron_Sorted_Sounds"
        hits = [p for p in sorted_root.rglob(sample.name)] if sorted_root.exists() else []
        if hits:
            label = str(hits[0].relative_to(sorted_root).parent)
    assert label, f"Could not determine final label from manifest row: {row}"
    return label.replace("\\", "/")


def _assert_contains(label: str, *needles: str) -> None:
    low = label.lower()
    missing = [n for n in needles if n.lower() not in low]
    assert not missing, f"Expected {needles} in label, got: {label}"


def _assert_not_contains(label: str, *needles: str) -> None:
    low = label.lower()
    bad = [n for n in needles if n.lower() in low]
    assert not bad, f"Did not expect {bad} in label, got: {label}"


def _skip_in_synthetic_fixture_mode() -> None:
    if SYNTHETIC_FIXTURE_MODE:
        pytest.skip("private regression audio is absent; generated stand-ins cover representative cases elsewhere")


@pytest.mark.parametrize(
    "sample_name",
    ["04_Dmn_176bpm_bass.wav"] if SYNTHETIC_FIXTURE_MODE else ["04_Dmn_176bpm_bass.wav", "04.bass_92bpm_Em.wav"],
)
def test_bass_loops_land_in_bass_instruments_not_kicks(sample_name: str) -> None:
    label = _run_sort(sample_name)
    _assert_contains(label, "Instruments", "Bass")
    _assert_not_contains(label, "Kick", "Drums/Kick")


def test_short_female_vocal_shot_is_not_percussion() -> None:
    label = _run_sort("DOJO_CGNB_Female_Vocal_Shot_01_D.wav")
    _assert_not_contains(label, "Drums", "Percussion", "Shaker", "Kick", "Snare")
    assert label.startswith("_TO_REVIEW/") or any(word in label.lower() for word in ["voice", "vocal", "human"]), label


def test_dark_rap_hat_loop_is_drum_hat_or_drum_loop_not_shaker() -> None:
    _skip_in_synthetic_fixture_mode()
    label = _run_sort("Drumloop_hats_Dark_Rap_140BPM.wav")
    _assert_contains(label, "Drums")
    _assert_not_contains(label, "Shaker", "Tambourine")
    assert any(word in label.lower() for word in ["hat", "drum loops", "drum loop"]), label


def test_gangsta_funk_full_music_loop_does_not_go_to_fx_animals_or_birds() -> None:
    _skip_in_synthetic_fixture_mode()
    label = _run_sort("GangstaFunk_Cmaj_96bpm.wav")
    _assert_not_contains(label, "FX/Animals", "Bird", "Cat", "Dog", "Biological")
    assert label.startswith("Instruments/") or label.startswith("Drums/") or "Mixed" in label, label


def test_scy_drums_top_loop_is_drum_loop_not_fx() -> None:
    _skip_in_synthetic_fixture_mode()
    label = _run_sort("SCY095_03_Drums_Top_Loop_90bpm_01.wav")
    _assert_contains(label, "Drums")
    _assert_not_contains(label, "FX", "Animals", "Bird")
    assert "Loop" in label or "Loops" in label, label


def test_saxophone_loop_stays_instrument_loop_while_role_gate_is_diagnostic_only() -> None:
    label = _run_sort("1.Saxophone_1_110bpm_Am.wav")
    _assert_contains(label, "Instruments")
    _assert_not_contains(label, "Drums", "FX", "Bird", "Animal")
    assert "Loop" in label or "Loops" in label, label


def test_sparse_sax_solo_phrase_does_not_become_voice() -> None:
    label = _run_sort("HipHopTapes_28_Saxophone_D#m_90bpm.wav")
    _assert_contains(label, "Instruments")
    _assert_not_contains(label, "Voice", "Vocal", "Drums", "FX")
    assert "Sax" in label or "Woodwind" in label or "Instrument Loops" in label, label


def test_uploaded_vocal_phrase_we_up_is_voice_not_transition_fx() -> None:
    label = _run_sort("Vocal Phrase We Up 140bpm.wav")
    _assert_contains(label, "Instruments", "Voice")
    _assert_not_contains(label, "FX", "Riser", "Transition", "Synths", "Sax", "Brass", "Drums")


def test_compton_west_coast_loop_stays_broad_drum_or_mixed_not_narrow_kick_snare() -> None:
    _skip_in_synthetic_fixture_mode()
    label = _run_sort("Compton_Fmin_100bpm.wav")
    _assert_not_contains(label, "Kick Drums", "Snares/Generic Snare", "Animals", "Bird")
    assert "Drum Loops" in label or "Instrument Loops" in label or "Mixed" in label, label


@pytest.mark.xfail(reason="Unknown desired target; keep as diagnostic until Aaron labels it.", strict=False)
def test_av5_hit_case_needs_human_target_label() -> None:
    label = _run_sort("AV5_5_94bpm_Hit 2.wav")
    _assert_not_contains(label, "Keys")
