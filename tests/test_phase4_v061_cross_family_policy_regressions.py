from __future__ import annotations

import csv
import zipfile
from pathlib import Path

from tests.synthetic_audio_fixtures import ensure_fixture_wav, ensure_scratch_brain_zip

ROOT = Path(__file__).resolve().parents[1]
from aaron_sound_sorter import api as mod


def _fixture_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


def _extract_brain(tmp_path: Path) -> Path:
    brain_zip = ensure_scratch_brain_zip(_fixture_dir())
    with zipfile.ZipFile(brain_zip, "r") as zf:
        zf.extract("phase3_pure_scratch_brain.json", tmp_path)
    return tmp_path / "phase3_pure_scratch_brain.json"


def _run_sort(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    inp = tmp_path / "input"
    inp.mkdir()
    for name in ["Piano_G.wav", "Acoustic Guitar Rhythm_25_KeyDm_100bpm.wav"]:
        src = _fixture_dir() / name
        ensure_fixture_wav(src)
        (inp / name).write_bytes(src.read_bytes())
    brain = _extract_brain(tmp_path)
    out = tmp_path / "out"
    args = type(
        "Args",
        (),
        {
            "project_dir": str(root),
            "input": str(inp),
            "output": str(out),
            "brain": str(brain),
            "training_root": "",
            "min_similarity": 0.52,
            "min_margin": 1.0,
            "random_seed": 17,
            "make_zip": False,
        },
    )()
    assert mod.run_sort_command(args) == 0
    manifest = out / "Aaron_Sorted_Sounds_manifest.csv"
    assert manifest.exists()
    with manifest.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_piano_not_auto_placed_as_riser_after_cross_family_switch(tmp_path: Path) -> None:
    rows = _run_sort(tmp_path)
    row = next(r for r in rows if r["source_path"].endswith("Piano_G.wav"))
    assert row["top_match_1"].startswith("Instruments/"), row["top_match_1"]
    # This regression locks the cross-family safety rule. A correct/safe
    # instrument placement is no longer a failure; only Drums/FX theft is.
    assert row["final_top"] in {"Instruments", "_TO_REVIEW"}
    assert not row["final_label"].startswith(("Drums/", "FX/"))


def test_acoustic_guitar_rhythm_not_auto_placed_as_any_forbidden_wrong_family(tmp_path: Path) -> None:
    rows = _run_sort(tmp_path)
    row = next(r for r in rows if r["source_path"].endswith("Acoustic Guitar Rhythm_25_KeyDm_100bpm.wav"))
    # The exact leaf may change as the tournament is repaired. The invariant is
    # that this pitch-confident guitar loop must not be sent to Drums or FX.
    assert row["final_top"] in {"Instruments", "_TO_REVIEW"}
    assert row["final_label"].startswith(("_TO_REVIEW/", "Instruments/")) or row["final_top"] == "_TO_REVIEW"
    assert not row["final_label"].startswith(("Drums/", "FX/"))
