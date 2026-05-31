from __future__ import annotations

import csv
import zipfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

from tests.synthetic_audio_fixtures import ensure_fixture_wav, ensure_scratch_brain_zip

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
from aaron_sound_sorter import api as mod

FORBIDDEN_AUTOPLACE_TOPS = {"Drums", "FX"}


def _extract_scratch_brain(tmp_path: Path) -> dict:
    brain_zip = ensure_scratch_brain_zip(FIXTURES)
    with zipfile.ZipFile(brain_zip, "r") as zf:
        zf.extract("phase3_pure_scratch_brain.json", tmp_path)
    brain = mod.read_json(tmp_path / "phase3_pure_scratch_brain.json")
    mod.validate_v052_brain_or_die(brain)
    return brain


def _fingerprint_for_fixture(name: str):
    path = FIXTURES / name
    ensure_fixture_wav(path)
    fp, duration, status = mod.make_fingerprint_safe(path)
    assert status == "ok"
    return fp, duration


def _raw_rows(brain: dict, fingerprint: Iterable[float]) -> list[tuple[str, float, float]]:
    weights = np.asarray(brain["feature_weights"], dtype=np.float32)
    raw_x = np.asarray(list(fingerprint), dtype=np.float32)
    rows: list[tuple[str, float, float]] = []
    for label in brain["labels"]:
        mean, std = mod.scaler_for_label(brain, label)
        xw = ((raw_x - mean) / std) * weights
        raw_score, raw_distance, _mode = mod.label_model_distance_score(brain, label, xw)
        rows.append((str(label), float(raw_score), float(raw_distance)))
    return sorted(rows, key=lambda t: (t[1], t[0]))


def _sort_actual_bad_files(tmp_path: Path) -> list[dict[str, str]]:
    inp = tmp_path / "input"
    inp.mkdir()
    for name in ["Piano_G.wav", "Acoustic Guitar Rhythm_25_KeyDm_100bpm.wav"]:
        src = FIXTURES / name
        ensure_fixture_wav(src)
        (inp / name).write_bytes(src.read_bytes())
    ensure_scratch_brain_zip(FIXTURES)
    with zipfile.ZipFile(FIXTURES / "phase3_pure_scratch_brain.json.zip", "r") as zf:
        zf.extract("phase3_pure_scratch_brain.json", tmp_path)
    out = tmp_path / "out"
    args = type(
        "Args",
        (),
        {
            "project_dir": str(ROOT),
            "input": str(inp),
            "output": str(out),
            "brain": str(tmp_path / "phase3_pure_scratch_brain.json"),
            "training_root": "",
            "min_similarity": 0.52,
            "min_margin": 1.0,
            "random_seed": 17,
            "make_zip": False,
        },
    )()
    assert mod.run_sort_command(args) == 0
    with (out / "Aaron_Sorted_Sounds_manifest.csv").open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _top(label: str, brain: dict | None = None) -> str:
    if brain and isinstance(brain.get("top_by_label"), dict):
        return str(brain["top_by_label"].get(label, mod.top_for_public_label(label)))
    return mod.top_for_public_label(label)


def test_piano_fixture_raw_brain_is_instrument_before_any_tournament(tmp_path: Path) -> None:
    brain = _extract_scratch_brain(tmp_path)
    fp, _dur = _fingerprint_for_fixture("Piano_G.wav")
    rows = _raw_rows(brain, fp)
    assert rows[0][0].startswith("Instruments/"), rows[:8]


def test_piano_fixture_full_pipeline_cannot_autoplace_drums_or_fx(tmp_path: Path) -> None:
    rows = _sort_actual_bad_files(tmp_path)
    row = next(r for r in rows if r["source_path"].endswith("Piano_G.wav"))
    assert row["top_match_1"].startswith("Instruments/"), row["top_match_1"]
    assert row["final_top"] in {"Instruments", "_TO_REVIEW"}
    assert not row["final_label"].startswith(("Drums/", "FX/"))


def test_acoustic_guitar_fixture_raw_brain_has_instrument_candidate(tmp_path: Path) -> None:
    brain = _extract_scratch_brain(tmp_path)
    fp, dur = _fingerprint_for_fixture("Acoustic Guitar Rhythm_25_KeyDm_100bpm.wav")
    rows = _raw_rows(brain, fp)
    assert dur > 0.0
    guitar_hits = [
        (idx + 1, row) for idx, row in enumerate(rows) if row[0] == "Instruments/Guitar/Acoustic Guitar/Loops"
    ]
    assert guitar_hits, "scratch brain has no acoustic guitar loop label"
    assert guitar_hits[0][0] <= 10, guitar_hits[0]


def test_acoustic_guitar_fixture_full_pipeline_forbids_wrong_family_autoplace(tmp_path: Path) -> None:
    rows = _sort_actual_bad_files(tmp_path)
    row = next(r for r in rows if r["source_path"].endswith("Acoustic Guitar Rhythm_25_KeyDm_100bpm.wav"))
    assert row["final_top"] in {"Instruments", "_TO_REVIEW"}
    assert not row["final_label"].startswith(("Drums/", "FX/"))
    # The attempted leaf may shift as the code changes. The product contract is
    # that none of those wrong Drums/FX leaves may become confident placement.
    assert not (row["confidence_status"] == "auto_place" and row["final_top"] in FORBIDDEN_AUTOPLACE_TOPS)


def _tournament_brain() -> dict:
    labels = {
        "inst_synth_chord": "Instruments",
        "inst_organ_chord": "Instruments",
        "inst_piano_arp": "Instruments",
        "inst_guitar_loop": "Instruments",
        "drum_cymbal": "Drums",
        "drum_snare_loop": "Drums",
        "drum_roll_fill": "Drums",
        "fx_riser": "FX",
        "fx_applause": "FX",
        "fx_door": "FX",
    }
    return {
        "model_ensemble_enabled": True,
        "model_ensemble_min_switch_advantage": 0.05,
        "model_ensemble_max_score_penalty_ratio": 10.0,
        "model_ensemble_allow_cross_family_switch": True,  # old brain may request this
        "model_ensemble_explicit_cross_family_override_enabled": False,  # code must ignore old request
        "top_by_label": labels,
        "structure_by_label": {
            "inst_synth_chord": "one_shot",
            "inst_organ_chord": "one_shot",
            "inst_piano_arp": "loop",
            "inst_guitar_loop": "loop",
            "drum_cymbal": "one_shot",
            "drum_snare_loop": "loop",
            "drum_roll_fill": "loop",
            "fx_riser": "transition",
            "fx_applause": "one_shot",
            "fx_door": "one_shot",
        },
    }


def _row(
    label: str,
    current: float,
    *,
    neutral: float | None = None,
    spread: float = 9.0,
    anchor: float = 9.0,
    linear_rank: float = 5.0,
    router_rank: float = 1.0,
    blocked: bool = False,
) -> dict:
    return {
        "label": label,
        "current_score": float(current),
        "support_neutral_score": float(current if neutral is None else neutral),
        "spread_norm_score": float(spread),
        "anchor_score": float(anchor),
        "linear_rank_score": float(linear_rank),
        "frontend_router_rank_score": float(router_rank),
        "membership_blocked": bool(blocked),
        "membership_score": 0.0 if blocked else 1.0,
    }


SCENARIOS = [
    ("Piano_G", "inst_synth_chord", ["drum_cymbal", "fx_riser"]),
    ("OrganChord1_Bmin", "inst_organ_chord", ["drum_cymbal", "fx_riser"]),
    ("PlinkyPianoArp1_keyBbmin", "inst_piano_arp", ["drum_cymbal", "fx_riser"]),
    ("Acoustic Guitar Rhythm_25_KeyDm_100bpm", "inst_guitar_loop", ["fx_applause", "drum_snare_loop", "fx_door"]),
    ("Fill_4_130", "drum_roll_fill", ["fx_riser", "drum_cymbal"]),
]


@pytest.mark.parametrize("case_name, raw_label, bad_labels", SCENARIOS)
def test_each_known_failed_scenario_tournament_cannot_launder_to_wrong_family(
    case_name: str, raw_label: str, bad_labels: list[str]
) -> None:
    brain = _tournament_brain()
    rows = [_row(raw_label, 1.00, spread=8.0, anchor=8.0, linear_rank=5.0)]
    for i, bad in enumerate(bad_labels):
        rows.append(_row(bad, 1.03 + i * 0.01, spread=0.10 + i * 0.05, anchor=0.01, linear_rank=-12.0 + i))
    final, meta = mod.ensemble_tournament_choose_label(brain, rows, raw_label)
    assert _top(final, brain) == _top(raw_label, brain), f"{case_name} laundered into {final}; meta={meta}"
    for bad in bad_labels:
        if _top(bad, brain) != _top(raw_label, brain):
            assert f"{bad}:cross_family_blocked" in meta["rejected_proposals"]


@pytest.mark.parametrize("case_name, raw_label, bad_labels", SCENARIOS)
def test_each_known_failed_scenario_blocked_candidate_cannot_win_even_inside_candidate_pool(
    case_name: str, raw_label: str, bad_labels: list[str]
) -> None:
    brain = _tournament_brain()
    # Make the bad label look absurdly attractive to every helper head, but blocked.
    bad = bad_labels[0]
    rows = [
        _row(raw_label, 1.00, spread=7.0, anchor=7.0, linear_rank=5.0),
        _row(bad, 1.01, spread=0.01, anchor=0.01, linear_rank=-20.0, blocked=True),
    ]
    final, meta = mod.ensemble_tournament_choose_label(brain, rows, raw_label)
    assert final == raw_label, f"{case_name} allowed blocked candidate {bad}; meta={meta}"
    assert f"{bad}:membership_blocked" in meta["rejected_proposals"]
