from __future__ import annotations

import csv
from pathlib import Path

from tools.named_sample_coverage_panel import (
    audit_case,
    find_named_candidates,
    panel_categories,
    read_manifest_rows,
    score_path_for_category,
    select_panel_samples,
)


def touch_audio(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-audio")


def category_by_id(category_id: str):
    return {category.category_id: category for category in panel_categories(10)}[category_id]


def test_score_path_rejects_eval_files() -> None:
    category = category_by_id("drums_kick_one_shots")

    score, matched_terms = score_path_for_category(Path("/samples/Kick_eval_01.wav"), category)

    assert score == 0
    assert matched_terms == ()


def test_find_named_candidates_covers_core_families(tmp_path: Path) -> None:
    touch_audio(tmp_path / "Pack A" / "Kick Clean 01.wav")
    touch_audio(tmp_path / "Pack B" / "Warm Guitar Loop 90bpm.wav")
    touch_audio(tmp_path / "Pack C" / "Dark Riser FX.wav")

    candidates = find_named_candidates(tmp_path, panel_categories(10))

    assert candidates["drums_kick_one_shots"]
    assert candidates["instruments_guitar"]
    assert candidates["fx_risers_builds"]


def test_select_panel_samples_spreads_categories(tmp_path: Path) -> None:
    touch_audio(tmp_path / "Pack A" / "Kick Clean 01.wav")
    touch_audio(tmp_path / "Pack B" / "Warm Guitar Loop 90bpm.wav")
    candidates = find_named_candidates(tmp_path, panel_categories(10))

    selected = select_panel_samples(candidates, panel_categories(10), panel_size=2, seed=11)

    assert len(selected) == 2
    assert {sample.category_id for sample in selected} == {"drums_kick_one_shots", "instruments_guitar"}


def test_audit_case_accepts_folder_prefix() -> None:
    expected = {
        "panel_filename": "kick.wav",
        "category_id": "drums_kick_one_shots",
        "accepted_prefixes": "Drums/Kick Drums",
        "source_path": "/samples/kick.wav",
    }
    row = {
        "source_path": "/panel/kick.wav",
        "final_label": "Drums/Kick Drums/Generic Kick/One Shots",
        "consensus_status": "strong_consensus",
    }

    result = audit_case(expected, row)

    assert result.status == "PASS"


def test_audit_case_flags_wrong_top() -> None:
    expected = {
        "panel_filename": "bass.wav",
        "category_id": "instruments_bass",
        "accepted_prefixes": "Instruments/Bass",
        "source_path": "/samples/bass.wav",
    }
    row = {"source_path": "/panel/bass.wav", "final_label": "FX/Designed Noise FX/Blip/One Shots"}

    result = audit_case(expected, row)

    assert result.status == "FAIL_WRONG_TOP"


def test_audit_case_flags_wrong_prefix_inside_top() -> None:
    expected = {
        "panel_filename": "piano.wav",
        "category_id": "instruments_piano",
        "accepted_prefixes": "Instruments/Keys/Piano",
        "source_path": "/samples/piano.wav",
    }
    row = {"source_path": "/panel/piano.wav", "final_label": "Instruments/Woodwinds/Saxophone/Loops"}

    result = audit_case(expected, row)

    assert result.status == "FAIL_WRONG_PREFIX"


def test_audit_case_flags_review() -> None:
    expected = {
        "panel_filename": "riser.wav",
        "category_id": "fx_risers_builds",
        "accepted_prefixes": "FX/Structural and Transitional FX/Risers and Builds",
        "source_path": "/samples/riser.wav",
    }
    row = {"source_path": "/panel/riser.wav", "final_label": "_TO_REVIEW/Measured Role Conflict"}

    result = audit_case(expected, row)

    assert result.status == "FAIL_REVIEW"


def test_read_manifest_rows_uses_source_filename(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "final_label"])
        writer.writeheader()
        writer.writerow(
            {
                "source_path": "/tmp/input_panel/00001_drums_kick.wav",
                "final_label": "Drums/Kick Drums/Generic Kick/One Shots",
            }
        )

    rows = read_manifest_rows(manifest)

    assert "00001_drums_kick.wav" in rows
