"""Tests for curated FX smoke manifest expected-bucket audit.

The audit may use source paths as ground truth because it is a post-sort test
oracle.  Production sorting code must remain source-name blind.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.audit_fx_smoke_expected_buckets import audit_manifest


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["source_path", "final_top", "folder_path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_fx_smoke_audit_fails_bass_loop_flattened_to_instrument_loops(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            {
                "source_path": "/x/FX_Aaron2/Loop/Bass/03_bass_Emn_178bpm.wav",
                "final_top": "Instruments",
                "folder_path": "Instruments/Instrument Loops/Loops",
            }
        ],
    )
    _total, failures = audit_manifest(manifest)
    assert failures
    assert failures[0]["rule"] == "loop_bass_to_bass_loops"


def test_fx_smoke_audit_fails_review_for_curated_smoke_file(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            {
                "source_path": "/x/FX_Aaron2/Loop/Brass_Woodwind/Intro Sax_FMinor_96BPM.wav",
                "final_top": "_TO_REVIEW",
                "folder_path": "_TO_REVIEW/Measured Role Conflict",
            }
        ],
    )
    _total, failures = audit_manifest(manifest)
    assert failures
    assert failures[0]["rule"] == "no_review_allowed_for_curated_fx_smoke"


def test_fx_smoke_audit_passes_broad_correct_buckets(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            {
                "source_path": "/x/FX_Aaron2/Loop/Bass/04.bass_92bpm_Em.wav",
                "final_top": "Instruments",
                "folder_path": "Instruments/Bass/Bass Loops",
            },
            {
                "source_path": "/x/FX_Aaron2/Loop/Drums/SCY095_03_Drums_Top_Loop_90bpm_01.wav",
                "final_top": "Drums",
                "folder_path": "Drums/Drum Loops/Loops",
            },
            {
                "source_path": "/x/FX_Aaron2/Loop/Brass_Woodwind/Intro Sax_FMinor_96BPM.wav",
                "final_top": "Instruments",
                "folder_path": "Instruments/Instrument Loops/Loops",
            },
        ],
    )
    _total, failures = audit_manifest(manifest)
    assert failures == []
