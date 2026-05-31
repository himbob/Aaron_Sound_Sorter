from __future__ import annotations

import csv
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import category_stability_gate as gate


def write_results(path: Path, actual_path: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "filename",
                "protected",
                "status",
                "actual_path",
                "anchor_pack",
                "target_category",
                "scenario",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "voice_positive",
                "filename": "voice.wav",
                "protected": "true",
                "status": "PASS",
                "actual_path": actual_path,
                "anchor_pack": "voice",
                "target_category": "Instruments/Voice",
                "scenario": "category_positive",
            }
        )


def test_protected_output_drift_fails_without_change_budget(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    write_results(baseline / "actual_results.csv", "Instruments/Voice/Phrase/One Shots")
    write_results(current / "actual_results.csv", "Instruments/Instrument Loops/Loops")

    exit_code, results = gate.compare_output_drift(
        baseline,
        current,
        gate.ChangeBudget("", (), (), (), ()),
    )

    assert exit_code == 1
    assert results[0].comparison == gate.REGRESSION_OUTPUT_DRIFT


def test_change_budget_can_approve_intentional_case_drift(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    current = tmp_path / "current"
    write_results(baseline / "actual_results.csv", "Instruments/Voice/Phrase/One Shots")
    write_results(current / "actual_results.csv", "Instruments/Voice/Vocal Loops/Loops")

    exit_code, results = gate.compare_output_drift(
        baseline,
        current,
        gate.ChangeBudget("voice leaf work", ("voice_positive",), (), (), ()),
    )

    assert exit_code == 0
    assert results[0].comparison == "APPROVED_OUTPUT_DRIFT"
