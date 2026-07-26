#!/usr/bin/env python3
"""Promote one small neural category group only after independent gates pass."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.neural_audio.authority_promotion import (  # noqa: E402
    SUPPORTED_AUTHORITY_GROUPS,
    assess_group_promotion,
    category_authority_group,
)
from aaron_sound_sorter.neural_audio.heldout_metrics import (  # noqa: E402
    evaluate_heldout_predictions,
    read_heldout_prediction_csv,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit limited-rollout command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--group", choices=sorted(SUPPORTED_AUTHORITY_GROUPS), required=True)
    parser.add_argument("--heldout-predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate and, only on success, record one promoted category group."""
    args = build_parser().parse_args(argv)
    root = args.project_root.expanduser().resolve()
    calibration_path = root / "neural_artifacts/calibration/current/calibration_status.json"
    if not calibration_path.is_file():
        raise FileNotFoundError(f"calibration status is missing: {calibration_path}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in read_heldout_prediction_csv(args.heldout_predictions_csv.expanduser().resolve())
        if category_authority_group(row.expected_label) == args.group
    ]
    metrics = evaluate_heldout_predictions(rows)
    assessment = assess_group_promotion(
        group=args.group,
        calibration_status=calibration,
        heldout_metrics={
            "example_count": metrics.example_count,
            "top1_accuracy": metrics.top1_accuracy,
            "parent_family_accuracy": metrics.parent_family_accuracy,
            "incorrect_auto_placement_rate": metrics.incorrect_auto_placement_rate,
        },
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    assessment["reviewed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report_path = output_dir / f"{args.group}_authority_promotion.json"
    report_path.write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if assessment["passed"]:
        enabled = set(str(group) for group in calibration.get("enabled_authority_groups", []))
        enabled.add(args.group)
        calibration["production_authority_enabled"] = True
        calibration["enabled_authority_groups"] = sorted(enabled)
        calibration["last_promotion_report"] = str(report_path)
        temporary = calibration_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(calibration_path)
    print(json.dumps(assessment, indent=2, sort_keys=True))
    return 0 if assessment["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
