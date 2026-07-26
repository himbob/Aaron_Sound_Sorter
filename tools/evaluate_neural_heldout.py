#!/usr/bin/env python3
"""Build production-oriented metrics from an explicit held-out prediction CSV."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.neural_audio.heldout_metrics import (  # noqa: E402
    evaluate_heldout_predictions,
    read_heldout_prediction_csv,
    write_heldout_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate held-out rows and write transparent metrics."""
    args = build_parser().parse_args(argv)
    rows = read_heldout_prediction_csv(args.predictions_csv.expanduser().resolve())
    metrics = evaluate_heldout_predictions(rows)
    write_heldout_metrics(metrics, args.output_dir.expanduser().resolve())
    print(json.dumps(asdict(metrics), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
