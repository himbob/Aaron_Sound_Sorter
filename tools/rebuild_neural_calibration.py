#!/usr/bin/env python3
"""Build a versioned confidence calibrator from reviewed non-leaking outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.neural_audio.calibration_rebuild import (  # noqa: E402
    load_excluded_hashes,
    rebuild_confidence_calibration,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--excluded-hashes-csv", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Deduplicate outcomes, exclude held-out hashes, and fit when eligible."""
    args = build_parser().parse_args(argv)
    feedback_path = args.feedback_jsonl.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    excluded = load_excluded_hashes(args.excluded_hashes_csv)
    status = rebuild_confidence_calibration(feedback_path, output_dir, excluded_hashes=excluded)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
