#!/usr/bin/env python3
"""Build a versioned confidence calibrator from reviewed non-leaking outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.neural_audio.calibration import (  # noqa: E402
    ConfidenceCalibrator,
    ReviewFeedbackStore,
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
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded = load_excluded_hashes(args.excluded_hashes_csv)
    all_rows = ReviewFeedbackStore(feedback_path).read_all()
    latest_by_hash = {row.file_sha256: row for row in all_rows if row.file_sha256 not in excluded}
    rows = tuple(latest_by_hash[file_hash] for file_hash in sorted(latest_by_hash))
    accepted_count = sum(row.accepted for row in rows)
    rejected_count = len(rows) - accepted_count
    status = {
        "schema_version": 1,
        "reviewed_unique_hash_count": len(rows),
        "excluded_hash_count": len({row.file_sha256 for row in all_rows} & excluded),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "source_name_policy": "content hashes and reviewed evidence only",
        "production_authority_enabled": False,
    }
    if len(rows) < 20 or not accepted_count or not rejected_count:
        status.update(
            {
                "status": "insufficient_data",
                "message": "Need at least 20 unique non-leaking reviews with accepted and rejected outcomes.",
            }
        )
        write_status(output_dir / "calibration_status.json", status)
        print(status["message"])
        return 0
    calibrator = ConfidenceCalibrator()
    calibrator.fit(rows)
    artifact_path = output_dir / "confidence_calibrator.json"
    calibrator.save(
        artifact_path,
        metadata={
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "reviewed_unique_hash_count": len(rows),
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
        },
    )
    status.update({"status": "built", "artifact_path": str(artifact_path), "method": calibrator.method})
    write_status(output_dir / "calibration_status.json", status)
    print(f"Calibrator: {artifact_path}")
    return 0


def load_excluded_hashes(path: Path | None) -> set[str]:
    """Load file hashes reserved for evaluation from an optional CSV."""
    if path is None:
        return set()
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return set()
    with resolved.open(encoding="utf-8", newline="") as handle:
        return {str(row.get("file_sha256", "")).strip() for row in csv.DictReader(handle)} - {""}


def write_status(path: Path, payload: dict[str, object]) -> None:
    """Write one stable calibration status artifact."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
