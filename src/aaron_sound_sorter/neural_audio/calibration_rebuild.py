"""Rebuild versioned confidence calibration from reviewed, non-leaking events."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from .calibration import ConfidenceCalibrator, ReviewFeedback, ReviewFeedbackStore


def rebuild_confidence_calibration(
    feedback_path: Path,
    output_dir: Path,
    *,
    excluded_hashes: set[str] | None = None,
) -> dict[str, object]:
    """Fit an inspectable calibrator when reviewed outcomes are sufficient.

    Args:
        feedback_path: Append-only reviewed outcome JSONL.
        output_dir: Versioned artifact destination.
        excluded_hashes: Training or final-held-out hashes forbidden from this
            calibration build.

    Returns:
        Honest build status including counts and artifact path when fitted.

    Side Effects:
        Writes ``calibration_status.json`` and, when eligible, a dependency-free
        ``confidence_calibrator.json``.
    """
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    excluded = set(excluded_hashes or ())
    all_rows = ReviewFeedbackStore(Path(feedback_path)).read_all()
    latest_by_hash = {row.file_sha256: row for row in all_rows if row.file_sha256 and row.file_sha256 not in excluded}
    rows = tuple(latest_by_hash[file_hash] for file_hash in sorted(latest_by_hash))
    accepted_count = sum(row.accepted for row in rows)
    rejected_count = len(rows) - accepted_count
    status: dict[str, object] = {
        "schema_version": 1,
        "reviewed_unique_hash_count": len(rows),
        "excluded_hash_count": len({row.file_sha256 for row in all_rows} & excluded),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "source_name_policy": "content hashes and reviewed evidence only",
        "production_authority_enabled": False,
    }
    if len(rows) < 20 or not accepted_count or not rejected_count:
        for stale_artifact in (
            destination / "confidence_calibrator.json",
            destination / "parent_family_calibrator.json",
        ):
            stale_artifact.unlink(missing_ok=True)
        status.update(
            {
                "status": "insufficient_data",
                "message": "Need at least 20 unique non-leaking reviews with accepted and corrected outcomes.",
            }
        )
        write_calibration_status(destination / "calibration_status.json", status)
        return status
    calibrator = ConfidenceCalibrator()
    calibrator.fit(rows)
    artifact_path = destination / "confidence_calibrator.json"
    calibrator.save(
        artifact_path,
        metadata={
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "reviewed_unique_hash_count": len(rows),
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
        },
    )
    built_models = ["prediction_correct"]
    parent_positive = sum(row.parent_family_accepted for row in rows)
    parent_negative = len(rows) - parent_positive
    if parent_positive and parent_negative:
        parent_calibrator = ConfidenceCalibrator()
        parent_calibrator.fit(rows, outcome="parent_family_correct")
        parent_calibrator.save(
            destination / "parent_family_calibrator.json",
            metadata={
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "reviewed_unique_hash_count": len(rows),
                "positive_count": parent_positive,
                "negative_count": parent_negative,
            },
        )
        built_models.append("parent_family_correct")
    status.update(
        {
            "status": "built",
            "artifact_path": str(artifact_path),
            "method": calibrator.method,
            "message": "Calibration model built. Production authority remains separately gated.",
            "built_models": built_models,
            "review_required_output": "one_minus_prediction_correct",
        }
    )
    write_calibration_status(destination / "calibration_status.json", status)
    return status


def load_excluded_hashes(path: Path | None) -> set[str]:
    """Load evaluation hashes that may not train confidence calibration."""
    if path is None:
        return set()
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return set()
    with resolved.open(encoding="utf-8", newline="") as handle:
        return {str(row.get("file_sha256", "")).strip() for row in csv.DictReader(handle)} - {""}


def write_calibration_status(path: Path, payload: dict[str, object]) -> None:
    """Write one stable calibration status artifact."""
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def calibration_feedback_counts(rows: tuple[ReviewFeedback, ...]) -> tuple[int, int]:
    """Return accepted and corrected counts for reporting tests."""
    accepted = sum(row.accepted for row in rows)
    return accepted, len(rows) - accepted
