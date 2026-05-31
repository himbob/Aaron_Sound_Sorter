#!/usr/bin/env python3
"""Analyze brain-lane competence CSVs from an existing sort run.

This tool does not read audio and does not affect routing. It lets Aaron study
which brain lane is useful for which kind of sample using an already completed
sort output folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aaron_sound_sorter.infrastructure.brain_lane_validation import (  # noqa: E402
    LANE_SPECS,
    add_lane_columns,
    brain_lane_group_winner_rows,
    competence_summary_rows,
    expected_profile_from_source_path,
    validation_matrix_fields,
    write_competence_summary_rows,
    write_group_winners,
)

MATRIX_NAME = "Aaron_Brain_Lane_Validation_Matrix.csv"
SUMMARY_NAME = "Aaron_Brain_Lane_Competence_Summary.csv"
WINNERS_NAME = "Aaron_Brain_Lane_Group_Winners.csv"
STUDY_NAME = "Aaron_Brain_Lane_Study.md"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries."""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_run_dir(value: str) -> Path:
    """Resolve a run directory from either the run folder or sorted-root folder."""
    path = Path(value).expanduser().resolve()
    if path.name == "Aaron_Sorted_Sounds":
        return path.parent
    return path


def ensure_summary_rows(run_dir: Path, *, rebuild: bool) -> list[dict[str, str]]:
    """Load or rebuild competence summary rows from an existing run directory."""
    summary_path = run_dir / SUMMARY_NAME
    matrix_path = run_dir / MATRIX_NAME
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Missing {MATRIX_NAME}. Expected it under: {run_dir}\n"
            "Run a sort with the current code or pass a folder containing the brain-lane CSVs."
        )
    matrix_rows = read_csv_rows(matrix_path)
    if not matrix_rows:
        write_competence_summary_rows(summary_path, [])
        return []
    missing_fields = [field for field in validation_matrix_fields() if field not in matrix_rows[0]]
    if missing_fields and not all(field.startswith("expected_") for field in missing_fields):
        raise ValueError(f"Validation matrix is missing expected columns: {missing_fields[:8]}")

    should_rebuild = rebuild or not summary_path.exists()
    if summary_path.exists() and not should_rebuild:
        existing = read_csv_rows(summary_path)
        if existing and "expected_source_quality" in existing[0]:
            return existing
        should_rebuild = True

    matrix_rows = enrich_legacy_matrix_rows(matrix_rows)
    summary_rows = competence_summary_rows(matrix_rows)
    write_competence_summary_rows(summary_path, summary_rows)
    return summary_rows


def enrich_legacy_matrix_rows(matrix_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Add v31.95 expected-quality fields to old validation matrices."""
    enriched: list[dict[str, str]] = []
    for row in matrix_rows:
        item = dict(row)
        if not item.get("expected_source_quality") or not item.get("expected_group_reason"):
            profile = expected_profile_from_source_path(Path(item.get("source_path", "")))
            item["expected_group"] = profile["expected_group"]
            item["expected_source_quality"] = profile["expected_source_quality"]
            item["expected_group_reason"] = profile["expected_group_reason"]
            item["expected_profile_version"] = "rebuilt_from_legacy_matrix"
            recompute_lane_columns(item)
        enriched.append(item)
    return enriched


def recompute_lane_columns(row: dict[str, str]) -> None:
    """Recompute lane correctness columns after expected group is rebuilt."""
    expected_group = row.get("expected_group", "unknown") or "unknown"
    lanes = ["physics", "final"] + [lane for lane, _key in LANE_SPECS]
    for lane in lanes:
        labels = labels_from_existing_row(row, lane)
        add_lane_columns(row, lane, labels, expected_group)


def labels_from_existing_row(row: dict[str, str], lane: str) -> list[str]:
    """Return top labels from an existing validation matrix row."""
    raw_json = row.get(f"{lane}_top5_json", "")
    if raw_json:
        try:
            values = json.loads(raw_json)
            if isinstance(values, list):
                return [str(value) for value in values if str(value)]
        except json.JSONDecodeError:
            pass
    top1 = row.get(f"{lane}_top1", "")
    return [top1] if top1 else []


def format_metric(value: str) -> str:
    """Format a decimal metric for markdown."""
    try:
        return f"{float(value):.1%}"
    except ValueError:
        return value or ""


def write_markdown_study(path: Path, winner_rows: Iterable[dict[str, str]]) -> None:
    """Write a readable lane-study markdown file."""
    lines = [
        "# Aaron Brain Lane Study",
        "",
        "This is diagnostic only. It ranks completed lane outputs and does not alter sorting.",
        "",
        "| Expected group | Source quality | Examples | Best brain strict | Strict | Best brain broad | Broad safe | Dead lanes | Watch lanes |",
        "|---|---|---:|---|---:|---|---:|---|---|",
    ]
    for row in winner_rows:
        lines.append(
            "| {expected_group} | {quality} | {total_examples} | {strict_lane} | {strict_acc} | {broad_lane} | {broad_acc} | {dead} | {watch} |".format(
                expected_group=row.get("expected_group", ""),
                quality=row.get("expected_source_quality", ""),
                total_examples=row.get("total_examples", ""),
                strict_lane=row.get("best_brain_strict_lane", ""),
                strict_acc=format_metric(row.get("best_brain_strict_accuracy", "")),
                broad_lane=row.get("best_brain_broad_lane", ""),
                broad_acc=format_metric(row.get("best_brain_broad_safe_accuracy", "")),
                dead=row.get("dead_or_missing_brain_lanes", ""),
                watch=row.get("watch_lanes", ""),
            )
        )
        rankings_json = row.get("brain_lane_rankings_json", "[]")
        try:
            rankings = json.loads(rankings_json)
        except json.JSONDecodeError:
            rankings = []
        if rankings:
            lines.append("")
            lines.append(f"## {row.get('expected_group', '')} / {row.get('expected_source_quality', '')}")
            lines.append("")
            for ranking in rankings:
                lines.append(
                    "- {lane}: strict {strict}, broad-safe {broad}, wrong {wrong}, dead {dead}, common wrong group: {wrong_group}, recommendation: {recommendation}".format(
                        lane=ranking.get("lane_name", ""),
                        strict=format_metric(ranking.get("strict_accuracy", "")),
                        broad=format_metric(ranking.get("broad_safe_accuracy", "")),
                        wrong=ranking.get("wrong_count", ""),
                        dead=ranking.get("dead_lane_count", ""),
                        wrong_group=ranking.get("common_wrong_group", ""),
                        recommendation=ranking.get("recommendation", ""),
                    )
                )
            lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze brain-lane competence files from an existing sort run.")
    parser.add_argument("run_dir", help="Sort run folder, e.g. /Volumes/T9/.../fx_YYYYMMDD_HHMMSS")
    parser.add_argument(
        "--rebuild-summary",
        action="store_true",
        help="Rebuild the competence summary from the matrix before making winners.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the existing-run analyzer."""
    args = parse_args(argv)
    run_dir = resolve_run_dir(args.run_dir)
    summary_rows = ensure_summary_rows(run_dir, rebuild=bool(args.rebuild_summary))
    winners_path = run_dir / WINNERS_NAME
    write_group_winners(winners_path, summary_rows)
    winner_rows = brain_lane_group_winner_rows(summary_rows)
    study_path = run_dir / STUDY_NAME
    write_markdown_study(study_path, winner_rows)
    print(f"Read: {run_dir / SUMMARY_NAME}")
    print(f"Wrote: {winners_path}")
    print(f"Wrote: {study_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
