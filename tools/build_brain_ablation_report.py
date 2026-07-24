#!/usr/bin/env python3
"""Summarize an explicit-label brain-ensemble ablation replay."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

LANE_COLUMNS = {
    "brain_ensemble": "brain_ensemble_vote_1",
    "full_brain": "full_brain_vote_1",
    "core_baby": "core_baby_vote_1",
    "spread_baby": "spread_baby_vote_1",
    "outlier_baby": "outlier_baby_vote_1",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV rows."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def expected_labels_by_materialized_path(experiment_manifest: Path) -> dict[str, str]:
    """Return explicit human-seed labels keyed by resolved materialized path."""
    expected: dict[str, str] = {}
    for row in read_csv(experiment_manifest):
        if row.get("split") != "review_preview":
            continue
        expected[str(Path(row["materialized_path"]).resolve())] = row["subgroup"]
    return expected


def manifest_rows_by_resolved_source(path: Path) -> dict[str, dict[str, str]]:
    """Return sort rows keyed by resolved input path."""
    return {str(Path(row["source_path"]).resolve()): row for row in read_csv(path)}


def classify_seconds(run_dir: Path) -> float:
    """Return wall-clock classification-stage seconds."""
    payload = json.loads((run_dir / "Aaron_Sort_Timing_Profile.json").read_text(encoding="utf-8"))
    return float(payload.get("run_stages", {}).get("classify_files", {}).get("total_seconds", 0.0))


def configuration_row(
    name: str,
    run_dir: Path,
    expected: dict[str, str],
    *,
    comparison_rows: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build one actual configuration-ablation row."""
    rows = manifest_rows_by_resolved_source(run_dir / "Aaron_Sorted_Sounds_manifest.csv")
    exact = sum(row.get("folder_path", "") == expected.get(source, "") for source, row in rows.items())
    broad = sum(
        row.get("folder_path", "").split("/", 1)[0] == expected.get(source, "").split("/", 1)[0]
        for source, row in rows.items()
    )
    review = sum(row.get("folder_path", "").startswith("_TO_REVIEW") for row in rows.values())
    changed = 0
    if comparison_rows is not None:
        changed = sum(
            row.get("folder_path", "") != comparison_rows.get(source, {}).get("folder_path", "")
            for source, row in rows.items()
        )
    return {
        "lane_or_configuration": name,
        "ablation_kind": "actual_configuration_replay",
        "heldout_or_panel_count": len(rows),
        "exact_correct_count": exact,
        "exact_accuracy": exact / len(rows) if rows else 0.0,
        "broad_family_correct_count": broad,
        "broad_family_accuracy": broad / len(rows) if rows else 0.0,
        "review_count": review,
        "final_placement_changes_vs_baseline": changed,
        "unique_correct_wins": "not_attributable_at_configuration_level",
        "unique_safety_vetoes": "not_measured",
        "duplicate_top1_votes": "",
        "classify_seconds": classify_seconds(run_dir),
        "retirement_eligible": False,
        "recommendation": "retain; this is one narrow explicit-label panel",
    }


def diagnostic_lane_rows(
    baseline_rows: dict[str, dict[str, str]],
    expected: dict[str, str],
) -> list[dict[str, Any]]:
    """Summarize per-lane top-one evidence without claiming a real ablation."""
    correct_lanes_by_source: dict[str, set[str]] = {}
    for source, row in baseline_rows.items():
        correct_lanes_by_source[source] = {
            lane for lane, column in LANE_COLUMNS.items() if row.get(column, "") == expected.get(source, "")
        }
    lane_rows: list[dict[str, Any]] = []
    ordered_lanes = list(LANE_COLUMNS)
    for position, lane in enumerate(ordered_lanes):
        column = LANE_COLUMNS[lane]
        predictions = [row.get(column, "") for row in baseline_rows.values()]
        exact = sum(baseline_rows[source].get(column, "") == expected.get(source, "") for source in baseline_rows)
        broad = sum(
            baseline_rows[source].get(column, "").split("/", 1)[0] == expected.get(source, "").split("/", 1)[0]
            for source in baseline_rows
        )
        unique = sum(lanes == {lane} for lanes in correct_lanes_by_source.values())
        earlier_columns = [LANE_COLUMNS[name] for name in ordered_lanes[:position]]
        duplicate_votes = sum(
            bool(prediction)
            and any(prediction == baseline_rows[source].get(other_column, "") for other_column in earlier_columns)
            for (source, _row), prediction in zip(baseline_rows.items(), predictions)
        )
        lane_rows.append(
            {
                "lane_or_configuration": lane,
                "ablation_kind": "diagnostic_top1_proxy_not_disabled_replay",
                "heldout_or_panel_count": len(baseline_rows),
                "exact_correct_count": exact,
                "exact_accuracy": exact / len(baseline_rows) if baseline_rows else 0.0,
                "broad_family_correct_count": broad,
                "broad_family_accuracy": broad / len(baseline_rows) if baseline_rows else 0.0,
                "review_count": "not_applicable_to_raw_lane",
                "final_placement_changes_vs_baseline": "not_run",
                "unique_correct_wins": unique,
                "unique_safety_vetoes": "not_measured",
                "duplicate_top1_votes": duplicate_votes,
                "classify_seconds": "",
                "retirement_eligible": False,
                "recommendation": "retain until an independent-disable replay measures safety and review impact",
            }
        )
    return lane_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write report rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_recommendations(
    path: Path,
    baseline: dict[str, Any],
    disabled: dict[str, Any],
    brain_bytes: int,
) -> None:
    """Write honest retirement and rollback guidance."""
    runtime_delta = float(baseline["classify_seconds"]) - float(disabled["classify_seconds"])
    content = f"""# Brain retirement recommendations — 2026-07-24

## Actual replay

- Panel: 45 explicit locked human-seed labels spanning Drums, Instruments, and FX.
- Baseline exact placements: {baseline["exact_correct_count"]}/45.
- All legacy baby brains disabled: {disabled["exact_correct_count"]}/45.
- Final placement changes: {disabled["final_placement_changes_vs_baseline"]}.
- Review change: {disabled["review_count"] - baseline["review_count"]}.
- Classification time: {baseline["classify_seconds"]:.3f}s baseline versus {disabled["classify_seconds"]:.3f}s disabled.
- Measured runtime delta on this replay: {runtime_delta:.3f}s.
- Core/spread/outlier brain-file footprint: {brain_bytes} bytes.

## Decision

Do not retire a lane yet. This replay proves the baby ensemble supplied no unique
final placement on this one trusted panel, but it did not measure independent
core/spread/outlier removal, unique safety vetoes, the larger contaminated
trainer corpus, or fresh source-pack diversity.

## Next reversible ablation

1. Add diagnostic independent-disable switches for core, spread, and outlier.
2. Replay the locked panel, held-out vocal negatives, fresh drums, instruments,
   and FX, plus the active-learning queue.
3. Record unique correct wins, safety vetoes, Review changes, time, and memory.
4. Retire only a lane with no protected value across all panels.
5. Keep every current brain JSON and a pre-migration Git tag until the smaller
   learned-owner replacement passes the same replay.

Rollback is simply restoring the retained brain paths and re-enabling the lane;
this pass deletes no brain.
"""
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-manifest", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--no-baby-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--brain-file", type=Path, action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build ablation CSV and retirement guidance."""
    args = build_parser().parse_args(argv)
    expected = expected_labels_by_materialized_path(args.experiment_manifest)
    baseline_rows = manifest_rows_by_resolved_source(args.baseline_run / "Aaron_Sorted_Sounds_manifest.csv")
    baseline = configuration_row("baseline_all_baby_lanes", args.baseline_run, expected)
    disabled = configuration_row(
        "all_legacy_baby_lanes_disabled",
        args.no_baby_run,
        expected,
        comparison_rows=baseline_rows,
    )
    rows = [baseline, disabled, *diagnostic_lane_rows(baseline_rows, expected)]
    write_csv(args.output_dir / "brain_lane_ablation.csv", rows)
    brain_bytes = sum(path.stat().st_size for path in args.brain_file if path.is_file())
    write_recommendations(
        args.output_dir / "brain_retirement_recommendations.md",
        baseline,
        disabled,
        brain_bytes,
    )
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
