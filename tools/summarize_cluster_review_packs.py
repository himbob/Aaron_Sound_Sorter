#!/usr/bin/env python3
"""Build one coverage-aware priority sheet from hash-only cluster packs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

MINIMUM_GENERALIZATION_EXAMPLES = 3


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, action="append", required=True)
    parser.add_argument("--active-index-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write a combined review priority CSV and concise guide."""
    args = build_parser().parse_args(argv)
    active_counts = load_active_counts(args.active_index_json.expanduser().resolve())
    rows: list[dict[str, str]] = []
    for pack_path in args.pack:
        pack = pack_path.expanduser().resolve()
        with (pack / "cluster_manifest.csv").open(encoding="utf-8", newline="") as handle:
            rows.extend(prioritize_rows(pack.name, list(csv.DictReader(handle)), active_counts))
    rows.sort(key=lambda row: (-float(row["priority_score"]), row["file_sha256"]))
    for rank, row in enumerate(rows, start=1):
        row["priority_rank"] = str(rank)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_priority_csv(output_dir / "REVIEW_PRIORITY.csv", rows)
    (output_dir / "README_AUDIO_EVIDENCE_ONLY.txt").write_text(
        review_guide(rows, len(args.pack)),
        encoding="utf-8",
    )
    print(f"Review candidates: {len(rows)}")
    print(f"Priority sheet: {output_dir / 'REVIEW_PRIORITY.csv'}")
    return 0


def load_active_counts(index_json: Path) -> dict[str, int]:
    """Load approved-example counts from one active prototype index."""
    import json

    payload = json.loads(index_json.read_text(encoding="utf-8"))
    counts = payload.get("metadata", {}).get("label_example_counts", {})
    return {str(label): max(0, int(count)) for label, count in dict(counts).items()}


def prioritize_rows(
    pack_name: str,
    rows: list[dict[str, str]],
    active_counts: dict[str, int],
) -> list[dict[str, str]]:
    """Rank review questions from audio evidence and training coverage only."""
    prioritized: list[dict[str, str]] = []
    for row in rows:
        suggestion = str(row.get("suggested_category", ""))
        active_examples = active_counts.get(suggestion, 0)
        examples_needed = max(0, MINIMUM_GENERALIZATION_EXAMPLES - active_examples)
        known = str(row.get("known_distribution", "")) == "1"
        margin = _float(row.get("margin", "0"))
        membership = str(row.get("membership", "boundary"))
        uncertainty = 0.0 if known else 2.0
        uncertainty += max(0.0, 0.12 - margin) * 5.0
        boundary_bonus = {"safe_core": 0.0, "boundary": 0.5, "outlier": 1.0}.get(membership, 0.5)
        priority_score = examples_needed * 3.0 + uncertainty + boundary_bonus
        prioritized.append(
            {
                "priority_rank": "",
                "priority_score": f"{priority_score:.6f}",
                "pack": str(pack_name),
                "cluster_id": str(row.get("cluster_id", "")),
                "review_file": str(row.get("review_file", "")),
                "file_sha256": str(row.get("file_sha256", "")),
                "membership": membership,
                "structure_bucket": str(row.get("structure_bucket", "")),
                "broad_measured_family": str(row.get("broad_measured_family", "")),
                "suggested_category_question": suggestion,
                "active_examples_for_suggestion": str(active_examples),
                "examples_needed_for_generalization": str(examples_needed),
                "known_distribution": "1" if known else "0",
                "margin": f"{margin:.8f}",
            }
        )
    return prioritized


def write_priority_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the combined stable priority sheet."""
    fields = list(rows[0]) if rows else ["priority_rank", "priority_score", "file_sha256"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def review_guide(rows: list[dict[str, str]], pack_count: int) -> str:
    """Return concise source-name-blind review instructions."""
    sparse = sum(int(row["examples_needed_for_generalization"]) > 0 for row in rows)
    return (
        "Aaron audio-evidence-only review set\n\n"
        f"- Packs: {pack_count}\n"
        f"- Unique review clips: {len(rows)}\n"
        f"- Sparse-category questions: {sparse}\n"
        "- Start at priority rank 1.\n"
        "- Listen first; the suggested category is only a question.\n"
        "- Use the narrowest correct category, or Review when unsure.\n"
        "- Hash-only filenames prevent source-name hints.\n"
        "- Nothing here trains automatically. Your explicit GUI approval does.\n"
    )


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
