#!/usr/bin/env python3
"""Build per-hash foundation panels and dataset conflict reports."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.neural_audio.foundation_panel import build_foundation_panel  # noqa: E402
from aaron_sound_sorter.neural_audio.panns_mapping import PannsMappingRegistry  # noqa: E402
from aaron_sound_sorter.neural_audio.runtime import _prediction_from_mapping  # noqa: E402
from aaron_sound_sorter.taxonomy_registry import TaxonomyRegistry  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--predictions-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write JSONL panels plus compact summary and conflict CSV files."""
    args = build_parser().parse_args(argv)
    root = args.project_root.expanduser().resolve()
    predictions_path = resolve_path(root, args.predictions_json)
    output_dir = resolve_path(root, args.output_dir)
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    raw_predictions = payload.get("predictions", [])
    if not isinstance(raw_predictions, list):
        raise ValueError("predictions must be a list")
    taxonomy = TaxonomyRegistry.load(
        root / "config/canonical_taxonomy.json",
        root / "config/taxonomy_aliases.json",
    )
    panns_mapping = PannsMappingRegistry.load(root / "config/panns_audioset_mapping.json", taxonomy)
    panels = [
        build_foundation_panel(
            _prediction_from_mapping(raw_prediction),
            taxonomy_version=taxonomy.taxonomy_version,
            brain_version=str(payload.get("index_path", "")),
            panns_mapping=panns_mapping,
        )
        for raw_prediction in raw_predictions
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    panels_path = output_dir / "foundation_panels.jsonl"
    panels_path.write_text(
        "".join(json.dumps(panel.to_mapping(), sort_keys=True) + "\n" for panel in panels),
        encoding="utf-8",
    )
    summary_rows = [summary_row(panel) for panel in panels]
    summary_path = output_dir / "foundation_panel_summary.csv"
    write_rows(summary_path, summary_rows)
    conflicts_path = output_dir / "foundation_panel_conflicts.csv"
    write_rows(conflicts_path, [row for row in summary_rows if row["conflict_count"] != "0"])
    print(f"Foundation panels: {len(panels)}")
    print(f"Conflicts: {sum(bool(panel.conflicts) for panel in panels)}")
    print(f"Panels: {panels_path}")
    print(f"Summary: {summary_path}")
    return 0


def summary_row(panel: object) -> dict[str, str]:
    """Return a compact CSV row from a foundation panel."""
    from aaron_sound_sorter.neural_audio.foundation_contracts import FoundationEvidencePanel

    if not isinstance(panel, FoundationEvidencePanel):
        raise TypeError("summary row requires FoundationEvidencePanel")
    return {
        "audio_sha256": panel.audio_sha256,
        "suggested_category": panel.suggested_category,
        "conflict_count": str(len(panel.conflicts)),
        "conflicts_json": json.dumps(panel.conflicts),
        "review_reason": panel.review_reason,
        "aaron_status": panel.lanes["aaron_prototype"].status,
        "clap_broad_status": panel.lanes["clap_broad"].status,
        "clap_prompt_status": panel.lanes["clap_prompt"].status,
        "panns_status": panel.lanes["panns"].status,
        "human_memory_status": panel.lanes["human_memory"].status,
    }


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """Write stable CSV rows, including a header for an empty conflict set."""
    fieldnames = [
        "audio_sha256",
        "suggested_category",
        "conflict_count",
        "conflicts_json",
        "review_reason",
        "aaron_status",
        "clap_broad_status",
        "clap_prompt_status",
        "panns_status",
        "human_memory_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(project_root: Path, path: Path) -> Path:
    """Resolve a project-relative or absolute path."""
    expanded = Path(path).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root / expanded).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
