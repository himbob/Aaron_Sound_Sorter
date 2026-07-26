#!/usr/bin/env python3
"""Replay claimed trusted audio through the GUI using hash-only filenames."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.gui.models import PreviewRow  # noqa: E402
from aaron_sound_sorter.gui.preview_service import SortPreviewService  # noqa: E402
from aaron_sound_sorter.neural_audio.hashing import sha256_file  # noqa: E402

DEFAULT_SEED_MANIFEST = Path("tests/acceptance/locked_smoke_v1/trusted_training_seed_v1.json")
DEFAULT_REPORT_ROOT = Path("_reports/trusted_audio_runtime_audit")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--seed-manifest", type=Path, default=DEFAULT_SEED_MANIFEST)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a source-name-blind GUI replay and write claimed-label comparisons."""
    args = build_parser().parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    manifest_path = resolve_project_path(project_root, args.seed_manifest)
    report_root = resolve_project_path(project_root, args.report_root)
    run_dir = report_root / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    input_dir = run_dir / "hash_named_input"
    input_dir.mkdir(parents=True, exist_ok=False)

    cases = load_seed_cases(manifest_path)
    case_by_hash = materialize_hash_named_audio(project_root, input_dir, cases)
    session = SortPreviewService(project_root=project_root).classify_input(input_dir)
    rows = comparison_rows(session.rows, case_by_hash)
    report_path = run_dir / "trusted_audio_runtime_comparison.csv"
    write_comparison_report(report_path, rows)
    status_counts = Counter(row["consensus_status"] for row in rows)
    print(f"Audited claimed trusted cases: {len(rows)}")
    print(f"Exact claimed-label placements: {sum(row['matches_claimed_label'] == '1' for row in rows)}")
    print(f"Review placements: {sum(row['final_top'] == '_TO_REVIEW' for row in rows)}")
    print(f"Statuses: {dict(sorted(status_counts.items()))}")
    print(f"GUI report: {session.run_dir}")
    print(f"Audit report: {report_path}")
    return 0


def resolve_project_path(project_root: Path, path: Path) -> Path:
    """Resolve an absolute or project-relative path."""
    expanded = Path(path).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root / expanded).resolve()


def load_seed_cases(manifest_path: Path) -> list[dict[str, str]]:
    """Load explicit trusted-label claims without inferring from path text."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases", [])
    if not isinstance(raw_cases, list):
        raise ValueError("trusted seed manifest cases must be a list")
    cases: list[dict[str, str]] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("every trusted seed case must be an object")
        case_id = str(raw_case.get("id", "")).strip()
        audio_path = str(raw_case.get("audio_path", "")).strip()
        training_label = str(raw_case.get("training_label", "")).strip()
        if not case_id or not audio_path or not training_label:
            raise ValueError("trusted seed cases require id, audio_path, and training_label")
        cases.append({"id": case_id, "audio_path": audio_path, "training_label": training_label})
    return cases


def materialize_hash_named_audio(
    project_root: Path,
    input_dir: Path,
    cases: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Copy seed bytes under hash-only names and return claims keyed by hash."""
    case_by_hash: dict[str, dict[str, str]] = {}
    for case in cases:
        raw_path = Path(case["audio_path"]).expanduser()
        source_path = raw_path.resolve() if raw_path.is_absolute() else (project_root / raw_path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        file_sha256 = sha256_file(source_path)
        existing_case = case_by_hash.get(file_sha256)
        if existing_case is not None and existing_case["training_label"] != case["training_label"]:
            raise ValueError(f"conflicting claimed labels for audio hash {file_sha256}")
        case_record = {
            **case,
            "source_locator": str(source_path),
            "file_sha256": file_sha256,
        }
        case_by_hash[file_sha256] = case_record
        destination = input_dir / f"audio_{file_sha256}{source_path.suffix.casefold()}"
        if not destination.exists():
            shutil.copyfile(source_path, destination)
    return case_by_hash


def comparison_rows(rows: list[PreviewRow], case_by_hash: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    """Join GUI results to explicit label claims by content hash."""
    comparisons: list[dict[str, str]] = []
    for row in rows:
        file_sha256 = sha256_file(row.source_path)
        case = case_by_hash[file_sha256]
        prompt_labels = [str(suggestion.get("path", "")) for suggestion in row.neural_prompt_suggestions]
        comparisons.append(
            {
                "case_id": case["id"],
                "file_sha256": file_sha256,
                "source_locator": case["source_locator"],
                "runtime_input_name": row.source_path.name,
                "claimed_training_label": case["training_label"],
                "final_label": row.proposed_folder,
                "final_top": row.final_top,
                "consensus_status": row.consensus_status,
                "neural_label": row.neural_folder,
                "neural_exact_training_match": str(row.neural_exact_training_match),
                "semantic_family": row.neural_semantic_family,
                "semantic_top_score": str(row.neural_semantic_score),
                "prompt_brain_status": row.neural_prompt_status,
                "prompt_top_label": prompt_labels[0] if prompt_labels else "",
                "prompt_top3_labels_json": json.dumps(prompt_labels[:3]),
                "prompt_top1_matches_claim": (
                    "1" if prompt_labels and prompt_labels[0] == case["training_label"] else "0"
                ),
                "prompt_top3_matches_claim": "1" if case["training_label"] in prompt_labels[:3] else "0",
                "matches_claimed_label": "1" if row.proposed_folder == case["training_label"] else "0",
            }
        )
    return comparisons


def write_comparison_report(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the trusted-audio runtime comparison CSV."""
    fieldnames = [
        "case_id",
        "file_sha256",
        "source_locator",
        "runtime_input_name",
        "claimed_training_label",
        "final_label",
        "final_top",
        "consensus_status",
        "neural_label",
        "neural_exact_training_match",
        "semantic_family",
        "semantic_top_score",
        "prompt_brain_status",
        "prompt_top_label",
        "prompt_top3_labels_json",
        "prompt_top1_matches_claim",
        "prompt_top3_matches_claim",
        "matches_claimed_label",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
