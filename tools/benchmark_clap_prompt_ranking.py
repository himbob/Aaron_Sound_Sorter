#!/usr/bin/env python3
"""Benchmark detailed CLAP prompt ranking on explicit trusted labels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.neural_audio.gui_training import configured_clap_trainer  # noqa: E402
from aaron_sound_sorter.neural_audio.prompt_brain import (  # noqa: E402
    ClapPromptIndex,
    PromptCategoryScore,
)

DEFAULT_SEED_MANIFEST = Path("tests/acceptance/locked_smoke_v1/trusted_training_seed_v1.json")
DEFAULT_POINTER = Path("config/runtime/neural_clap_prompt_index_path.txt")
DEFAULT_REPORT_ROOT = Path("_reports/foundation_prompt_brain/benchmarks")
NEGATIVE_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--seed-manifest", type=Path, default=DEFAULT_SEED_MANIFEST)
    parser.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate ranking rules without enabling prompt-based ownership."""
    args = build_parser().parse_args(argv)
    root = args.project_root.expanduser().resolve()
    seed_path = resolve_path(root, args.seed_manifest)
    pointer_path = resolve_path(root, args.pointer)
    index_path = resolve_path(root, Path(pointer_path.read_text(encoding="utf-8").strip()))
    index = ClapPromptIndex.load(index_path)
    trainer = configured_clap_trainer(root)
    cases = load_cases(seed_path)
    metrics = {weight_key(weight): empty_metrics() for weight in NEGATIVE_WEIGHTS}
    rows: list[dict[str, Any]] = []
    for case in cases:
        audio_path = resolve_path(root, Path(case["audio_path"]))
        record = trainer.cache.get_or_compute(audio_path, trainer.provider)
        all_scores = index.predict(record, top_k=len(index.categories))
        row: dict[str, Any] = {
            "case_id": case["id"],
            "audio_sha256": record.file_sha256,
            "claimed_label": case["training_label"],
            "rankings": {},
        }
        for weight in NEGATIVE_WEIGHTS:
            ranked = rank_prompt_scores(all_scores, negative_weight=weight)
            update_metrics(metrics[weight_key(weight)], ranked, case["training_label"])
            row["rankings"][weight_key(weight)] = [asdict(score) for score in ranked[:3]]
        rows.append(row)
    report = {
        "schema_version": 1,
        "case_count": len(cases),
        "model_id": index.metadata.model_id,
        "prompt_catalog_version": index.metadata.prompt_catalog_version,
        "production_ownership_enabled": False,
        "source_name_policy": "audio waveforms and explicit trusted labels only; source names forbidden",
        "metrics": metrics,
        "rows": rows,
    }
    report_root = resolve_path(root, args.report_root)
    report_dir = report_root / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    report_dir.mkdir(parents=True, exist_ok=False)
    report_path = report_dir / "prompt_ranking_benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "report_path": str(report_path)}, indent=2, sort_keys=True))
    return 0


def rank_prompt_scores(
    scores: tuple[PromptCategoryScore, ...],
    *,
    negative_weight: float,
) -> tuple[PromptCategoryScore, ...]:
    """Rank category prompts with a visible, benchmarkable confusion penalty."""
    return tuple(
        sorted(
            scores,
            key=lambda row: (
                -(row.positive_score - negative_weight * row.negative_score),
                -row.positive_score,
                row.path,
            ),
        )
    )


def load_cases(path: Path) -> list[dict[str, str]]:
    """Load explicit trusted labels without deriving targets from paths."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("trusted seed cases must be a list")
    return [
        {
            "id": str(case["id"]),
            "audio_path": str(case["audio_path"]),
            "training_label": str(case["training_label"]),
        }
        for case in cases
        if isinstance(case, dict)
    ]


def empty_metrics() -> dict[str, int]:
    """Return zeroed exact, identity, and broad-family counters."""
    return {
        "exact_top1": 0,
        "exact_top3": 0,
        "identity_top1": 0,
        "identity_top3": 0,
        "top_level_top1": 0,
        "top_level_top3": 0,
    }


def update_metrics(
    metrics: dict[str, int],
    ranked: tuple[PromptCategoryScore, ...],
    claimed_label: str,
) -> None:
    """Update benchmark counters for one explicit trusted claim."""
    top_paths = [score.path for score in ranked[:3]]
    claimed_identity = identity_path(claimed_label)
    claimed_top_level = claimed_label.split("/", 1)[0]
    metrics["exact_top1"] += int(bool(top_paths) and top_paths[0] == claimed_label)
    metrics["exact_top3"] += int(claimed_label in top_paths)
    metrics["identity_top1"] += int(bool(top_paths) and identity_path(top_paths[0]) == claimed_identity)
    metrics["identity_top3"] += int(any(identity_path(path) == claimed_identity for path in top_paths))
    metrics["top_level_top1"] += int(bool(top_paths) and top_paths[0].split("/", 1)[0] == claimed_top_level)
    metrics["top_level_top3"] += int(any(path.split("/", 1)[0] == claimed_top_level for path in top_paths))


def identity_path(label: str) -> str:
    """Return a category identity without its objective structure terminal."""
    return str(label).rsplit("/", 1)[0]


def weight_key(weight: float) -> str:
    """Return a stable JSON key for one confusion-penalty weight."""
    return f"negative_weight_{weight:.2f}"


def resolve_path(project_root: Path, path: Path) -> Path:
    """Resolve a project-relative or absolute path."""
    expanded = Path(path).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root / expanded).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
