#!/usr/bin/env python3
"""Cluster-first active learning queue for source-name-blind neural audio training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf

from .contracts import LabelPrediction
from .dataset import AUDIO_SUFFIXES
from .gui_training import configured_clap_trainer
from .hashing import sha256_file
from .prototype_index import PrototypeIndex

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass(frozen=True)
class ActiveLearningCandidate:
    """One candidate ranked for cluster-first human review."""

    path: Path
    file_sha256: str
    prediction: LabelPrediction
    vector: np.ndarray = field(repr=False)
    priority_score: float
    review_reason: str


@dataclass(frozen=True)
class ClusterReviewPackSummary:
    """Summary of an active learning pack build."""

    selected_count: int
    candidate_count: int
    selected_by_label: dict[str, int]
    active_label_counts: dict[str, int]
    destination: str
    active_index_path: str
    status: str
    sample_library_root: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--sample-library-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--max-per-label", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_candidates < 1 or args.max_per_label < 1:
        raise ValueError("candidate limits must be positive")
    if not args.dry_run and args.destination is None:
        raise ValueError("--destination is required unless --dry-run is used")

    project_root = args.project_root.expanduser().resolve()
    sample_library_root = args.sample_library_root.expanduser().resolve()
    destination = args.destination.expanduser().resolve() if args.destination else None

    trainer = configured_clap_trainer(project_root)
    index_path = _active_index_path(project_root, trainer.pointer_path)
    selected = build_cluster_review_pack(
        project_root=project_root,
        sample_library_root=sample_library_root,
        max_candidates=args.max_candidates,
        max_per_label=args.max_per_label,
        trainer=trainer,
    )

    copied_rows: list[dict[str, Any]] = []
    if not args.dry_run:
        assert destination is not None
        copied_rows = _copy_review_pack(destination, selected)

    summary = {
        "schema_version": 1,
        "status": "dry_run" if args.dry_run else "built",
        "destination": str(destination) if destination else "",
        "active_index_path": str(index_path),
        "selected_count": len(selected),
        "candidate_count": len(selected),
        "selected_by_label": {label: sum(row.prediction.predicted_label == label for row in selected) for label in sorted({row.prediction.predicted_label for row in selected})},
        "sample_library_root": str(sample_library_root),
        "candidates": copied_rows if copied_rows else [_candidate_payload(row) for row in selected],
    }
    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "candidates"}, indent=2))
    return 0


def build_cluster_review_pack(
    *,
    project_root: Path,
    sample_library_root: Path,
    max_candidates: int = 80,
    max_per_label: int = 8,
    similarity_threshold: float = 0.92,
    trainer: Optional[Any] = None,
) -> list[ActiveLearningCandidate]:
    project_root = project_root.expanduser().resolve()
    sample_library_root = sample_library_root.expanduser().resolve()
    if trainer is None:
        trainer = configured_clap_trainer(project_root)
    index_path = _active_index_path(project_root, trainer.pointer_path)
    index = PrototypeIndex.load(index_path)
    active_hashes = set(index.metadata.training_hashes)

    candidates = _collect_candidates(
        sample_library_root=sample_library_root,
        active_hashes=active_hashes,
        trainer=trainer,
        index=index,
    )
    return _select_clustered_candidates(
        candidates,
        max_candidates=max_candidates,
        max_per_label=max_per_label,
        similarity_threshold=similarity_threshold,
    )


def _active_index_path(project_root: Path, pointer_path: Path) -> Path:
    value = pointer_path.read_text(encoding="utf-8").strip()
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _collect_candidates(
    *,
    sample_library_root: Path,
    active_hashes: set[str],
    trainer: Any,
    index: PrototypeIndex,
) -> list[ActiveLearningCandidate]:
    candidates: list[ActiveLearningCandidate] = []
    seen_hashes: set[str] = set()
    for path in sorted(sample_library_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        try:
            digest = sha256_file(path)
        except (OSError, RuntimeError, ValueError):
            continue
        if digest in active_hashes or digest in seen_hashes:
            continue
        try:
            info = sf.info(path)
            duration_sec = float(info.duration)
            if not math.isfinite(duration_sec) or not 0.05 <= duration_sec <= 180.0:
                continue
            record = trainer.cache.get_or_compute(path, trainer.provider)
            prediction = index.predict(record)
        except (OSError, RuntimeError, ValueError):
            continue
        seen_hashes.add(digest)
        priority_score, review_reason = _priority_score(prediction, index.metadata.label_example_counts)
        candidates.append(
            ActiveLearningCandidate(
                path=path,
                file_sha256=digest,
                prediction=prediction,
                vector=record.vector,
                priority_score=priority_score,
                review_reason=review_reason,
            )
        )
    return candidates


def _priority_score(prediction: LabelPrediction, active_label_counts: dict[str, int]) -> tuple[float, str]:
    label_example_count = int(prediction.evidence.get("label_example_count", 0))
    margin = prediction.margin
    top_similarity = prediction.top_similarity
    under_support = max(0.0, 4.0 - float(label_example_count)) / 4.0
    low_margin = max(0.0, 0.25 - margin) / 0.25
    low_similarity = max(0.0, 0.95 - top_similarity) / 0.95
    ood = 1.0 if not prediction.known_distribution else 0.0

    if ood:
        reason = "outside_learned_radius"
    elif margin < 0.10:
        reason = "near_decision_boundary"
    elif label_example_count < 3:
        reason = "sparse_label"
    else:
        reason = "cluster_novelty"

    score = 3.0 * ood + 2.0 * low_margin + 1.5 * under_support + 1.0 * low_similarity
    if prediction.predicted_label in active_label_counts and active_label_counts[prediction.predicted_label] < 3:
        score += 0.5
    return float(score), reason


def _select_clustered_candidates(
    candidates: list[ActiveLearningCandidate],
    *,
    max_candidates: int,
    max_per_label: int,
    similarity_threshold: float,
) -> list[ActiveLearningCandidate]:
    ordered = sorted(
        candidates,
        key=lambda row: (
            -row.priority_score,
            -row.prediction.margin,
            -row.prediction.top_similarity,
            row.file_sha256,
        ),
    )
    selected: list[ActiveLearningCandidate] = []
    selected_by_label: dict[str, int] = defaultdict(int)
    selected_vectors_by_label: dict[str, list[np.ndarray]] = defaultdict(list)

    for candidate in ordered:
        if len(selected) >= max_candidates:
            break
        label = candidate.prediction.predicted_label
        if selected_by_label[label] >= max_per_label:
            continue
        if _is_redundant(candidate.vector, selected_vectors_by_label[label], similarity_threshold):
            continue
        selected.append(candidate)
        selected_by_label[label] += 1
        selected_vectors_by_label[label].append(candidate.vector)

    return selected


def _is_redundant(vector: np.ndarray, selected_vectors: list[np.ndarray], threshold: float) -> bool:
    for selected_vector in selected_vectors:
        if float(np.dot(vector, selected_vector)) >= threshold:
            return True
    return False


def _copy_review_pack(destination: Path, selected: list[ActiveLearningCandidate]) -> list[dict[str, Any]]:
    if destination.exists():
        raise FileExistsError(f"review destination already exists: {destination}")
    destination.mkdir(parents=True)
    payloads: list[dict[str, Any]] = []
    for position, candidate in enumerate(selected, start=1):
        safe_name = _safe_filename(candidate.path.stem)
        copied_name = f"candidate_{position:03d}_{candidate.file_sha256[:8]}_{safe_name}{candidate.path.suffix.lower()}"
        copied_path = destination / copied_name
        shutil.copy2(candidate.path, copied_path)
        payload = _candidate_payload(candidate)
        payload.update(
            {
                "candidate_file": copied_name,
                "priority_reason": candidate.review_reason,
            }
        )
        payloads.append(payload)
    _write_manifest(destination / "REVIEW_MANIFEST.csv", payloads)
    (destination / "README.txt").write_text(
        "Aaron_Active_Learning_Review_Pack\n\n"
        "1. Listen to every audio file.\n"
        "2. The candidate label is a neural prediction, not a human-approved ground truth.\n"
        "3. Use the file and prediction only as a suggestion for review, not as training truth.\n"
        "4. Correct the category in the GUI if the label is wrong.\n\n"
        "Filenames and folders are only used to gather candidates; the neural model is source-name-blind.\n",
        encoding="utf-8",
    )
    return payloads


def _candidate_payload(candidate: ActiveLearningCandidate) -> dict[str, Any]:
    return {
        "path": str(candidate.path),
        "file_sha256": candidate.file_sha256,
        "predicted_label": candidate.prediction.predicted_label,
        "second_label": candidate.prediction.second_label,
        "top_similarity": candidate.prediction.top_similarity,
        "second_similarity": candidate.prediction.second_similarity,
        "margin": candidate.prediction.margin,
        "known_distribution": candidate.prediction.known_distribution,
        "prototype_id": candidate.prediction.prototype_id,
        "radius_ratio": candidate.prediction.radius_ratio,
        "priority_score": candidate.priority_score,
        "review_reason": candidate.review_reason,
        "label_example_count": int(candidate.prediction.evidence.get("label_example_count", 0)),
    }


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _safe_filename(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in " ._-" else "_" for character in value)
    return "_".join(cleaned.split())[:120] or "audio"


if __name__ == "__main__":
    raise SystemExit(main())
