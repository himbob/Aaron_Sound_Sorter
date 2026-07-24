"""Leakage-safe curated dataset discovery and splitting."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Union

from .contracts import (
    HELDOUT_EVAL_SPLIT,
    REVIEW_PREVIEW_SPLIT,
    EvaluationSplit,
    LabeledAudioExample,
)
from .hashing import decoded_audio_sha256, sha256_file

AUDIO_SUFFIXES = frozenset({".wav", ".aif", ".aiff", ".flac", ".ogg", ".au", ".mp3", ".m4a"})
CuratedRow = Union[tuple[Path, str], tuple[Path, str, str]]


def discover_curated_examples(
    root: Path,
    *,
    allow_label_conflicts: bool = False,
) -> dict[str, list[tuple[Path, str, str]]]:
    """Read labels from deliberately curated relative folder paths.

    File names are never inspected for label evidence.  Duplicate bytes assigned
    to different labels are rejected because they would poison both training and
    held-out evaluation.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"curated training root is not a directory: {root}")

    by_label: dict[str, list[tuple[Path, str, str]]] = defaultdict(list)
    hash_labels: dict[str, set[str]] = defaultdict(set)
    decoded_hash_labels: dict[str, set[str]] = defaultdict(set)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        if path.name.startswith(".") or any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        relative_parent = path.relative_to(root).parent
        label = "/".join(relative_parent.parts).strip("/")
        if not label:
            raise ValueError(f"audio file must be inside a curated label folder: {path}")
        digest = sha256_file(path)
        try:
            decoded_digest = decoded_audio_sha256(path)
        except (OSError, RuntimeError):
            decoded_digest = ""
        by_label[label].append((path, digest, decoded_digest))
        hash_labels[digest].add(label)
        if decoded_digest:
            decoded_hash_labels[decoded_digest].add(label)

    conflicts = {digest: labels for digest, labels in hash_labels.items() if len(labels) > 1}
    decoded_conflicts = {digest: labels for digest, labels in decoded_hash_labels.items() if len(labels) > 1}
    if conflicts and not allow_label_conflicts:
        preview = ", ".join(f"{digest[:10]}:{sorted(labels)}" for digest, labels in list(conflicts.items())[:5])
        raise ValueError(f"same audio bytes assigned to multiple labels: {preview}")
    if decoded_conflicts and not allow_label_conflicts:
        preview = ", ".join(f"{digest[:10]}:{sorted(labels)}" for digest, labels in list(decoded_conflicts.items())[:5])
        raise ValueError(f"same decoded audio assigned to multiple labels: {preview}")
    return dict(by_label)


def _stable_rank(seed: str, label: str, file_sha256: str) -> str:
    return hashlib.sha256(f"{seed}\0{label}\0{file_sha256}".encode()).hexdigest()


def build_evaluation_split(
    by_label: Mapping[str, Sequence[CuratedRow]],
    *,
    holdout_fraction: float = 0.20,
    seed: str = "aaron-neural-v1",
    minimum_train_examples: int = 2,
) -> EvaluationSplit:
    """Create disjoint preview and held-out sets per label."""
    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be between 0 and 0.5")
    if minimum_train_examples < 1:
        raise ValueError("minimum_train_examples must be positive")

    preview: list[LabeledAudioExample] = []
    heldout: list[LabeledAudioExample] = []
    warnings: list[str] = []
    global_seen: set[str] = set()
    global_decoded_seen: set[str] = set()

    for label, rows in sorted(by_label.items()):
        unique: dict[str, tuple[Path, str]] = {}
        for row in rows:
            path, digest, decoded_digest = _curated_row_parts(row)
            content_key = decoded_digest or digest
            unique.setdefault(content_key, (Path(path), digest))
        duplicates_removed = len(rows) - len(unique)
        if duplicates_removed:
            warnings.append(f"{label}: removed {duplicates_removed} duplicate-audio example(s)")

        ranked = sorted(
            ((digest, path, content_key) for content_key, (path, digest) in unique.items()),
            key=lambda item: _stable_rank(seed, label, item[0]),
        )
        n = len(ranked)
        if n <= minimum_train_examples:
            holdout_count = 0
            warnings.append(f"{label}: {n} example(s); prototype-only, no held-out evaluation")
        else:
            holdout_count = max(1, int(round(n * holdout_fraction)))
            holdout_count = min(holdout_count, n - minimum_train_examples)

        heldout_hashes = {digest for digest, _, _ in ranked[:holdout_count]}
        for digest, path, content_key in ranked:
            if digest in global_seen or content_key in global_decoded_seen:
                continue
            global_seen.add(digest)
            global_decoded_seen.add(content_key)
            split = HELDOUT_EVAL_SPLIT if digest in heldout_hashes else REVIEW_PREVIEW_SPLIT
            example = LabeledAudioExample(
                label=label,
                path=path,
                file_sha256=digest,
                split=split,
                decoded_audio_sha256=content_key,
            )
            (heldout if split == HELDOUT_EVAL_SPLIT else preview).append(example)

    return EvaluationSplit(tuple(preview), tuple(heldout), tuple(warnings))


def build_explicit_evaluation_split(
    review_preview_by_label: Mapping[str, Sequence[CuratedRow]],
    heldout_eval_by_label: Mapping[str, Sequence[CuratedRow]],
) -> EvaluationSplit:
    """Build a leakage-checked split from deliberately separate roots.

    Args:
        review_preview_by_label: Human-curated prototype-building examples.
        heldout_eval_by_label: Human-curated evaluation examples that were not
            used to construct prototypes.

    Returns:
        Explicit split with byte-hash and decoded-audio leakage protection.

    Raises:
        ValueError: If held-out labels lack training support or either byte or
            decoded-content fingerprints cross the partition boundary.
    """
    missing_labels = sorted(set(heldout_eval_by_label) - set(review_preview_by_label))
    if missing_labels:
        raise ValueError(f"held-out labels have no preview support: {missing_labels}")

    preview, preview_warnings = _examples_for_explicit_split(
        review_preview_by_label,
        split=REVIEW_PREVIEW_SPLIT,
    )
    heldout, heldout_warnings = _examples_for_explicit_split(
        heldout_eval_by_label,
        split=HELDOUT_EVAL_SPLIT,
    )
    return EvaluationSplit(
        review_preview=tuple(preview),
        heldout_eval=tuple(heldout),
        warnings=tuple(preview_warnings + heldout_warnings),
    )


def _curated_row_parts(row: CuratedRow) -> tuple[Path, str, str]:
    if len(row) == 2:
        path, digest = row
        path = Path(path)
        try:
            decoded_digest = decoded_audio_sha256(path) if path.is_file() else ""
        except (OSError, RuntimeError):
            decoded_digest = ""
        return path, str(digest), decoded_digest
    path, digest, decoded_digest = row
    return Path(path), str(digest), str(decoded_digest)


def _examples_for_explicit_split(
    by_label: Mapping[str, Sequence[CuratedRow]],
    *,
    split: str,
) -> tuple[list[LabeledAudioExample], list[str]]:
    examples: list[LabeledAudioExample] = []
    warnings: list[str] = []
    global_seen: set[str] = set()
    global_decoded_seen: set[str] = set()
    for label, rows in sorted(by_label.items()):
        kept = 0
        for row in sorted(rows, key=lambda value: (_curated_row_parts(value)[1], str(_curated_row_parts(value)[0]))):
            path, digest, decoded_digest = _curated_row_parts(row)
            content_key = decoded_digest or digest
            if digest in global_seen or content_key in global_decoded_seen:
                continue
            global_seen.add(digest)
            global_decoded_seen.add(content_key)
            examples.append(
                LabeledAudioExample(
                    label=label,
                    path=path,
                    file_sha256=digest,
                    split=split,
                    decoded_audio_sha256=content_key,
                )
            )
            kept += 1
        duplicates_removed = len(rows) - kept
        if duplicates_removed:
            warnings.append(f"{split}/{label}: removed {duplicates_removed} duplicate-audio example(s)")
    return examples, warnings


def write_split_manifest(split: EvaluationSplit, output_dir: Path) -> None:
    """Write separate machine-readable preview and held-out manifests."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(name: str, rows: Iterable[LabeledAudioExample]) -> None:
        path = output_dir / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["label", "path", "file_sha256", "decoded_audio_sha256", "split"])
            for row in rows:
                writer.writerow(
                    [
                        row.label,
                        str(row.path),
                        row.file_sha256,
                        row.decoded_audio_sha256,
                        row.split,
                    ]
                )

    write_csv("review_preview.csv", split.review_preview)
    write_csv("heldout_eval.csv", split.heldout_eval)
    summary = {
        "schema_version": 1,
        "review_preview_count": len(split.review_preview),
        "heldout_eval_count": len(split.heldout_eval),
        "warnings": list(split.warnings),
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
