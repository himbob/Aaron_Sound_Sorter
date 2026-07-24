"""Source-blind contamination audit for human-curated neural trainers."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .contracts import EmbeddingRecord, TrainerAuditRow


def _robust_floor(values: Sequence[float], *, minimum_slack: float = 0.02) -> float | None:
    """Return a conservative lower support bound from a label's own examples."""
    if len(values) < 4:
        return None
    arr = np.asarray(values, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    robust_sigma = 1.4826 * mad
    return float(max(-1.0, median - max(minimum_slack, 3.0 * robust_sigma)))


def audit_labeled_embeddings(
    labeled_embeddings: Mapping[str, Sequence[EmbeddingRecord]],
    *,
    cross_label_margin: float = 0.02,
) -> tuple[TrainerAuditRow, ...]:
    """Audit trainers using leave-one-out neighborhood evidence.

    The audit does not estimate model accuracy and must not be reported as held-out
    evaluation.  It finds duplicate-label conflicts, unsupported singletons,
    category outliers, and examples represented more strongly by another label.
    """
    if not labeled_embeddings:
        raise ValueError("no labeled embeddings supplied")
    if cross_label_margin < 0.0:
        raise ValueError("cross_label_margin cannot be negative")

    flattened: list[tuple[str, EmbeddingRecord]] = []
    for label, records in sorted(labeled_embeddings.items()):
        for record in sorted(records, key=lambda item: item.file_sha256):
            flattened.append((label, record))
    if not flattened:
        raise ValueError("no embedding records supplied")

    provider_pairs = {(record.provider_id, record.model_id) for _, record in flattened}
    if len(provider_pairs) != 1:
        raise ValueError("one trainer audit may contain exactly one provider/model pair")
    provider_id, model_id = next(iter(provider_pairs))
    dimensions = {record.dimension for _, record in flattened}
    if len(dimensions) != 1:
        raise ValueError("all embeddings must have the same dimension")

    hash_labels: dict[str, set[str]] = defaultdict(set)
    for label, record in flattened:
        hash_labels[record.file_sha256].add(label)

    # NumPy 2.0's macOS Accelerate GEMM can emit spurious overflow warnings for
    # this otherwise bounded unit-vector matrix.  The explicit contraction
    # avoids that backend path and keeps leave-one-out comparisons deterministic.
    matrix = np.vstack([record.vector for _, record in flattened]).astype(np.float64)
    similarities = np.einsum("ik,jk->ij", matrix, matrix, optimize=False)
    labels = [label for label, _ in flattened]

    nearest_same: list[float | None] = []
    nearest_other: list[tuple[str, float] | None] = []
    for index, label in enumerate(labels):
        same_candidates = [j for j, other_label in enumerate(labels) if j != index and other_label == label]
        other_candidates = [j for j, other_label in enumerate(labels) if other_label != label]
        same_score = max((float(similarities[index, j]) for j in same_candidates), default=None)
        if other_candidates:
            other_idx = max(
                other_candidates, key=lambda j: (float(similarities[index, j]), labels[j], flattened[j][1].file_sha256)
            )
            other = (labels[other_idx], float(similarities[index, other_idx]))
        else:
            other = None
        nearest_same.append(same_score)
        nearest_other.append(other)

    same_by_label: dict[str, list[float]] = defaultdict(list)
    for label, score in zip(labels, nearest_same):
        if score is not None:
            same_by_label[label].append(score)
    floors = {label: _robust_floor(values) for label, values in same_by_label.items()}
    label_counts = Counter(labels)

    rows: list[TrainerAuditRow] = []
    for index, (label, record) in enumerate(flattened):
        same_score = nearest_same[index]
        other = nearest_other[index]
        other_label = other[0] if other else ""
        other_score = other[1] if other else None
        margin = None if same_score is None or other_score is None else same_score - other_score
        floor = floors.get(label)
        reasons: list[str] = []

        if len(hash_labels[record.file_sha256]) > 1:
            status = "duplicate_hash_label_conflict"
            reasons.append("identical audio bytes are assigned to multiple labels")
        elif label_counts[label] == 1:
            status = "prototype_only_singleton"
            reasons.append("one example cannot establish category variation or verify label support")
        elif same_score is not None and other_score is not None and other_score >= same_score + cross_label_margin:
            status = "cross_label_conflict"
            reasons.append("another label supports this trainer more strongly than its assigned label")
        elif floor is not None and same_score is not None and same_score < floor:
            status = "possible_label_outlier"
            reasons.append("same-label support falls below the label's robust leave-one-out floor")
        elif same_score is not None and other_score is not None and other_score > same_score:
            status = "ambiguous_cross_label_support"
            reasons.append("another label is slightly closer; manual review is required")
        elif label_counts[label] < 4:
            status = "limited_support"
            reasons.append("too few examples for a robust category outlier threshold")
        else:
            status = "supported"

        rows.append(
            TrainerAuditRow(
                provider_id=provider_id,
                model_id=model_id,
                label=label,
                file_sha256=record.file_sha256,
                label_example_count=label_counts[label],
                nearest_same_similarity=same_score,
                nearest_other_label=other_label,
                nearest_other_similarity=other_score,
                support_margin=margin,
                robust_same_label_floor=floor,
                status=status,
                reasons=tuple(reasons),
            )
        )
    return tuple(sorted(rows, key=lambda row: (row.status == "supported", row.status, row.label, row.file_sha256)))


def write_trainer_audit_csv(
    rows: Sequence[TrainerAuditRow],
    path: Path,
    *,
    display_names_by_hash: Mapping[str, str] | None = None,
) -> None:
    """Write disagreements and weak trainers first for efficient human review."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "provider_id",
                "model_id",
                "label",
                "display_name",
                "file_sha256",
                "label_example_count",
                "nearest_same_similarity",
                "nearest_other_label",
                "nearest_other_similarity",
                "support_margin",
                "robust_same_label_floor",
                "status",
                "reasons",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.provider_id,
                    row.model_id,
                    row.label,
                    (display_names_by_hash or {}).get(row.file_sha256, ""),
                    row.file_sha256,
                    row.label_example_count,
                    "" if row.nearest_same_similarity is None else f"{row.nearest_same_similarity:.8f}",
                    row.nearest_other_label,
                    "" if row.nearest_other_similarity is None else f"{row.nearest_other_similarity:.8f}",
                    "" if row.support_margin is None else f"{row.support_margin:.8f}",
                    "" if row.robust_same_label_floor is None else f"{row.robust_same_label_floor:.8f}",
                    row.status,
                    "; ".join(row.reasons),
                ]
            )


def trainer_audit_status_counts(rows: Sequence[TrainerAuditRow]) -> dict[str, int]:
    return dict(sorted(Counter(row.status for row in rows).items()))
