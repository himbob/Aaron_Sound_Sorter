"""Held-out evaluation and per-label representation selection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from .contracts import EmbeddingRecord, LabelLaneChoice, LaneEvaluation
from .prototype_index import PrototypeIndex


def evaluate_heldout(
    index: PrototypeIndex,
    labeled_heldout: Mapping[str, Sequence[EmbeddingRecord]],
) -> tuple[LaneEvaluation, ...]:
    """Evaluate only examples whose hashes were excluded from index training."""
    training_hashes = set(index.metadata.training_hashes)
    results: list[LaneEvaluation] = []
    for label, records in sorted(labeled_heldout.items()):
        if not records:
            continue
        overlap = training_hashes & {record.file_sha256 for record in records}
        if overlap:
            raise ValueError(f"held-out evaluation reused {len(overlap)} training example(s) for {label}")
        predictions = [index.predict(record) for record in records]
        correct = [prediction for prediction in predictions if prediction.predicted_label == label]
        results.append(
            LaneEvaluation(
                provider_id=index.metadata.provider_id,
                label=label,
                heldout_count=len(records),
                correct_count=len(correct),
                top1_accuracy=len(correct) / len(records),
                mean_correct_similarity=float(np.mean([p.top_similarity for p in correct])) if correct else 0.0,
                mean_margin=float(np.mean([p.margin for p in predictions])),
                known_distribution_rate=float(np.mean([1.0 if p.known_distribution else 0.0 for p in predictions])),
            )
        )
    return tuple(results)


def choose_per_label_lanes(evaluations: Iterable[LaneEvaluation]) -> tuple[LabelLaneChoice, ...]:
    """Choose one provider per label; do not silently average weak lanes."""
    by_label: dict[str, list[LaneEvaluation]] = defaultdict(list)
    for evaluation in evaluations:
        by_label[evaluation.label].append(evaluation)

    choices: list[LabelLaneChoice] = []
    for label, rows in sorted(by_label.items()):
        ranked = sorted(
            rows,
            key=lambda row: (
                -row.top1_accuracy,
                -row.known_distribution_rate,
                -row.mean_margin,
                -row.heldout_count,
                row.provider_id,
            ),
        )
        winner = ranked[0]
        score = (
            0.70 * winner.top1_accuracy
            + 0.20 * winner.known_distribution_rate
            + 0.10 * min(1.0, max(0.0, winner.mean_margin * 5.0))
        )
        choices.append(
            LabelLaneChoice(
                label=label,
                provider_id=winner.provider_id,
                score=float(score),
                heldout_count=winner.heldout_count,
                reason="best held-out accuracy; ties resolved by OOD coverage and margin",
            )
        )
    return tuple(choices)
