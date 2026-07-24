"""Per-label encoder leaderboards and evidence-gated fusion policy."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from pathlib import Path

from .contracts import FusionGateDecision, LabelLaneChoice, LaneEvaluation
from .evaluation import choose_per_label_lanes


def ranked_lane_evaluations(evaluations: Iterable[LaneEvaluation]) -> tuple[LaneEvaluation, ...]:
    """Return a stable per-label leaderboard without blending providers."""
    return tuple(
        sorted(
            evaluations,
            key=lambda row: (
                row.label,
                -row.top1_accuracy,
                -row.known_distribution_rate,
                -row.mean_margin,
                -row.heldout_count,
                row.provider_id,
            ),
        )
    )


def write_lane_leaderboard(evaluations: Iterable[LaneEvaluation], path: Path) -> tuple[LabelLaneChoice, ...]:
    """Persist every lane result plus the held-out winner for each label."""
    rows = ranked_lane_evaluations(evaluations)
    choices = choose_per_label_lanes(rows)
    winners = {choice.label: choice.provider_id for choice in choices}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "provider_id",
                "heldout_count",
                "correct_count",
                "top1_accuracy",
                "mean_correct_similarity",
                "mean_margin",
                "known_distribution_rate",
                "selected_for_label",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.label,
                    row.provider_id,
                    row.heldout_count,
                    row.correct_count,
                    f"{row.top1_accuracy:.8f}",
                    f"{row.mean_correct_similarity:.8f}",
                    f"{row.mean_margin:.8f}",
                    f"{row.known_distribution_rate:.8f}",
                    row.provider_id == winners.get(row.label),
                ]
            )
    return choices


def gate_fusion_candidate(
    single_lane_evaluations: Sequence[LaneEvaluation],
    fused_evaluation: LaneEvaluation,
    *,
    minimum_heldout_count: int = 10,
    minimum_accuracy_gain: float = 0.02,
    maximum_ood_rate_loss: float = 0.05,
) -> FusionGateDecision:
    """Allow fusion only when held-out evidence beats the best single lane.

    A fused lane must use the same label and held-out count as the comparison
    lanes.  It is rejected when the evaluation set is too small, top-1 gain is
    below the declared threshold, or OOD coverage degrades materially.
    """
    if not single_lane_evaluations:
        raise ValueError("at least one single-lane evaluation is required")
    if minimum_heldout_count < 1:
        raise ValueError("minimum_heldout_count must be positive")
    if minimum_accuracy_gain < 0.0 or maximum_ood_rate_loss < 0.0:
        raise ValueError("fusion thresholds cannot be negative")
    labels = {row.label for row in single_lane_evaluations} | {fused_evaluation.label}
    if len(labels) != 1:
        raise ValueError("fusion comparisons must use one label")
    heldout_counts = {row.heldout_count for row in single_lane_evaluations} | {fused_evaluation.heldout_count}
    if len(heldout_counts) != 1:
        raise ValueError("fusion comparisons must use the same held-out examples")

    best = sorted(
        single_lane_evaluations,
        key=lambda row: (-row.top1_accuracy, -row.known_distribution_rate, -row.mean_margin, row.provider_id),
    )[0]
    gain = fused_evaluation.top1_accuracy - best.top1_accuracy
    ood_loss = best.known_distribution_rate - fused_evaluation.known_distribution_rate

    if fused_evaluation.heldout_count < minimum_heldout_count:
        accepted = False
        reason = "insufficient held-out examples for a fusion decision"
    elif gain < minimum_accuracy_gain:
        accepted = False
        reason = "fusion did not beat the strongest single lane by the required held-out accuracy gain"
    elif ood_loss > maximum_ood_rate_loss:
        accepted = False
        reason = "fusion reduced known-distribution coverage beyond the allowed loss"
    else:
        accepted = True
        reason = "fusion measurably beat the strongest single lane on the same held-out examples"

    return FusionGateDecision(
        label=fused_evaluation.label,
        best_single_provider_id=best.provider_id,
        fused_provider_id=fused_evaluation.provider_id,
        heldout_count=fused_evaluation.heldout_count,
        best_single_accuracy=best.top1_accuracy,
        fused_accuracy=fused_evaluation.top1_accuracy,
        accuracy_gain=gain,
        accepted=accepted,
        reason=reason,
    )
