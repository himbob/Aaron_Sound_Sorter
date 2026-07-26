"""Comprehensive held-out metrics for safe neural ownership decisions."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class HeldoutPrediction:
    """One independent expected label and source-name-blind prediction."""

    file_sha256: str
    expected_label: str
    predicted_labels: tuple[str, ...]
    known_distribution: bool
    automatically_placed: bool


@dataclass(frozen=True)
class HeldoutMetrics:
    """Production-oriented metrics over one independent evaluation panel."""

    example_count: int
    top1_accuracy: float
    top3_accuracy: float
    parent_family_accuracy: float
    review_coverage: float
    incorrect_auto_placement_rate: float
    out_of_distribution_rate: float
    catastrophic_auto_placement_count: int
    confusion_counts: tuple[tuple[str, str, int], ...]


def evaluate_heldout_predictions(rows: list[HeldoutPrediction]) -> HeldoutMetrics:
    """Calculate accuracy, review, OOD, and catastrophic-placement metrics."""
    if not rows:
        return HeldoutMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, ())
    top1_correct = 0
    top3_correct = 0
    parent_correct = 0
    review_count = 0
    ood_count = 0
    catastrophic = 0
    confusions: Counter[tuple[str, str]] = Counter()
    for row in rows:
        predicted = row.predicted_labels[0] if row.predicted_labels else ""
        exact = predicted == row.expected_label
        top1_correct += int(exact)
        top3_correct += int(row.expected_label in row.predicted_labels[:3])
        parent_correct += int(_parent_family(predicted) == _parent_family(row.expected_label))
        review_count += int(not row.automatically_placed)
        ood_count += int(not row.known_distribution)
        if row.automatically_placed and not exact:
            catastrophic += 1
            confusions[(row.expected_label, predicted)] += 1
    count = len(rows)
    return HeldoutMetrics(
        example_count=count,
        top1_accuracy=top1_correct / count,
        top3_accuracy=top3_correct / count,
        parent_family_accuracy=parent_correct / count,
        review_coverage=review_count / count,
        incorrect_auto_placement_rate=catastrophic / count,
        out_of_distribution_rate=ood_count / count,
        catastrophic_auto_placement_count=catastrophic,
        confusion_counts=tuple(
            (expected, predicted, confusion_count)
            for (expected, predicted), confusion_count in sorted(
                confusions.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
    )


def read_heldout_prediction_csv(path: Path) -> list[HeldoutPrediction]:
    """Read existing neural held-out predictions without filename evidence."""
    rows: list[HeldoutPrediction] = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            labels = tuple(
                label
                for label in (
                    str(raw.get("predicted_label", "")).strip(),
                    str(raw.get("second_label", "")).strip(),
                    str(raw.get("third_label", "")).strip(),
                )
                if label
            )
            known = _truthy(raw.get("known_distribution"))
            automatically_placed = _truthy(raw.get("automatically_placed")) if "automatically_placed" in raw else known
            rows.append(
                HeldoutPrediction(
                    file_sha256=str(raw.get("file_sha256", "")).strip(),
                    expected_label=str(raw.get("expected_label", "")).strip(),
                    predicted_labels=labels,
                    known_distribution=known,
                    automatically_placed=automatically_placed,
                )
            )
    return rows


def write_heldout_metrics(metrics: HeldoutMetrics, output_dir: Path) -> None:
    """Write transparent summary and confusion artifacts."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = asdict(metrics)
    payload["confusion_counts"] = [
        {"expected_label": expected, "predicted_label": predicted, "count": count}
        for expected, predicted, count in metrics.confusion_counts
    ]
    (destination / "heldout_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (destination / "heldout_confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["expected_label", "predicted_label", "count"])
        writer.writerows(metrics.confusion_counts)


def _parent_family(label: str) -> str:
    parts = str(label).split("/")
    if len(parts) <= 2:
        return parts[0] if parts else ""
    return "/".join(parts[:2])


def _truthy(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "on"}
