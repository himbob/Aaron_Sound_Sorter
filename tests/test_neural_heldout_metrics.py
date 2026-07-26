from __future__ import annotations

import json
from pathlib import Path

from aaron_sound_sorter.neural_audio.heldout_metrics import (
    HeldoutPrediction,
    evaluate_heldout_predictions,
    write_heldout_metrics,
)


def test_heldout_metrics_prioritize_incorrect_automatic_placements(tmp_path: Path) -> None:
    rows = [
        HeldoutPrediction("a" * 64, "Instruments/Keys", ("Instruments/Keys",), True, True),
        HeldoutPrediction(
            "b" * 64,
            "Instruments/Woodwinds",
            ("Instruments/Voice", "Instruments/Woodwinds"),
            True,
            True,
        ),
        HeldoutPrediction("c" * 64, "Drums/Snares", ("Drums/Kick Drums",), False, False),
    ]

    metrics = evaluate_heldout_predictions(rows)

    assert metrics.top1_accuracy == 1 / 3
    assert metrics.top3_accuracy == 2 / 3
    assert metrics.parent_family_accuracy == 1.0
    assert metrics.review_coverage == 1 / 3
    assert metrics.out_of_distribution_rate == 1 / 3
    assert metrics.incorrect_auto_placement_rate == 1 / 3
    assert metrics.catastrophic_auto_placement_count == 1

    write_heldout_metrics(metrics, tmp_path)
    payload = json.loads((tmp_path / "heldout_metrics.json").read_text(encoding="utf-8"))
    assert payload["catastrophic_auto_placement_count"] == 1
    assert (tmp_path / "heldout_confusion_matrix.csv").is_file()
