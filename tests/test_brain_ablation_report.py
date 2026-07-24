from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_brain_ablation_report import expected_labels_by_materialized_path  # noqa: E402


def test_ablation_uses_explicit_manifest_label_not_input_name(tmp_path: Path) -> None:
    audio = tmp_path / "opaque.wav"
    audio.write_bytes(b"audio")
    manifest = tmp_path / "experiment.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "materialized_path", "subgroup"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "review_preview",
                "materialized_path": str(audio),
                "subgroup": "Instruments/Keys/Piano/One Shots",
            }
        )

    expected = expected_labels_by_materialized_path(manifest)

    assert expected[str(audio.resolve())] == "Instruments/Keys/Piano/One Shots"
