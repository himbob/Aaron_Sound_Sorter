from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from benchmark_neural_encoders import benchmark_paths  # noqa: E402


def test_benchmark_panel_is_selected_by_content_hash(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "file_sha256", "materialized_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "heldout_eval",
                "file_sha256": "f" * 64,
                "materialized_path": str(first),
            }
        )
        writer.writerow(
            {
                "split": "heldout_eval",
                "file_sha256": "0" * 64,
                "materialized_path": str(second),
            }
        )

    assert benchmark_paths(manifest, 1) == [second]
