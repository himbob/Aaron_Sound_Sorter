#!/usr/bin/env python3
"""Predict a batch with the active source-name-blind neural prototype index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.neural_audio.gui_training import configured_clap_trainer  # noqa: E402
from aaron_sound_sorter.neural_audio.prototype_index import PrototypeIndex  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run neural inference for operational paths in one request manifest."""
    args = build_parser().parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    request = json.loads(args.request_json.expanduser().resolve().read_text(encoding="utf-8"))
    items = request.get("items", [])
    if not isinstance(items, list):
        raise ValueError("request items must be a list")
    trainer = configured_clap_trainer(project_root)
    pointer_value = trainer.pointer_path.read_text(encoding="utf-8").strip()
    index_path = Path(pointer_value).expanduser()
    if not index_path.is_absolute():
        index_path = project_root / index_path
    index = PrototypeIndex.load(index_path)

    predictions: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each request item must be an object")
        audio_path = Path(str(item["audio_path"])).expanduser().resolve()
        record = trainer.cache.get_or_compute(audio_path, trainer.provider)
        prediction = index.predict(record)
        predictions.append(
            {
                "row_id": str(item["row_id"]),
                "file_sha256": record.file_sha256,
                "predicted_label": prediction.predicted_label,
                "second_label": prediction.second_label,
                "top_similarity": prediction.top_similarity,
                "second_similarity": prediction.second_similarity,
                "margin": prediction.margin,
                "radius_ratio": prediction.radius_ratio,
                "known_distribution": prediction.known_distribution,
            }
        )

    output_payload = {
        "schema_version": 1,
        "status": "predicted",
        "message": f"Predicted {len(predictions)} audio waveform(s).",
        "index_path": str(index_path),
        "source_name_policy": "audio waveform only; path and row ID are operational metadata",
        "predictions": predictions,
    }
    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
