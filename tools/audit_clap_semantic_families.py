#!/usr/bin/env python3
"""Audit audio-only CLAP semantic families for a GUI neural request."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.neural_audio.gui_training import configured_clap_trainer  # noqa: E402
from aaron_sound_sorter.neural_audio.semantic_panel import (  # noqa: E402
    flattened_semantic_prompts,
    predict_semantic_family,
)


def main() -> int:
    """Run the source-name-blind semantic panel and write one CSV row per audio."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    request = json.loads(args.request_json.expanduser().resolve().read_text(encoding="utf-8"))
    items = request.get("items", [])
    if not isinstance(items, list):
        raise ValueError("request items must be a list")
    trainer = configured_clap_trainer(project_root)
    prompts, prompt_families = flattened_semantic_prompts()
    text_embeddings = trainer.provider.embed_texts(prompts)

    output_path = args.output_csv.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_id",
        "file_sha256",
        "semantic_family",
        "second_family",
        "top_score",
        "second_score",
        "margin",
        "family_scores_json",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for position, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                raise ValueError("each request item must be an object")
            audio_path = Path(str(item["audio_path"])).expanduser().resolve()
            record = trainer.cache.get_or_compute(audio_path, trainer.provider)
            prediction = predict_semantic_family(record, text_embeddings, prompt_families)
            writer.writerow(
                {
                    "row_id": str(item["row_id"]),
                    "file_sha256": record.file_sha256,
                    "semantic_family": prediction.predicted_family,
                    "second_family": prediction.second_family,
                    "top_score": f"{prediction.top_score:.8f}",
                    "second_score": f"{prediction.second_score:.8f}",
                    "margin": f"{prediction.margin:.8f}",
                    "family_scores_json": json.dumps(prediction.family_scores, sort_keys=True),
                }
            )
            print(f"[{position}/{len(items)}] {item['row_id']}: {prediction.predicted_family}", flush=True)
    print(f"Report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
