#!/usr/bin/env python3
"""Predict read-only detailed taxonomy suggestions with the CLAP prompt brain."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.neural_audio.gui_training import configured_clap_trainer  # noqa: E402
from aaron_sound_sorter.neural_audio.prompt_brain import ClapPromptIndex  # noqa: E402

DEFAULT_POINTER = Path("config/runtime/neural_clap_prompt_index_path.txt")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Score request audio and write separate uncalibrated prompt evidence."""
    args = build_parser().parse_args(argv)
    if args.top_k < 1:
        raise ValueError("top-k must be positive")
    root = args.project_root.expanduser().resolve()
    request_path = resolve_path(root, args.request_json)
    output_path = resolve_path(root, args.output_json)
    pointer_path = resolve_path(root, args.pointer)
    index_path_value = pointer_path.read_text(encoding="utf-8").strip()
    index_path = resolve_path(root, Path(index_path_value))
    index = ClapPromptIndex.load(index_path)
    trainer = configured_clap_trainer(root)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    items = request.get("items", [])
    if not isinstance(items, list):
        raise ValueError("request items must be a list")
    predictions = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each request item must be an object")
        audio_path = Path(str(item["audio_path"])).expanduser().resolve()
        record = trainer.cache.get_or_compute(audio_path, trainer.provider)
        predictions.append(
            {
                "row_id": str(item["row_id"]),
                "file_sha256": record.file_sha256,
                "scores": [asdict(score) for score in index.predict(record, top_k=args.top_k)],
            }
        )
    payload = {
        "schema_version": 1,
        "status": "predicted",
        "lane_id": "clap_prompt",
        "model_id": index.metadata.model_id,
        "taxonomy_version": index.metadata.taxonomy_version,
        "production_ownership_enabled": False,
        "source_name_policy": "audio waveform and taxonomy prompts only",
        "predictions": predictions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Predicted prompt evidence for {len(predictions)} audio waveform(s).")
    print(f"Output: {output_path}")
    return 0


def resolve_path(project_root: Path, path: Path) -> Path:
    """Resolve a project-relative or absolute path."""
    expanded = Path(path).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root / expanded).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
