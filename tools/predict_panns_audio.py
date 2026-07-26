#!/usr/bin/env python3
"""Predict broad AudioSet events with the optional pinned PANNs model."""

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

from aaron_sound_sorter.neural_audio.panns_provider import configured_panns_provider  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write PANNs events while degrading cleanly when the lane is absent."""
    args = build_parser().parse_args(argv)
    if args.top_k < 1:
        raise ValueError("top-k must be positive")
    root = args.project_root.expanduser().resolve()
    request_path = resolve_path(root, args.request_json)
    output_path = resolve_path(root, args.output_json)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    items = request.get("items", [])
    if not isinstance(items, list):
        raise ValueError("request items must be a list")
    predictions = []
    try:
        provider = configured_panns_provider(root)
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("each request item must be an object")
            prediction = provider.predict_file(Path(str(item["audio_path"])), top_k=args.top_k)
            predictions.append(
                {
                    "row_id": str(item["row_id"]),
                    "audio_sha256": prediction.audio_sha256,
                    "segment_count": prediction.segment_count,
                    "top_events": [asdict(event) for event in prediction.top_events],
                }
            )
        status = "predicted"
        message = f"Predicted PANNs events for {len(predictions)} audio waveform(s)."
        model_id = provider.model_id
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        status = "unavailable"
        message = str(exc)
        model_id = ""
        predictions = []
    payload = {
        "schema_version": 1,
        "status": status,
        "message": message,
        "lane_id": "panns",
        "model_id": model_id,
        "production_ownership_enabled": False,
        "source_name_policy": "decoded audio waveform only; paths are I/O metadata",
        "predictions": predictions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(message)
    return 0


def resolve_path(project_root: Path, path: Path) -> Path:
    """Resolve a project-relative or absolute path."""
    expanded = Path(path).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root / expanded).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
