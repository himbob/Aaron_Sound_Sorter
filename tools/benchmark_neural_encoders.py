#!/usr/bin/env python3
"""Benchmark pinned neural providers on a shared deterministic real-audio panel."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.neural_audio.providers import (  # noqa: E402
    HuggingFaceAudioEncoderProvider,
    HuggingFaceClapProvider,
)

CLAP_REVISION = "195c3a3e68faebb3e2088b9a79e79b43ddbda76b"
MERT_REVISION = "7d1bb4c6894b70c0f958a550c20dc861c83b25c3"


def benchmark_paths(manifest_path: Path, limit: int) -> list[Path]:
    """Select a stable shared panel by content hash, never by display name."""
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == "heldout_eval"]
    rows.sort(key=lambda row: row.get("file_sha256", ""))
    paths = [Path(row["materialized_path"]) for row in rows[:limit]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"benchmark materialized audio is missing: {missing[0]}")
    return paths


def directory_size(path: Path) -> int:
    """Return the total byte size of regular files below a directory."""
    return sum(candidate.stat().st_size for candidate in path.rglob("*") if candidate.is_file())


def benchmark_provider(provider, paths: list[Path], model_path: Path) -> dict[str, object]:
    """Measure cold first-use and warmed per-file inference."""
    durations: list[float] = []
    dimension = 0
    model_id = provider.model_id
    for path in paths:
        started = time.perf_counter()
        record = provider.embed_file(path)
        durations.append(time.perf_counter() - started)
        dimension = record.dimension
        model_id = record.model_id
    warm = durations[1:] if len(durations) > 1 else durations
    return {
        "provider_id": provider.provider_id,
        "model_id": model_id,
        "sample_count": len(paths),
        "cold_first_embed_seconds": durations[0],
        "warm_mean_embed_seconds": float(np.mean(warm)),
        "warm_p95_embed_seconds": float(np.percentile(warm, 95)),
        "total_seconds": float(sum(durations)),
        "embedding_dimension": dimension,
        "float32_vector_bytes": dimension * 4,
        "local_model_bytes": directory_size(model_path),
        "device": "cpu",
    }


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument(
        "--clap-model",
        type=Path,
        default=PROJECT_ROOT / "_models" / "laion_larger_clap_music_and_speech",
    )
    parser.add_argument(
        "--mert-model",
        type=Path,
        default=PROJECT_ROOT / "_models" / "mert_v1_95m_7d1bb4c",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run both encoders on exactly the same real audio."""
    args = build_parser().parse_args(argv)
    paths = benchmark_paths(args.experiment_manifest, args.sample_count)
    clap = HuggingFaceClapProvider(
        model_name_or_path=str(args.clap_model),
        source_model_id="laion/larger_clap_music_and_speech",
        model_revision=CLAP_REVISION,
        device="cpu",
        allow_network=False,
    )
    mert = HuggingFaceAudioEncoderProvider(
        provider_name="hf_mert",
        model_name_or_path=str(args.mert_model),
        source_model_id="m-a-p/MERT-v1-95M",
        model_revision=MERT_REVISION,
        layer=6,
        device="cpu",
        allow_network=False,
        allow_remote_code=True,
    )
    report = {
        "schema_version": 1,
        "selection_policy": "same held-out files selected by content hash",
        "results": [
            benchmark_provider(clap, paths, args.clap_model),
            benchmark_provider(mert, paths, args.mert_model),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
