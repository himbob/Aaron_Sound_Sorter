#!/usr/bin/env python3
"""Build a source-blind supplemental vocal-confuser shadow set."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.neural_audio.dataset import AUDIO_SUFFIXES
from aaron_sound_sorter.neural_audio.hashing import sha256_file

SUPPLEMENT_GROUP_ROOTS = {
    "Altered Voice FX": ("FX/Human and Voice FX/Altered Voice",),
    "Breath and Mouth Sounds": (
        "FX/Human and Voice FX/Breath",
        "FX/Human and Voice FX/Mouth Sounds",
    ),
    "Mixed Instrument Loops": ("Instruments/Mixed Musical Loops",),
    "Sax and Woodwind": ("Instruments/Woodwinds/Saxophone",),
    "Spoken Voice": ("FX/Human and Voice FX/Spoken Voice",),
    "Synth Leads and Pads": (
        "Instruments/Synths/Synth Lead",
        "Instruments/Synths/Synth Pad",
        "Instruments/Synths/Pads",
    ),
    "Texture Beds": ("FX/Textures",),
}


def excluded_hashes(manifest_path: Path) -> set[str]:
    """Return every byte hash already used by the primary experiment."""
    with Path(manifest_path).open(encoding="utf-8", newline="") as handle:
        return {row["file_sha256"] for row in csv.DictReader(handle)}


def group_candidates(training_root: Path, relative_roots: Sequence[str]) -> list[tuple[str, Path]]:
    """Find candidate audio by explicit curated taxonomy folders."""
    candidates: list[tuple[str, Path]] = []
    for relative_root in relative_roots:
        root = Path(training_root) / relative_root
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
                candidates.append((sha256_file(path), path))
    return sorted(candidates, key=lambda pair: (pair[0], str(pair[1])))


def build_supplement(
    *,
    training_root: Path,
    exclude_manifest: Path,
    output_dir: Path,
    max_per_group: int,
) -> dict[str, object]:
    """Materialize a non-overlapping diagnostic set from legacy labels.

    Folder names are used only as explicit legacy taxonomy labels during set
    construction. The neural encoder receives audio waveforms only.
    """
    if max_per_group < 1:
        raise ValueError("max_per_group must be positive")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"supplement output must be absent or empty: {output_dir}")
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    excluded = excluded_hashes(exclude_manifest)
    selected_hashes: set[str] = set()
    manifest_rows: list[dict[str, str]] = []
    group_counts: dict[str, int] = {}

    for group, relative_roots in sorted(SUPPLEMENT_GROUP_ROOTS.items()):
        selected: list[tuple[str, Path]] = []
        for digest, source_path in group_candidates(training_root, relative_roots):
            if digest in excluded or digest in selected_hashes:
                continue
            selected.append((digest, source_path))
            selected_hashes.add(digest)
            if len(selected) >= max_per_group:
                break
        group_counts[group] = len(selected)
        for index, (digest, source_path) in enumerate(selected, start=1):
            suffix = source_path.suffix.lower()
            materialized_path = audio_dir / f"supplement_{len(manifest_rows) + 1:03d}{suffix}"
            os.symlink(source_path.resolve(), materialized_path)
            legacy_label = str(source_path.parent.relative_to(training_root))
            manifest_rows.append(
                {
                    "group": group,
                    "legacy_label": legacy_label,
                    "source_path": str(source_path.resolve()),
                    "materialized_path": str(materialized_path.absolute()),
                    "file_sha256": digest,
                    "group_position": str(index),
                }
            )

    manifest_path = output_dir / "supplement_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "legacy_label",
                "source_path",
                "materialized_path",
                "file_sha256",
                "group_position",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "schema_version": 1,
        "trainer_root": str(Path(training_root).resolve()),
        "excluded_primary_experiment_hash_count": len(excluded),
        "selected_file_count": len(manifest_rows),
        "selected_group_counts": group_counts,
        "formant_fx_status": "No explicit curated Formant FX taxonomy folder exists; altered-voice evidence is reported instead.",
        "warning": (
            "These labels come from the legacy curated corpus and are disputed. "
            "Results are diagnostic false-claim rates, not clean held-out accuracy."
        ),
    }
    (output_dir / "supplement_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-group", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build the supplement and print its summary."""
    args = parse_args(argv)
    summary = build_supplement(
        training_root=args.training_root,
        exclude_manifest=args.exclude_manifest,
        output_dir=args.output,
        max_per_group=args.max_per_group,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
