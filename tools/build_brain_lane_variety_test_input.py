#!/usr/bin/env python3
"""Build a report-only brain-lane variety test input folder.

The builder creates an input folder with diagnostic expected-group names. It does
not train, move, delete, or rewrite source libraries. Directory files are linked
when possible; ZIP members are extracted into the report run folder.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".au"}

GROUP_ORDER = [
    "drums",
    "bass",
    "sax_reed",
    "guitar",
    "keys",
    "synth",
    "strings",
    "voice",
    "true_transition_fx",
    "melody_loop_inside_fx_pack",
    "multi_instrument_loop",
    "string_or_synth_loop",
    "texture",
]


def normalize_text(value: object) -> str:
    """Normalize a path/name for token checks."""
    text = str(value or "")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.replace("\\", "/").replace("_", " ").replace("-", " ").lower()
    return re.sub(r"[^a-z0-9#/. ]+", " ", text)


def any_token(text: str, *needles: str) -> bool:
    """Return True when any normalized fragment appears in text."""
    return any(needle in text for needle in needles)


def is_audio_file(path: Path) -> bool:
    """Return True for supported audio files while skipping macOS junk."""
    if path.name.startswith("._") or path.name.startswith("."):
        return False
    if "__MACOSX" in path.parts:
        return False
    return path.suffix.lower() in AUDIO_EXTS


def is_audio_member(name: str) -> bool:
    """Return True for supported audio members in ZIP files."""
    path = Path(name.replace("\\", "/"))
    if any(part == "__MACOSX" for part in path.parts):
        return False
    if any(part.startswith("._") or part.startswith(".") for part in path.parts):
        return False
    return path.suffix.lower() in AUDIO_EXTS


def classify_expected_group(text: str, *, from_fx_pack: bool) -> tuple[str, str]:
    """Classify a source path into a diagnostic group and source quality."""
    if from_fx_pack:
        split_group = split_fx_pack_group(text)
        if split_group != "unknown":
            quality = (
                "weak_expected_label" if split_group == "true_transition_fx" else "dirty_or_ambiguous_expected_label"
            )
            return split_group, quality

    group = "unknown"
    if any_token(text, "bass drum") or any_token(
        text,
        "kick",
        "snare",
        "clap",
        "hat",
        "hats",
        "drum",
        "drums",
        "drumloop",
        "percussion",
        "perc",
        "tom",
        "cymbal",
        "shaker",
        "tambourine",
        "bongo",
        "conga",
        "rim",
        "sidestick",
    ):
        group = "drums"
    elif any_token(
        text,
        "sax",
        "saxophone",
        "woodwind",
        "woodwinds",
        "flute",
        "clarinet",
        "bassoon",
        "reed",
        "brass",
        "horn",
        "trumpet",
        "trombone",
    ):
        group = "sax_reed"
    elif any_token(text, "vocal", "vocals", "vox", "voice", "choir", "spoken", "breath"):
        group = "voice"
    elif any_token(text, "bass", "808", "sub"):
        group = "bass"
    elif any_token(text, "guitar", "gtr"):
        group = "guitar"
    elif any_token(text, "piano", "rhodes", "keys", "keyboard", "organ", "wurlitzer", "electric piano"):
        group = "keys"
    elif any_token(text, "string", "strings", "violin", "viola", "cello"):
        group = "strings"
    elif any_token(text, "synth", "pad", "lead", "pluck", "arp"):
        group = "synth"
    elif any_token(
        text,
        "drone",
        "texture",
        "atmosphere",
        "ambience",
        "ambient",
        "field recording",
        "noise",
        "static",
        "wind",
        "ocean",
    ):
        group = "texture"
    elif any_token(
        text,
        "riser",
        "downlifter",
        "uplifter",
        "whoosh",
        "sweep",
        "impact",
        "glitch",
        "stutter",
        "alarm",
        "siren",
        "reverse",
        "build",
    ):
        group = "true_transition_fx"

    if group == "unknown":
        return "unknown", "unknown_expected_label"
    if from_fx_pack and group != "true_transition_fx":
        return group, "dirty_or_ambiguous_expected_label"
    if group == "true_transition_fx":
        return group, "weak_expected_label"
    return group, "strong_expected_label"


def split_fx_pack_group(text: str) -> str:
    """Split ambiguous FX-pack members into better diagnostic groups."""
    instrument_count = sum(
        1
        for family_tokens in (
            ("sax", "woodwind", "brass", "horn", "flute", "clarinet"),
            ("guitar", "gtr"),
            ("piano", "keys", "rhodes", "organ"),
            ("synth", "pad", "lead", "pluck", "arp"),
            ("string", "strings", "violin", "cello"),
            ("bass", "808", "sub"),
            ("vocal", "voice", "vox", "choir"),
        )
        if any_token(text, *family_tokens)
    )
    loopish = any_token(
        text, "loop", "loops", "bpm", "melody", "melodic", "chord", "progression", "phrase", "full mix", "fullmix"
    )
    concrete_fx = any_token(
        text,
        "riser",
        "downlifter",
        "uplifter",
        "whoosh",
        "sweep",
        "impact",
        "glitch",
        "stutter",
        "alarm",
        "siren",
        "reverse",
        "build",
        "drop",
    )
    string_or_synth = any_token(text, "synth", "pad", "lead", "arp", "string", "strings", "violin", "viola", "cello")
    melody_loop = any_token(text, "melody", "melodic", "chord", "progression", "music loop", "musical loop")
    multi = instrument_count >= 2 or any_token(
        text, "multi", "full mix", "fullmix", "construction", "songstarter", "song starter"
    )

    if concrete_fx and not instrument_count and not melody_loop:
        return "true_transition_fx"
    if multi and loopish:
        return "multi_instrument_loop"
    if string_or_synth and loopish:
        return "string_or_synth_loop"
    if melody_loop and loopish:
        return "melody_loop_inside_fx_pack"
    if any_token(text, "fx", "sfx", "effect", "effects") and concrete_fx:
        return "true_transition_fx"
    return "unknown"


def group_folder_name(group: str, quality: str) -> str:
    """Build a source folder name that the lane-validation report can parse."""
    if quality == "strong_expected_label":
        prefix = "strong_expected"
    elif quality == "dirty_or_ambiguous_expected_label":
        prefix = "dirty_ambiguous_expected"
    elif quality == "weak_expected_label":
        prefix = "weak_expected"
    else:
        prefix = "unknown_expected"
    return f"{prefix}_{group}"


def target_name(index: int, source_name: str) -> str:
    """Return a stable, readable test-file name."""
    clean = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(source_name).name).strip(" ._")
    return f"{index:04d}_{clean or 'sample.wav'}"


def collect_from_directory(
    source_root: Path, output_root: Path, max_per_group: int, counts: Counter[str], rows: list[list[str]]
) -> None:
    """Add matching files from a directory tree using symlinks when possible."""
    if not source_root.exists() or not source_root.is_dir():
        return
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or not is_audio_file(path):
            continue
        text = normalize_text(path)
        group, quality = classify_expected_group(text, from_fx_pack=False)
        if group not in GROUP_ORDER or counts[group] >= max_per_group:
            continue
        counts[group] += 1
        folder = output_root / group_folder_name(group, quality)
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / target_name(counts[group], path.name)
        try:
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            os.symlink(path, dest)
            mode = "symlink"
        except OSError:
            shutil.copy2(path, dest)
            mode = "copy"
        rows.append([group, quality, mode, str(path), str(dest)])


def collect_from_zip(
    zip_path: Path, output_root: Path, max_per_group: int, counts: Counter[str], rows: list[list[str]]
) -> None:
    """Extract matching ZIP members into expected-group folders."""
    if not zip_path.exists() or not zip_path.is_file():
        return
    from_fx_pack = "fx" in normalize_text(zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            if info.is_dir() or not is_audio_member(info.filename):
                continue
            member_text = normalize_text(f"{zip_path.name}/{info.filename}")
            group, quality = classify_expected_group(member_text, from_fx_pack=from_fx_pack)
            if group not in GROUP_ORDER or counts[group] >= max_per_group:
                continue
            counts[group] += 1
            folder = output_root / group_folder_name(group, quality)
            folder.mkdir(parents=True, exist_ok=True)
            dest = folder / target_name(counts[group], Path(info.filename).name)
            with archive.open(info, "r") as source, dest.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            rows.append([group, quality, "zip_extract", f"{zip_path}!{info.filename}", str(dest)])


def write_manifest(path: Path, rows: list[list[str]]) -> None:
    """Write source selection manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["expected_group", "expected_source_quality", "mode", "source", "test_path"])
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build a brain-lane variety test input folder.")
    parser.add_argument("--output", required=True, help="Output input folder to create.")
    parser.add_argument(
        "--source-root", action="append", default=[], help="Good sorted/training folder to sample. Can repeat."
    )
    parser.add_argument("--zip", action="append", default=[], help="Project ZIP to sample. Can repeat.")
    parser.add_argument("--max-per-group", type=int, default=60, help="Maximum files to include per expected group.")
    parser.add_argument(
        "--manifest", default="", help="Selection manifest path. Default: output/_source_selection_manifest.csv"
    )
    return parser.parse_args()


def main() -> int:
    """Build the test input folder."""
    args = parse_args()
    output_root = Path(args.output).expanduser().resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_roots = [Path(value).expanduser() for value in args.source_root]
    zip_paths = [Path(value).expanduser() for value in args.zip]
    counts: Counter[str] = Counter()
    rows: list[list[str]] = []

    for source_root in source_roots:
        collect_from_directory(source_root, output_root, args.max_per_group, counts, rows)
    for zip_path in zip_paths:
        collect_from_zip(zip_path, output_root, args.max_per_group, counts, rows)

    manifest_path = (
        Path(args.manifest).expanduser() if args.manifest else output_root / "_source_selection_manifest.csv"
    )
    write_manifest(manifest_path, rows)

    print("Built brain-lane variety input:", output_root)
    print("Selection manifest:", manifest_path)
    for group in GROUP_ORDER:
        print(f"  {group}: {counts[group]}")
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
