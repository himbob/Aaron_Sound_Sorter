#!/usr/bin/env python3
"""Destructive cleanup for Aaron Sound Sorter training symlinks.

Deletes ONLY symlinks inside the training tree.

Removes:
  1. Broken symlinks.
  2. Duplicate symlinks pointing to the same real sample.
  3. Loop-like symlinks placed inside _ONE_SHOTS.
  4. Tiny audio symlinks at or below --tiny-sec.

Does NOT remove:
  - Real audio files.
  - Anything in /Volumes/T9/music_production/samples.
  - Non-symlink files.

Default mode is destructive. Use --dry-run to preview.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

try:
    import aifc
except Exception:
    aifc = None

try:
    import soundfile as sf
except Exception:
    sf = None


DEFAULT_PROJECT_DIR = Path("/Volumes/T9/testbed/Aaron_Sound_Sorter")
DEFAULT_TRAINING_ROOT = DEFAULT_PROJECT_DIR / "training" / "locked_curated_v1"
AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".au"}


@dataclass
class LinkRecord:
    link_path: Path
    rel_path: Path
    structure: str
    target_text: str
    target_path: Path | None
    target_key: str
    duration_sec: float | None
    desired_structure: str
    delete_reason: str = ""


def clean_text(text: str) -> str:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text))
    text = text.lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9#]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_audio_path(path: Path) -> bool:
    if path.name.startswith("._") or path.name.startswith("."):
        return False
    return path.suffix.lower() in AUDIO_EXTS


def read_duration_seconds(path: Path) -> float | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".wav":
            with wave.open(str(path), "rb") as handle:
                rate = handle.getframerate()
                frames = handle.getnframes()
                if rate > 0:
                    return float(frames) / float(rate)
        if suffix in {".aif", ".aiff"} and aifc is not None:
            with aifc.open(str(path), "rb") as handle:
                rate = handle.getframerate()
                frames = handle.getnframes()
                if rate > 0:
                    return float(frames) / float(rate)
        if sf is not None:
            info = sf.info(str(path))
            if int(info.samplerate) > 0:
                return float(info.frames) / float(info.samplerate)
    except Exception:
        return None
    return None


def structure_from_rel(rel_path: Path) -> str:
    for part in rel_path.parts:
        if part in {"_ONE_SHOTS", "_LOOPS", "_LONG_FX"}:
            return part
    return ""


def is_fx_rel(rel_path: Path) -> bool:
    parts = [p.lower() for p in rel_path.parts]
    return bool(parts and parts[0] == "fx")


def has_loop_text(text: str) -> bool:
    return bool(
        re.search(
            r"\b(loop|loops|looped|drumloop|beatloop|groove|grooves|riff|riffs|"
            r"phrase|phrases|pattern|patterns|sequence|sequences|arp|arpeggio|"
            r"full mix|fullmix)\b",
            text,
        )
        or re.search(r"\b\d{2,3}\s*bpm\b", text)
        or re.search(r"\b(bpm\s*\d{2,3})\b", text)
    )


def has_one_shot_text(text: str) -> bool:
    return bool(
        re.search(
            r"\b(one shot|oneshot|one-shot|single hit|hit|hits|stab|stabs|pluck|"
            r"plucks|kick|snare|clap|snap|hat|hihat|hi hat|tom|rim|crash|cymbal)\b",
            text,
        )
    )


def classify_desired_structure(
    rel_path: Path,
    target_text: str,
    duration_sec: float | None,
    tiny_sec: float,
    loop_min_sec: float,
    fx_long_min_sec: float,
) -> str:
    if duration_sec is not None and duration_sec <= tiny_sec:
        return "TINY"

    fx = is_fx_rel(rel_path)
    loop_text = has_loop_text(target_text)
    one_text = has_one_shot_text(target_text)

    if fx:
        if loop_text or (duration_sec is not None and duration_sec >= fx_long_min_sec and not one_text):
            return "_LONG_FX"
        return "_ONE_SHOTS"

    if loop_text:
        return "_LOOPS"

    if duration_sec is not None and duration_sec >= loop_min_sec and not one_text:
        return "_LOOPS"

    return "_ONE_SHOTS"


def link_target_key(link_path: Path) -> tuple[Path | None, str, str]:
    try:
        raw = os.readlink(link_path)
    except OSError:
        return None, "", ""
    try:
        resolved = link_path.resolve(strict=True)
        key = str(resolved)
    except Exception:
        resolved = None
        key = raw
    return resolved, key, raw


def collect_links(
    training_root: Path,
    tiny_sec: float,
    loop_min_sec: float,
    fx_long_min_sec: float,
) -> list[LinkRecord]:
    records: list[LinkRecord] = []
    for link_path in sorted(training_root.rglob("*")):
        if not link_path.is_symlink():
            continue
        rel_path = link_path.relative_to(training_root)
        structure = structure_from_rel(rel_path)

        target_path, target_key, raw_target = link_target_key(link_path)
        if target_path is None:
            duration = None
            target_text = clean_text(raw_target + " " + link_path.name)
            desired = "BROKEN"
        else:
            duration = read_duration_seconds(target_path) if is_audio_path(target_path) else None
            target_text = clean_text(str(target_path))
            desired = classify_desired_structure(
                rel_path=rel_path,
                target_text=target_text,
                duration_sec=duration,
                tiny_sec=tiny_sec,
                loop_min_sec=loop_min_sec,
                fx_long_min_sec=fx_long_min_sec,
            )

        records.append(
            LinkRecord(
                link_path=link_path,
                rel_path=rel_path,
                structure=structure,
                target_text=target_text,
                target_path=target_path,
                target_key=target_key,
                duration_sec=duration,
                desired_structure=desired,
            )
        )
    return records


def keeper_score(record: LinkRecord) -> tuple[int, int, int, str]:
    structure_match = 1 if record.structure == record.desired_structure else 0
    known_structure = 1 if record.structure in {"_ONE_SHOTS", "_LOOPS", "_LONG_FX"} else 0
    shorter = -len(record.rel_path.parts)
    return (structure_match, known_structure, shorter, str(record.rel_path))


def mark_deletions(records: list[LinkRecord]) -> list[LinkRecord]:
    to_delete: list[LinkRecord] = []

    for record in records:
        if record.desired_structure == "BROKEN":
            record.delete_reason = "broken_symlink"
            to_delete.append(record)

    for record in records:
        if record.delete_reason:
            continue
        if record.desired_structure == "TINY":
            record.delete_reason = "tiny_audio"
            to_delete.append(record)

    for record in records:
        if record.delete_reason:
            continue
        if record.structure == "_ONE_SHOTS" and record.desired_structure in {"_LOOPS", "_LONG_FX"}:
            record.delete_reason = f"wrong_structure_{record.structure}_should_be_{record.desired_structure}"
            to_delete.append(record)

    groups: dict[str, list[LinkRecord]] = {}
    for record in records:
        if record.delete_reason:
            continue
        if not record.target_key:
            continue
        groups.setdefault(record.target_key, []).append(record)

    for group in groups.values():
        if len(group) <= 1:
            continue
        keeper = sorted(group, key=keeper_score, reverse=True)[0]
        for record in group:
            if record is keeper:
                continue
            record.delete_reason = f"duplicate_target_keep={keeper.link_path}"
            to_delete.append(record)

    seen = set()
    unique: list[LinkRecord] = []
    for record in to_delete:
        key = str(record.link_path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def write_report(report_path: Path, rows: Sequence[LinkRecord]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "delete_reason",
                "link_path",
                "structure",
                "desired_structure",
                "duration_sec",
                "target_path",
                "target_key",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.delete_reason,
                    str(row.link_path),
                    row.structure,
                    row.desired_structure,
                    "" if row.duration_sec is None else f"{row.duration_sec:.6f}",
                    "" if row.target_path is None else str(row.target_path),
                    row.target_key,
                ]
            )


def delete_links(rows: Sequence[LinkRecord]) -> int:
    deleted = 0
    for row in rows:
        try:
            if row.link_path.is_symlink():
                row.link_path.unlink()
                deleted += 1
        except FileNotFoundError:
            continue
    return deleted


def remove_empty_dirs(training_root: Path) -> int:
    removed = 0
    for path in sorted(
        (p for p in training_root.rglob("*") if p.is_dir() and not p.is_symlink()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        try:
            if not any(path.iterdir()):
                path.rmdir()
                removed += 1
        except Exception:
            pass
    return removed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Destructively delete duplicate, broken, tiny, and loop-in-one-shot training symlinks."
    )
    parser.add_argument("--training-root", default=str(DEFAULT_TRAINING_ROOT))
    parser.add_argument("--project-dir", default=str(DEFAULT_PROJECT_DIR))
    parser.add_argument("--tiny-sec", type=float, default=0.06)
    parser.add_argument("--loop-min-sec", type=float, default=2.0)
    parser.add_argument("--fx-long-min-sec", type=float, default=1.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    training_root = Path(args.training_root).expanduser().resolve()
    project_dir = Path(args.project_dir).expanduser().resolve()

    if not training_root.exists():
        raise SystemExit(f"Training root not found: {training_root}")

    records = collect_links(
        training_root=training_root,
        tiny_sec=float(args.tiny_sec),
        loop_min_sec=float(args.loop_min_sec),
        fx_long_min_sec=float(args.fx_long_min_sec),
    )
    doomed = mark_deletions(records)

    report_dir = project_dir / "_reports" / "training_destructive_cleanup"
    report_path = report_dir / "deleted_training_symlinks_report.csv"
    write_report(report_path, doomed)

    counts: dict[str, int] = {}
    for row in doomed:
        reason = row.delete_reason.split("_keep=", 1)[0]
        counts[reason] = counts.get(reason, 0) + 1

    print(f"Training root: {training_root}")
    print(f"Symlinks scanned: {len(records)}")
    print(f"Symlinks selected for deletion: {len(doomed)}")
    for reason, count in sorted(counts.items()):
        print(f"  {reason}: {count}")
    print(f"Report: {report_path}")

    if args.dry_run:
        print("DRY RUN: no links deleted.")
        return 0

    deleted = delete_links(doomed)
    removed_dirs = remove_empty_dirs(training_root)
    print(f"Deleted symlinks: {deleted}")
    print(f"Removed empty folders: {removed_dirs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
