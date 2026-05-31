#!/usr/bin/env python3
"""
aaron_metallic_percussion_reviewer.py

Safe review helper for Aaron Sound Sorter training data.

Purpose:
  Study the metallic percussion / cymbal area and create a listening review pack
  that separates likely cymbals from likely pitched rings, chimes, bells, and
  questionable metallic samples.  The default mode does NOT move training data.
  It creates symlinks plus CSV evidence so Aaron can listen and decide.

Why this exists:
  Very bright pitched rings can look cymbal-like to the prototype brain because
  both are short, metallic, high-frequency one-shots.  This helper exposes the
  physical split using pitch/harmonic stability, entropy/flatness, tail pitch,
  and high-band balance.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import through the project runner instead of individual component modules.
# Several component functions expect the synchronized public API globals that the
# runner provides for tests and tools.
import importlib.util  # noqa: E402

RUNNER = ROOT / "Aaron_Sound_Sorter.py"
SPEC = importlib.util.spec_from_file_location("aaron_sound_sorter_runner_for_metallic_review", RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load runner: {RUNNER}")
_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_mod)

make_fingerprint_safe = _mod.make_fingerprint_safe
FEATURE_NAMES = _mod.FEATURE_NAMES
fingerprint_physics_dict = _mod.fingerprint_physics_dict
fingerprint_physics_summary = _mod.fingerprint_physics_summary
fingerprint_physics_tags = _mod.fingerprint_physics_tags

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".au"}


def is_audio(path: Path) -> bool:
    if path.name.startswith("._") or path.name.startswith("."):
        return False
    if any(part == "__MACOSX" for part in path.parts):
        return False
    return path.suffix.lower() in AUDIO_EXTS


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._ -]+", "_", str(text)).strip()
    text = re.sub(r"\s+", "_", text)
    return text[:180] or "sample"


def iter_target_files(root: Path, scan_all: bool = False) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not is_audio(path):
            continue
        if scan_all:
            yield path
            continue
        text = "/".join(path.parts).lower()
        if any(
            word in text
            for word in [
                "cymbal",
                "cymbals",
                "hat",
                "hi hat",
                "hi-hat",
                "ride",
                "crash",
                "bell",
                "bells",
                "metallic",
                "chime",
                "chimes",
                "ring",
                "rings",
                "triangle",
                "cowbell",
                "gong",
                "tine",
            ]
        ):
            yield path


def get_float(d: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default) or default)
    except Exception:
        return default


def feature_value(fp, name: str, default: float = 0.0) -> float:
    try:
        return float(fp[FEATURE_NAMES.index(name)])
    except Exception:
        return default


def classify_metallic_sample(path: Path, fp, duration: float, read_status: str) -> tuple[str, str, dict[str, str]]:
    """Return suggested bucket, reason, physics dict.

    This is not the sorter.  It is a curation helper.  It intentionally uses
    broad physics buckets and sends uncertain cases to review.
    """
    phys = fingerprint_physics_dict(fp, duration) if read_status == "ok" else {}
    if read_status != "ok":
        return "_QUARANTINE/Unreadable", f"read_status={read_status}", phys

    events = get_float(phys, "primary_event_count_est")
    presence = get_float(phys, "presence_ratio_2000_8000hz")
    air = get_float(phys, "air_ratio_gt_8000hz")
    high = get_float(phys, "high_total_ratio_gt_2000hz")
    entropy = get_float(phys, "spectral_entropy_mean")
    flatness = get_float(phys, "spectral_flatness_mean")
    tail = get_float(phys, "tail_energy_ratio")
    get_float(phys, "temporal_centroid_ratio")
    attack = get_float(phys, "attack_rise_time_norm")
    pitch = get_float(phys, "pitch_confidence")
    zcr = get_float(phys, "zcr_mean")

    hnr = feature_value(fp, "harmonic_to_noise_ratio")
    body_pitch = feature_value(fp, "body_pitch_confidence")
    tail_pitch = feature_value(fp, "tail_pitch_confidence")
    harmonic_energy = feature_value(fp, "harmonic_energy_ratio")
    feature_value(fp, "f0_voiced_ratio")
    inharmonic = feature_value(fp, "inharmonicity_proxy")
    noise_ms = feature_value(fp, "noise_burst_duration_ms")

    source_text = "/".join(path.parts).lower()
    from_cymbal_folder = any(w in source_text for w in ["cymbal", "crash", "ride"])
    from_bell_folder = any(w in source_text for w in ["bell", "chime", "ring", "cowbell", "triangle"])

    one_shot = duration <= 2.5 and events <= 3
    very_short = duration <= 0.35
    pitched_tail = pitch >= 0.48 and (body_pitch >= 0.52 or tail_pitch >= 0.52 or hnr >= 1.0)
    strongly_pitched_tail = pitch >= 0.56 and (body_pitch >= 0.65 or tail_pitch >= 0.65 or hnr >= 1.25)
    cymbal_noise = high >= 0.70 and (entropy >= 0.50 or flatness >= 0.34 or air >= 0.18) and not strongly_pitched_tail
    high_ring = one_shot and high >= 0.65 and presence >= 0.40 and entropy <= 0.48 and strongly_pitched_tail
    bell_or_chime = one_shot and pitched_tail and entropy <= 0.55 and flatness <= 0.42 and tail_pitch >= 0.45
    cowbell_like = one_shot and pitched_tail and 0.08 <= presence <= 0.75 and entropy <= 0.60 and attack <= 0.18
    questionable = one_shot and high >= 0.55 and pitch >= 0.40 and entropy <= 0.60

    details = (
        f"dur={duration:.3f} events={events:.1f} high={high:.3f} presence={presence:.3f} air={air:.3f} "
        f"entropy={entropy:.3f} flatness={flatness:.3f} pitch={pitch:.3f} body_pitch={body_pitch:.3f} "
        f"tail_pitch={tail_pitch:.3f} hnr={hnr:.3f} harmonic_energy={harmonic_energy:.3f} "
        f"tail={tail:.3f} zcr={zcr:.3f} noise_ms={noise_ms:.1f} inharmonic={inharmonic:.3f}"
    )

    if high_ring:
        bucket = "Drums/Percussion/Bells and Metallic Percussion/High Rings and Chimes/One Shots"
        if from_cymbal_folder:
            bucket = "_QUARANTINE/Cymbals_Possible_High_Rings_or_Chimes"
        return bucket, "strong_high_ring_chime_physics; " + details, phys

    if bell_or_chime:
        bucket = "Drums/Percussion/Bells and Metallic Percussion/Bells Chimes and Rings/One Shots"
        if from_cymbal_folder:
            bucket = "_QUARANTINE/Cymbals_Possible_Bells_or_Rings"
        return bucket, "pitched_metallic_tail_more_bell_than_cymbal; " + details, phys

    if cowbell_like:
        bucket = "Drums/Percussion/Bells and Metallic Percussion/Cowbells and Blocks/One Shots"
        if from_cymbal_folder:
            bucket = "_QUARANTINE/Cymbals_Possible_Cowbell_or_Metallic_Percussion"
        return bucket, "cowbell_or_metallic_percussion_candidate; " + details, phys

    if cymbal_noise:
        bucket = "Drums/Cymbals/Noisy Cymbals/One Shots"
        if from_bell_folder:
            bucket = "_QUARANTINE/Bells_Possible_Cymbals"
        return bucket, "noisy_bright_cymbal_like; " + details, phys

    if very_short and high >= 0.50:
        return "_QUARANTINE/Too_Short_Bright_Metallic", "too_short_to_trust_for_training; " + details, phys

    if questionable:
        return "_REVIEW/Metallic_Ambiguous_Pitched_vs_Noisy", "ambiguous_pitched_metallic; " + details, phys

    return "_REVIEW/Metallic_Other_or_Weak_Evidence", "weak_or_other_metallic_evidence; " + details, phys


def link_or_copy(src: Path, dest: Path, mode: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if mode == "copy":
        shutil.copy2(src, dest)
    else:
        try:
            dest.symlink_to(src.resolve())
        except Exception:
            shutil.copy2(src, dest)


def main() -> int:
    ap = argparse.ArgumentParser(description="Create metallic/cymbal/bell curation review pack.")
    ap.add_argument("--root", required=True, help="Training root or sample folder to scan")
    ap.add_argument("--out", default="", help="Output review folder. Default: reports/metallic_review/run_TIMESTAMP")
    ap.add_argument(
        "--scan-all", action="store_true", help="Scan every audio file, not just metallic/cymbal/bell-ish paths"
    )
    ap.add_argument("--limit", type=int, default=0, help="Optional max files to analyze")
    ap.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Create symlinks by default to avoid wasting disk",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")
    if args.out:
        out = Path(args.out).expanduser().resolve()
    else:
        out = ROOT / "reports" / "metallic_percussion_review" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    files = list(iter_target_files(root, scan_all=bool(args.scan_all)))
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    for i, path in enumerate(files, 1):
        fp, duration, read_status = make_fingerprint_safe(path)
        bucket, reason, phys = classify_metallic_sample(path, fp, duration, read_status)
        rel = path.resolve()
        suffix = safe_name(path.name)
        dest = out / "review_tree" / bucket / f"{i:05d}__{suffix}"
        link_or_copy(rel, dest, args.mode)
        rows.append(
            {
                "index": str(i),
                "source_path": str(path),
                "review_bucket": bucket,
                "review_link_or_copy": str(dest),
                "read_status": read_status,
                "duration_sec": f"{float(duration):.6f}",
                "reason": reason,
                "physics_tags": fingerprint_physics_tags(fp, duration) if read_status == "ok" else "",
                "physics_summary": fingerprint_physics_summary(fp, duration) if read_status == "ok" else "",
                **{k: str(v) for k, v in phys.items()},
            }
        )
        if i % 50 == 0:
            print(f"Analyzed {i}/{len(files)} files...", flush=True)

    csv_path = out / "metallic_percussion_review_manifest.csv"
    fieldnames = [
        "index",
        "source_path",
        "review_bucket",
        "review_link_or_copy",
        "read_status",
        "duration_sec",
        "reason",
        "physics_tags",
        "physics_summary",
        "primary_event_count_est",
        "high_total_ratio_gt_2000hz",
        "presence_ratio_2000_8000hz",
        "air_ratio_gt_8000hz",
        "spectral_entropy_mean",
        "spectral_flatness_mean",
        "pitch_confidence",
        "tail_energy_ratio",
        "temporal_centroid_ratio",
        "attack_rise_time_norm",
        "zcr_mean",
    ]
    extra = sorted({k for row in rows for k in row} - set(fieldnames))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames + extra)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["review_bucket"]] = counts.get(row["review_bucket"], 0) + 1
    summary = out / "README_METALLIC_REVIEW.txt"
    with summary.open("w", encoding="utf-8") as f:
        f.write("Metallic Percussion Review Pack\n")
        f.write("==============================\n\n")
        f.write(f"Source root: {root}\n")
        f.write(f"Output root: {out}\n")
        f.write(f"Files analyzed: {len(rows)}\n")
        f.write(f"Mode: {args.mode}\n\n")
        f.write("This tool does not move or delete training data. It creates a review tree using measured physics.\n")
        f.write("Listen before using these folders as new training labels.\n\n")
        f.write("Suggested next training folders, after listening:\n")
        f.write("  Drums/Percussion/Bells and Metallic Percussion/High Rings and Chimes/One Shots\n")
        f.write("  Drums/Percussion/Bells and Metallic Percussion/Bells Chimes and Rings/One Shots\n")
        f.write("  Drums/Percussion/Bells and Metallic Percussion/Cowbells and Blocks/One Shots\n")
        f.write("  Drums/Cymbals/Noisy Cymbals/One Shots\n\n")
        f.write("Bucket counts:\n")
        for bucket, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            f.write(f"  {count:5d}  {bucket}\n")
        f.write(f"\nCSV manifest: {csv_path}\n")

    print("DONE")
    print(f"Review folder: {out}")
    print(f"Manifest: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
