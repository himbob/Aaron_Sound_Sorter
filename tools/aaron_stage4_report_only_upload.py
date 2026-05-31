#!/usr/bin/env python3
"""
Aaron Stage 4 report-only upload packer.

Purpose:
  Replace unsafe giant upload packs with a small ZIP containing only text/report files.
  This does not modify training data and does not use training labels for decisions.

It intentionally excludes:
  - audio files
  - symlinks
  - brain/model dumps
  - previous upload ZIPs
  - copied preview/review audio trees
  - files over --max-file-mb

Optional:
  --quarantine-heavy-upload-zips moves huge generated upload ZIPs out of the run folder
  and into reports/stage4_heavy_uploads_quarantine/. It does not delete them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".au", ".mp3", ".m4a", ".aac"}
REPORT_EXTS = {".csv", ".tsv", ".json", ".jsonl", ".txt", ".md", ".log", ".out", ".err"}
BINARY_EXTS = {".npy", ".npz", ".pkl", ".pickle", ".joblib"}
SKIP_DIR_TOKENS = {
    "review_audio",
    "preview_audio",
    "real_preview_audio",
    "copied_audio",
    "sorted_audio",
    "audio_review",
    "symlink",
    "symlinks",
    "upload",
    "brain",
    "model",
    "models",
}


def project_root_from_arg(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "reports" / "stage4_big_real_reviews").exists():
        return cwd
    if cwd.name == "commands" and (cwd.parent / "reports" / "stage4_big_real_reviews").exists():
        return cwd.parent
    return Path(
        "/Users/aaron/Documents/Codex/2026-04-25/files-mentioned-by-the-user-create/Aaron_Sound_Sorter"
    ).resolve()


def resolve_run_dir(project_root: Path, run_arg: str) -> Path:
    base = project_root / "reports" / "stage4_big_real_reviews"
    if run_arg == "latest":
        runs = sorted((p for p in base.glob("run_*") if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            raise SystemExit(f"No run_* folders found under {base}")
        return runs[0]
    p = Path(run_arg).expanduser()
    if p.exists():
        return p.resolve()
    p2 = base / run_arg
    if p2.exists():
        return p2.resolve()
    raise SystemExit(f"Run folder not found: {run_arg}\nTried: {p2}")


def file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def should_skip_path(path: Path, run_dir: Path, max_bytes: int) -> tuple[bool, str]:
    rel = path.relative_to(run_dir)
    lower_parts = [p.lower() for p in rel.parts[:-1]]
    lower_name = path.name.lower()
    suffix = path.suffix.lower()
    size = file_size(path)
    if path.is_symlink():
        return True, "symlink"
    if suffix in AUDIO_EXTS:
        return True, "audio_file"
    if suffix in BINARY_EXTS:
        return True, "binary_or_model_dump"
    if size > max_bytes:
        return True, "over_max_file_size"
    if any(any(token in part for token in SKIP_DIR_TOKENS) for part in lower_parts):
        # Keep text files directly under a reports dir, but skip generated upload/brain/audio areas.
        return True, "skipped_generated_area"
    if lower_name.startswith("._") or "__macosx" in lower_name:
        return True, "macos_junk"
    if suffix not in REPORT_EXTS:
        return True, "extension_not_report"
    return False, ""


def copy_report_files(run_dir: Path, stage_dir: Path, max_file_mb: int) -> dict[str, int]:
    max_bytes = max_file_mb * 1024 * 1024
    manifest_path = stage_dir / "_REPORTS_ONLY_MANIFEST.csv"
    skipped_path = stage_dir / "_SKIPPED_FILES.csv"
    counts: dict[str, int] = {"included": 0, "skipped": 0}
    with (
        manifest_path.open("w", newline="", encoding="utf-8") as mf,
        skipped_path.open("w", newline="", encoding="utf-8") as sf,
    ):
        mw = csv.writer(mf)
        sw = csv.writer(sf)
        mw.writerow(["relative_path", "size_bytes"])
        sw.writerow(["relative_path", "size_bytes", "reason"])
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(run_dir)
            except ValueError:
                continue
            skip, reason = should_skip_path(path, run_dir, max_bytes)
            size = file_size(path)
            if skip:
                sw.writerow([str(rel), size, reason])
                counts["skipped"] += 1
                continue
            dest = stage_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest, follow_symlinks=False)
            mw.writerow([str(rel), size])
            counts["included"] += 1
    return counts


def write_inventory(run_dir: Path, stage_dir: Path, counts: dict[str, int]) -> None:
    ext_counts: dict[str, int] = {}
    symlinks = 0
    total_files = 0
    for path in run_dir.rglob("*"):
        if path.is_symlink():
            symlinks += 1
        if path.is_file():
            total_files += 1
            ext = path.suffix.lower() or "[no_ext]"
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
    disk_rows = []
    for child in sorted(run_dir.iterdir()):
        try:
            size = sum(file_size(p) for p in child.rglob("*") if p.is_file()) if child.is_dir() else file_size(child)
            disk_rows.append((child.name, size))
        except Exception:
            pass
    payload = {
        "run_dir": str(run_dir),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "included_files": counts.get("included", 0),
        "skipped_files": counts.get("skipped", 0),
        "original_regular_file_count": total_files,
        "original_symlink_count": symlinks,
        "extension_counts": dict(sorted(ext_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "top_level_size_bytes": dict(sorted(disk_rows, key=lambda kv: (-kv[1], kv[0]))),
    }
    (stage_dir / "_RUN_INVENTORY.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "Stage 4 report-only upload bundle",
        "",
        f"Run folder: {run_dir}",
        f"Generated: {payload['generated_at']}",
        "",
        f"Included report files: {counts.get('included', 0)}",
        f"Skipped files: {counts.get('skipped', 0)}",
        f"Original symlink count: {symlinks}",
        "",
        "This bundle excludes audio, symlinks, brain/model dumps, previous upload ZIPs, and oversized files.",
    ]
    (stage_dir / "README_REPORTS_ONLY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_zip(stage_dir: Path, out_zip: Path) -> None:
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(stage_dir.rglob("*")):
            if path.is_file() and not path.is_symlink():
                zf.write(path, path.relative_to(stage_dir.parent))


def quarantine_heavy_upload_zips(project_root: Path, run_dir: Path, min_mb: int) -> list[tuple[str, str, int]]:
    min_bytes = min_mb * 1024 * 1024
    moved: list[tuple[str, str, int]] = []
    quarantine_root = project_root / "reports" / "stage4_heavy_uploads_quarantine" / run_dir.name
    for upload_dir in run_dir.rglob("upload"):
        if not upload_dir.is_dir():
            continue
        for z in upload_dir.glob("*.zip"):
            size = file_size(z)
            if size < min_bytes:
                continue
            quarantine_root.mkdir(parents=True, exist_ok=True)
            dest = quarantine_root / z.name
            if dest.exists():
                dest = quarantine_root / f"{z.stem}_{int(time.time())}{z.suffix}"
            shutil.move(str(z), str(dest))
            moved.append((str(z), str(dest), size))
    return moved


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a report-only upload ZIP for a Stage 4 run.")
    ap.add_argument("run", nargs="?", default="latest", help="run folder path, run name, or latest")
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--max-file-mb", type=int, default=50)
    ap.add_argument("--quarantine-heavy-upload-zips", action="store_true")
    ap.add_argument("--heavy-upload-mb", type=int, default=200)
    args = ap.parse_args()

    project_root = project_root_from_arg(args.project_root)
    run_dir = resolve_run_dir(project_root, args.run)
    out_root = project_root / "reports" / "stage4_report_uploads"
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stage_dir = out_root / f"{run_dir.name}_REPORTS_ONLY_{stamp}"
    out_zip = out_root / f"{run_dir.name}_REPORTS_ONLY_{stamp}.zip"
    stage_dir.mkdir(parents=True, exist_ok=False)

    counts = copy_report_files(run_dir, stage_dir, max(1, args.max_file_mb))
    moved = []
    if args.quarantine_heavy_upload_zips:
        moved = quarantine_heavy_upload_zips(project_root, run_dir, max(1, args.heavy_upload_mb))
        qpath = stage_dir / "_HEAVY_UPLOAD_ZIPS_QUARANTINED.csv"
        with qpath.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["old_path", "new_path", "size_bytes"])
            for row in moved:
                w.writerow(row)
    write_inventory(run_dir, stage_dir, counts)
    make_zip(stage_dir, out_zip)

    print("Stage 4 report-only ZIP created:")
    print(f"  {out_zip}")
    print(f"Included files: {counts.get('included', 0)}")
    print(f"Skipped files:  {counts.get('skipped', 0)}")
    if moved:
        print(f"Quarantined heavy upload ZIPs: {len(moved)}")
    print(f"ZIP size bytes: {file_size(out_zip)}")
    if sys.platform == "darwin":
        os.system(f'open "{out_root}" >/dev/null 2>&1')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
