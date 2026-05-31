#!/usr/bin/env python3
"""Run Aaron Sound Sorter one ZIP member at a time and keep logs only.

The tool extracts one selected audio member into a temporary input folder, runs the
sorter, copies the manifest/summary/stdout into a report folder, then deletes the
temporary audio.  It is built for timeout-safe regression sweeps where the AI or
Aaron needs evidence without uploading sample audio back.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

csv.field_size_limit(sys.maxsize)
AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".au"}


@dataclass(frozen=True)
class SelectedMember:
    member: str
    category_key: str


def is_audio_member(name: str) -> bool:
    parts = [part for part in name.replace("\\", "/").split("/") if part]
    if not parts:
        return False
    if any(part == "__MACOSX" for part in parts):
        return False
    if any(part.startswith(".") or part.startswith("._") for part in parts):
        return False
    return Path(parts[-1]).suffix.lower() in AUDIO_EXTS


def category_for_member(member: str, depth_from_leaf: int) -> str:
    parts = [part for part in member.replace("\\", "/").split("/") if part]
    folders = parts[:-1]
    if not folders:
        return "ROOT"
    index = max(0, len(folders) - depth_from_leaf)
    return "/".join(folders[: index + 1])


def select_members(
    zip_path: Path, per_category: int, seed: int, max_total: int, depth_from_leaf: int, contains: str = ""
) -> list[SelectedMember]:
    rng = random.Random(seed)
    buckets: dict[str, list[str]] = {}
    with zipfile.ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            if info.is_dir() or not is_audio_member(info.filename):
                continue
            if contains and contains.lower() not in info.filename.lower():
                continue
            key = category_for_member(info.filename, depth_from_leaf)
            buckets.setdefault(key, []).append(info.filename)
    selected: list[SelectedMember] = []
    for key in sorted(buckets):
        names = sorted(buckets[key])
        rng.shuffle(names)
        for name in names[:per_category]:
            selected.append(SelectedMember(member=name, category_key=key))
    if max_total > 0:
        selected = selected[:max_total]
    return selected


def read_first_manifest_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def parse_role(row: dict[str, str]) -> str:
    try:
        payload = json.loads(row.get("parent_role_audit_json", "") or "{}")
    except Exception:
        return ""
    return str(payload.get("detected_parent_role", ""))


def suspicious_reason(row: dict[str, str], expected_top: str) -> str:
    folder = row.get("folder_path", "") or row.get("placed_path", "")
    final_top = row.get("final_top", "") or (folder.split("/", 1)[0] if folder else "")
    if expected_top and final_top and final_top != expected_top and final_top != "_TO_REVIEW":
        return f"expected_top_{expected_top}_got_{final_top}"
    if "Human and Voice FX" in folder and "vocal" not in Path(row.get("source_path", "")).name.lower():
        return "non_vocal_name_to_human_voice_fx"
    if "Alarm" in folder and not any(
        word in Path(row.get("source_path", "")).name.lower() for word in ("alarm", "siren", "alert")
    ):
        return "non_alarm_name_to_alarm"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--brain", default="stage4_folder_brain.json")
    parser.add_argument("--out-root", type=Path, default=Path("reports/single_file_log_panel"))
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--depth-from-leaf", type=int, default=1)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--max-total", type=int, default=36)
    parser.add_argument("--timeout-sec", type=float, default=75.0)
    parser.add_argument("--expected-top", default="")
    parser.add_argument("--contains", default="")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    zip_path = args.zip.expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = (project_root / args.out_root / f"run_{stamp}").resolve()
    cases_dir = report_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    selected = select_members(
        zip_path, args.per_category, args.seed, args.max_total, args.depth_from_leaf, args.contains
    )
    results: list[list[str]] = []
    suspicious: list[list[str]] = []
    header = [
        "index",
        "status",
        "member",
        "category_key",
        "final_top",
        "folder_path",
        "consensus_status",
        "detected_parent_role",
        "shape_vote",
        "shape_confidence",
        "suspicious_reason",
        "seconds",
        "case_report_dir",
    ]

    for index, item in enumerate(selected, start=1):
        safe = hashlib.sha1(item.member.encode("utf-8")).hexdigest()[:10]
        case_dir = cases_dir / f"{index:04d}_{safe}"
        case_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = case_dir / "sorter_stdout.log"
        manifest_copy = case_dir / "Aaron_Sorted_Sounds_manifest.csv"
        summary_copy = case_dir / "Aaron_Sorted_Sounds_summary.txt"
        status = "not_run"
        seconds = 0.0
        row: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix="aaron_single_member_") as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "output"
            input_dir.mkdir()
            member_name = Path(item.member.replace("\\", "/")).name
            extracted = input_dir / member_name
            with zipfile.ZipFile(zip_path, "r") as archive:
                with archive.open(item.member, "r") as src, extracted.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            cmd = [
                sys.executable,
                "Aaron_Sound_Sorter.py",
                "sort",
                str(input_dir),
                str(output_dir),
                "--brain",
                args.brain,
                "--no-zip",
            ]
            start = time.monotonic()
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=args.timeout_sec,
                )
                seconds = time.monotonic() - start
                stdout_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
                status = "pass" if completed.returncode == 0 else f"exit_{completed.returncode}"
            except subprocess.TimeoutExpired as error:
                seconds = time.monotonic() - start
                out = error.stdout or ""
                if isinstance(out, bytes):
                    out = out.decode("utf-8", errors="replace")
                stdout_path.write_text(out + "\nTIMEOUT\n", encoding="utf-8", errors="replace")
                status = "timeout"
            manifest = output_dir / "Aaron_Sorted_Sounds_manifest.csv"
            summary = output_dir / "Aaron_Sorted_Sounds_summary.txt"
            if manifest.exists():
                shutil.copy2(manifest, manifest_copy)
                row = read_first_manifest_row(manifest_copy)
            if summary.exists():
                shutil.copy2(summary, summary_copy)
        folder = row.get("folder_path", "") or row.get("placed_path", "")
        final_top = row.get("final_top", "") or (folder.split("/", 1)[0] if folder else "")
        reason = suspicious_reason(row, args.expected_top) if row else "no_manifest"
        result = [
            str(index),
            status,
            item.member,
            item.category_key,
            final_top,
            folder,
            row.get("consensus_status", ""),
            parse_role(row),
            row.get("shape_vote", ""),
            row.get("shape_confidence", ""),
            reason,
            f"{seconds:.3f}",
            str(case_dir),
        ]
        results.append(result)
        if reason or status != "pass":
            suspicious.append(result)
        print(f"{index:04d}/{len(selected):04d} {status:8s} {member_name} => {folder or 'NO_MANIFEST'}")

    write_csv(report_dir / "single_file_results.csv", header, results)
    write_csv(report_dir / "single_file_suspicious.csv", header, suspicious)
    write_text_summary(report_dir, zip_path, selected, results, suspicious, args)
    upload_zip = shutil.make_archive(str(report_dir / "single_file_log_panel_upload_back"), "zip", report_dir)
    print(f"Report folder: {report_dir}")
    print(f"Upload-back logs zip: {upload_zip}")
    return 0


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_text_summary(
    report_dir: Path,
    zip_path: Path,
    selected: list[SelectedMember],
    results: list[list[str]],
    suspicious: list[list[str]],
    args: argparse.Namespace,
) -> None:
    counts: dict[str, int] = {}
    for row in results:
        counts[row[4] or "NO_TOP"] = counts.get(row[4] or "NO_TOP", 0) + 1
    with (report_dir / "single_file_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"ZIP: {zip_path}\n")
        handle.write(f"Selected files: {len(selected)}\n")
        handle.write(f"Suspicious rows: {len(suspicious)}\n")
        handle.write(f"Expected top: {args.expected_top or '(none)'}\n")
        handle.write("\nTop folder counts:\n")
        for top, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            handle.write(f"{count:5d}  {top}\n")


if __name__ == "__main__":
    raise SystemExit(main())
