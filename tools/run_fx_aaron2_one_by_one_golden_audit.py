#!/usr/bin/env python3
"""Run every audio file in FX_Aaron2.zip one at a time and audit buckets.

This is a TEST HARNESS, not sorter logic.  It may use ZIP member paths as a
post-sort oracle because FX_Aaron2 is a curated regression pack.  Production
sorting code must remain source-name blind.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from tools.audit_fx_smoke_expected_buckets import audit_manifest, write_failures

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".au"}


def is_audio_member(name: str) -> bool:
    normalized = name.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return False
    if any(part == "__MACOSX" for part in parts):
        return False
    if any(part.startswith("._") or part.startswith(".") for part in parts):
        return False
    return Path(parts[-1]).suffix.lower() in AUDIO_EXTS


def resolve_fx_zip(project_root: Path, explicit: str) -> Path:
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.exists():
            return candidate.resolve()
    candidates = [
        project_root / "FX_Aaron2.zip",
        Path("/path/to/sample-library/FX_Aaron2.zip"),
        Path("/mnt/data/FX_Aaron2.zip"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise SystemExit("FX_Aaron2.zip not found. Pass --fx-zip or set FX_ZIP.")


def read_single_manifest_row(manifest: Path) -> dict[str, str]:
    with manifest.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one manifest row in {manifest}, got {len(rows)}")
    return rows[0]


def write_combined_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    preferred = [
        "zip_member",
        "case_status",
        "case_seconds",
        "source_path",
        "final_top",
        "folder_path",
        "consensus_status",
        "decision_reason",
    ]
    for field in preferred:
        seen.add(field)
        fields.append(field)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--fx-zip", default="")
    parser.add_argument("--brain", default="stage4_folder_brain.json")
    parser.add_argument("--out-root", type=Path, default=Path("_real_sort_tests/fx_one_by_one_golden"))
    parser.add_argument("--limit", type=int, default=0, help="0 means all audio members")
    parser.add_argument("--timeout-sec", type=float, default=0.0, help="0 means no subprocess timeout")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    fx_zip = resolve_fx_zip(project_root, args.fx_zip)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (project_root / args.out_root / f"run_{stamp}").resolve()
    cases_dir = run_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(fx_zip, "r") as zf:
        members = [info.filename for info in zf.infolist() if not info.is_dir() and is_audio_member(info.filename)]
        members.sort()
        if args.limit and args.limit > 0:
            members = members[: args.limit]

        combined_rows: list[dict[str, str]] = []
        for index, member in enumerate(members, start=1):
            start = time.monotonic()
            case_dir = cases_dir / f"{index:04d}_{Path(member).stem[:70].replace(' ', '_')}"
            case_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = case_dir / "sort_stdout.log"
            with tempfile.TemporaryDirectory(prefix="aaron_fx_one_by_one_") as tmp_name:
                tmp = Path(tmp_name)
                input_root = tmp / "input"
                output_root = tmp / "output"
                staged = input_root / member
                staged.parent.mkdir(parents=True, exist_ok=True)
                staged.write_bytes(zf.read(member))
                cmd = [
                    sys.executable,
                    str(project_root / "Aaron_Sound_Sorter.py"),
                    "sort",
                    str(input_root),
                    str(output_root),
                    "--brain",
                    str(project_root / args.brain),
                    "--no-zip",
                ]
                timeout = None if args.timeout_sec <= 0 else args.timeout_sec
                try:
                    completed = subprocess.run(
                        cmd,
                        cwd=project_root,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        timeout=timeout,
                    )
                    stdout_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
                    status = "pass" if completed.returncode == 0 else f"exit_{completed.returncode}"
                    manifest = output_root / "Aaron_Sorted_Sounds_manifest.csv"
                    if manifest.exists():
                        row = read_single_manifest_row(manifest)
                        shutil.copy2(manifest, case_dir / "Aaron_Sorted_Sounds_manifest.csv")
                    else:
                        row = {}
                except subprocess.TimeoutExpired as error:
                    out = error.stdout or ""
                    if isinstance(out, bytes):
                        out = out.decode("utf-8", errors="replace")
                    stdout_path.write_text(out + "\nTIMEOUT\n", encoding="utf-8", errors="replace")
                    status = "timeout"
                    row = {}

            elapsed = time.monotonic() - start
            row = dict(row)
            row["zip_member"] = member
            row["case_status"] = status
            row["case_seconds"] = f"{elapsed:.3f}"
            # Make the audit oracle independent of temp-dir paths.
            row["source_path"] = f"{fx_zip.name}/{member}"
            combined_rows.append(row)
            folder = row.get("folder_path", "NO_MANIFEST")
            print(f"{index:04d}/{len(members):04d} {status:8s} {member} => {folder}", flush=True)

    combined_manifest = run_dir / "FX_Aaron2_one_by_one_combined_manifest.csv"
    write_combined_manifest(combined_manifest, combined_rows)
    total, failures = audit_manifest(combined_manifest)
    failures_csv = run_dir / "FX_Aaron2_one_by_one_golden_failures.csv"
    write_failures(failures_csv, failures)

    summary_path = run_dir / "FX_Aaron2_one_by_one_golden_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"FX zip: {fx_zip}\n")
        handle.write(f"Rows checked: {total}\n")
        handle.write(f"Golden failures: {len(failures)}\n")
        handle.write(f"Combined manifest: {combined_manifest}\n")
        handle.write(f"Failures CSV: {failures_csv}\n")
    print(f"Run folder: {run_dir}")
    print(f"Combined manifest: {combined_manifest}")
    print(f"Golden failures: {len(failures)}")
    if failures:
        for failure in failures[:40]:
            print(f"{failure['row']}: {failure['rule']}: {failure['source_path']} => {failure['folder_path']}")
        return 1
    print("PASS: every FX_Aaron2 audio member passed the golden smoke audit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
