#!/usr/bin/env python3
"""Diff two Aaron Sound Sorter manifests and write log-only regression reports.

This tool does not read audio and does not change sorter behavior.  It compares
old/new manifest rows by source filename/path and flags changed placements,
changed consensus status, and broad family regressions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

TOPS = ("Drums", "Instruments", "FX", "Textures", "_TO_REVIEW")


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row_key(row)
            if key:
                rows[key] = row
    return rows


def row_key(row: dict[str, str]) -> str:
    source = first(row, "source_path", "original_path", "input_path")
    if source:
        return Path(source).name
    placed = first(row, "placed_path", "new_relative_path", "folder_path")
    return Path(placed).name if placed else ""


def first(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name, "")
        if value:
            return value
    return ""


def top_of(path: str, row: dict[str, str]) -> str:
    top = first(row, "final_top", "top_level")
    if top:
        return top
    normalized = str(path).replace("\\", "/")
    return normalized.split("/", 1)[0] if normalized else ""


def folder_of(row: dict[str, str]) -> str:
    return first(row, "folder_path", "new_relative_path", "placed_path", "final_label")


def audit_role(row: dict[str, str]) -> str:
    text = row.get("parent_role_audit_json", "")
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except Exception:
        return ""
    return str(payload.get("detected_parent_role", ""))


def classify_change(key: str, old: dict[str, str] | None, new: dict[str, str] | None) -> str:
    if old is None:
        return "NEW_ROW"
    if new is None:
        return "MISSING_NEW_ROW"
    old_folder = folder_of(old)
    new_folder = folder_of(new)
    old_top = top_of(old_folder, old)
    new_top = top_of(new_folder, new)
    if old_folder == new_folder and first(old, "consensus_status") == first(new, "consensus_status"):
        return "UNCHANGED"
    if old_top == "FX" and new_top == "Instruments":
        return "LIKELY_GOOD_FX_TO_INSTRUMENTS"
    if old_top == "Instruments" and new_top == "FX":
        return "RISK_INSTRUMENTS_TO_FX"
    if old_top != new_top:
        return f"TOP_CHANGED_{old_top}_TO_{new_top}"
    if old_folder != new_folder:
        return "LEAF_CHANGED_SAME_TOP"
    return "STATUS_CHANGED"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    old_rows = read_manifest(args.old)
    new_rows = read_manifest(args.new)
    args.out.mkdir(parents=True, exist_ok=True)

    all_keys = sorted(set(old_rows) | set(new_rows))
    diff_rows: list[list[str]] = []
    changed_rows: list[list[str]] = []
    suspicious_rows: list[list[str]] = []
    summary: dict[str, int] = {}

    header = [
        "severity",
        "source_name",
        "old_top",
        "new_top",
        "old_folder",
        "new_folder",
        "old_status",
        "new_status",
        "old_role",
        "new_role",
        "old_reason",
        "new_reason",
    ]
    for key in all_keys:
        old = old_rows.get(key)
        new = new_rows.get(key)
        severity = classify_change(key, old, new)
        summary[severity] = summary.get(severity, 0) + 1
        old_folder = folder_of(old or {})
        new_folder = folder_of(new or {})
        row = [
            severity,
            key,
            top_of(old_folder, old or {}),
            top_of(new_folder, new or {}),
            old_folder,
            new_folder,
            first(old or {}, "consensus_status"),
            first(new or {}, "consensus_status"),
            audit_role(old or {}),
            audit_role(new or {}),
            first(old or {}, "decision_reason", "reason")[:400],
            first(new or {}, "decision_reason", "reason")[:400],
        ]
        diff_rows.append(row)
        if severity != "UNCHANGED":
            changed_rows.append(row)
        if severity.startswith("RISK") or severity.startswith("TOP_CHANGED") or severity == "MISSING_NEW_ROW":
            suspicious_rows.append(row)

    write_csv(args.out / "manifest_diff_all.csv", header, diff_rows)
    write_csv(args.out / "manifest_diff_changed.csv", header, changed_rows)
    write_csv(args.out / "manifest_diff_suspicious.csv", header, suspicious_rows)
    with (args.out / "manifest_diff_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"Old rows: {len(old_rows)}\nNew rows: {len(new_rows)}\n\n")
        for name, count in sorted(summary.items(), key=lambda item: (-item[1], item[0])):
            handle.write(f"{count:5d}  {name}\n")
    print(f"Wrote diff reports to: {args.out}")
    return 0


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
