#!/usr/bin/env python3
"""Audit a sorter manifest and fail if any final placement is _TO_REVIEW.

This is a report/audit tool only.  It reads manifest outputs after sorting.  It
must never feed filenames or folder names back into sorting logic.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def newest_manifest(project_root: Path) -> Path:
    candidates = list(project_root.rglob("Aaron_Sorted_Sounds_manifest.csv"))
    candidates = [p for p in candidates if "_backup_before" not in str(p)]
    if not candidates:
        raise SystemExit("No Aaron_Sorted_Sounds_manifest.csv found. Pass --manifest.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> int:
    csv.field_size_limit(sys.maxsize)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--show", type=int, default=50)
    args = parser.parse_args()
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else newest_manifest(args.project_root.expanduser().resolve())
    )
    rows = []
    with manifest.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            final_top = str(row.get("final_top") or "")
            folder_path = str(row.get("folder_path") or row.get("new_relative_path") or "")
            if final_top == "_TO_REVIEW" or folder_path.startswith("_TO_REVIEW"):
                rows.append(row)
    print(f"Manifest: {manifest}")
    print(f"Review rows: {len(rows)}")
    if rows:
        print("\nFirst review rows:")
        for row in rows[: max(0, args.show)]:
            print("---")
            print(f"source_path: {row.get('source_path', '')}")
            print(f"folder_path: {row.get('folder_path', '')}")
            print(f"consensus_status: {row.get('consensus_status', '')}")
            print(f"decision_reason: {row.get('decision_reason', '')}")
            print(f"brain_vote_1: {row.get('brain_vote_1', '')}")
            print(f"physics_vote_1: {row.get('physics_vote_1', '')}")
            print(f"shape_vote: {row.get('shape_vote', '')}")
        return 1
    print("PASS: manifest has no _TO_REVIEW rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
