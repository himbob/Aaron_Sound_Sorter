#!/usr/bin/env python3
"""Post-sort audit for curated FX smoke expected broad buckets.

This tool is a test oracle, not production sorting logic.  It may inspect source
paths to validate known curated smoke files after sorting has already happened.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path


def _norm(text: str) -> str:
    return str(text or "").replace("\\", "/").lower()


def _expected_rule_for_source(source_path: str) -> tuple[str, str] | None:
    source = _norm(source_path)
    if "/loop/bass/" in source or "loop/bass" in source:
        return "loop_bass_to_bass_loops", "instruments/bass/bass loops"
    if "/loop/drums/" in source or "loop/drums" in source:
        return "loop_drums_to_drum_loops", "drums/drum loops"
    if "/loop/brass_woodwind/" in source or "brass_woodwind" in source:
        return "loop_brass_woodwind_to_instruments", "instruments/"
    return None


def _row_path(row: dict[str, str]) -> str:
    return _norm(row.get("folder_path") or row.get("new_relative_path") or row.get("output_path") or "")


def audit_manifest(manifest_path: Path) -> tuple[int, list[dict[str, str]]]:
    """Return ``(rows_checked, failures)`` for a sorter manifest CSV."""
    failures: list[dict[str, str]] = []
    total = 0
    with Path(manifest_path).open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            source = row.get("source_path") or row.get("original_path") or row.get("input_path") or ""
            expected = _expected_rule_for_source(source)
            if expected is None:
                continue
            total += 1
            folder_path = _row_path(row)
            final_top = _norm(row.get("final_top") or row.get("top_level") or "")
            if final_top.startswith("_to_review") or folder_path.startswith("_to_review"):
                failures.append(
                    {
                        "source_path": source,
                        "folder_path": folder_path,
                        "rule": "no_review_allowed_for_curated_fx_smoke",
                        "expected": expected[1],
                    }
                )
                continue
            rule, expected_fragment = expected
            if expected_fragment not in folder_path:
                failures.append(
                    {
                        "source_path": source,
                        "folder_path": folder_path,
                        "rule": rule,
                        "expected": expected_fragment,
                    }
                )
    return total, failures


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    total, failures = audit_manifest(args.manifest)
    print(f"Checked curated FX smoke rows: {total}")
    if failures:
        print(f"Failures: {len(failures)}")
        for failure in failures:
            print(f"- {failure['rule']}: {failure['source_path']} -> {failure['folder_path']}")
        return 1
    print("PASS: curated FX smoke expected buckets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
