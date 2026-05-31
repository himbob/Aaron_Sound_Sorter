#!/usr/bin/env python3
"""Rollback symlinks added by one training apply manifest.

This is intentionally manifest-based instead of date-based. The bad apply run
knows exactly which symlinks it added, while creation time can also catch
taxonomy folders or unrelated manual cleanup.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import time
from collections.abc import Sequence
from pathlib import Path

DEFAULT_PROJECT_DIR = Path("/Volumes/T9/testbed/Aaron_Sound_Sorter")
DEFAULT_TRAINING_ROOT = DEFAULT_PROJECT_DIR / "training" / "locked_curated_v1"
DEFAULT_BAD_APPLY_MANIFEST = (
    DEFAULT_PROJECT_DIR
    / "_reports"
    / "training_reseed"
    / "apply_reviewed_candidates"
    / "run_20260520_065746"
    / "apply_reviewed_candidates_manifest.csv"
)


def safe_rel(path: Path, root: Path) -> Path:
    try:
        return path.absolute().relative_to(root.absolute())
    except Exception:
        return Path(str(path).strip("/").replace("/", "__"))


def unique_path(path: Path) -> Path:
    if not path.exists() and not path.is_symlink():
        return path
    for index in range(2, 100000):
        candidate = path.with_name(f"{path.stem}__{index}{path.suffix}")
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise RuntimeError(f"Could not create unique rollback path for {path}")


def read_manifest_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def link_matches_expected_target(link: Path, expected: Path) -> bool:
    try:
        actual = Path(os.readlink(link))
    except OSError:
        return False
    if actual == expected:
        return True
    try:
        return actual.resolve() == expected.resolve()
    except Exception:
        return False


def plan_rollback(manifest: Path, training_root: Path, rollback_root: Path) -> list[dict[str, str]]:
    planned: list[dict[str, str]] = []
    for row in read_manifest_rows(manifest):
        if row.get("action") != "add_reviewed_training_symlink" or row.get("applied") != "True":
            continue

        source_target = Path(row.get("source_target", ""))
        training_link = Path(row.get("training_link", ""))
        if not training_link:
            continue

        try:
            training_link.absolute().relative_to(training_root.absolute())
        except Exception:
            planned.append(
                {
                    "action": "skip_outside_training_root",
                    "training_link": str(training_link),
                    "rollback_link": "",
                    "source_target": str(source_target),
                    "current_target": "",
                    "label_rel": row.get("label_rel", ""),
                    "structure": row.get("structure", ""),
                }
            )
            continue

        rollback_link = rollback_root / "moved_from_training" / safe_rel(training_link, training_root)
        current_target = os.readlink(training_link) if training_link.is_symlink() else ""
        if not training_link.exists() and not training_link.is_symlink():
            action = "skip_missing"
        elif not training_link.is_symlink():
            action = "skip_not_symlink"
        elif not link_matches_expected_target(training_link, source_target):
            action = "skip_target_mismatch"
        else:
            action = "move_symlink"

        planned.append(
            {
                "action": action,
                "training_link": str(training_link),
                "rollback_link": str(rollback_link),
                "source_target": str(source_target),
                "current_target": current_target,
                "label_rel": row.get("label_rel", ""),
                "structure": row.get("structure", ""),
            }
        )
    return planned


def write_report(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "action",
        "training_link",
        "rollback_link",
        "source_target",
        "current_target",
        "label_rel",
        "structure",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def apply_rollback(rows: Sequence[dict[str, str]]) -> int:
    moved = 0
    for row in rows:
        if row["action"] != "move_symlink":
            continue
        source = Path(row["training_link"])
        target = unique_path(Path(row["rollback_link"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        row["rollback_link"] = str(target)
        moved += 1
    return moved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rollback symlinks from a mistaken training apply manifest")
    parser.add_argument("--manifest", default=str(DEFAULT_BAD_APPLY_MANIFEST))
    parser.add_argument("--training-root", default=str(DEFAULT_TRAINING_ROOT))
    parser.add_argument("--project-dir", default=str(DEFAULT_PROJECT_DIR))
    parser.add_argument(
        "--apply", action="store_true", help="Actually move matching symlinks into the rollback report folder."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = Path(args.manifest).expanduser().resolve()
    training_root = Path(args.training_root).expanduser().resolve()
    project_dir = Path(args.project_dir).expanduser().resolve()

    if not manifest.exists():
        raise SystemExit(f"Apply manifest not found: {manifest}")
    if not training_root.exists():
        raise SystemExit(f"Training root not found: {training_root}")

    stamp = time.strftime("run_%Y%m%d_%H%M%S")
    rollback_root = project_dir / "_reports" / "training_rollback" / stamp
    rows = plan_rollback(manifest, training_root, rollback_root)
    movable = sum(1 for row in rows if row["action"] == "move_symlink")
    skipped = len(rows) - movable

    moved = apply_rollback(rows) if args.apply else 0
    report = rollback_root / "rollback_manifest.csv"
    write_report(report, rows)
    readme = rollback_root / "README_ROLLBACK.txt"
    readme.write_text(
        "Aaron training apply rollback\n"
        f"Mode: {'APPLY' if args.apply else 'DRY RUN'}\n"
        f"Source apply manifest: {manifest}\n"
        f"Training root: {training_root}\n"
        f"Rows inspected: {len(rows)}\n"
        f"Movable symlinks: {movable}\n"
        f"Skipped rows: {skipped}\n"
        f"Moved symlinks: {moved}\n"
        f"Manifest: {report}\n",
        encoding="utf-8",
    )

    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"Rows inspected: {len(rows)}")
    print(f"Movable symlinks: {movable}")
    print(f"Skipped rows: {skipped}")
    print(f"Moved symlinks: {moved}")
    print(f"Report folder: {rollback_root}")
    print(f"Manifest: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
