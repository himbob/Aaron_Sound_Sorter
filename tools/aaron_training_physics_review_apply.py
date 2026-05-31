#!/usr/bin/env python3
"""
aaron_training_physics_review_apply.py

Applies an Aaron training physics review plan.

Safety model:
  - It only acts on symlinks still present in the review tree.
  - If Aaron deleted a review symlink, that original training entry is left alone.
  - PROPOSED_MOVES: the current folder containing the review symlink is the destination.
  - PROPOSED_REMOVALS_TO_QUARANTINE: remaining symlinks are moved out of training into quarantine.
  - If the original training entry is a symlink, this moves the symlink itself, not the target audio.
  - Removals are quarantined by default, not permanently deleted.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from datetime import datetime
from pathlib import Path


def lexists(path: Path) -> bool:
    return os.path.lexists(str(path))


def target_of_symlink(link: Path) -> Path:
    raw = os.readlink(str(link))
    p = Path(raw)
    if not p.is_absolute():
        p = link.parent / p
    return p


def lexical_abs(path: Path) -> Path:
    """Return an absolute lexical path without resolving the final file.

    Critical detail for Aaron's training tree:
      training entries are often symlinks to /Volumes/T9 audio.
      We must move the training-tree symlink itself, not the target audio.

    Plain abspath is not enough on macOS temp paths because /var and /private/var
    can refer to the same parent directory.  So we realpath only the parent
    directories, then append the final basename lexically.  This normalizes
    /var -> /private/var without following the final symlink.
    """
    expanded = Path(os.path.expanduser(str(path)))
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    parent = Path(os.path.realpath(str(expanded.parent)))
    return parent / expanded.name


def is_lexically_under(path: Path, root: Path) -> bool:
    try:
        lexical_abs(path).relative_to(lexical_abs(root))
        return True
    except Exception:
        return False


def unique_log_path(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    if not p.exists():
        return p
    for i in range(2, 9999):
        alt = folder / f"{p.stem}_{i:03d}{p.suffix}"
        if not alt.exists():
            return alt
    return p


def move_training_entry(src: Path, dst: Path, dry_run: bool) -> tuple[str, str]:
    if not lexists(src):
        return "SKIP_MISSING_ORIGINAL", "original training path does not exist"
    if lexists(dst):
        return "SKIP_DEST_EXISTS", "destination already exists"
    if src == dst:
        return "SKIP_SAME_PATH", "source equals destination"
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    return ("DRY_RUN_WOULD_MOVE" if dry_run else "MOVED"), ""


def scan_symlinks(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.is_symlink():
                out.append(p)
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply a reviewed training physics symlink plan.")
    ap.add_argument("--training-root", type=Path, required=True)
    ap.add_argument("--plan-dir", type=Path, required=True)
    ap.add_argument("--apply", action="store_true", help="Actually move training entries. Without this, dry-run only.")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    training_root = args.training_root.expanduser().resolve()
    plan_dir = args.plan_dir.expanduser().resolve()
    if not (plan_dir / ".AARON_TRAINING_REVIEW_PLAN.json").exists():
        raise SystemExit(f"Refusing to apply. Not a recognized review plan folder: {plan_dir}")

    dry_run = not args.apply
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    apply_reports = plan_dir / "APPLY_REPORTS"
    log_path = unique_log_path(apply_reports, f"apply_log_{stamp}{'_DRY_RUN' if dry_run else ''}.csv")
    summary_path = log_path.with_suffix(".txt")
    quarantine_root = training_root / "_QUARANTINED_BY_PHYSICS_REVIEW" / stamp

    moves_root = plan_dir / "PROPOSED_MOVES"
    removals_root = plan_dir / "PROPOSED_REMOVALS_TO_QUARANTINE" / "TO_BE_REMOVED_REVIEW"

    rows: list[dict[str, str]] = []
    seen_originals = set()

    for link in scan_symlinks(moves_root):
        original = lexical_abs(target_of_symlink(link).expanduser())
        if not is_lexically_under(original, training_root):
            rows.append(
                {
                    "action": "move",
                    "status": "SKIP_ORIGINAL_NOT_UNDER_TRAINING_ROOT",
                    "review_symlink": str(link),
                    "original_training_path": str(original),
                    "destination_path": "",
                    "note": "review symlink target is outside training root",
                }
            )
            continue
        if str(original) in seen_originals:
            rows.append(
                {
                    "action": "move",
                    "status": "SKIP_DUPLICATE_ORIGINAL",
                    "review_symlink": str(link),
                    "original_training_path": str(original),
                    "destination_path": "",
                    "note": "another remaining review symlink already proposed this original",
                }
            )
            continue
        seen_originals.add(str(original))
        dest_parent_rel = link.parent.relative_to(moves_root)
        dest = training_root / dest_parent_rel / original.name
        status, note = move_training_entry(original, dest, dry_run)
        rows.append(
            {
                "action": "move",
                "status": status,
                "review_symlink": str(link),
                "original_training_path": str(original),
                "destination_path": str(dest),
                "note": note,
            }
        )

    for link in scan_symlinks(removals_root):
        original = lexical_abs(target_of_symlink(link).expanduser())
        if not is_lexically_under(original, training_root):
            rows.append(
                {
                    "action": "quarantine",
                    "status": "SKIP_ORIGINAL_NOT_UNDER_TRAINING_ROOT",
                    "review_symlink": str(link),
                    "original_training_path": str(original),
                    "destination_path": "",
                    "note": "review symlink target is outside training root",
                }
            )
            continue
        if str(original) in seen_originals:
            rows.append(
                {
                    "action": "quarantine",
                    "status": "SKIP_DUPLICATE_ORIGINAL",
                    "review_symlink": str(link),
                    "original_training_path": str(original),
                    "destination_path": "",
                    "note": "another remaining review symlink already proposed this original",
                }
            )
            continue
        seen_originals.add(str(original))
        rel = original.relative_to(training_root)
        dest = quarantine_root / rel
        status, note = move_training_entry(original, dest, dry_run)
        rows.append(
            {
                "action": "quarantine",
                "status": status,
                "review_symlink": str(link),
                "original_training_path": str(original),
                "destination_path": str(dest),
                "note": note,
            }
        )

    fields = ["action", "status", "review_symlink", "original_training_path", "destination_path", "note"]
    apply_reports.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    summary = [
        "Aaron Training Physics Review Apply",
        "===================================",
        "",
        f"Plan: {plan_dir}",
        f"Training root: {training_root}",
        f"Mode: {'APPLY' if args.apply else 'DRY RUN'}",
        f"Log: {log_path}",
        "",
        "Counts:",
    ]
    for k in sorted(counts):
        summary.append(f"  {k}: {counts[k]}")
    if not rows:
        summary.append("  No remaining review symlinks found.")
    summary += [
        "",
        "Reminder: deleted review symlinks produce no action.",
        "Quarantine removals are moved under:",
        f"  {quarantine_root}",
    ]
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    if args.open:
        os.system(f'open "{apply_reports}" >/dev/null 2>&1 || true')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
