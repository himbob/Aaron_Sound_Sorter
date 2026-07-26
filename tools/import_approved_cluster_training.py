#!/usr/bin/env python3
"""Import or roll back explicit cluster safe-core training approval."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.neural_audio.cluster_approval import (  # noqa: E402
    import_cluster_approval,
    rollback_cluster_import,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--approval-manifest", type=Path)
    action.add_argument("--rollback-record", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run an explicit import or rollback without rebuilding prototypes."""
    args = build_parser().parse_args(argv)
    root = args.project_root.expanduser().resolve()
    if args.rollback_record is not None:
        rollback_cluster_import(args.rollback_record)
        print(f"Rolled back cluster import: {args.rollback_record}")
        return 0
    if args.output_dir is None:
        output_dir = root / "neural_artifacts/approved_cluster_manifests" / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = args.output_dir.expanduser().resolve()
    summary = import_cluster_approval(root, args.approval_manifest, output_dir)
    print(
        json.dumps(
            {
                "cluster_id": summary.cluster_id,
                "approved_category": summary.approved_category,
                "imported_count": summary.imported_count,
                "current_inbox": str(summary.intake_summary.current_manifest_path),
                "rollback_record": str(summary.rollback_record_path),
                "prototype_rebuild_performed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
