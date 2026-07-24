#!/usr/bin/env python3
"""Queue GUI corrections and rebuild the configured CLAP prototypes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.neural_audio.gui_training import (  # noqa: E402
    NeuralTrainingInbox,
    configured_clap_trainer,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--gui-import-manifest",
        type=Path,
        action="append",
        default=[],
        help="GUI training-import CSV to commit before rebuilding. Repeat as needed.",
    )
    parser.add_argument("--queue-only", action="store_true", help="Commit intake without rebuilding prototypes.")
    parser.add_argument("--output-json", type=Path, help="Optional durable JSON result path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run durable intake and the configured versioned prototype rebuild."""
    args = build_parser().parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    inbox = NeuralTrainingInbox(project_root)
    intake_payloads: list[dict[str, object]] = []
    for manifest in args.gui_import_manifest:
        manifest_path = manifest if manifest.is_absolute() else project_root / manifest
        summary = inbox.queue_import_manifest(manifest_path)
        intake_payloads.append(
            {
                **asdict(summary),
                "current_manifest_path": str(summary.current_manifest_path),
                "event_log_path": str(summary.event_log_path),
            }
        )
    if args.queue_only:
        payload = {"intake": intake_payloads, "status": "queued"}
        _emit_payload(payload, args.output_json)
        return 0

    build_summary = configured_clap_trainer(project_root).rebuild()
    payload = asdict(build_summary)
    for key in ("index_path", "pointer_path", "report_path"):
        payload[key] = str(payload[key]) if payload[key] is not None else ""
    _emit_payload({"intake": intake_payloads, "build": payload}, args.output_json)
    return 0 if build_summary.status in {"built", "unchanged"} else 1


def _emit_payload(payload: dict[str, object], output_path: Path | None) -> None:
    """Print a result and optionally persist it atomically enough for one job."""
    content = json.dumps(payload, indent=2, default=str) + "\n"
    if output_path is not None:
        resolved = output_path.expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
    print(content, end="")


if __name__ == "__main__":
    raise SystemExit(main())
