"""Seed dedicated voter-memory brains from explicit trusted audio panels.

This tool is intentionally narrower than the normal training commands.  It
does not infer labels from filenames, source folders, or accepted-prefix test
rules.  A case is trainable only when the panel explicitly declares an
``approved_label`` or ``training_label``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aaron_sound_sorter.gui.incremental_brain_update import IncrementalBrainUpdater, IncrementalCorrection

DEFAULT_PANEL = Path("tests/acceptance/locked_smoke_v1/expected_results.json")
DEFAULT_SAMPLES_DIR = Path("tests/acceptance/locked_smoke_v1/samples")
DEFAULT_EVIDENCE_WEIGHT = 240


@dataclass(frozen=True)
class TrustedSeedCase:
    """One explicit trusted memory-training case.

    Args:
        case_id: Stable case identifier for reports.
        audio_path: Audio file to fingerprint.
        approved_label: Human-approved taxonomy label to train.
        source_description: Display-only origin such as panel filename.

    Side Effects:
        None.
    """

    case_id: str
    audio_path: Path
    approved_label: str
    source_description: str


@dataclass(frozen=True)
class TrustedSeedLoadResult:
    """Loaded trainable cases and skipped panel rows.

    Args:
        trainable_cases: Cases safe to feed into memory brains.
        skipped_rows: Human-readable skip records for reports.

    Side Effects:
        None.
    """

    trainable_cases: list[TrustedSeedCase]
    skipped_rows: list[dict[str, str]]


def load_trusted_seed_cases(
    panel_path: Path,
    *,
    project_root: Path,
    samples_dir: Path | None = None,
) -> TrustedSeedLoadResult:
    """Load explicit memory seed cases from JSON or CSV.

    Args:
        panel_path: JSON or CSV panel file.
        project_root: Repository root used to resolve relative paths.
        samples_dir: Optional sample folder used for JSON cases with
            ``filename`` but no ``audio_path``.

    Returns:
        Trainable cases plus skipped rows with reasons.

    Raises:
        ValueError: If the panel extension is unsupported or the JSON root is
            not a mapping/list.
    """

    panel_path = resolve_project_path(project_root, panel_path)
    if panel_path.suffix.lower() == ".json":
        records = load_json_panel_records(panel_path)
    elif panel_path.suffix.lower() == ".csv":
        records = load_csv_panel_records(panel_path)
    else:
        raise ValueError(f"Unsupported trusted memory panel type: {panel_path.suffix}")
    return cases_from_records(records, panel_path=panel_path, project_root=project_root, samples_dir=samples_dir)


def load_json_panel_records(panel_path: Path) -> list[dict[str, Any]]:
    """Return case dictionaries from a JSON panel."""
    payload = json.loads(panel_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        cases = payload.get("cases", [])
        if not isinstance(cases, list):
            raise ValueError("JSON panel field 'cases' must be a list")
        return [row for row in cases if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError("JSON panel must be an object with 'cases' or a list")


def load_csv_panel_records(panel_path: Path) -> list[dict[str, Any]]:
    """Return case dictionaries from a CSV panel."""
    with panel_path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def cases_from_records(
    records: Iterable[dict[str, Any]],
    *,
    panel_path: Path,
    project_root: Path,
    samples_dir: Path | None,
) -> TrustedSeedLoadResult:
    """Convert raw panel rows into explicit trainable cases."""
    resolved_samples_dir = resolve_samples_dir(project_root, panel_path, samples_dir)
    trainable: list[TrustedSeedCase] = []
    skipped: list[dict[str, str]] = []
    for index, record in enumerate(records, start=1):
        case_id = str(record.get("id") or record.get("case_id") or f"row_{index:04d}")
        approved_label = str(record.get("training_label") or record.get("approved_label") or "").strip()
        if not is_trainable_label(approved_label):
            skipped.append({"case_id": case_id, "status": "skipped", "reason": "missing_explicit_training_label"})
            continue
        audio_path = audio_path_from_record(record, project_root=project_root, samples_dir=resolved_samples_dir)
        if audio_path is None:
            skipped.append({"case_id": case_id, "status": "skipped", "reason": "missing_audio_path_or_filename"})
            continue
        trainable.append(
            TrustedSeedCase(
                case_id=case_id,
                audio_path=audio_path,
                approved_label=approved_label,
                source_description=str(record.get("filename") or record.get("audio_path") or audio_path.name),
            )
        )
    return TrustedSeedLoadResult(trainable_cases=trainable, skipped_rows=skipped)


def is_trainable_label(label: str) -> bool:
    """Return true for explicit non-review labels."""
    parts = [part for part in str(label).split("/") if part]
    return bool(parts and parts[0] in {"Drums", "Instruments", "FX"} and "_TO_REVIEW" not in parts[0])


def audio_path_from_record(record: dict[str, Any], *, project_root: Path, samples_dir: Path) -> Path | None:
    """Resolve one row's audio path without using the path as evidence."""
    raw_audio = str(record.get("audio_path") or record.get("path") or "").strip()
    if raw_audio:
        return resolve_project_path(project_root, Path(raw_audio))
    filename = str(record.get("filename") or "").strip()
    if filename:
        return samples_dir / filename
    return None


def resolve_samples_dir(project_root: Path, panel_path: Path, samples_dir: Path | None) -> Path:
    """Resolve a sample directory for filename-only cases."""
    if samples_dir is not None:
        return resolve_project_path(project_root, samples_dir)
    default_dir = project_root / DEFAULT_SAMPLES_DIR
    if panel_path.parent.joinpath("samples").is_dir():
        return panel_path.parent / "samples"
    return default_dir


def resolve_project_path(project_root: Path, raw_path: Path) -> Path:
    """Resolve ``raw_path`` against ``project_root`` when relative."""
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else project_root / path


def corrections_from_cases(cases: list[TrustedSeedCase], *, evidence_weight: int) -> list[IncrementalCorrection]:
    """Build incremental correction rows for trainable cases."""
    return [
        IncrementalCorrection(
            label=case.approved_label,
            audio_path=case.audio_path,
            source_path=case.audio_path,
            row_id=case.case_id,
            status="trusted_memory_seed",
            evidence_weight=evidence_weight,
        )
        for case in cases
    ]


def write_case_manifest(path: Path, load_result: TrustedSeedLoadResult) -> None:
    """Write the loaded/skipped case manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "status", "approved_label", "audio_path", "source_description", "reason"])
        for case in load_result.trainable_cases:
            status = "trainable" if case.audio_path.is_file() else "missing_audio"
            reason = "" if case.audio_path.is_file() else "audio file does not exist"
            writer.writerow(
                [
                    case.case_id,
                    status,
                    case.approved_label,
                    str(case.audio_path),
                    case.source_description,
                    reason,
                ]
            )
        for row in load_result.skipped_rows:
            writer.writerow([row.get("case_id", ""), row.get("status", ""), "", "", "", row.get("reason", "")])


def write_seed_summary(
    path: Path,
    *,
    panel_path: Path,
    load_result: TrustedSeedLoadResult,
    applied_count: int,
    dry_run: bool,
) -> None:
    """Write a compact text summary."""
    missing_audio_count = sum(1 for case in load_result.trainable_cases if not case.audio_path.is_file())
    lines = [
        "Trusted memory seed",
        f"Panel: {panel_path}",
        f"Dry run: {dry_run}",
        f"Trainable rows: {len(load_result.trainable_cases)}",
        f"Missing audio rows: {missing_audio_count}",
        f"Skipped rows: {len(load_result.skipped_rows)}",
        f"Corrections applied: {applied_count}",
        "",
        "Only rows with explicit approved_label/training_label are trainable.",
        "Filenames and folders are used only to locate files, never as sorting evidence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_trusted_memory_seed(args: argparse.Namespace) -> int:
    """Run the trusted memory seed workflow."""
    project_root = Path(args.project_root).expanduser().resolve()
    panel_path = resolve_project_path(project_root, Path(args.panel))
    stamp = time.strftime("run_%Y%m%d_%H%M%S")
    report_dir = project_root / "_reports" / "trusted_memory_seed" / stamp
    backup_dir = report_dir / "brain_backups_before_seed"
    load_result = load_trusted_seed_cases(
        panel_path,
        project_root=project_root,
        samples_dir=Path(args.samples_dir) if args.samples_dir else None,
    )
    write_case_manifest(report_dir / "trusted_memory_seed_cases.csv", load_result)
    existing_cases = [case for case in load_result.trainable_cases if case.audio_path.is_file()]
    applied_count = 0
    if existing_cases and not args.dry_run:
        updater = IncrementalBrainUpdater(project_root)
        summary = updater.apply(
            corrections_from_cases(existing_cases, evidence_weight=int(args.evidence_weight)),
            report_dir=report_dir,
            backup_dir=backup_dir,
        )
        applied_count = int(summary.applied_correction_count)
    write_seed_summary(
        report_dir / "trusted_memory_seed_summary.txt",
        panel_path=panel_path,
        load_result=load_result,
        applied_count=applied_count,
        dry_run=bool(args.dry_run),
    )
    print(f"Report folder: {report_dir}")
    print(f"Trainable rows: {len(load_result.trainable_cases)}")
    print(f"Applied rows: {applied_count}")
    print(f"Skipped rows: {len(load_result.skipped_rows)}")
    return 0 if applied_count or args.dry_run or not existing_cases else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/Volumes/T9/testbed/Aaron_Sound_Sorter")
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--samples-dir", default="")
    parser.add_argument("--evidence-weight", type=int, default=DEFAULT_EVIDENCE_WEIGHT)
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(func=run_trusted_memory_seed)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run the selected command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
