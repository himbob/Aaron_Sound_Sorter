"""Seed ShapeVoter memory from explicit audio/label teacher panels.

This tool is intentionally opt-in. It does not train from arbitrary tests or
filenames. A seed panel must explicitly name each audio fixture and approved
taxonomy label, then runtime ShapeVoter matching uses measured fingerprints.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aaron_audio_intelligence.shape_memory_brain import (
    SHAPE_MEMORY_BRAIN_NAME,
    build_empty_shape_memory_brain,
    shape_target_for_label,
    update_shape_memory_with_row,
)
from aaron_sound_sorter.core import FP_SIZE, FeatureRow
from aaron_sound_sorter.features import make_fingerprint_safe
from aaron_sound_sorter.infrastructure.brain_repository import BrainRepository
from aaron_sound_sorter.io_utils import source_pack_key_for_path
from aaron_sound_sorter.training_labels import label_default_structure, top_for_public_label

DEFAULT_PANEL = "tests/shape_memory_seed_v1/shape_seed_panel.json"
DEFAULT_BASE_BRAIN = "stage4_folder_brain.json"


@dataclass(frozen=True)
class ShapeMemorySeedCase:
    """One explicit teacher example for ShapeVoter memory.

    Args:
        case_id: Stable panel case identifier.
        source_path: Audio path to measure.
        approved_label: Human-approved public taxonomy label.
        structure: Optional explicit structure, such as ``loop`` or
            ``one_shot``.
        expected_shape: Optional expected broad shape used for audit.
        evidence_weight: Human support weight to apply to this seed.

    Side Effects:
        None.
    """

    case_id: str
    source_path: Path
    approved_label: str
    structure: str
    expected_shape: str
    evidence_weight: int


@dataclass(frozen=True)
class ShapeMemorySeedRow:
    """Measured seed row and audit fields.

    Args:
        seed_case: Original seed panel case.
        feature_row: Measured feature row ready for shape-memory update.
        learned_shape: Shape target derived from the approved label.
        status: ``would_apply``, ``applied``, or an error status.
        message: Human-readable audit message.

    Side Effects:
        None.
    """

    seed_case: ShapeMemorySeedCase
    feature_row: FeatureRow | None
    learned_shape: str
    status: str
    message: str


@dataclass(frozen=True)
class ShapeMemorySeedSummary:
    """Result of one seed-panel run.

    Args:
        report_dir: Folder containing seed reports.
        manifest_path: CSV report path.
        applied: True when the shape-memory brain was written.
        readable_count: Number of readable seed examples.
        error_count: Number of failed seed examples.
        shape_memory_path: Target shape-memory brain path.
        backup_path: Backup path written for apply runs, when present.

    Side Effects:
        None.
    """

    report_dir: Path
    manifest_path: Path
    applied: bool
    readable_count: int
    error_count: int
    shape_memory_path: Path
    backup_path: Path | None


def load_seed_cases(panel_path: Path, project_root: Path, default_weight: int) -> list[ShapeMemorySeedCase]:
    """Load explicit seed cases from a JSON panel.

    Args:
        panel_path: JSON panel path.
        project_root: Project root for resolving relative audio paths.
        default_weight: Evidence weight used when a case omits one.

    Returns:
        Parsed seed cases.

    Raises:
        ValueError: If the panel is malformed.
        FileNotFoundError: If the panel file is missing.

    Side Effects:
        None.
    """
    payload = json.loads(panel_path.read_text(encoding="utf-8"))
    cases_payload = payload.get("cases")
    if not isinstance(cases_payload, list):
        raise ValueError("shape seed panel must contain a cases list")
    cases: list[ShapeMemorySeedCase] = []
    for index, raw_case in enumerate(cases_payload, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"case {index} is not an object")
        case_id = str(raw_case.get("id") or f"case_{index}").strip()
        source_value = str(raw_case.get("source_path") or "").strip()
        label = str(raw_case.get("approved_label") or "").strip()
        if not source_value or not label:
            raise ValueError(f"{case_id}: source_path and approved_label are required")
        source_path = resolve_project_path(project_root, source_value)
        structure = str(raw_case.get("structure") or label_default_structure(label) or "").strip()
        expected_shape = str(raw_case.get("expected_shape") or "").strip()
        evidence_weight = bounded_seed_weight(raw_case.get("evidence_weight", default_weight), default_weight)
        cases.append(
            ShapeMemorySeedCase(
                case_id=case_id,
                source_path=source_path,
                approved_label=label,
                structure=structure,
                expected_shape=expected_shape,
                evidence_weight=evidence_weight,
            )
        )
    return cases


def measure_seed_case(seed_case: ShapeMemorySeedCase) -> ShapeMemorySeedRow:
    """Measure one seed case into a ``FeatureRow``.

    Args:
        seed_case: Explicit audio/label teacher case.

    Returns:
        Measured seed row or an error row.

    Side Effects:
        Reads the seed audio file.
    """
    learned_shape = shape_target_for_label(seed_case.approved_label, seed_case.structure)
    if seed_case.expected_shape and learned_shape != seed_case.expected_shape:
        return ShapeMemorySeedRow(
            seed_case=seed_case,
            feature_row=None,
            learned_shape=learned_shape,
            status="ERROR_SHAPE_MISMATCH",
            message=f"expected {seed_case.expected_shape}, derived {learned_shape}",
        )
    if not seed_case.source_path.is_file():
        return ShapeMemorySeedRow(
            seed_case=seed_case,
            feature_row=None,
            learned_shape=learned_shape,
            status="ERROR_MISSING_AUDIO",
            message=str(seed_case.source_path),
        )
    fingerprint, duration_sec, read_status = make_fingerprint_safe(seed_case.source_path)
    if read_status != "ok" or len(fingerprint) != FP_SIZE:
        return ShapeMemorySeedRow(
            seed_case=seed_case,
            feature_row=None,
            learned_shape=learned_shape,
            status="ERROR_UNREADABLE_AUDIO",
            message=read_status,
        )
    feature_row = FeatureRow(
        path=str(seed_case.source_path),
        group_key=seed_case.approved_label,
        label=seed_case.approved_label,
        top=top_for_public_label(seed_case.approved_label),
        structure=seed_case.structure or label_default_structure(seed_case.approved_label),
        duration_sec=float(duration_sec),
        fingerprint=np.asarray(fingerprint, dtype=np.float32).astype(float).tolist(),
        read_status="ok",
        source_pack=source_pack_key_for_path(str(seed_case.source_path)),
        training_active_status="ACTIVE_SHAPE_MEMORY_SEED",
        label_source_group_count=1,
        label_clean_available=1,
    )
    return ShapeMemorySeedRow(
        seed_case=seed_case,
        feature_row=feature_row,
        learned_shape=learned_shape,
        status="would_apply",
        message="ok",
    )


def run_shape_memory_seed(
    *,
    project_root: Path,
    panel_path: Path,
    base_brain_path: Path,
    shape_memory_path: Path,
    report_dir: Path,
    apply: bool,
    default_weight: int,
) -> ShapeMemorySeedSummary:
    """Run one shape-memory seed panel.

    Args:
        project_root: Repository root.
        panel_path: Explicit seed panel JSON.
        base_brain_path: Full brain used for feature metadata.
        shape_memory_path: Shape-memory brain to create or update.
        report_dir: Report output directory.
        apply: True to write the shape-memory brain.
        default_weight: Evidence weight used by cases without one.

    Returns:
        Summary of the seed run.

    Raises:
        FileNotFoundError: If the base brain or panel is missing.
        ValueError: If the panel is malformed.

    Side Effects:
        Always writes a report manifest. When ``apply`` is true, backs up and
        rewrites the shape-memory brain.
    """
    repository = BrainRepository()
    project_root = project_root.expanduser().resolve()
    panel_path = resolve_project_path(project_root, str(panel_path))
    base_brain_path = resolve_project_path(project_root, str(base_brain_path))
    shape_memory_path = resolve_project_path(project_root, str(shape_memory_path))
    report_dir = report_dir.expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    base_brain = repository.load(base_brain_path)
    if shape_memory_path.exists():
        shape_memory = repository.load(shape_memory_path)
    else:
        shape_memory = build_empty_shape_memory_brain(base_brain)

    cases = load_seed_cases(panel_path, project_root, default_weight)
    measured_rows = [measure_seed_case(seed_case) for seed_case in cases]
    readable_rows = [row for row in measured_rows if row.feature_row is not None]
    backup_path: Path | None = None
    if apply and readable_rows:
        if shape_memory_path.exists():
            backup_dir = report_dir / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / shape_memory_path.name
            shutil.copy2(shape_memory_path, backup_path)
        for measured in readable_rows:
            update_shape_memory_with_row(
                shape_memory,
                measured.feature_row,
                evidence_weight=measured.seed_case.evidence_weight,
            )
        repository.save(shape_memory_path, shape_memory)
        measured_rows = [
            ShapeMemorySeedRow(
                seed_case=row.seed_case,
                feature_row=row.feature_row,
                learned_shape=row.learned_shape,
                status="applied" if row.feature_row is not None else row.status,
                message=row.message,
            )
            for row in measured_rows
        ]

    manifest_path = report_dir / "shape_memory_seed_manifest.csv"
    write_seed_manifest(manifest_path, measured_rows)
    write_seed_summary(
        report_dir / "shape_memory_seed_summary.txt",
        applied=apply,
        readable_count=len(readable_rows),
        error_count=len(measured_rows) - len(readable_rows),
        shape_memory_path=shape_memory_path,
        backup_path=backup_path,
    )
    return ShapeMemorySeedSummary(
        report_dir=report_dir,
        manifest_path=manifest_path,
        applied=apply,
        readable_count=len(readable_rows),
        error_count=len(measured_rows) - len(readable_rows),
        shape_memory_path=shape_memory_path,
        backup_path=backup_path,
    )


def write_seed_manifest(path: Path, rows: list[ShapeMemorySeedRow]) -> None:
    """Write one CSV manifest for a seed run.

    Args:
        path: Manifest destination.
        rows: Measured seed rows.

    Returns:
        None.

    Side Effects:
        Writes ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "status",
                "source_path",
                "approved_label",
                "structure",
                "learned_shape",
                "expected_shape",
                "duration_sec",
                "evidence_weight",
                "message",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row.seed_case.case_id,
                    "status": row.status,
                    "source_path": str(row.seed_case.source_path),
                    "approved_label": row.seed_case.approved_label,
                    "structure": row.seed_case.structure,
                    "learned_shape": row.learned_shape,
                    "expected_shape": row.seed_case.expected_shape,
                    "duration_sec": f"{row.feature_row.duration_sec:.6f}" if row.feature_row else "",
                    "evidence_weight": str(row.seed_case.evidence_weight),
                    "message": row.message,
                }
            )


def write_seed_summary(
    path: Path,
    *,
    applied: bool,
    readable_count: int,
    error_count: int,
    shape_memory_path: Path,
    backup_path: Path | None,
) -> None:
    """Write a plain-text seed summary.

    Args:
        path: Summary destination.
        applied: True when the brain was written.
        readable_count: Number of readable examples.
        error_count: Number of failed examples.
        shape_memory_path: Target shape-memory brain path.
        backup_path: Optional backup path.

    Returns:
        None.

    Side Effects:
        Writes ``path``.
    """
    lines = [
        f"Mode: {'APPLY' if applied else 'DRY_RUN'}",
        f"Readable seed examples: {readable_count}",
        f"Errors: {error_count}",
        f"Shape memory brain: {shape_memory_path}",
    ]
    if backup_path is not None:
        lines.append(f"Backup: {backup_path}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_project_path(project_root: Path, value: str) -> Path:
    """Resolve an absolute or project-relative path.

    Args:
        project_root: Repository root.
        value: Absolute path or project-relative path.

    Returns:
        Resolved path.

    Side Effects:
        None.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def bounded_seed_weight(value: object, default_weight: int) -> int:
    """Return a conservative positive seed evidence weight.

    Args:
        value: Raw JSON value.
        default_weight: Fallback value.

    Returns:
        Integer between 1 and 1200.

    Side Effects:
        None.
    """
    try:
        weight = int(float(value))
    except Exception:
        weight = int(default_weight)
    return max(1, min(1200, weight))


def default_report_dir(project_root: Path) -> Path:
    """Return a timestamped report folder for a seed run.

    Args:
        project_root: Repository root.

    Returns:
        Report directory path.

    Side Effects:
        None.
    """
    stamp = time.strftime("run_%Y%m%d_%H%M%S")
    return project_root / "_reports" / "shape_memory_seed" / stamp


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/path/to/Aaron_Sound_Sorter")
    parser.add_argument("--panel", default=DEFAULT_PANEL)
    parser.add_argument("--base-brain", default=DEFAULT_BASE_BRAIN)
    parser.add_argument("--shape-memory-brain", default=SHAPE_MEMORY_BRAIN_NAME)
    parser.add_argument("--report-dir", default="")
    parser.add_argument("--default-weight", type=int, default=96)
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the shape-memory seed command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve() if args.report_dir else default_report_dir(project_root)
    summary = run_shape_memory_seed(
        project_root=project_root,
        panel_path=Path(args.panel),
        base_brain_path=Path(args.base_brain),
        shape_memory_path=Path(args.shape_memory_brain),
        report_dir=report_dir,
        apply=bool(args.apply),
        default_weight=int(args.default_weight),
    )
    print(f"Mode: {'APPLY' if summary.applied else 'DRY_RUN'}")
    print(f"Readable seed examples: {summary.readable_count}")
    print(f"Errors: {summary.error_count}")
    print(f"Report: {summary.report_dir}")
    print(f"Manifest: {summary.manifest_path}")
    if summary.applied:
        print(f"Updated: {summary.shape_memory_path}")
    return 0 if summary.error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
