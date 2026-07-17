"""Build low-trust ShapeVoter starter memory from trusted training trees.

The GUI correction path is the high-trust teacher lane.  This module gives the
command-line brain trainers a separate, lower-trust ShapeVoter starter brain so
curated folder-truth examples can teach broad shape behavior without
overwriting or diluting explicit human GUI corrections.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aaron_audio_intelligence.shape_memory_brain import (
    SHAPE_STARTER_MEMORY_BRAIN_NAME,
    build_empty_shape_memory_brain,
    shape_target_for_label,
    update_shape_memory_with_row,
)
from aaron_sound_sorter.features import extract_feature_rows, scan_training_tree_rows, select_rows
from aaron_sound_sorter.infrastructure.brain_repository import BrainRepository
from aaron_sound_sorter.io_utils import write_csv
from aaron_sound_sorter.training_labels import public_label, top_for_public_label

DEFAULT_STARTER_SHAPE_MEMORY_WEIGHT = 8
DEFAULT_STARTER_SHAPE_MEMORY_EXAMPLES_PER_LABEL = 3


@dataclass(frozen=True)
class ShapeMemoryStarterTrainingSummary:
    """Summary for one command-line starter shape-memory training run.

    Args:
        report_dir: Folder containing starter-memory reports.
        output_brain_path: Shape starter memory JSON path.
        manifest_path: CSV manifest with per-row decisions.
        summary_path: Human-readable summary text path.
        source_row_count: Raw rows scanned from the trusted tree.
        selected_row_count: Rows selected for starter memory measurement.
        readable_row_count: Selected rows with readable audio fingerprints.
        applied_row_count: Rows written into the starter memory payload.
        skipped_row_count: Rows skipped after selection.
        error_count: Rows that failed due to missing shape targets or read
            errors.
        shape_count: Number of learned broad shape buckets.
        applied: True when the output brain file was written.

    Side Effects:
        None.
    """

    report_dir: Path
    output_brain_path: Path
    manifest_path: Path
    summary_path: Path
    source_row_count: int
    selected_row_count: int
    readable_row_count: int
    applied_row_count: int
    skipped_row_count: int
    error_count: int
    shape_count: int
    applied: bool


def train_shape_memory_starter_from_tree(
    *,
    training_root: Path,
    base_brain_path: Path,
    output_brain_path: Path,
    report_dir: Path,
    allowed_top: str | set[str],
    max_examples_per_label: int = DEFAULT_STARTER_SHAPE_MEMORY_EXAMPLES_PER_LABEL,
    evidence_weight: int = DEFAULT_STARTER_SHAPE_MEMORY_WEIGHT,
    random_seed: int = 20260503,
    apply: bool = True,
) -> ShapeMemoryStarterTrainingSummary:
    """Train a low-trust starter ShapeVoter memory brain from folder truth.

    Args:
        training_root: Trusted training folder tree to scan.
        base_brain_path: Active full brain whose feature scaler metadata should
            be copied into the shape-memory brain.
        output_brain_path: Starter shape-memory brain JSON path to write.
        report_dir: Folder for CSV/text audit reports.
        allowed_top: Comma-separated or set of allowed top-level training
            families.
        max_examples_per_label: Maximum selected teacher examples per label.
        evidence_weight: Low-trust support weight for each selected example.
        random_seed: Deterministic selection seed.
        apply: When false, write reports only and do not save the brain.

    Returns:
        Training summary with report paths and counts.

    Side Effects:
        Reads the trusted training tree, writes reports, and optionally writes
        ``output_brain_path``. It never mutates GUI correction memory.

    Important Constraints:
        Folder paths are used only as supervised training labels. Runtime
        matching uses numeric audio fingerprints, not source filenames.
    """
    training_root = Path(training_root).expanduser().resolve()
    base_brain_path = Path(base_brain_path).expanduser().resolve()
    output_brain_path = Path(output_brain_path).expanduser().resolve()
    report_dir = Path(report_dir).expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    source_rows = scan_training_tree_rows(training_root, source_name="shape_memory_starter_tree")
    selected_rows, _eval_rows, skipped_selection_rows = select_rows(
        source_rows,
        parse_allowed_top(allowed_top),
        max_train_per_group=max(0, int(max_examples_per_label)),
        max_eval_per_label=0,
        max_eval_total=0,
        random_seed=int(random_seed),
    )
    feature_rows = extract_feature_rows(selected_rows, report_dir, "shape_memory_starter_train")
    base_brain = BrainRepository().load(base_brain_path)
    shape_brain = build_empty_shape_memory_brain(base_brain)
    shape_brain["brain_name"] = SHAPE_STARTER_MEMORY_BRAIN_NAME
    shape_brain["shape_memory_policy"] = "training_tree_teaches_shape_voter_low_trust_starter_memory"
    shape_brain["shape_memory_trust_level"] = "starter_low_trust"
    shape_brain["shape_memory_training_root"] = str(training_root)
    shape_brain["shape_memory_evidence_weight"] = int(evidence_weight)
    shape_brain["shape_memory_max_examples_per_label"] = int(max_examples_per_label)

    manifest_rows: list[dict[str, str]] = []
    applied_count = 0
    readable_count = 0
    error_count = 0
    skipped_count = len(skipped_selection_rows)
    for row in feature_rows:
        status = "SKIPPED_UNREADABLE"
        message = str(row.read_status or "")
        learned_shape = ""
        public = public_label(row.label)
        if row.read_status == "ok":
            readable_count += 1
            learned_shape = shape_target_for_label(public, row.structure)
            if learned_shape:
                learned_shape = update_shape_memory_with_row(
                    shape_brain,
                    row,
                    evidence_weight=int(evidence_weight),
                    memory_source="training_tree_starter",
                )
                applied_count += 1
                status = "APPLIED" if apply else "WOULD_APPLY"
                message = "trusted training tree example teaches low-trust starter shape memory"
            else:
                error_count += 1
                skipped_count += 1
                status = "SKIPPED_NO_SHAPE_TARGET"
                message = "approved label does not map to a ShapeVoter target"
        else:
            error_count += 1
            skipped_count += 1
        manifest_rows.append(
            {
                "status": status,
                "message": message,
                "source_path": str(row.path),
                "group_key": str(row.group_key),
                "public_label": public,
                "top": top_for_public_label(public),
                "structure": str(row.structure),
                "learned_shape": learned_shape,
                "duration_sec": f"{float(row.duration_sec):.6f}",
                "read_status": str(row.read_status),
                "evidence_weight": str(int(evidence_weight)),
                "memory_source": "training_tree_starter",
            }
        )

    if apply:
        BrainRepository().save(output_brain_path, shape_brain)
    manifest_path = report_dir / "shape_memory_starter_manifest.csv"
    summary_path = report_dir / "shape_memory_starter_summary.txt"
    write_csv(manifest_path, manifest_rows, shape_memory_starter_manifest_fields())
    write_shape_memory_starter_summary(
        summary_path,
        summary=ShapeMemoryStarterTrainingSummary(
            report_dir=report_dir,
            output_brain_path=output_brain_path,
            manifest_path=manifest_path,
            summary_path=summary_path,
            source_row_count=len(source_rows),
            selected_row_count=len(selected_rows),
            readable_row_count=readable_count,
            applied_row_count=applied_count,
            skipped_row_count=skipped_count,
            error_count=error_count,
            shape_count=len(shape_brain.get("labels", []) if isinstance(shape_brain.get("labels", []), list) else []),
            applied=bool(apply),
        ),
    )
    return ShapeMemoryStarterTrainingSummary(
        report_dir=report_dir,
        output_brain_path=output_brain_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
        source_row_count=len(source_rows),
        selected_row_count=len(selected_rows),
        readable_row_count=readable_count,
        applied_row_count=applied_count,
        skipped_row_count=skipped_count,
        error_count=error_count,
        shape_count=len(shape_brain.get("labels", []) if isinstance(shape_brain.get("labels", []), list) else []),
        applied=bool(apply),
    )


def parse_allowed_top(allowed_top: str | set[str]) -> set[str]:
    """Return normalized allowed top-level family names.

    Args:
        allowed_top: Comma-separated family names or an existing set.

    Returns:
        Non-empty set of family names.

    Side Effects:
        None.
    """
    if isinstance(allowed_top, set):
        return {str(value).strip() for value in allowed_top if str(value).strip()}
    return {part.strip() for part in str(allowed_top or "").split(",") if part.strip()}


def shape_memory_starter_manifest_fields() -> list[str]:
    """Return stable manifest columns for starter shape-memory training."""
    return [
        "status",
        "message",
        "source_path",
        "group_key",
        "public_label",
        "top",
        "structure",
        "learned_shape",
        "duration_sec",
        "read_status",
        "evidence_weight",
        "memory_source",
    ]


def write_shape_memory_starter_summary(path: Path, *, summary: ShapeMemoryStarterTrainingSummary) -> None:
    """Write a short text summary for starter shape-memory training.

    Args:
        path: Text file to write.
        summary: Counts and paths to report.

    Returns:
        None.

    Side Effects:
        Writes ``path``.
    """
    lines = [
        "Aaron Sound Sorter ShapeVoter Starter Memory Training",
        "",
        f"Applied: {summary.applied}",
        f"Output brain: {summary.output_brain_path}",
        f"Source rows scanned: {summary.source_row_count}",
        f"Selected rows: {summary.selected_row_count}",
        f"Readable rows: {summary.readable_row_count}",
        f"Applied rows: {summary.applied_row_count}",
        f"Skipped rows: {summary.skipped_row_count}",
        f"Errors: {summary.error_count}",
        f"Learned shape buckets: {summary.shape_count}",
        f"Manifest: {summary.manifest_path}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
