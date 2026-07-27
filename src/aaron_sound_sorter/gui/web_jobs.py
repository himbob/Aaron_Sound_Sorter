"""Background web-GUI jobs and JSON payload adapters."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from aaron_sound_sorter.gui.brain_family_files import active_gui_brain_file_names
from aaron_sound_sorter.gui.incremental_brain_update import (
    IncrementalBrainUpdater,
    corrections_from_import_manifest,
)
from aaron_sound_sorter.gui.models import PreviewRow, SortPreviewSession, TrainingImportSummary
from aaron_sound_sorter.gui.neural_explanations import neural_evidence_lines
from aaron_sound_sorter.neural_audio.runtime import run_configured_neural_rebuild


@dataclass
class PreviewJob:
    """Background preview job state."""

    job_id: str
    status: str = "queued"
    message: str = "Queued"
    session_id: str = ""
    error: str = ""
    completed_files: int = 0
    total_files: int = 0
    latest_file: str = ""
    partial_rows: list[dict[str, Any]] = field(default_factory=list)
    completed_rows: dict[int, PreviewRow] = field(default_factory=dict, repr=False)
    row_updates: list[tuple[int, dict[str, Any]]] = field(default_factory=list, repr=False)
    row_revision: int = 0
    keep_completed_on_stop: bool = False
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)


@dataclass
class ExportJob:
    """Background export job state for long copy, move, or symlink runs."""

    job_id: str
    status: str = "queued"
    message: str = "Queued"
    error: str = ""
    exported_count: int = 0
    corrected_count: int = 0
    sorted_root: str = ""
    approved_plan_path: str = ""
    corrections_path: str = ""
    errors: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class BrainTrainingJob:
    """Background job state for GUI correction-driven brain training."""

    job_id: str
    status: str = "queued"
    message: str = "Queued"
    corrected_count: int = 0
    staged_count: int = 0
    reused_existing_count: int = 0
    trainable_count: int = 0
    skipped_count: int = 0
    returncode: int | None = None
    training_root: str = ""
    report_dir: str = ""
    manifest_path: str = ""
    evidence_path: str = ""
    backup_dir: str = ""
    log_path: str = ""
    update_manifest_path: str = ""
    neural_status: str = "queued"
    neural_intake_path: str = ""
    neural_index_path: str = ""
    neural_report_path: str = ""
    neural_training_example_count: int = 0
    neural_label_count: int = 0
    neural_pending_conflict_count: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


def update_preview_job_progress(job: PreviewJob, completed_files: int, total_files: int, latest_file: str) -> None:
    """Update preview job progress for the browser status bar."""
    if job.cancel_event.is_set():
        return
    job.completed_files = int(completed_files)
    job.total_files = int(total_files)
    job.latest_file = str(latest_file)
    job.updated_at = time.time()
    if total_files <= 0:
        job.message = "Preparing audio files..."
        return
    if completed_files <= 0:
        job.message = f"Prepared {total_files} files. Starting final audio analysis..."
        return
    job.message = f"Finalized {completed_files} of {total_files}: {latest_file}"


def cancel_preview_job(job: PreviewJob) -> None:
    """Mark a live preview job as cancelled and request worker shutdown."""
    if job.status in {"done", "error", "stopped"}:
        return
    job.cancel_event.set()
    job.status = "cancelled"
    job.message = f"Preview cancelled after {job.completed_files} of {job.total_files or '?'} files."
    job.updated_at = time.time()


def request_stop_preview_job(job: PreviewJob) -> None:
    """Stop new analysis while preserving completed rows for review."""
    if job.status in {"done", "error", "cancelled", "stopped"}:
        return
    job.keep_completed_on_stop = True
    job.cancel_event.set()
    job.status = "stopping"
    job.message = f"Stopping after the {len(job.completed_rows)} completed sounds…"
    job.updated_at = time.time()


def create_export_job() -> ExportJob:
    """Create a background export job for browser polling."""
    return ExportJob(
        job_id=uuid.uuid4().hex,
        status="queued",
        message="Export queued...",
    )


def create_brain_training_job(import_summary: TrainingImportSummary) -> BrainTrainingJob:
    """Create a background brain-training job from an import summary."""
    trainable_count = len(corrections_from_import_manifest(import_summary.manifest_path))
    return BrainTrainingJob(
        job_id=uuid.uuid4().hex,
        status="running",
        message=(
            f"Imported {trainable_count} trainable correction(s) "
            f"({import_summary.staged_count} copied, {import_summary.reused_existing_count} already in training). "
            "Updating active brains incrementally..."
        ),
        corrected_count=import_summary.staged_count
        + import_summary.reused_existing_count
        + import_summary.skipped_count,
        staged_count=import_summary.staged_count,
        reused_existing_count=import_summary.reused_existing_count,
        trainable_count=trainable_count,
        skipped_count=import_summary.skipped_count,
        training_root=str(import_summary.training_root),
        report_dir=str(import_summary.report_dir),
        manifest_path=str(import_summary.manifest_path),
        evidence_path=str(import_summary.correction_evidence_path),
        backup_dir=str(import_summary.report_dir / "brain_backups_before_training"),
        log_path=str(import_summary.report_dir / "Aaron_GUI_Brain_Training.log"),
        neural_intake_path=str(import_summary.neural_intake_path),
        errors=list(import_summary.errors),
    )


def run_brain_training_job(
    job: BrainTrainingJob,
    project_root: Path,
    *,
    corrections_loader: Callable[[Path], list[Any]] = corrections_from_import_manifest,
    neural_rebuild: Callable[[Path, Path], dict[str, Any]] = run_configured_neural_rebuild,
    updater_class: type[Any] = IncrementalBrainUpdater,
) -> None:
    """Train neural prototypes first, then refresh transitional memories."""
    log_path = Path(job.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        job.status = "running"
        job.updated_at = time.time()
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write("Aaron GUI incremental brain update\n")
            log_handle.write(f"Project root: {project_root}\n")
            log_handle.write(f"Import manifest: {job.manifest_path}\n\n")
            corrections = corrections_loader(Path(job.manifest_path))
            job.trainable_count = len(corrections)
            if not corrections:
                job.returncode = 1
                job.status = "error"
                job.message = "No trainable corrections were available for incremental brain update."
                job.updated_at = time.time()
                log_handle.write(job.message + "\n")
                return
            log_handle.write("Rebuilding versioned CLAP prototypes from neural training intake...\n")
            log_handle.flush()
            try:
                neural_summary = neural_rebuild(project_root, Path(job.report_dir))
                job.neural_status = str(neural_summary.get("status", "error"))
                job.neural_index_path = str(neural_summary.get("index_path", ""))
                job.neural_report_path = str(neural_summary.get("report_path", ""))
                job.neural_training_example_count = int(neural_summary.get("training_example_count", 0))
                job.neural_label_count = int(neural_summary.get("label_count", 0))
                pending_conflicts = list(neural_summary.get("pending_training_conflicts", []))
                job.neural_pending_conflict_count = len(pending_conflicts)
                log_handle.write(
                    f"Neural status: {job.neural_status}; "
                    f"examples={job.neural_training_example_count}; "
                    f"labels={job.neural_label_count}\n"
                )
                log_handle.write(f"Neural index: {job.neural_index_path or '-'}\n")
                log_handle.write(f"Neural report: {job.neural_report_path or '-'}\n")
                for conflict in pending_conflicts:
                    log_handle.write(
                        "Neural relabel awaiting one more confirmation: "
                        f"locked={conflict.get('locked_label', '')}; "
                        f"proposed={conflict.get('proposed_label', '')}\n"
                    )
                for prediction in neural_summary.get("correction_predictions", []):
                    log_handle.write(
                        "Neural correction verification: "
                        f"approved={prediction.get('approved_label', '')}; "
                        f"before={prediction.get('predicted_before', '') or '-'}; "
                        f"after={prediction.get('predicted_after', '')}; "
                        f"known_distribution={prediction.get('known_distribution_after', False)}\n"
                    )
            except Exception as exc:
                job.neural_status = "queued_only"
                warning = f"Neural intake was saved, but automatic CLAP prototype rebuilding failed: {exc}"
                job.errors.append(warning)
                log_handle.write(f"WARNING: {warning}\n")
            log_handle.write("\nRefreshing transitional legacy memories...\n")
            log_handle.flush()
            legacy_summary = None
            try:
                backup_dir = backup_active_brain_family(project_root, Path(job.backup_dir))
                log_handle.write(f"Backed up active brain files to: {backup_dir}\n\n")
                legacy_summary = updater_class(project_root).apply(
                    corrections,
                    report_dir=Path(job.report_dir),
                    backup_dir=Path(job.backup_dir),
                )
                job.update_manifest_path = str(legacy_summary.manifest_path)
                job.errors.extend(legacy_summary.errors)
                for result in legacy_summary.updated_brains:
                    log_handle.write(
                        f"Updated {result.brain_path.name}: "
                        f"labels {result.label_count_before}->{result.label_count_after}; "
                        f"corrections={result.corrections_applied}; "
                        f"touched={', '.join(result.labels_touched)}\n"
                    )
                    for warning in result.warnings:
                        log_handle.write(f"WARNING {result.brain_path.name}: {warning}\n")
                if legacy_summary.errors:
                    log_handle.write("\nLegacy memory warnings:\n")
                    for error in legacy_summary.errors:
                        log_handle.write(f"- {error}\n")
            except Exception as exc:
                warning = f"CLAP training was independent, but transitional legacy memory refreshing failed: {exc}"
                job.errors.append(warning)
                log_handle.write(f"WARNING: {warning}\n")
        neural_built = job.neural_status in {"built", "unchanged"}
        legacy_built = legacy_summary is not None
        if not neural_built and not legacy_built:
            job.returncode = 1
            job.status = "error"
            job.message = "Both CLAP rebuilding and transitional memory refreshing failed."
            job.updated_at = time.time()
            return
        job.returncode = 0
        job.status = "done"
        neural_message = (
            f"CLAP ready with {job.neural_training_example_count} example(s)."
            if neural_built
            else "CLAP evidence was queued for a later rebuild."
        )
        if job.neural_pending_conflict_count:
            neural_message += (
                f" {job.neural_pending_conflict_count} locked-label relabel(s) need the same choice "
                "one more time before neural training changes them."
            )
        legacy_message = (
            f" Transitional memories refreshed from {legacy_summary.applied_correction_count} correction(s)."
            if legacy_summary is not None
            else " Transitional memories were not refreshed."
        )
        job.message = f"{neural_message}{legacy_message}"
        job.updated_at = time.time()
    except Exception as exc:
        job.status = "error"
        job.returncode = 1
        job.message = f"Incremental brain update failed before completion: {exc}"
        job.updated_at = time.time()


def training_job_to_payload(job: BrainTrainingJob) -> dict[str, Any]:
    """Convert one brain-training job to a JSON-safe payload."""
    return {
        "job_id": job.job_id,
        "status": job.status,
        "message": job.message,
        "corrected_count": job.corrected_count,
        "staged_count": job.staged_count,
        "reused_existing_count": job.reused_existing_count,
        "trainable_count": job.trainable_count,
        "skipped_count": job.skipped_count,
        "returncode": job.returncode,
        "training_root": job.training_root,
        "report_dir": job.report_dir,
        "manifest_path": job.manifest_path,
        "evidence_path": job.evidence_path,
        "backup_dir": job.backup_dir,
        "log_path": job.log_path,
        "update_manifest_path": job.update_manifest_path,
        "neural_status": job.neural_status,
        "neural_intake_path": job.neural_intake_path,
        "neural_index_path": job.neural_index_path,
        "neural_report_path": job.neural_report_path,
        "neural_training_example_count": job.neural_training_example_count,
        "neural_label_count": job.neural_label_count,
        "neural_pending_conflict_count": job.neural_pending_conflict_count,
        "errors": job.errors,
        "started_at": job.started_at,
        "updated_at": job.updated_at,
    }


def export_job_to_payload(job: ExportJob) -> dict[str, Any]:
    """Convert one export job to a JSON-safe payload."""
    return {
        "job_id": job.job_id,
        "status": job.status,
        "message": job.message,
        "error": job.error,
        "exported_count": job.exported_count,
        "corrected_count": job.corrected_count,
        "sorted_root": job.sorted_root,
        "approved_plan_path": job.approved_plan_path,
        "corrections_path": job.corrections_path,
        "errors": job.errors,
        "started_at": job.started_at,
        "updated_at": job.updated_at,
    }


def backup_active_brain_family(project_root: Path, backup_dir: Path) -> Path:
    """Copy active GUI brain JSON files before correction-driven brain updates.

    Args:
        project_root: Project root containing active brain files.
        backup_dir: Destination folder inside the GUI training report.

    Returns:
        Backup folder path.

    Side Effects:
        Creates ``backup_dir`` and copies any existing active brain JSON files.
    """
    root = Path(project_root).expanduser().resolve()
    backup_dir.mkdir(parents=True, exist_ok=True)
    for brain_name in active_gui_brain_file_names():
        source_path = root / brain_name
        if source_path.exists() and source_path.is_file():
            shutil.copy2(source_path, backup_dir / brain_name)
    return backup_dir


def reveal_path_in_finder(path: Path) -> Path:
    """Open a folder or reveal a file in the macOS Finder.

    Args:
        path: Existing path to open or reveal.

    Returns:
        Resolved path that Finder was asked to show.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        RuntimeError: If macOS ``open`` is not available.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Path not found: {resolved}")
    open_binary = Path("/usr/bin/open")
    if not open_binary.exists():
        raise RuntimeError("Finder reveal is available only on macOS with /usr/bin/open")
    command = [str(open_binary), str(resolved)] if resolved.is_dir() else [str(open_binary), "-R", str(resolved)]
    subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return resolved


def session_to_payload(session: SortPreviewSession) -> dict[str, Any]:
    """Convert a preview session to JSON-safe data."""
    return {
        "run_dir": str(session.run_dir),
        "input_path": str(session.input_path),
        "available_labels": session.available_labels,
        "rows": [row_to_payload(index, row) for index, row in enumerate(session.rows)],
    }


def add_preview_job_row(job: PreviewJob, row: PreviewRow) -> None:
    """Add or replace one completed row in a live preview job."""
    row_index = preview_row_payload_index(row)
    payload = row_to_payload(row_index, row)
    job.completed_rows[row_index] = row
    job.partial_rows = [existing for existing in job.partial_rows if int(existing.get("index", -1)) != row_index]
    job.partial_rows.append(payload)
    job.partial_rows.sort(key=lambda existing: int(existing.get("index", 0)))
    job.row_revision += 1
    job.row_updates.append((job.row_revision, payload))
    job.updated_at = time.time()


def preview_row_payload_index(row: PreviewRow) -> int:
    """Return the zero-based final row index encoded in a preview row id."""
    try:
        return max(0, int(row.row_id) - 1)
    except ValueError:
        return 0


def preview_job_to_payload(job: PreviewJob, *, after_revision: int = 0) -> dict[str, Any]:
    """Convert one live preview job to a JSON-safe browser payload."""
    if after_revision > 0:
        partial_rows = [payload for revision, payload in job.row_updates if revision > after_revision]
    else:
        partial_rows = sorted(job.partial_rows, key=lambda row: int(row.get("index", 0)))
    return {
        "job_id": job.job_id,
        "status": job.status,
        "message": job.message,
        "session_id": job.session_id,
        "error": job.error,
        "completed_files": job.completed_files,
        "total_files": job.total_files,
        "latest_file": job.latest_file,
        "cancel_requested": job.cancel_event.is_set(),
        "partial_rows": partial_rows,
        "row_revision": job.row_revision,
        "started_at": job.started_at,
        "updated_at": job.updated_at,
    }


def row_to_payload(index: int, row: PreviewRow) -> dict[str, Any]:
    """Convert one preview row to JSON-safe data."""
    return {
        "index": index,
        "row_id": row.row_id,
        "source_path": str(row.source_path),
        "display_name": row.display_name,
        "proposed_folder": row.proposed_folder,
        "approved_folder": row.approved_folder,
        "final_top": row.final_top,
        "consensus_status": row.consensus_status,
        "confidence": row.confidence,
        "duration_sec": row.duration_sec,
        "read_status": row.read_status,
        "decision_reason": row.decision_reason,
        "diagnostic_summary": row.diagnostic_summary,
        "candidate_folders": row.candidate_folders,
        "neural_decision_state": row.neural_decision_state,
        "neural_runtime_status": row.neural_runtime_status,
        "neural_runtime_message": row.neural_runtime_message,
        "neural_row_error": row.neural_row_error,
        "neural_folder": row.neural_folder,
        "neural_known_distribution": row.neural_known_distribution,
        "neural_ownership_ready": row.neural_ownership_ready,
        "neural_ownership_reason": row.neural_ownership_reason,
        "neural_label_example_count": row.neural_label_example_count,
        "neural_exact_training_match": row.neural_exact_training_match,
        "neural_similarity": row.neural_similarity,
        "neural_margin": row.neural_margin,
        "neural_radius_ratio": row.neural_radius_ratio,
        "neural_semantic_status": row.neural_semantic_status,
        "neural_semantic_family": row.neural_semantic_family,
        "neural_semantic_second_family": row.neural_semantic_second_family,
        "neural_semantic_score": row.neural_semantic_score,
        "neural_semantic_second_score": row.neural_semantic_second_score,
        "neural_semantic_margin": row.neural_semantic_margin,
        "neural_semantic_family_scores": row.neural_semantic_family_scores,
        "neural_prompt_status": row.neural_prompt_status,
        "neural_prompt_suggestions": row.neural_prompt_suggestions,
        "panns_status": row.panns_status,
        "panns_model_id": row.panns_model_id,
        "panns_events": row.panns_events,
        "panns_family_scores": row.panns_family_scores,
        "panns_support_score": row.panns_support_score,
        "panns_contradiction_score": row.panns_contradiction_score,
        "panns_supporting_events": row.panns_supporting_events,
        "panns_contradicting_events": row.panns_contradicting_events,
        "neural_explanation_lines": neural_evidence_lines(row),
        "is_corrected": row.is_corrected,
    }


def apply_overrides(session: SortPreviewSession, rows_payload: Any) -> None:
    """Apply approved-folder overrides from the browser."""
    if not isinstance(rows_payload, list):
        return
    for row_payload in rows_payload:
        if not isinstance(row_payload, dict):
            continue
        index = int(row_payload.get("index", -1))
        approved = str(row_payload.get("approved_folder", "")).strip()
        if 0 <= index < len(session.rows) and approved:
            session.rows[index].approved_folder = approved
