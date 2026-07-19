"""Local browser GUI for previewing and approving sort results."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO, Callable
from urllib.parse import parse_qs, urlparse

from aaron_audio_intelligence.physics_memory_brain import PHYSICS_MEMORY_BRAIN_NAME
from aaron_audio_intelligence.shape_memory_brain import SHAPE_MEMORY_BRAIN_NAME, SHAPE_STARTER_MEMORY_BRAIN_NAME
from aaron_audio_intelligence.user_memory_brain import USER_MEMORY_BRAIN_NAME
from aaron_audio_intelligence.voter_memory_brain import VOTER_MEMORY_BRAIN_NAME
from aaron_sound_sorter.gui.incremental_brain_update import (
    IncrementalBrainUpdater,
    corrections_from_import_manifest,
)
from aaron_sound_sorter.gui.models import PreviewRow, SortPreviewSession, TrainingImportSummary
from aaron_sound_sorter.gui.preview_service import (
    SortPlanExporter,
    SortPreviewService,
    TrainingCorrectionImporter,
    gui_worker_count,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


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
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


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
    errors: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class WebGuiState:
    """Mutable state shared by local web-GUI request handlers."""

    project_root: Path
    preview_service: SortPreviewService = field(default_factory=SortPreviewService)
    exporter: SortPlanExporter = field(default_factory=SortPlanExporter)
    training_importer: TrainingCorrectionImporter = field(default_factory=TrainingCorrectionImporter)
    sessions: dict[str, SortPreviewSession] = field(default_factory=dict)
    jobs: dict[str, PreviewJob] = field(default_factory=dict)
    export_jobs: dict[str, ExportJob] = field(default_factory=dict)
    training_jobs: dict[str, BrainTrainingJob] = field(default_factory=dict)


class SorterWebApp:
    """Local HTTP app that exposes the sorter GUI in a browser.

    Args:
        project_root: Project root used for default paths and report output.
        host: Local interface to bind.
        port: Preferred port. The server tries following ports when busy.
        open_browser: Whether to open the default browser after binding.

    Side Effects:
        Starts a local HTTP server and may open the default browser.
    """

    def __init__(
        self,
        project_root: Path,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        open_browser: bool = True,
    ) -> None:
        self.project_root = project_root
        self.host = host
        self.port = port
        self.open_browser = open_browser
        self.state = WebGuiState(
            project_root=project_root,
            preview_service=SortPreviewService(project_root=project_root),
            exporter=SortPlanExporter(project_root=project_root),
            training_importer=TrainingCorrectionImporter(project_root=project_root),
        )
        self.server: ThreadingHTTPServer | None = None

    def run(self) -> int:
        """Run the local web GUI until interrupted."""
        handler = self._handler_class()
        self.server = bind_server(self.host, self.port, handler)
        url = f"http://{self.server.server_address[0]}:{self.server.server_address[1]}/"
        print(f"Aaron Sound Sorter GUI: {url}")
        print("Press Ctrl+C to stop the GUI server.")
        if self.open_browser:
            webbrowser.open(url)
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping Aaron Sound Sorter GUI.")
        finally:
            self.server.server_close()
        return 0

    def probe(self) -> int:
        """Validate that the web GUI can render its HTML shell."""
        brain_config = self.state.preview_service.load_brain_family_config()
        html = render_index_html(
            brain_config_path=str(brain_config.config_path),
            brain_summary=brain_config.display_summary(),
        )
        print(f"html_bytes={len(html.encode('utf-8'))}")
        print(f"has_preview_button={'Preview Sort' in html}")
        print(f"has_export_button={'Export Approved Sort' in html}")
        print(f"has_audio_player={'audioPlayer' in html}")
        print(f"has_category_chooser={'categoryModal' in html}")
        print(f"has_progress_bar={'progressBar' in html}")
        print(f"has_finder_button={'Open Sorted Folder' in html}")
        print(f"has_train_brains_button={'Train Brains From Corrections' in html}")
        required_markers = [
            "Preview Sort",
            "Export Approved Sort",
            "audioPlayer",
            "categoryModal",
            "progressBar",
            "Open Sorted Folder",
            "Train Brains From Corrections",
        ]
        return 0 if all(marker in html for marker in required_markers) else 2

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        state = self.state

        class Handler(SorterRequestHandler):
            gui_state = state

        return Handler


class SorterRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the local web GUI."""

    gui_state: WebGuiState

    def do_GET(self) -> None:
        """Serve the HTML shell and JSON status endpoints."""
        parsed = urlparse(self.path)
        if parsed.path == "/":
            brain_config = self.gui_state.preview_service.load_brain_family_config()
            self.send_html(
                render_index_html(
                    brain_config_path=str(brain_config.config_path),
                    brain_summary=brain_config.display_summary(),
                )
            )
            return
        if parsed.path.startswith("/api/audio/"):
            self.send_audio_from_path(parsed.path)
            return
        if parsed.path.startswith("/api/job-audio/"):
            self.send_job_audio_from_path(parsed.path)
            return
        if parsed.path.startswith("/api/job/"):
            self.send_job(parsed.path.rsplit("/", 1)[-1])
            return
        if parsed.path.startswith("/api/export-job/"):
            self.send_export_job(parsed.path.rsplit("/", 1)[-1])
            return
        if parsed.path.startswith("/api/session/"):
            self.send_session(parsed.path.rsplit("/", 1)[-1])
            return
        if parsed.path.startswith("/api/training-job/"):
            self.send_training_job(parsed.path.rsplit("/", 1)[-1])
            return
        if parsed.path == "/api/dialog":
            query = parse_qs(parsed.query)
            kind = query.get("kind", ["file"])[0]
            current_path = query.get("current", [""])[0]
            self.send_json({"path": choose_path_with_osascript(kind, current_path=current_path)})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        """Handle preview and export requests."""
        parsed = urlparse(self.path)
        payload = self.read_json()
        if parsed.path == "/api/preview":
            self.start_preview(payload)
            return
        if parsed.path == "/api/export":
            self.export_session(payload)
            return
        if parsed.path == "/api/train-corrections":
            self.train_from_corrections(payload)
            return
        if parsed.path == "/api/reveal":
            self.reveal_path(payload)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format_string: str, *args: Any) -> None:
        """Print compact request logs."""
        print(f"{self.address_string()} - {format_string % args}")

    def read_json(self) -> dict[str, Any]:
        """Read a JSON request body."""
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def send_html(self, html: str) -> None:
        """Send an HTML response."""
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        """Send a JSON response."""
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_audio_from_path(self, request_path: str) -> None:
        """Stream a preview-row audio file to the browser audio player."""
        parts = request_path.removeprefix("/api/audio/").split("/")
        if len(parts) != 2:
            self.send_error(HTTPStatus.BAD_REQUEST, "Audio URL must include session and row index")
            return
        session_id, row_index_text = parts
        session = self.gui_state.sessions.get(session_id)
        if session is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown session")
            return
        try:
            row_index = int(row_index_text)
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Audio row index must be an integer")
            return
        if row_index < 0 or row_index >= len(session.rows):
            self.send_error(HTTPStatus.NOT_FOUND, "Audio row not found")
            return
        source_audio_path = session.rows[row_index].source_path
        if not source_audio_path.exists() or not source_audio_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Audio file not found")
            return
        try:
            audio_path = browser_preview_audio_path(source_audio_path, session.run_dir)
        except Exception as exc:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Audio preview conversion failed: {exc}")
            return
        self.send_audio_file(audio_path)

    def send_job_audio_from_path(self, request_path: str) -> None:
        """Stream audio for a row that has completed inside a live preview job."""
        parts = request_path.removeprefix("/api/job-audio/").split("/")
        if len(parts) != 2:
            self.send_error(HTTPStatus.BAD_REQUEST, "Live audio URL must include job and row index")
            return
        job_id, row_index_text = parts
        job = self.gui_state.jobs.get(job_id)
        if job is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown preview job")
            return
        try:
            row_index = int(row_index_text)
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Live audio row index must be an integer")
            return
        row_payload = next((row for row in job.partial_rows if int(row.get("index", -1)) == row_index), None)
        if row_payload is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Live audio row not ready")
            return
        source_audio_path = Path(str(row_payload.get("source_path", "")))
        if not source_audio_path.exists() or not source_audio_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Audio file not found")
            return
        live_cache_dir = self.gui_state.project_root / "_reports" / "gui_preview" / "_live_audio_cache" / job.job_id
        try:
            audio_path = browser_preview_audio_path(source_audio_path, live_cache_dir)
        except Exception as exc:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Audio preview conversion failed: {exc}")
            return
        self.send_audio_file(audio_path)

    def send_audio_file(self, audio_path: Path) -> None:
        """Send one local audio file with basic HTTP range support."""
        file_size = audio_path.stat().st_size
        content_type = audio_content_type(audio_path)
        if file_size <= 0:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        byte_range = parse_range_header(self.headers.get("Range"), file_size)
        if byte_range is None:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.end_headers()
            return
        start_byte, end_byte, is_partial = byte_range
        self.send_response(HTTPStatus.PARTIAL_CONTENT if is_partial else HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end_byte - start_byte + 1))
        if is_partial:
            self.send_header("Content-Range", f"bytes {start_byte}-{end_byte}/{file_size}")
        self.end_headers()
        write_audio_range_to_stream(audio_path, self.wfile, start_byte=start_byte, end_byte=end_byte)

    def send_job(self, job_id: str) -> None:
        """Send preview job state."""
        job = self.gui_state.jobs.get(job_id)
        if job is None:
            self.send_json({"error": "Unknown job"}, HTTPStatus.NOT_FOUND)
            return
        self.send_json(preview_job_to_payload(job))

    def send_export_job(self, job_id: str) -> None:
        """Send export job state."""
        job = self.gui_state.export_jobs.get(job_id)
        if job is None:
            self.send_json({"error": "Unknown export job"}, HTTPStatus.NOT_FOUND)
            return
        self.send_json(export_job_to_payload(job))

    def send_session(self, session_id: str) -> None:
        """Send a completed preview session."""
        session = self.gui_state.sessions.get(session_id)
        if session is None:
            self.send_json({"error": "Unknown session"}, HTTPStatus.NOT_FOUND)
            return
        self.send_json(session_to_payload(session))

    def send_training_job(self, job_id: str) -> None:
        """Send correction-driven brain training job state."""
        job = self.gui_state.training_jobs.get(job_id)
        if job is None:
            self.send_json({"error": "Unknown training job"}, HTTPStatus.NOT_FOUND)
            return
        self.send_json(training_job_to_payload(job))

    def start_preview(self, payload: dict[str, Any]) -> None:
        """Start preview classification in a background thread."""
        input_path = Path(str(payload.get("input_path", "")).strip())
        if not input_path:
            self.send_json({"error": "Missing input path"}, HTTPStatus.BAD_REQUEST)
            return
        job = PreviewJob(job_id=uuid.uuid4().hex, status="running", message="Classifying preview...")
        self.gui_state.jobs[job.job_id] = job
        thread = threading.Thread(target=self.run_preview_job, args=(job, input_path), daemon=True)
        thread.start()
        self.send_json({"job_id": job.job_id})

    def run_preview_job(self, job: PreviewJob, input_path: Path) -> None:
        """Background preview worker."""
        try:
            job.status = "running"
            job.updated_at = time.time()
            session = self.gui_state.preview_service.classify_input(
                input_path,
                sort_workers=gui_worker_count(),
                progress_callback=lambda completed, total, latest: update_preview_job_progress(
                    job,
                    completed,
                    total,
                    latest,
                ),
                row_callback=lambda row: add_preview_job_row(job, row),
            )
            session_id = uuid.uuid4().hex
            self.gui_state.sessions[session_id] = session
            job.status = "done"
            job.completed_files = len(session.rows)
            job.total_files = len(session.rows)
            job.message = f"Preview ready: {len(session.rows)} files"
            job.session_id = session_id
            job.updated_at = time.time()
        except Exception as exc:
            job.status = "error"
            job.error = str(exc)
            job.message = "Preview failed"
            job.updated_at = time.time()

    def export_session(self, payload: dict[str, Any]) -> None:
        """Start an approved-folder export in a background thread."""
        session_id = str(payload.get("session_id", ""))
        session = self.gui_state.sessions.get(session_id)
        if session is None:
            self.send_json({"error": "Unknown session"}, HTTPStatus.NOT_FOUND)
            return
        destination = Path(str(payload.get("destination_path", "")).strip())
        if not str(destination):
            self.send_json({"error": "Missing destination path"}, HTTPStatus.BAD_REQUEST)
            return
        apply_overrides(session, payload.get("rows", []))
        mode = str(payload.get("mode", "copy"))
        job = create_export_job()
        self.gui_state.export_jobs[job.job_id] = job
        thread = threading.Thread(
            target=self.run_export_job,
            args=(job, session, destination, mode),
            daemon=True,
        )
        thread.start()
        self.send_json(export_job_to_payload(job))

    def run_export_job(
        self,
        job: ExportJob,
        session: SortPreviewSession,
        destination: Path,
        mode: str,
    ) -> None:
        """Background export worker for slow copy, move, or symlink runs."""
        try:
            job.status = "running"
            job.message = "Export is running..."
            job.updated_at = time.time()
            summary = self.gui_state.exporter.export(session, destination, mode=mode)  # type: ignore[arg-type]
            job.exported_count = summary.exported_count
            job.corrected_count = summary.corrected_count
            job.sorted_root = str(summary.sorted_root)
            job.approved_plan_path = str(summary.approved_plan_path)
            job.corrections_path = str(summary.corrections_path)
            job.errors = list(summary.errors)
            job.status = "done"
            job.message = f"Exported {summary.exported_count} files."
            job.updated_at = time.time()
        except Exception as exc:
            job.status = "error"
            job.error = str(exc)
            job.message = "Export failed"
            job.updated_at = time.time()

    def train_from_corrections(self, payload: dict[str, Any]) -> None:
        """Stage corrected rows and start incremental brain updates."""
        session_id = str(payload.get("session_id", ""))
        session = self.gui_state.sessions.get(session_id)
        if session is None:
            self.send_json({"error": "Unknown session"}, HTTPStatus.NOT_FOUND)
            return
        try:
            apply_overrides(session, payload.get("rows", []))
            import_summary = self.gui_state.training_importer.import_session(session)
            trainable_corrections = corrections_from_import_manifest(import_summary.manifest_path)
            if not trainable_corrections:
                self.send_json(
                    {
                        "error": "No trainable corrected audio files were available for brain updating.",
                        "errors": import_summary.errors,
                        "report_dir": str(import_summary.report_dir),
                    },
                    HTTPStatus.BAD_REQUEST,
                )
                return
            job = create_brain_training_job(import_summary)
            self.gui_state.training_jobs[job.job_id] = job
            thread = threading.Thread(
                target=run_brain_training_job,
                args=(job, self.gui_state.project_root),
                daemon=True,
            )
            thread.start()
            self.send_json(training_job_to_payload(job))
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def reveal_path(self, payload: dict[str, Any]) -> None:
        """Ask macOS Finder to open or reveal a path."""
        raw_path = str(payload.get("path", "")).strip()
        if not raw_path:
            self.send_json({"error": "Missing path"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            opened_path = reveal_path_in_finder(Path(raw_path))
            self.send_json({"opened_path": str(opened_path)})
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def bind_server(
    host: str,
    preferred_port: int,
    handler: Callable[..., BaseHTTPRequestHandler],
) -> ThreadingHTTPServer:
    """Bind a local HTTP server, trying nearby ports when needed."""
    for port_offset in range(20):
        try:
            server = ThreadingHTTPServer((host, preferred_port + port_offset), handler)
            server.daemon_threads = True
            return server
        except OSError:
            continue
    raise OSError(f"Could not bind {host}:{preferred_port}-{preferred_port + 19}")


def update_preview_job_progress(job: PreviewJob, completed_files: int, total_files: int, latest_file: str) -> None:
    """Update preview job progress for the browser status bar."""
    job.completed_files = int(completed_files)
    job.total_files = int(total_files)
    job.latest_file = str(latest_file)
    job.updated_at = time.time()
    if total_files <= 0:
        job.message = "Preparing audio files..."
        return
    if completed_files <= 0:
        job.message = f"Prepared {total_files} files. Starting classification..."
        return
    job.message = f"Classified {completed_files} of {total_files}: {latest_file}"


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
        errors=list(import_summary.errors),
    )


def run_brain_training_job(job: BrainTrainingJob, project_root: Path) -> None:
    """Run an incremental brain update for a correction-driven GUI job."""
    log_path = Path(job.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        job.status = "running"
        job.updated_at = time.time()
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write("Aaron GUI incremental brain update\n")
            log_handle.write(f"Project root: {project_root}\n")
            log_handle.write(f"Import manifest: {job.manifest_path}\n\n")
            corrections = corrections_from_import_manifest(Path(job.manifest_path))
            job.trainable_count = len(corrections)
            if not corrections:
                job.returncode = 1
                job.status = "error"
                job.message = "No trainable corrections were available for incremental brain update."
                job.updated_at = time.time()
                log_handle.write(job.message + "\n")
                return
            backup_dir = backup_active_brain_family(project_root, Path(job.backup_dir))
            log_handle.write(f"Backed up active brain files to: {backup_dir}\n\n")
            log_handle.flush()
            summary = IncrementalBrainUpdater(project_root).apply(
                corrections,
                report_dir=Path(job.report_dir),
                backup_dir=Path(job.backup_dir),
            )
            job.update_manifest_path = str(summary.manifest_path)
            job.errors.extend(summary.errors)
            for result in summary.updated_brains:
                log_handle.write(
                    f"Updated {result.brain_path.name}: "
                    f"labels {result.label_count_before}->{result.label_count_after}; "
                    f"corrections={result.corrections_applied}; "
                    f"touched={', '.join(result.labels_touched)}\n"
                )
                for warning in result.warnings:
                    log_handle.write(f"WARNING {result.brain_path.name}: {warning}\n")
            if summary.errors:
                log_handle.write("\nErrors/warnings:\n")
                for error in summary.errors:
                    log_handle.write(f"- {error}\n")
        job.returncode = 0
        job.status = "done"
        job.message = (
            f"Incrementally updated {len(summary.updated_brains)} active brain file(s) "
            f"from {summary.applied_correction_count} correction(s)."
        )
        job.updated_at = time.time()
    except Exception as exc:
        job.status = "error"
        job.returncode = 1
        job.message = f"Incremental brain update failed before completion: {exc}"
        job.updated_at = time.time()


def build_brain_family_training_command(project_root: Path, training_root: Path) -> list[str]:
    """Build the command that rebuilds the full/core/spread/outlier brains.

    Args:
        project_root: Project root containing ``Aaron_Sound_Sorter.py``.
        training_root: Trusted curated training tree.

    Returns:
        Command argument list suitable for ``subprocess.run``.

    Side Effects:
        None.
    """
    root = Path(project_root).expanduser().resolve()
    return [
        str(Path(sys.executable).expanduser()),
        str(root / "Aaron_Sound_Sorter.py"),
        "train-brain-family",
        str(Path(training_root).expanduser().resolve()),
        "--project-dir",
        str(root),
        "--save-full",
        str(root / "stage4_folder_brain.json"),
        "--save-core-baby",
        str(root / "stage4_folder_brain_core_baby.json"),
        "--save-spread-baby",
        str(root / "stage4_folder_brain_spread_baby.json"),
        "--save-outlier-baby",
        str(root / "stage4_folder_brain_outlier_baby.json"),
        "--core-anchors",
        "3",
        "--spread-anchors",
        "3",
        "--outlier-anchors",
        "3",
        "--full-max-centroids",
        "6",
        "--baby-max-centroids",
        "3",
        "--max-files-per-label-to-scan",
        "0",
        "--fingerprint-timeout-sec",
        "45",
        "--training-preview-per-label",
        "3",
        "--min-active-train-per-label",
        "1",
    ]


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


def active_gui_brain_file_names() -> list[str]:
    """Return brain JSON filenames that the GUI may use or overwrite."""
    return [
        "stage4_folder_brain.json",
        "stage4_folder_brain_baby.json",
        "stage4_folder_brain_core_baby.json",
        "stage4_folder_brain_spread_baby.json",
        "stage4_folder_brain_outlier_baby.json",
        USER_MEMORY_BRAIN_NAME,
        PHYSICS_MEMORY_BRAIN_NAME,
        VOTER_MEMORY_BRAIN_NAME,
        SHAPE_STARTER_MEMORY_BRAIN_NAME,
        SHAPE_MEMORY_BRAIN_NAME,
        "stage4_folder_brain_harmonic_core_baby.json",
        "stage4_folder_brain_harmonic_spread_baby.json",
        "stage4_folder_brain_harmonic_outlier_baby.json",
    ]


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
    job.partial_rows = [existing for existing in job.partial_rows if int(existing.get("index", -1)) != row_index]
    job.partial_rows.append(payload)
    job.partial_rows.sort(key=lambda existing: int(existing.get("index", 0)))
    job.updated_at = time.time()


def preview_row_payload_index(row: PreviewRow) -> int:
    """Return the zero-based final row index encoded in a preview row id."""
    try:
        return max(0, int(row.row_id) - 1)
    except ValueError:
        return 0


def preview_job_to_payload(job: PreviewJob) -> dict[str, Any]:
    """Convert one live preview job to a JSON-safe browser payload."""
    return {
        "job_id": job.job_id,
        "status": job.status,
        "message": job.message,
        "session_id": job.session_id,
        "error": job.error,
        "completed_files": job.completed_files,
        "total_files": job.total_files,
        "latest_file": job.latest_file,
        "partial_rows": sorted(job.partial_rows, key=lambda row: int(row.get("index", 0))),
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


def choose_path_with_osascript(kind: str, current_path: str = "") -> str:
    """Open a native macOS chooser and return the selected path when possible."""
    if os.name != "posix" or not Path("/usr/bin/osascript").exists():
        return ""
    script = """
on run argv
    set chooserKind to item 1 of argv
    set defaultPath to item 2 of argv
    set defaultLocation to missing value
    if defaultPath is not "" then
        try
            set defaultLocation to POSIX file defaultPath
        end try
    end if

    if chooserKind is "folder" then
        if defaultLocation is missing value then
            return POSIX path of (choose folder with prompt "Choose the exact sample folder to sort")
        end if
        return POSIX path of (choose folder with prompt "Choose the exact sample folder to sort" default location defaultLocation)
    end if

    if chooserKind is "destination" then
        if defaultLocation is missing value then
            return POSIX path of (choose folder with prompt "Choose the exact destination folder")
        end if
        return POSIX path of (choose folder with prompt "Choose the exact destination folder" default location defaultLocation)
    end if

    if chooserKind is "brain" then
        if defaultLocation is missing value then
            return POSIX path of (choose file with prompt "Choose brain JSON")
        end if
        return POSIX path of (choose file with prompt "Choose brain JSON" default location defaultLocation)
    end if

    if defaultLocation is missing value then
        return POSIX path of (choose file with prompt "Choose ZIP or audio file")
    end if
    return POSIX path of (choose file with prompt "Choose ZIP or audio file" default location defaultLocation)
end run
"""
    initial_path = chooser_default_location(kind, current_path)
    try:
        completed = subprocess.run(
            ["/usr/bin/osascript", "-e", script, kind, initial_path],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def chooser_default_location(kind: str, current_path: str) -> str:
    """Return a safe default location for the native macOS chooser."""
    raw_path = str(current_path or "").strip()
    if not raw_path:
        return ""
    try:
        path = Path(raw_path).expanduser()
        if kind in {"folder", "destination"}:
            if path.is_file():
                path = path.parent
            return str(path.resolve()) if path.exists() else ""
        if path.is_dir():
            return str(path.resolve())
        if path.parent.exists():
            return str(path.parent.resolve())
    except Exception:
        return ""
    return ""


def browser_preview_audio_path(source_audio_path: Path, run_dir: Path) -> Path:
    """Return an audio path that the browser can reliably preview.

    Args:
        source_audio_path: Preview-row source audio file.
        run_dir: GUI preview run directory used for generated preview cache
            files.

    Returns:
        The original source path for browser-native formats, or a cached WAV
        preview for AIFF/AIF files.

    Side Effects:
        May create ``audio_preview_cache`` under ``run_dir`` and write a WAV
        preview file.

    Important Constraints:
        This helper exists only for browser playback. It does not alter sorting,
        training evidence, exports, or classifier inputs.
    """
    resolved_source = Path(source_audio_path).expanduser().resolve()
    if resolved_source.suffix.lower() not in {".aif", ".aiff"}:
        return resolved_source
    cache_dir = Path(run_dir).expanduser().resolve() / "audio_preview_cache"
    return cached_wav_preview_path(resolved_source, cache_dir)


def cached_wav_preview_path(source_audio_path: Path, cache_dir: Path) -> Path:
    """Return a cached WAV preview path for one source audio file."""
    stat = source_audio_path.stat()
    cache_key = hashlib.sha256(f"{source_audio_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()[
        :16
    ]
    target_path = cache_dir / f"{safe_audio_cache_stem(source_audio_path)}_{cache_key}.wav"
    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_wav_preview_from_audio(source_audio_path, target_path)
    return target_path


def safe_audio_cache_stem(audio_path: Path) -> str:
    """Return a compact filesystem-safe stem for a generated preview file."""
    safe_chars = [char if char.isalnum() or char in {"-", "_"} else "_" for char in audio_path.stem]
    return ("".join(safe_chars).strip("_") or "audio")[:80]


def write_wav_preview_from_audio(source_audio_path: Path, target_path: Path) -> None:
    """Decode source audio and write a browser-friendly WAV preview."""
    import soundfile as sf

    audio_data, samplerate = sf.read(source_audio_path, always_2d=False)
    sf.write(target_path, audio_data, samplerate, format="WAV", subtype="PCM_16")


def write_audio_range_to_stream(audio_path: Path, stream: BinaryIO, *, start_byte: int, end_byte: int) -> bool:
    """Write one byte range to a browser stream.

    Args:
        audio_path: Local browser-preview audio path.
        stream: HTTP response stream.
        start_byte: Inclusive byte offset where streaming starts.
        end_byte: Inclusive byte offset where streaming stops.

    Returns:
        True when the whole requested byte range was written. False when the
        browser disconnected during playback.

    Side Effects:
        Reads from ``audio_path`` and writes bytes to ``stream``.

    Raises:
        Propagates file I/O errors other than normal browser disconnects.

    Important Constraints:
        Browser audio elements often cancel range requests when the user clicks
        another row, seeks, or the browser reissues a better range. Broken pipe
        and connection-reset errors are expected client disconnects, not sorter
        failures.
    """
    with audio_path.open("rb") as handle:
        handle.seek(start_byte)
        remaining_bytes = end_byte - start_byte + 1
        while remaining_bytes > 0:
            chunk = handle.read(min(262_144, remaining_bytes))
            if not chunk:
                break
            try:
                stream.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return False
            remaining_bytes -= len(chunk)
    return True


def audio_content_type(audio_path: Path) -> str:
    """Return a browser-friendly content type for an audio path."""
    suffix = audio_path.suffix.lower()
    explicit_types = {
        ".wav": "audio/wav",
        ".wave": "audio/wav",
        ".aif": "audio/aiff",
        ".aiff": "audio/aiff",
        ".flac": "audio/flac",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
    }
    if suffix in explicit_types:
        return explicit_types[suffix]
    guessed, _encoding = mimetypes.guess_type(str(audio_path))
    return guessed or "application/octet-stream"


def parse_range_header(range_header: str | None, file_size: int) -> tuple[int, int, bool] | None:
    """Parse an HTTP byte-range header for audio streaming.

    Args:
        range_header: Raw ``Range`` header from the browser.
        file_size: Total file size in bytes.

    Returns:
        ``(start_byte, end_byte, is_partial)`` or ``None`` for an invalid
        unsatisfiable range.
    """
    if not range_header:
        return (0, max(0, file_size - 1), False)
    if not range_header.startswith("bytes="):
        return None
    start_text, separator, end_text = range_header.removeprefix("bytes=").partition("-")
    if separator != "-":
        return None
    try:
        if not start_text:
            suffix_length = int(end_text)
            if suffix_length <= 0:
                return None
            start_byte = max(0, file_size - suffix_length)
            end_byte = file_size - 1
        else:
            start_byte = int(start_text)
            end_byte = int(end_text) if end_text else file_size - 1
    except ValueError:
        return None
    if start_byte < 0 or end_byte < start_byte or start_byte >= file_size:
        return None
    return (start_byte, min(end_byte, file_size - 1), True)


def render_index_html(*, brain_config_path: str, brain_summary: str) -> str:
    """Render the browser GUI shell."""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Aaron Sound Sorter</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f2ed;
      --panel: #ffffff;
      --text: #1d1d1f;
      --muted: #62666d;
      --line: #d7d2c8;
      --accent: #2357a5;
      --accent-dark: #173f7a;
      --soft: #eef3fb;
      --warn: #8b560d;
      --changed: #fff3bf;
      --changed-line: #d49a1d;
      --ok: #1f6f43;
      --details-pane-width: clamp(340px, 30vw, 560px);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scrollbar-gutter: stable; }}
    body {{
      margin: 0;
      padding-bottom: 86px;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    body.busy, body.busy button, body.busy input {{ cursor: progress; }}
    body.resizing-pane {{
      cursor: col-resize;
      user-select: none;
    }}
      header {{
        padding: 18px 22px 10px;
        border-bottom: 1px solid var(--line);
        background: #fbfaf7;
      }}
      h1 {{ margin: 0; font-size: 24px; }}
      header p {{ margin: 4px 0 0; color: var(--muted); }}
      main {{
        width: min(1920px, 100%);
        margin: 0 auto;
        padding: 16px 22px 22px;
        overflow-x: auto;
      }}
      .app-toolbar {{
        position: sticky;
        top: 0;
        z-index: 12;
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 10px;
        align-items: center;
        border-bottom: 1px solid var(--line);
        background: rgba(251, 250, 247, 0.96);
        padding: 10px 22px;
        backdrop-filter: blur(10px);
      }}
      .toolbar-label {{
        color: var(--muted);
        font-size: 12px;
        font-weight: 750;
        white-space: nowrap;
      }}
      .queue-scrollbar {{
        height: 16px;
        margin: 0 0 8px;
        overflow-x: auto;
        overflow-y: hidden;
        border: 1px solid #e7e1d7;
        border-radius: 999px;
        background: #f7f4ed;
      }}
      .queue-scrollbar[hidden] {{
        display: none;
      }}
      .queue-scrollbar-spacer {{
        width: 100%;
        min-width: 100%;
        height: 1px;
      }}
      .queue-scrollbar::-webkit-scrollbar,
      .table-wrap::-webkit-scrollbar {{
        height: 12px;
      }}
      .queue-scrollbar::-webkit-scrollbar-thumb,
      .table-wrap::-webkit-scrollbar-thumb {{
        background: #b7afa3;
        border: 3px solid #f7f4ed;
        border-radius: 999px;
      }}
      .queue-scrollbar::-webkit-scrollbar-track,
      .table-wrap::-webkit-scrollbar-track {{
        background: #eee9df;
      }}
      .toolbar-actions {{
        display: flex;
        gap: 8px;
        justify-content: flex-end;
        align-items: center;
      }}
      .panel {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 14px;
      }}
      .panel-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
        border-radius: 8px;
        transition:
          background 140ms ease,
          border-color 140ms ease;
      }}
      .panel-header[data-collapsible-header] {{
        cursor: pointer;
      }}
      .panel-header[data-collapsible-header]:hover {{
        background: #f8f7f2;
      }}
      .panel-title {{
        display: flex;
        align-items: baseline;
        gap: 10px;
        min-width: 0;
        font-weight: 800;
      }}
      .panel-title strong {{ font-size: 15px; }}
      .panel-title .small {{ overflow-wrap: anywhere; }}
      .sr-only {{
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
      }}
      .panel-toggle,
      .toolbar-icon-button {{
        display: inline-grid;
        place-items: center;
        width: 34px;
        height: 34px;
        min-height: 34px;
        padding: 0;
        border-radius: 999px;
        border: 1px solid transparent;
        background: color-mix(in srgb, var(--accent) 9%, white);
        color: #214b8e;
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.85),
          0 1px 2px rgba(15, 23, 42, 0.05);
        transition:
          background 140ms ease,
          border-color 140ms ease,
          box-shadow 140ms ease,
          transform 140ms ease;
      }}
      .panel-toggle:hover,
      .toolbar-icon-button:hover {{
        background: color-mix(in srgb, var(--accent) 15%, white);
        border-color: color-mix(in srgb, var(--accent) 30%, white);
        box-shadow: 0 6px 18px rgba(43, 95, 178, 0.13);
        transform: translateY(-1px);
      }}
      .panel-toggle:focus-visible,
      .toolbar-icon-button:focus-visible {{
        outline: 3px solid rgba(43, 95, 178, 0.24);
        outline-offset: 2px;
      }}
      .collapse-icon {{
        width: 9px;
        height: 9px;
        border-right: 2px solid currentColor;
        border-bottom: 2px solid currentColor;
        transform: translateY(-2px) rotate(45deg);
        transition: transform 160ms ease;
      }}
      .collapsed > .panel-header .collapse-icon {{
        transform: translateX(-1px) rotate(-45deg);
      }}
      .panel-toggle:hover .collapse-icon {{
        color: var(--accent-dark);
      }}
      .stack-icon {{
        position: relative;
        width: 16px;
        height: 14px;
      }}
      .stack-icon::before,
      .stack-icon::after {{
        content: "";
        position: absolute;
        left: 2px;
        width: 12px;
        height: 5px;
        border: 1.8px solid currentColor;
        border-radius: 3px;
        transition: transform 140ms ease;
      }}
      .stack-icon::before {{ top: 1px; }}
      .stack-icon::after {{ bottom: 1px; }}
      .collapse-all-icon::before {{ transform: translateY(3px); }}
      .collapse-all-icon::after {{ transform: translateY(-3px); }}
      .expand-all-icon::before {{ transform: translateY(-1px); }}
      .expand-all-icon::after {{ transform: translateY(1px); }}
      .toolbar-actions {{
        padding: 2px;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.7);
      }}
      .collapsible-panel.collapsed > .panel-body {{
        display: none;
      }}
      .collapsible-panel.collapsed > .panel-header {{
        margin-bottom: 0;
      }}
      .row {{
        display: grid;
        grid-template-columns: minmax(160px, 1fr) auto auto auto;
        gap: 8px;
      align-items: end;
      margin-bottom: 10px;
    }}
    label {{ display: block; font-size: 12px; font-weight: 700; color: var(--muted); margin-bottom: 4px; }}
    input[type="text"] {{
      width: 100%;
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 9px;
      font-size: 14px;
      background: white;
      color: var(--text);
    }}
    button {{
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 12px;
      background: #f2f2f0;
      color: var(--text);
      font-weight: 650;
      cursor: pointer;
    }}
    button.primary {{
      border-color: var(--accent);
      background: var(--accent);
      color: white;
    }}
    button.primary:hover {{ background: var(--accent-dark); }}
      button:disabled {{
        cursor: not-allowed;
        opacity: 0.55;
      }}
      .workspace-panel {{
        padding: 0;
        overflow: hidden;
      }}
      .workspace-panel > .panel-header {{
        padding: 12px 14px;
        margin-bottom: 0;
        border-bottom: 1px solid #ece8df;
        border-radius: 8px 8px 0 0;
      }}
      .workspace-panel.collapsed > .panel-header {{
        border-bottom-color: transparent;
        border-radius: 8px;
      }}
      .workspace-panel.collapsed > .workspace-body {{
        display: none;
      }}
      .workspace-body {{
        position: relative;
        display: grid;
        grid-template-columns: minmax(520px, 1fr) 14px minmax(320px, var(--details-pane-width));
        gap: 0;
        min-height: 390px;
        padding: 14px;
        align-items: start;
        transition: grid-template-columns 170ms ease;
      }}
      .workspace-body.details-collapsed {{
        grid-template-columns: minmax(520px, 1fr);
      }}
      .workspace-body.details-collapsed > .pane-resizer,
      .workspace-body.details-collapsed > #detailsPanel {{
        display: none;
      }}
      .workspace-body.queue-collapsed {{
        grid-template-columns: minmax(320px, 1fr);
      }}
      .workspace-body.queue-collapsed > #queuePanel,
      .workspace-body.queue-collapsed > .pane-resizer {{
        display: none;
      }}
      .details-reveal-button {{
        position: absolute;
        top: 14px;
        right: 14px;
        display: none;
        min-height: 34px;
        width: 34px;
        padding: 0;
        border-radius: 999px;
        background: color-mix(in srgb, var(--accent) 10%, white);
        color: var(--accent-dark);
        box-shadow: 0 8px 22px rgba(31, 35, 41, 0.12);
      }}
      .workspace-body.details-collapsed > .details-reveal-button {{
        display: inline-grid;
        place-items: center;
      }}
      .details-reveal-icon {{
        width: 9px;
        height: 9px;
        border-right: 2px solid currentColor;
        border-bottom: 2px solid currentColor;
        transform: rotate(135deg);
      }}
      .subpanel {{
        min-width: 0;
      }}
      .subpanel-frame {{
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #fff;
        padding: 12px;
        min-width: 0;
      }}
      #detailsPanel {{
        transition:
          opacity 160ms ease,
          transform 160ms ease;
      }}
      .pane-resizer {{
        position: relative;
        align-self: stretch;
        min-height: 220px;
        cursor: col-resize;
        touch-action: none;
      }}
      .pane-resizer::before {{
        content: "";
        position: absolute;
        top: 8px;
        bottom: 8px;
        left: 6px;
        width: 2px;
        border-radius: 999px;
        background: #ddd6ca;
        transition:
          background 140ms ease,
          box-shadow 140ms ease;
      }}
      .pane-resizer:hover::before,
      .pane-resizer:focus-visible::before,
      .workspace-body.resizing > .pane-resizer::before {{
        background: var(--accent);
        box-shadow: 0 0 0 4px rgba(35, 87, 165, 0.12);
      }}
      .pane-resizer:focus-visible {{
        outline: 0;
      }}
      .subpanel-frame > .panel-header[data-collapsible-header] {{
        margin: -4px -4px 10px;
        padding: 4px;
      }}
      .subpanel-frame.collapsed > .panel-header[data-collapsible-header] {{
        margin-bottom: -4px;
      }}
      .subpanel-frame.collapsed > .panel-body {{
        display: none;
      }}
      .table-wrap {{
        overflow: auto;
        border: 1px solid var(--line);
        border-radius: 6px;
        max-height: min(70vh, 820px);
        scrollbar-gutter: stable;
      }}
      table {{
        width: max(100%, 1120px);
        border-collapse: collapse;
        font-size: 13px;
      }}
      th, td {{ padding: 8px; border-bottom: 1px solid #ece8df; text-align: left; vertical-align: top; }}
      th {{ position: sticky; top: 0; background: #fbfaf7; z-index: 1; }}
    tr.selected {{ background: var(--soft); }}
    tr.corrected {{ background: #fffaf0; }}
    tr.selected.corrected {{ background: #fff1ce; }}
    td.approved {{ min-width: 340px; }}
    .approved-box {{
      display: flex;
      gap: 8px;
      align-items: center;
      min-width: 320px;
    }}
    .folder-text {{
      display: inline-block;
      max-width: 380px;
      overflow-wrap: anywhere;
    }}
    .folder-text.corrected {{
      border-left: 4px solid var(--changed-line);
      background: var(--changed);
      border-radius: 4px;
      padding: 4px 6px;
      font-weight: 700;
    }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 11px;
      font-weight: 750;
      background: var(--changed);
      color: #6f4604;
      border: 1px solid #ecc262;
    }}
    td.play {{ width: 72px; }}
    button.compact {{ min-height: 30px; padding: 5px 9px; font-size: 12px; }}
    .audio-box {{
      margin: 8px 0 10px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
    }}
    audio {{ width: 100%; height: 34px; }}
      .details {{
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 12px;
        background: #fbfbfa;
      white-space: pre-wrap;
      overflow: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 12px;
        line-height: 1.42;
        max-height: min(54vh, 660px);
      }}
      .category-hints {{
        margin-top: 8px;
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 10px;
        background: #fff;
        font-size: 12px;
      }}
      .category-hints summary {{
        cursor: pointer;
        font-weight: 750;
        color: var(--accent-dark);
      }}
      .hint-section {{
        margin-top: 10px;
        padding-top: 8px;
        border-top: 1px solid #ece8df;
      }}
      .hint-buttons {{
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 6px;
      }}
      .hint-button {{
        min-height: 28px;
        padding: 4px 7px;
        font-size: 11px;
        font-weight: 650;
        text-align: left;
        max-width: 100%;
        overflow-wrap: anywhere;
      }}
      .hint-tree {{
        margin-top: 8px;
        max-height: 260px;
        overflow: auto;
        border: 1px solid #ece8df;
        border-radius: 5px;
        padding: 8px;
        background: #fbfbfa;
      }}
      .hint-tree details {{ margin-left: 8px; }}
      .hint-tree summary {{ color: var(--text); font-weight: 700; }}
      .export-grid {{
      display: grid;
      grid-template-columns: minmax(180px, 1fr) auto auto auto auto;
      gap: 8px;
      align-items: end;
    }}
    .teach-grid {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto auto;
      gap: 10px;
      align-items: center;
    }}
    .radio-group {{
      display: flex;
      min-height: 36px;
      gap: 10px;
      align-items: center;
    }}
    .status {{
      color: var(--muted);
      font-size: 13px;
      min-height: 22px;
    }}
    .sticky-action-dock {{
      position: fixed;
      left: 50%;
      bottom: 14px;
      z-index: 18;
      display: flex;
      width: min(1040px, calc(100vw - 28px));
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 9px 10px;
      border: 1px solid rgba(139, 126, 106, 0.32);
      border-radius: 14px;
      background: rgba(251, 250, 247, 0.94);
      box-shadow: 0 16px 44px rgba(31, 35, 41, 0.14);
      transform: translateX(-50%);
      backdrop-filter: blur(12px);
    }}
    .sticky-action-group {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }}
    .sticky-action-status {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .progress-area {{
      margin-top: 10px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfbfa;
    }}
    .progress-header {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .progress-track {{
      width: 100%;
      height: 10px;
      border-radius: 999px;
      background: #e6e1d8;
      overflow: hidden;
    }}
    .progress-bar {{
      height: 100%;
      width: 0%;
      background: var(--accent);
      transition: width 160ms ease;
    }}
    .config-line {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 8px;
      align-items: baseline;
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
    }}
    .correction-notice {{
      border: 1px solid #e7b64c;
      border-left: 6px solid var(--changed-line);
      background: #fff7d8;
      color: #593b06;
      border-radius: 6px;
      padding: 10px 12px;
      margin-bottom: 14px;
      font-size: 13px;
    }}
    .modal-backdrop {{
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.32);
      display: grid;
      place-items: center;
      z-index: 20;
      padding: 20px;
    }}
    .modal-backdrop[hidden] {{ display: none; }}
    .modal-card {{
      width: min(880px, 96vw);
      max-height: 88vh;
      overflow: hidden;
      display: grid;
      grid-template-rows: auto auto minmax(220px, 1fr) auto;
      gap: 10px;
      border-radius: 8px;
      background: white;
      border: 1px solid var(--line);
      box-shadow: 0 18px 48px rgba(0, 0, 0, 0.22);
      padding: 14px;
    }}
    .modal-head {{
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 12px;
    }}
    .modal-actions {{
      display: flex;
      justify-content: flex-end;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .category-tree {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      overflow: auto;
      background: #fbfbfa;
    }}
    .category-tree details {{ margin-left: 10px; }}
      .category-tree summary {{
        cursor: pointer;
        font-weight: 700;
        padding: 3px 0;
    }}
    .category-leaf {{
      display: block;
      width: 100%;
      min-height: 28px;
      margin: 2px 0 2px 20px;
      padding: 4px 7px;
      text-align: left;
      border: 0;
      border-radius: 4px;
      background: transparent;
      font-weight: 500;
    }}
    .category-leaf:hover, .category-leaf.selected {{
      background: var(--soft);
      color: var(--accent-dark);
    }}
      .warn {{ color: var(--warn); }}
      .ok {{ color: var(--ok); }}
      .small {{ color: var(--muted); font-size: 12px; }}
      @media (max-width: 1100px) {{
        .workspace-body,
        .workspace-body.details-collapsed,
        .workspace-body.queue-collapsed {{
          grid-template-columns: 1fr;
        }}
        .workspace-body > .pane-resizer {{
          display: none;
        }}
        .row, .export-grid, .teach-grid {{ grid-template-columns: 1fr; }}
        .row button, .export-grid button, .teach-grid button {{ width: 100%; }}
        .panel-toggle, .toolbar-icon-button {{
          width: 34px;
          flex: 0 0 auto;
        }}
      }}
      @media (max-width: 760px) {{
        header {{ padding: 14px 14px 10px; }}
        main {{ padding: 12px 12px 18px; }}
        .app-toolbar {{
          grid-template-columns: 1fr auto;
          padding: 10px 12px;
        }}
        .sticky-action-dock {{
          align-items: stretch;
          flex-direction: column;
        }}
        .sticky-action-group {{
          display: grid;
          grid-template-columns: 1fr 1fr;
          width: 100%;
        }}
        .sticky-action-group button {{ width: 100%; }}
        table {{ width: max(100%, 980px); }}
      }}
    </style>
  </head>
  <body>
    <header>
      <h1>Aaron Sound Sorter</h1>
      <p>Preview, listen, fix folders, export the approved sort, and teach the brains from your corrections.</p>
    </header>
    <div class="app-toolbar" aria-label="Workspace controls">
      <span class="toolbar-label">Workspace controls</span>
      <div class="toolbar-actions">
        <button id="collapseAllPanels" class="toolbar-icon-button" type="button" aria-label="Collapse all panels" title="Collapse all panels">
          <span class="stack-icon collapse-all-icon" aria-hidden="true"></span>
        </button>
        <button id="expandAllPanels" class="toolbar-icon-button" type="button" aria-label="Expand all panels" title="Expand all panels">
          <span class="stack-icon expand-all-icon" aria-hidden="true"></span>
        </button>
      </div>
    </div>
    <main>
      <section class="panel collapsible-panel" data-panel-id="input">
        <div class="panel-header" data-collapsible-header>
          <div class="panel-title">
            <strong>Input</strong>
            <span class="small">Choose a folder, ZIP, or single audio file.</span>
          </div>
          <button class="panel-toggle" type="button" data-collapse-target="input" aria-expanded="true" aria-controls="inputPanelBody" aria-label="Collapse Input" title="Collapse Input">
            <span class="collapse-icon" aria-hidden="true"></span>
          </button>
        </div>
        <div id="inputPanelBody" class="panel-body">
          <div class="row">
            <div>
              <label for="inputPath">Input folder, ZIP, or audio file</label>
              <input id="inputPath" type="text" placeholder="/path/to/samples.zip" />
            </div>
            <button id="chooseFolder">Folder</button>
            <button id="chooseFile">ZIP/File</button>
            <button class="primary" id="previewButton">Preview Sort</button>
          </div>
          <div class="config-line">
            <strong>Brains</strong>
            <span>{escape_html(brain_summary)} from {escape_html(brain_config_path)}</span>
          </div>
          <div class="small">Browser security does not expose folder paths directly. Use the buttons for native macOS choosers, or paste paths.</div>
          <div id="progressArea" class="progress-area" hidden>
            <div class="progress-header">
              <span id="progressText">Preparing...</span>
              <span id="progressCount">0 / 0</span>
            </div>
            <div class="progress-track" aria-hidden="true"><div id="progressBar" class="progress-bar"></div></div>
          </div>
        </div>
      </section>

    <div id="correctionNotice" class="correction-notice" hidden>
      <strong id="correctionCount">0 corrections staged.</strong>
      Changed folders are already accepted for export. To teach the sorter, click Train Brains From Corrections.
    </div>

      <section class="panel workspace-panel collapsible-panel" data-panel-id="workspace">
        <div class="panel-header" data-collapsible-header>
          <div class="panel-title">
            <strong>Review Workspace</strong>
            <span id="rowCount" class="small">0 files</span>
          </div>
          <button class="panel-toggle" type="button" data-collapse-target="workspace" aria-expanded="true" aria-controls="workspaceBody" aria-label="Collapse Review Workspace" title="Collapse Review Workspace">
            <span class="collapse-icon" aria-hidden="true"></span>
          </button>
        </div>
        <div id="workspaceBody" class="workspace-body">
          <div id="queuePanel" class="subpanel subpanel-frame" data-panel-id="queue">
            <div class="panel-header" data-collapsible-header>
              <div class="panel-title">
                <strong>Preview Queue</strong>
                <span class="small">Scroll from the top or bottom of the queue.</span>
              </div>
              <button class="panel-toggle" type="button" data-collapse-target="queue" aria-expanded="true" aria-controls="queuePanelBody" aria-label="Collapse Preview Queue" title="Collapse Preview Queue">
                <span class="collapse-icon" aria-hidden="true"></span>
              </button>
            </div>
            <div id="queuePanelBody" class="panel-body">
              <div id="queueTopScroll" class="queue-scrollbar queue-scrollbar-top" aria-label="Preview queue horizontal scroll" tabindex="0" hidden>
                <div id="queueTopScrollSpacer" class="queue-scrollbar-spacer"></div>
              </div>
              <div id="tableWrap" class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Play</th>
                      <th>File</th>
                      <th>Approved Folder</th>
                      <th>Sorter Proposed</th>
                      <th>Decision</th>
                    </tr>
                  </thead>
                  <tbody id="rows"></tbody>
                </table>
              </div>
            </div>
          </div>
          <div id="detailsResizeHandle" class="pane-resizer" role="separator" aria-orientation="vertical" aria-label="Resize selected file panel" tabindex="0"></div>
          <button id="detailsRevealButton" class="details-reveal-button" type="button" data-collapse-target="details" aria-expanded="false" aria-controls="detailsPanelBody" aria-label="Expand Selected File" title="Expand Selected File">
            <span class="details-reveal-icon" aria-hidden="true"></span>
          </button>
          <aside id="detailsPanel" class="subpanel subpanel-frame" data-panel-id="details">
            <div class="panel-header" data-collapsible-header>
              <div class="panel-title"><strong>Selected File</strong></div>
              <button class="panel-toggle" type="button" data-collapse-target="details" aria-expanded="true" aria-controls="detailsPanelBody" aria-label="Collapse Selected File" title="Collapse Selected File">
                <span class="collapse-icon" aria-hidden="true"></span>
              </button>
            </div>
            <div id="detailsPanelBody" class="panel-body">
              <div class="audio-box">
                <label for="audioPlayer">Listen</label>
                <audio id="audioPlayer" controls preload="metadata"></audio>
              </div>
              <div id="details" class="details" style="margin-top:8px;">No file selected.</div>
              <div id="categoryHints" class="category-hints" hidden></div>
            </div>
          </aside>
        </div>
        </section>

      <section class="panel collapsible-panel" data-panel-id="export">
        <div class="panel-header" data-collapsible-header>
          <div class="panel-title">
            <strong>Export</strong>
            <span class="small">Write the approved plan after review.</span>
          </div>
          <button class="panel-toggle" type="button" data-collapse-target="export" aria-expanded="true" aria-controls="exportPanelBody" aria-label="Collapse Export" title="Collapse Export">
            <span class="collapse-icon" aria-hidden="true"></span>
          </button>
        </div>
        <div id="exportPanelBody" class="panel-body">
          <div class="export-grid">
            <div>
              <label for="destinationPath">Destination folder</label>
              <input id="destinationPath" type="text" placeholder="/path/to/output" />
            </div>
            <button id="chooseDestination">Destination</button>
            <div>
              <label>Mode</label>
              <div class="radio-group">
                <label><input type="radio" name="mode" value="copy" checked /> Copy</label>
                <label><input type="radio" name="mode" value="move" /> Move</label>
                <label><input type="radio" name="mode" value="symlink" /> Symlink</label>
              </div>
            </div>
            <button class="primary" id="exportButton">Export Approved Sort</button>
            <button id="openSortedButton" disabled>Open Sorted Folder</button>
            <button id="openReportButton" disabled>Open Report Folder</button>
          </div>
        </div>
      </section>

      <section class="panel collapsible-panel" data-panel-id="training">
        <div class="panel-header" data-collapsible-header>
          <div class="panel-title">
            <strong>Teach The Sorter</strong>
            <span class="small">Use staged corrections as measured training evidence.</span>
          </div>
          <button class="panel-toggle" type="button" data-collapse-target="training" aria-expanded="true" aria-controls="trainingPanelBody" aria-label="Collapse Teach The Sorter" title="Collapse Teach The Sorter">
            <span class="collapse-icon" aria-hidden="true"></span>
          </button>
        </div>
        <div id="trainingPanelBody" class="panel-body">
          <div class="teach-grid">
            <div>
              <div class="small">Copies changed rows into <code>training/locked_curated_v1</code>, backs up active brains, then applies a fast measured-evidence update.</div>
            </div>
            <button class="primary" id="trainCorrectionsButton" disabled>Train Brains From Corrections</button>
            <button id="openTrainingReportButton" disabled>Open Training Report</button>
          </div>
          <div id="trainingStatus" class="progress-area" hidden>
            <div class="progress-header">
              <span id="trainingText">Training not started.</span>
              <span id="trainingCount">0 corrections</span>
            </div>
            <div class="progress-track" aria-hidden="true"><div id="trainingBar" class="progress-bar"></div></div>
          </div>
        </div>
      </section>
    <div id="status" class="status">Ready.</div>
  </main>

  <div id="stickyActionDock" class="sticky-action-dock" role="region" aria-label="Always visible sort actions">
    <span id="stickyActionStatus" class="sticky-action-status">No preview loaded</span>
    <div class="sticky-action-group">
      <button id="quickExportButton" class="primary" type="button" disabled>Export Approved Sort</button>
      <button id="quickTrainCorrectionsButton" class="primary" type="button" disabled>Train Brains</button>
      <button id="quickOpenSortedButton" type="button" disabled>Open Sorted</button>
      <button id="quickOpenReportButton" type="button" disabled>Open Report</button>
      <button id="quickOpenTrainingReportButton" type="button" disabled>Open Training</button>
    </div>
  </div>

  <div id="categoryModal" class="modal-backdrop" hidden>
    <div class="modal-card" role="dialog" aria-modal="true" aria-labelledby="categoryModalTitle">
      <div class="modal-head">
        <div>
          <h2 id="categoryModalTitle" style="margin:0; font-size:18px;">Choose Approved Folder</h2>
          <div id="categoryModalFile" class="small" style="margin-top:4px;"></div>
        </div>
        <button id="closeCategoryModal" type="button">Close</button>
      </div>
      <div>
        <label for="categorySearch">Search taxonomy</label>
        <input id="categorySearch" type="text" placeholder="voice, kick, riser, piano..." />
      </div>
      <div id="categoryTree" class="category-tree"></div>
      <div>
        <label for="customCategory">Create or type a new approved category</label>
        <input id="customCategory" type="text" placeholder="Instruments/Synths/New Synth Folder/Loops" />
        <div id="selectedCategoryText" class="small" style="margin-top:6px;">No category selected yet.</div>
      </div>
      <div class="modal-actions">
        <button id="resetApprovedButton" type="button">Reset To Sorter Proposal</button>
        <button id="applyCategoryButton" class="primary" type="button">Apply Approved Folder</button>
      </div>
    </div>
  </div>

  <datalist id="labelOptions"></datalist>
<script>
const state = {{
  sessionId: "",
  rows: [],
  labels: [],
  selectedIndex: -1,
  categoryEditIndex: -1,
  selectedCategory: "",
  lastReportPath: "",
  lastSortedPath: "",
  lastTrainingReportPath: "",
  trainingJobId: "",
  exportJobId: "",
  livePreviewJobId: "",
  queueScrollSyncing: false
}};
const MAX_JOB_POLL_FAILURES = 40;
const PREVIEW_POLL_MS = 900;
const EXPORT_POLL_MS = 1500;
const TRAINING_POLL_MS = 2500;
const POLL_RETRY_MS = 2500;
const statusEl = document.getElementById("status");
const rowsEl = document.getElementById("rows");
const detailsEl = document.getElementById("details");
const categoryHints = document.getElementById("categoryHints");
const rowCountEl = document.getElementById("rowCount");
const audioPlayer = document.getElementById("audioPlayer");
const progressArea = document.getElementById("progressArea");
const progressText = document.getElementById("progressText");
const progressCount = document.getElementById("progressCount");
const progressBar = document.getElementById("progressBar");
const correctionNotice = document.getElementById("correctionNotice");
const correctionCount = document.getElementById("correctionCount");
const categoryModal = document.getElementById("categoryModal");
const categoryTree = document.getElementById("categoryTree");
const categorySearch = document.getElementById("categorySearch");
const customCategory = document.getElementById("customCategory");
const selectedCategoryText = document.getElementById("selectedCategoryText");
const openSortedButton = document.getElementById("openSortedButton");
const openReportButton = document.getElementById("openReportButton");
const openTrainingReportButton = document.getElementById("openTrainingReportButton");
const trainCorrectionsButton = document.getElementById("trainCorrectionsButton");
const trainingStatus = document.getElementById("trainingStatus");
  const trainingText = document.getElementById("trainingText");
  const trainingCount = document.getElementById("trainingCount");
  const trainingBar = document.getElementById("trainingBar");
  const tableWrap = document.getElementById("tableWrap");
  const queueTopScroll = document.getElementById("queueTopScroll");
  const queueTopScrollSpacer = document.getElementById("queueTopScrollSpacer");
  const workspaceBody = document.getElementById("workspaceBody");
  const detailsPanel = document.getElementById("detailsPanel");
  const detailsResizeHandle = document.getElementById("detailsResizeHandle");
  const exportButton = document.getElementById("exportButton");
  const quickExportButton = document.getElementById("quickExportButton");
  const quickTrainCorrectionsButton = document.getElementById("quickTrainCorrectionsButton");
  const quickOpenSortedButton = document.getElementById("quickOpenSortedButton");
  const quickOpenReportButton = document.getElementById("quickOpenReportButton");
  const quickOpenTrainingReportButton = document.getElementById("quickOpenTrainingReportButton");
  const stickyActionStatus = document.getElementById("stickyActionStatus");

  function setStatus(text, isWarn=false) {{
    statusEl.textContent = text;
    statusEl.className = isWarn ? "status warn" : "status";
  }}

  function panelElement(panelId) {{
    return document.querySelector(`[data-panel-id="${{panelId}}"]`);
  }}

  function panelDisplayName(panel) {{
    const title = panel?.querySelector(".panel-title strong");
    return title?.textContent?.trim() || "panel";
  }}

  function setPanelCollapsed(panelId, collapsed) {{
    const panel = panelElement(panelId);
    if (!panel) return;
    panel.classList.toggle("collapsed", collapsed);
    const panelName = panelDisplayName(panel);
    document.querySelectorAll(`[data-collapse-target="${{panelId}}"]`).forEach(button => {{
      const action = collapsed ? "Expand" : "Collapse";
      const label = `${{action}} ${{panelName}}`;
      button.setAttribute("aria-expanded", collapsed ? "false" : "true");
      button.setAttribute("aria-label", label);
      button.setAttribute("title", label);
    }});
    updateWorkspacePanelState();
    updateQueueScrollbars();
  }}

  function togglePanel(panelId) {{
    const panel = panelElement(panelId);
    if (!panel) return;
    setPanelCollapsed(panelId, !panel.classList.contains("collapsed"));
  }}

  function updateWorkspacePanelState() {{
    if (!workspaceBody) return;
    workspaceBody.classList.toggle("queue-collapsed", Boolean(panelElement("queue")?.classList.contains("collapsed")));
    workspaceBody.classList.toggle("details-collapsed", Boolean(panelElement("details")?.classList.contains("collapsed")));
  }}

  function setAllPanelsCollapsed(collapsed) {{
    ["input", "workspace", "queue", "details", "export", "training"].forEach(panelId => setPanelCollapsed(panelId, collapsed));
  }}

  function shouldIgnoreHeaderClick(event) {{
    const eventTarget = event.target instanceof Element ? event.target : null;
    return Boolean(eventTarget?.closest("button, a, input, select, textarea, label"));
  }}

  function updateQueueScrollbars() {{
    if (!tableWrap || !queueTopScroll || !queueTopScrollSpacer) return;
    const queueIsCollapsed = Boolean(panelElement("queue")?.classList.contains("collapsed"));
    const maxScroll = Math.max(0, tableWrap.scrollWidth - tableWrap.clientWidth);
    queueTopScroll.hidden = queueIsCollapsed || maxScroll <= 0;
    queueTopScrollSpacer.style.width = `${{Math.max(tableWrap.scrollWidth, tableWrap.clientWidth)}}px`;
    if (state.queueScrollSyncing) return;
    state.queueScrollSyncing = true;
    queueTopScroll.scrollLeft = tableWrap.scrollLeft;
    window.requestAnimationFrame(() => {{
      state.queueScrollSyncing = false;
    }});
  }}

  function syncQueueScrollbars(source, target) {{
    if (!source || !target || state.queueScrollSyncing) return;
    state.queueScrollSyncing = true;
    target.scrollLeft = source.scrollLeft;
    window.requestAnimationFrame(() => {{
      state.queueScrollSyncing = false;
      updateQueueScrollbars();
    }});
  }}

  function clampDetailsPaneWidth(width) {{
    const viewportLimit = Math.max(320, window.innerWidth - 520);
    return Math.max(300, Math.min(Number(width || 0), Math.min(780, viewportLimit)));
  }}

  function setDetailsPaneWidth(width, persist=true) {{
    const nextWidth = clampDetailsPaneWidth(width);
    document.documentElement.style.setProperty("--details-pane-width", `${{nextWidth}}px`);
    if (persist) localStorage.setItem("aaronDetailsPaneWidth", String(nextWidth));
  }}

  function restoreDetailsPaneWidth() {{
    const savedWidth = Number(localStorage.getItem("aaronDetailsPaneWidth") || 0);
    if (savedWidth > 0) setDetailsPaneWidth(savedWidth, false);
  }}

  function initDetailsResizer() {{
    if (!detailsResizeHandle || !detailsPanel) return;
    let startX = 0;
    let startWidth = 0;
    let activePointerId = null;
    function endResize() {{
      if (activePointerId === null) return;
      activePointerId = null;
      document.body.classList.remove("resizing-pane");
      workspaceBody.classList.remove("resizing");
      setDetailsPaneWidth(detailsPanel.getBoundingClientRect().width, true);
      updateQueueScrollbars();
    }}
    detailsResizeHandle.addEventListener("pointerdown", event => {{
      if (panelElement("details")?.classList.contains("collapsed")) return;
      activePointerId = event.pointerId;
      startX = event.clientX;
      startWidth = detailsPanel.getBoundingClientRect().width;
      detailsResizeHandle.setPointerCapture(event.pointerId);
      document.body.classList.add("resizing-pane");
      workspaceBody.classList.add("resizing");
      event.preventDefault();
    }});
    detailsResizeHandle.addEventListener("pointermove", event => {{
      if (activePointerId !== event.pointerId) return;
      setDetailsPaneWidth(startWidth + startX - event.clientX, false);
      updateQueueScrollbars();
    }});
    detailsResizeHandle.addEventListener("pointerup", endResize);
    detailsResizeHandle.addEventListener("pointercancel", endResize);
    detailsResizeHandle.addEventListener("keydown", event => {{
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      const delta = event.key === "ArrowLeft" ? 32 : -32;
      setDetailsPaneWidth(detailsPanel.getBoundingClientRect().width + delta, true);
      updateQueueScrollbars();
      event.preventDefault();
    }});
  }}

  function syncStickyActionDock() {{
    if (!stickyActionStatus) return;
    const correctedCount = correctionRows().length;
    const rowText = state.rows.length ? `${{state.rows.length}} files loaded` : "No preview loaded";
    const correctionText = correctedCount ? ` · ${{correctedCount}} staged` : "";
    stickyActionStatus.textContent = `${{rowText}}${{correctionText}}`;
    quickExportButton.disabled = exportButton.disabled;
    quickTrainCorrectionsButton.disabled = trainCorrectionsButton.disabled;
    quickOpenSortedButton.disabled = openSortedButton.disabled;
    quickOpenReportButton.disabled = openReportButton.disabled;
    quickOpenTrainingReportButton.disabled = openTrainingReportButton.disabled;
  }}

function setBusy(isBusy) {{
  document.body.classList.toggle("busy", isBusy);
  document.getElementById("previewButton").disabled = isBusy;
  exportButton.disabled = isBusy || !state.sessionId;
  trainCorrectionsButton.disabled = isBusy || correctionRows().length === 0 || !state.sessionId || Boolean(state.trainingJobId) || Boolean(state.exportJobId);
  syncStickyActionDock();
}}

async function jsonFetch(url, options={{}}) {{
  const response = await fetch(url, {{
    headers: {{ "Content-Type": "application/json" }},
    ...options
  }});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}}

function jobAgeSeconds(job) {{
  const updatedAt = Number(job.updated_at || 0);
  if (!updatedAt) return 0;
  return Math.max(0, Math.round((Date.now() / 1000) - updatedAt));
}}

function retryLongJobPoll(kind, retryCount, retryCallback) {{
  if (retryCount >= MAX_JOB_POLL_FAILURES) {{
    setStatus(`${{kind}} status could not be reached. The local GUI server may have stopped.`, true);
    setBusy(false);
    state.trainingJobId = "";
    state.exportJobId = "";
    trainCorrectionsButton.disabled = correctionRows().length === 0 || !state.sessionId;
    syncStickyActionDock();
    return;
  }}
  setStatus(`${{kind}} is still running; retrying status check (${{retryCount + 1}}/${{MAX_JOB_POLL_FAILURES}})...`, true);
  window.setTimeout(() => retryCallback(retryCount + 1), POLL_RETRY_MS);
}}

async function choosePath(kind, targetId) {{
  setStatus("Opening native chooser...");
  const target = document.getElementById(targetId);
  const current = target ? target.value.trim() : "";
  const payload = await jsonFetch(
    `/api/dialog?kind=${{encodeURIComponent(kind)}}&current=${{encodeURIComponent(current)}}`
  );
  if (payload.path) document.getElementById(targetId).value = payload.path;
  const selectedMessage = kind === "folder"
    ? `Selected exact folder: ${{payload.path}}. Preview scans this folder recursively.`
    : `Selected path: ${{payload.path}}`;
  setStatus(payload.path ? selectedMessage : "No path selected.");
}}

function resetProgress() {{
  progressArea.hidden = true;
  progressText.textContent = "Preparing...";
  progressCount.textContent = "0 / 0";
  progressBar.style.width = "0%";
}}

function updateProgress(job) {{
  const total = Number(job.total_files || 0);
  const completed = Number(job.completed_files || 0);
  progressArea.hidden = false;
  progressText.textContent = job.message || "Working...";
  progressCount.textContent = total > 0 ? `${{completed}} / ${{total}}` : "Preparing";
  const percent = total > 0 ? Math.max(0, Math.min(100, Math.round((completed / total) * 100))) : 5;
  progressBar.style.width = `${{percent}}%`;
}}

function showPreviewFailure(message) {{
  const text = message || "Preview failed.";
  progressArea.hidden = false;
  progressText.textContent = "Preview failed";
  progressCount.textContent = "Error";
  progressBar.style.width = "100%";
  detailsEl.textContent = text;
  setStatus(text, true);
  setBusy(false);
}}

function rowStableKey(row) {{
  return String(row?.row_id || row?.source_path || row?.display_name || row?.index || "");
}}

function rowAudioUrl(row) {{
  if (!row) return "";
  if (state.sessionId) return `/api/audio/${{encodeURIComponent(state.sessionId)}}/${{row.index}}`;
  if (state.livePreviewJobId) return `/api/job-audio/${{encodeURIComponent(state.livePreviewJobId)}}/${{row.index}}`;
  return "";
}}

function audioIsActivelyPlaying() {{
  return Boolean(audioPlayer.currentTime > 0 && !audioPlayer.paused && !audioPlayer.ended);
}}

function clearAudioSource() {{
  audioPlayer.removeAttribute("src");
  audioPlayer.dataset.sourceUrl = "";
  audioPlayer.dataset.rowKey = "";
  audioPlayer.load();
}}

function setAudioSourceForRow(row) {{
  const nextSource = rowAudioUrl(row);
  const nextRowKey = rowStableKey(row);
  if (!nextSource) {{
    if (audioPlayer.dataset.sourceUrl) clearAudioSource();
    return;
  }}
  const currentSource = audioPlayer.dataset.sourceUrl || "";
  const currentRowKey = audioPlayer.dataset.rowKey || "";
  if (currentRowKey === nextRowKey && (currentSource === nextSource || audioIsActivelyPlaying())) return;
  audioPlayer.dataset.sourceUrl = nextSource;
  audioPlayer.dataset.rowKey = nextRowKey;
  audioPlayer.src = nextSource;
  audioPlayer.load();
}}

function correctedFolderMap(rows) {{
  const edits = new Map();
  rows.forEach(row => {{
    if (row && rowIsCorrected(row)) edits.set(rowStableKey(row), row.approved_folder);
  }});
  return edits;
}}

function rowsWithPreservedCorrections(incomingRows) {{
  const edits = correctedFolderMap(state.rows);
  return [...(incomingRows || [])]
    .sort((left, right) => Number(left.index || 0) - Number(right.index || 0))
    .map(row => {{
      const copy = {{ ...row }};
      const approved = edits.get(rowStableKey(copy));
      if (approved) copy.approved_folder = approved;
      return copy;
    }});
}}

function mergeLivePreviewRows(partialRows) {{
  if (!Array.isArray(partialRows) || !partialRows.length) return;
  const selectedKey = state.selectedIndex >= 0 ? rowStableKey(state.rows[state.selectedIndex]) : "";
  state.rows = rowsWithPreservedCorrections(partialRows);
  if (selectedKey) {{
    const selectedIndex = state.rows.findIndex(row => rowStableKey(row) === selectedKey);
    state.selectedIndex = selectedIndex >= 0 ? selectedIndex : Math.min(state.selectedIndex, state.rows.length - 1);
  }}
  renderRows();
  if (state.selectedIndex >= 0 && state.rows[state.selectedIndex]) refreshSelectedRowDetails();
}}

async function startPreview() {{
  const inputPath = document.getElementById("inputPath").value.trim();
  if (!inputPath) {{
    setStatus("Choose or paste an input path first.", true);
    return;
  }}
    rowsEl.innerHTML = "";
    detailsEl.textContent = "Preview running...";
    categoryHints.hidden = true;
    categoryHints.innerHTML = "";
    clearAudioSource();
  rowCountEl.textContent = "0 files";
  state.rows = [];
  state.labels = [];
  state.sessionId = "";
  state.selectedIndex = -1;
  state.lastReportPath = "";
  state.lastSortedPath = "";
    state.lastTrainingReportPath = "";
    state.trainingJobId = "";
    state.exportJobId = "";
    state.livePreviewJobId = "";
    openReportButton.disabled = true;
    openSortedButton.disabled = true;
    openTrainingReportButton.disabled = true;
    updateQueueScrollbars();
    trainingStatus.hidden = true;
    trainingBar.style.width = "0%";
  updateCorrectionNotice();
  resetProgress();
  setBusy(true);
  setStatus("Starting preview sort...");
  try {{
    const payload = await jsonFetch("/api/preview", {{
      method: "POST",
      body: JSON.stringify({{ input_path: inputPath }})
    }});
    state.livePreviewJobId = payload.job_id || "";
    pollJob(payload.job_id);
  }} catch (error) {{
    showPreviewFailure(error.message || "Could not start preview.");
  }}
}}

async function pollJob(jobId, retryCount=0) {{
  let job;
  try {{
    job = await jsonFetch(`/api/job/${{jobId}}`);
  }} catch (error) {{
    retryLongJobPoll("Preview", retryCount, nextRetry => pollJob(jobId, nextRetry));
    return;
  }}
  updateProgress(job);
  mergeLivePreviewRows(job.partial_rows || []);
  const heartbeat = jobAgeSeconds(job);
  setStatus(`${{job.message || job.status}}${{heartbeat > 20 ? ` · last update ${{heartbeat}}s ago` : ""}}`);
  if (job.status === "done") {{
    await loadSession(job.session_id);
    state.livePreviewJobId = "";
    setBusy(false);
    return;
  }}
  if (job.status === "error") {{
    showPreviewFailure(job.error || job.message || "Preview failed.");
    return;
  }}
  window.setTimeout(() => pollJob(jobId, 0), PREVIEW_POLL_MS);
}}

async function loadSession(sessionId) {{
  const session = await jsonFetch(`/api/session/${{sessionId}}`);
  const selectedKey = state.selectedIndex >= 0 ? rowStableKey(state.rows[state.selectedIndex]) : "";
  state.sessionId = sessionId;
  state.rows = rowsWithPreservedCorrections(session.rows);
  state.labels = session.available_labels || [];
  state.lastReportPath = session.run_dir || "";
  state.lastTrainingReportPath = "";
  state.trainingJobId = "";
  state.livePreviewJobId = "";
  if (selectedKey) {{
    const selectedIndex = state.rows.findIndex(row => rowStableKey(row) === selectedKey);
    state.selectedIndex = selectedIndex >= 0 ? selectedIndex : state.selectedIndex;
  }}
  openReportButton.disabled = !state.lastReportPath;
  openTrainingReportButton.disabled = true;
  exportButton.disabled = !state.sessionId;
  trainingStatus.hidden = true;
  const labels = document.getElementById("labelOptions");
  labels.innerHTML = "";
  state.labels.forEach(label => {{
    const option = document.createElement("option");
    option.value = label;
    labels.appendChild(option);
  }});
    renderRows();
    if (state.selectedIndex >= 0 && state.rows[state.selectedIndex]) refreshSelectedRowDetails();
    updateCorrectionNotice();
    window.requestAnimationFrame(updateQueueScrollbars);
    setStatus(`Preview ready: ${{state.rows.length}} files. Run folder: ${{session.run_dir}}`);
    syncStickyActionDock();
  }}

function renderRows() {{
  rowCountEl.textContent = `${{state.rows.length}} files`;
  const fragment = document.createDocumentFragment();
  state.rows.forEach((row, index) => {{
    const tr = document.createElement("tr");
    const classes = [];
    if (index === state.selectedIndex) classes.push("selected");
    if (rowIsCorrected(row)) classes.push("corrected");
    tr.className = classes.join(" ");
    tr.dataset.rowIndex = String(index);
    tr.innerHTML = `
      <td class="play"><button class="compact" type="button" data-play-index="${{index}}">Play</button></td>
      <td>${{escapeHtml(row.display_name)}}</td>
      <td class="approved">
        <div class="approved-box">
          <button class="compact" type="button" data-category-index="${{index}}">Choose</button>
          <span class="folder-text ${{rowIsCorrected(row) ? "corrected" : ""}}">${{escapeHtml(row.approved_folder)}}</span>
          ${{rowIsCorrected(row) ? '<span class="pill">staged</span>' : ""}}
        </div>
      </td>
      <td>${{escapeHtml(row.proposed_folder)}}</td>
      <td>${{escapeHtml(row.consensus_status)}}</td>
    `;
    fragment.appendChild(tr);
  }});
    rowsEl.replaceChildren(fragment);
    updateCorrectionNotice();
    if (state.rows.length && state.selectedIndex < 0) selectRow(0);
    window.requestAnimationFrame(updateQueueScrollbars);
  }}

function normalizeFolder(value) {{
  return String(value || "").replace(/\\\\/g, "/").split("/").map(part => part.trim()).filter(Boolean).join("/");
}}

function rowIsCorrected(row) {{
  return normalizeFolder(row.approved_folder) !== normalizeFolder(row.proposed_folder);
}}

function correctionRows() {{
  return state.rows.filter(row => rowIsCorrected(row));
}}

function candidateFolders(row) {{
  const labels = Array.isArray(row.candidate_folders) ? row.candidate_folders : [];
  const unique = [];
  const seen = new Set();
  [row.approved_folder, row.proposed_folder, ...labels].forEach(label => {{
    const normalized = normalizeFolder(label);
    if (!normalized || seen.has(normalized)) return;
    seen.add(normalized);
    unique.push(normalized);
  }});
  return unique;
}}

function possibleFxFolders(row) {{
  return candidateFolders(row).filter(label => label.startsWith("FX/"));
}}

function updateCorrectionNotice() {{
  const correctedCount = correctionRows().length;
  correctionNotice.hidden = correctedCount === 0;
  correctionCount.textContent = correctedCount === 1 ? "1 correction staged." : `${{correctedCount}} corrections staged.`;
  trainCorrectionsButton.disabled = correctedCount === 0 || !state.sessionId || Boolean(state.trainingJobId);
  syncStickyActionDock();
}}

function selectRow(index) {{
  state.selectedIndex = index;
  refreshSelectedRowDetails();
}}

function refreshSelectedRowDetails() {{
  const index = state.selectedIndex;
  const row = state.rows[index];
  if (!row) return;
  setAudioSourceForRow(row);
    detailsEl.textContent = [
    `File: ${{row.display_name}}`,
    "",
    `Approved: ${{row.approved_folder}}`,
    `Proposed:  ${{row.proposed_folder}}`,
    `Decision:  ${{row.consensus_status}}`,
    `Top:       ${{row.final_top}}`,
    `Duration:  ${{Number(row.duration_sec || 0).toFixed(2)}}s`,
    rowIsCorrected(row) ? "" : "",
    rowIsCorrected(row) ? "Correction staged: export will use the approved folder. Train Brains From Corrections teaches future runs." : "",
    "",
    "Voter Summary:",
    row.diagnostic_summary || "(none)",
    "",
    "Possible FX Matches:",
    possibleFxFolders(row).length ? possibleFxFolders(row).slice(0, 8).map(label => `- ${{label}}`).join("\\n") : "(none from voter candidates)",
    "",
    "Detected Alternatives:",
    candidateFolders(row).filter(label => label !== row.approved_folder && label !== row.proposed_folder).slice(0, 8).map(label => `- ${{label}}`).join("\\n") || "(none)",
    "",
    "Reason:",
    row.decision_reason || "(none)"
    ].join("\\n");
    renderCategoryHints(row, index);
    Array.from(rowsEl.children).forEach((tr, rowIndex) => {{
      const candidate = state.rows[rowIndex];
      const classes = [];
      if (rowIndex === index) classes.push("selected");
      if (candidate && rowIsCorrected(candidate)) classes.push("corrected");
      tr.className = classes.join(" ");
    }});
  }}

  function renderCategoryHints(row, displayIndex) {{
    const candidates = candidateFolders(row);
    if (!candidates.length) {{
      categoryHints.hidden = true;
      categoryHints.innerHTML = "";
      return;
    }}
    categoryHints.hidden = false;
    categoryHints.innerHTML = "";
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `Detected category options (${{candidates.length}})`;
    details.appendChild(summary);
    const note = document.createElement("div");
    note.className = "small";
    note.style.marginTop = "6px";
    note.textContent = "These are voter/proposal hints only. Click a folder to approve it for this file.";
    details.appendChild(note);
    const fxCandidates = possibleFxFolders(row);
    if (fxCandidates.length) {{
      const fxSection = document.createElement("div");
      fxSection.className = "hint-section";
      const title = document.createElement("strong");
      title.textContent = "Possible FX matches";
      fxSection.appendChild(title);
      const buttonWrap = document.createElement("div");
      buttonWrap.className = "hint-buttons";
      fxCandidates.slice(0, 12).forEach(label => buttonWrap.appendChild(categoryHintButton(displayIndex, label)));
      fxSection.appendChild(buttonWrap);
      details.appendChild(fxSection);
    }}
    const treeSection = document.createElement("div");
    treeSection.className = "hint-section";
    const treeTitle = document.createElement("strong");
    treeTitle.textContent = "All detected options";
    treeSection.appendChild(treeTitle);
    const tree = document.createElement("div");
    tree.className = "hint-tree";
    appendHintNodes(tree, buildLabelTree(candidates).children, "", displayIndex);
    treeSection.appendChild(tree);
    details.appendChild(treeSection);
    categoryHints.appendChild(details);
  }}

  function categoryHintButton(index, label) {{
    const button = document.createElement("button");
    button.type = "button";
    button.className = "hint-button";
    button.textContent = label;
    button.addEventListener("click", event => {{
      event.preventDefault();
      event.stopPropagation();
      stageApprovedFolder(index, label);
    }});
    return button;
  }}

  function appendHintNodes(parent, children, prefix, index) {{
    Object.keys(children).sort((left, right) => left.localeCompare(right)).forEach(part => {{
      const node = children[part];
      const childKeys = Object.keys(node.children);
      const currentPath = prefix ? `${{prefix}}/${{part}}` : part;
      if (childKeys.length) {{
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = part;
        details.appendChild(summary);
        if (node.label) details.appendChild(categoryHintButton(index, node.label));
        appendHintNodes(details, node.children, currentPath, index);
        parent.appendChild(details);
        return;
      }}
      parent.appendChild(categoryHintButton(index, node.label || currentPath));
    }});
  }}

  function waitForAudioReady(player) {{
    if (player.readyState >= 2) return Promise.resolve();
    return new Promise((resolve, reject) => {{
      const timeout = window.setTimeout(() => {{
        cleanup();
        reject(new Error("Audio preview is still loading. Press the audio control once it appears ready."));
      }}, 3500);
      function cleanup() {{
        window.clearTimeout(timeout);
        player.removeEventListener("canplay", onReady);
        player.removeEventListener("loadedmetadata", onReady);
        player.removeEventListener("error", onError);
      }}
      function onReady() {{
        cleanup();
        resolve();
      }}
      function onError() {{
        cleanup();
        reject(new Error("Audio preview could not be loaded for this file."));
      }}
      player.addEventListener("canplay", onReady, {{ once: true }});
      player.addEventListener("loadedmetadata", onReady, {{ once: true }});
      player.addEventListener("error", onError, {{ once: true }});
    }});
  }}

  async function playRow(index) {{
    selectRow(index);
  try {{
    await waitForAudioReady(audioPlayer);
    await audioPlayer.play();
    setStatus(`Playing ${{state.rows[index].display_name}}`);
  }} catch (error) {{
    setStatus(error.message || "Browser blocked playback until you press the audio control.", true);
  }}
}}

function openCategoryChooser(index) {{
  state.categoryEditIndex = index;
  const row = state.rows[index];
  state.selectedCategory = row.approved_folder || row.proposed_folder || "";
  document.getElementById("categoryModalFile").textContent = row.display_name;
  categorySearch.value = "";
  customCategory.value = state.selectedCategory;
  selectedCategoryText.textContent = state.selectedCategory ? `Selected: ${{state.selectedCategory}}` : "No category selected yet.";
  categoryModal.hidden = false;
  renderCategoryTree();
  categorySearch.focus();
}}

function closeCategoryChooser() {{
  categoryModal.hidden = true;
  state.categoryEditIndex = -1;
  state.selectedCategory = "";
}}

  function renderCategoryTree() {{
    categoryTree.innerHTML = "";
    const query = categorySearch.value.trim().toLowerCase();
    const labels = state.labels.filter(label => !query || label.toLowerCase().includes(query));
  if (!labels.length) {{
    const empty = document.createElement("div");
    empty.className = "small";
    empty.textContent = "No matching trained folders. Type a new category below if this sound needs one.";
    categoryTree.appendChild(empty);
    return;
    }}
    const tree = buildLabelTree(labels);
    appendCategoryNodes(categoryTree, tree.children, "", Boolean(query));
    scrollSelectedCategoryIntoView();
  }}

function buildLabelTree(labels) {{
  const root = {{ children: {{}}, label: "" }};
  labels.forEach(label => {{
    const parts = normalizeFolder(label).split("/").filter(Boolean);
    let current = root;
    parts.forEach(part => {{
      if (!current.children[part]) current.children[part] = {{ children: {{}}, label: "" }};
      current = current.children[part];
    }});
    current.label = normalizeFolder(label);
  }});
  return root;
}}

  function appendCategoryNodes(parent, children, prefix, expandMatches) {{
    Object.keys(children).sort((left, right) => left.localeCompare(right)).forEach(part => {{
      const node = children[part];
      const childKeys = Object.keys(node.children);
      const currentPath = prefix ? `${{prefix}}/${{part}}` : part;
      if (childKeys.length) {{
        const details = document.createElement("details");
        details.open = expandMatches || selectedCategoryContainsPath(currentPath);
        const summary = document.createElement("summary");
        summary.textContent = part;
        details.appendChild(summary);
        if (node.label) details.appendChild(categoryLeafButton(node.label));
        appendCategoryNodes(details, node.children, currentPath, expandMatches);
        parent.appendChild(details);
        return;
      }}
      parent.appendChild(categoryLeafButton(node.label || currentPath));
    }});
  }}

  function selectedCategoryContainsPath(path) {{
    const selected = normalizeFolder(state.selectedCategory);
    const current = normalizeFolder(path);
    return Boolean(selected && current && (selected === current || selected.startsWith(`${{current}}/`)));
  }}

  function scrollSelectedCategoryIntoView() {{
    const selectedButton = categoryTree.querySelector(".category-leaf.selected");
    if (selectedButton) selectedButton.scrollIntoView({{ block: "center" }});
  }}

  function categoryLeafButton(label) {{
    const button = document.createElement("button");
    button.type = "button";
    button.className = normalizeFolder(label) === normalizeFolder(state.selectedCategory) ? "category-leaf selected" : "category-leaf";
    button.textContent = label;
  button.addEventListener("click", () => {{
    state.selectedCategory = label;
    customCategory.value = label;
    selectedCategoryText.textContent = `Selected: ${{label}}`;
    renderCategoryTree();
  }});
    return button;
  }}

  function stageApprovedFolder(index, approvedFolder) {{
    if (index < 0 || index >= state.rows.length) return;
    const approved = normalizeFolder(approvedFolder);
    if (!approved) {{
      setStatus("Choose or type an approved folder first.", true);
      return;
    }}
    state.rows[index].approved_folder = approved;
    state.lastTrainingReportPath = "";
    openTrainingReportButton.disabled = true;
    renderRows();
    selectRow(index);
    setStatus("Approved folder staged. Export will use it; training can teach the brains from it.");
  }}

  function applyApprovedCategory() {{
    if (state.categoryEditIndex < 0 || state.categoryEditIndex >= state.rows.length) return;
    const approved = normalizeFolder(customCategory.value || state.selectedCategory);
    if (!approved) {{
      setStatus("Choose or type an approved folder first.", true);
      return;
    }}
    const editedIndex = state.categoryEditIndex;
    closeCategoryChooser();
    stageApprovedFolder(editedIndex, approved);
  }}

function resetApprovedCategory() {{
  if (state.categoryEditIndex < 0 || state.categoryEditIndex >= state.rows.length) return;
  state.rows[state.categoryEditIndex].approved_folder = state.rows[state.categoryEditIndex].proposed_folder;
  state.lastTrainingReportPath = "";
  openTrainingReportButton.disabled = true;
  const editedIndex = state.categoryEditIndex;
  closeCategoryChooser();
  renderRows();
  selectRow(editedIndex);
  setStatus("Approved folder reset to sorter proposal.");
}}

async function trainBrainsFromCorrections() {{
  if (!state.sessionId) {{
    setStatus("Run Preview Sort before training from corrections.", true);
    return;
  }}
  if (!correctionRows().length) {{
    setStatus("No staged corrections to train from.", true);
    return;
  }}
  trainCorrectionsButton.disabled = true;
  syncStickyActionDock();
  trainingStatus.hidden = false;
  trainingText.textContent = "Staging corrections into the training tree...";
  trainingCount.textContent = `${{correctionRows().length}} correction(s)`;
  trainingBar.style.width = "12%";
  setStatus("Starting incremental brain update from corrections...");
  try {{
    const payload = await jsonFetch("/api/train-corrections", {{
      method: "POST",
      body: JSON.stringify({{
        session_id: state.sessionId,
        rows: state.rows.map(row => ({{ index: row.index, approved_folder: row.approved_folder }}))
      }})
    }});
    state.trainingJobId = payload.job_id || "";
    state.lastTrainingReportPath = payload.report_dir || "";
    openTrainingReportButton.disabled = !state.lastTrainingReportPath;
    syncStickyActionDock();
    updateTrainingStatus(payload);
    if (state.trainingJobId) pollTrainingJob(state.trainingJobId);
  }} catch (error) {{
    state.trainingJobId = "";
    trainCorrectionsButton.disabled = correctionRows().length === 0 || !state.sessionId;
    syncStickyActionDock();
    setStatus(error.message, true);
  }}
}}

async function exportApproved() {{
  if (!state.sessionId) {{
    setStatus("Run Preview Sort before exporting.", true);
    return;
  }}
  const destinationPath = document.getElementById("destinationPath").value.trim();
  if (!destinationPath) {{
    setStatus("Choose or paste a destination folder first.", true);
    return;
  }}
  const mode = document.querySelector("input[name='mode']:checked").value;
  setBusy(true);
  setStatus("Starting export job...");
  try {{
    const payload = await jsonFetch("/api/export", {{
      method: "POST",
      body: JSON.stringify({{
        session_id: state.sessionId,
        destination_path: destinationPath,
        mode,
        rows: state.rows.map(row => ({{ index: row.index, approved_folder: row.approved_folder }}))
      }})
    }});
    state.exportJobId = payload.job_id || "";
    if (!state.exportJobId) throw new Error("Export did not return a job id.");
    setStatus(payload.message || "Export is running...");
    pollExportJob(state.exportJobId);
  }} catch (error) {{
    state.exportJobId = "";
    setBusy(false);
    setStatus(error.message || "Could not start export.", true);
  }}
}}

async function pollExportJob(jobId, retryCount=0) {{
  let job;
  try {{
    job = await jsonFetch(`/api/export-job/${{jobId}}`);
  }} catch (error) {{
    retryLongJobPoll("Export", retryCount, nextRetry => pollExportJob(jobId, nextRetry));
    return;
  }}
  const heartbeat = jobAgeSeconds(job);
  if (job.status === "done") {{
    state.exportJobId = "";
    state.lastSortedPath = job.sorted_root || "";
    openSortedButton.disabled = !state.lastSortedPath;
    const errors = job.errors && job.errors.length ? ` Errors: ${{job.errors.join("; ")}}` : "";
    setStatus(`Exported ${{job.exported_count}} files to ${{job.sorted_root}}.${{errors}}`, Boolean(errors));
    setBusy(false);
    syncStickyActionDock();
    return;
  }}
  if (job.status === "error") {{
    state.exportJobId = "";
    setStatus(job.error || job.message || "Export failed.", true);
    setBusy(false);
    syncStickyActionDock();
    return;
  }}
  setStatus(`${{job.message || "Export is running..."}}${{heartbeat > 20 ? ` · last update ${{heartbeat}}s ago` : ""}}`);
  window.setTimeout(() => pollExportJob(jobId, 0), EXPORT_POLL_MS);
}}

function updateTrainingStatus(job) {{
  trainingStatus.hidden = false;
  trainingText.textContent = job.message || "Updating brains...";
  const trainable = Number(job.trainable_count || 0);
  const reused = Number(job.reused_existing_count || 0);
  const skipped = Number(job.skipped_count || 0);
  const pieces = [`${{trainable}} trainable`];
  if (reused) pieces.push(`${{reused}} already in training`);
  pieces.push(`${{skipped}} skipped`);
  trainingCount.textContent = pieces.join(", ");
  if (job.status === "done") {{
    trainingBar.style.width = "100%";
    state.trainingJobId = "";
    trainCorrectionsButton.disabled = correctionRows().length === 0 || !state.sessionId;
    setStatus(job.message || "Brain update complete.");
    syncStickyActionDock();
    return;
  }}
  if (job.status === "error") {{
    trainingBar.style.width = "100%";
    state.trainingJobId = "";
    trainCorrectionsButton.disabled = correctionRows().length === 0 || !state.sessionId;
    setStatus(job.message || "Brain update failed.", true);
    syncStickyActionDock();
    return;
  }}
  trainingBar.style.width = "55%";
  const heartbeat = jobAgeSeconds(job);
  setStatus(`${{job.message || "Brain update is running..."}}${{heartbeat > 20 ? ` · last update ${{heartbeat}}s ago` : ""}}`);
}}

async function pollTrainingJob(jobId, retryCount=0) {{
  let job;
  try {{
    job = await jsonFetch(`/api/training-job/${{jobId}}`);
  }} catch (error) {{
    retryLongJobPoll("Training", retryCount, nextRetry => pollTrainingJob(jobId, nextRetry));
    return;
  }}
  if (job.report_dir) {{
    state.lastTrainingReportPath = job.report_dir;
    openTrainingReportButton.disabled = false;
    syncStickyActionDock();
  }}
  updateTrainingStatus(job);
  if (job.status === "running" || job.status === "queued") {{
    window.setTimeout(() => pollTrainingJob(jobId, 0), TRAINING_POLL_MS);
  }}
}}

async function revealPath(path) {{
  if (!path) {{
    setStatus("No folder available yet.", true);
    return;
  }}
  const payload = await jsonFetch("/api/reveal", {{
    method: "POST",
    body: JSON.stringify({{ path }})
  }});
  setStatus(`Opened in Finder: ${{payload.opened_path}}`);
}}

function escapeHtml(value) {{
  return String(value || "").replace(/[&<>"']/g, char => ({{ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }}[char]));
}}

function escapeAttr(value) {{
  return escapeHtml(value).replace(/`/g, "&#96;");
}}

document.getElementById("chooseFolder").addEventListener("click", () => choosePath("folder", "inputPath"));
document.getElementById("chooseFile").addEventListener("click", () => choosePath("file", "inputPath"));
document.getElementById("chooseDestination").addEventListener("click", () => choosePath("destination", "destinationPath"));
document.getElementById("previewButton").addEventListener("click", () => startPreview().catch(error => {{
  setBusy(false);
  setStatus(error.message, true);
}}));
exportButton.addEventListener("click", () => exportApproved().catch(error => setStatus(error.message, true)));
trainCorrectionsButton.addEventListener("click", () => trainBrainsFromCorrections().catch(error => setStatus(error.message, true)));
openSortedButton.addEventListener("click", () => revealPath(state.lastSortedPath).catch(error => setStatus(error.message, true)));
openReportButton.addEventListener("click", () => revealPath(state.lastReportPath).catch(error => setStatus(error.message, true)));
openTrainingReportButton.addEventListener("click", () => revealPath(state.lastTrainingReportPath).catch(error => setStatus(error.message, true)));
quickExportButton.addEventListener("click", () => exportButton.click());
quickTrainCorrectionsButton.addEventListener("click", () => trainCorrectionsButton.click());
quickOpenSortedButton.addEventListener("click", () => openSortedButton.click());
quickOpenReportButton.addEventListener("click", () => openReportButton.click());
quickOpenTrainingReportButton.addEventListener("click", () => openTrainingReportButton.click());
document.getElementById("closeCategoryModal").addEventListener("click", () => closeCategoryChooser());
document.getElementById("applyCategoryButton").addEventListener("click", () => applyApprovedCategory());
  document.getElementById("resetApprovedButton").addEventListener("click", () => resetApprovedCategory());
  rowsEl.addEventListener("click", event => {{
    const eventTarget = event.target instanceof Element ? event.target : null;
    if (!eventTarget) return;
    const categoryButton = eventTarget.closest("button[data-category-index]");
    if (categoryButton) {{
      event.stopPropagation();
      openCategoryChooser(Number(categoryButton.dataset.categoryIndex));
      return;
    }}
    const playButton = eventTarget.closest("button[data-play-index]");
    if (playButton) {{
      event.stopPropagation();
      playRow(Number(playButton.dataset.playIndex));
      return;
    }}
    const rowElement = eventTarget.closest("tr[data-row-index]");
    if (rowElement) selectRow(Number(rowElement.dataset.rowIndex));
  }});
  document.querySelectorAll("[data-collapse-target]").forEach(button => {{
    button.addEventListener("click", () => togglePanel(button.dataset.collapseTarget));
  }});
  document.querySelectorAll("[data-collapsible-header]").forEach(header => {{
    header.addEventListener("click", event => {{
      if (shouldIgnoreHeaderClick(event)) return;
      const panel = header.closest("[data-panel-id]");
      if (panel?.dataset?.panelId) togglePanel(panel.dataset.panelId);
    }});
  }});
  document.getElementById("collapseAllPanels").addEventListener("click", () => setAllPanelsCollapsed(true));
  document.getElementById("expandAllPanels").addEventListener("click", () => setAllPanelsCollapsed(false));
  queueTopScroll.addEventListener("scroll", () => syncQueueScrollbars(queueTopScroll, tableWrap));
  tableWrap.addEventListener("scroll", () => syncQueueScrollbars(tableWrap, queueTopScroll));
  window.addEventListener("resize", () => updateQueueScrollbars());
  restoreDetailsPaneWidth();
  initDetailsResizer();
  updateQueueScrollbars();
  syncStickyActionDock();
  categorySearch.addEventListener("input", () => renderCategoryTree());
customCategory.addEventListener("input", () => {{
  state.selectedCategory = customCategory.value;
  selectedCategoryText.textContent = customCategory.value.trim()
    ? `New/selected: ${{customCategory.value.trim()}}`
    : "No category selected yet.";
}});
categoryModal.addEventListener("click", event => {{
  if (event.target === categoryModal) closeCategoryChooser();
}});
</script>
</body>
</html>"""


def escape_html(value: str) -> str:
    """Escape a value for HTML text or attribute contexts."""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def main(*, project_root: Path | None = None, probe: bool = False, open_browser: bool = True) -> int:
    """Run or probe the local browser GUI."""
    root = project_root or Path(__file__).resolve().parents[3]
    app = SorterWebApp(root, open_browser=open_browser)
    if probe:
        return app.probe()
    return app.run()
