"""Local browser GUI for previewing and approving sort results."""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import uuid
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

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


@dataclass
class BrainTrainingJob:
    """Background job state for GUI correction-driven brain training."""

    job_id: str
    status: str = "queued"
    message: str = "Queued"
    corrected_count: int = 0
    staged_count: int = 0
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


@dataclass
class WebGuiState:
    """Mutable state shared by local web-GUI request handlers."""

    project_root: Path
    preview_service: SortPreviewService = field(default_factory=SortPreviewService)
    exporter: SortPlanExporter = field(default_factory=SortPlanExporter)
    training_importer: TrainingCorrectionImporter = field(default_factory=TrainingCorrectionImporter)
    sessions: dict[str, SortPreviewSession] = field(default_factory=dict)
    jobs: dict[str, PreviewJob] = field(default_factory=dict)
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
        if parsed.path.startswith("/api/job/"):
            self.send_job(parsed.path.rsplit("/", 1)[-1])
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
            self.send_json({"path": choose_path_with_osascript(kind)})
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
        audio_path = session.rows[row_index].source_path
        if not audio_path.exists() or not audio_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Audio file not found")
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
        with audio_path.open("rb") as handle:
            handle.seek(start_byte)
            remaining_bytes = end_byte - start_byte + 1
            while remaining_bytes > 0:
                chunk = handle.read(min(262_144, remaining_bytes))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining_bytes -= len(chunk)

    def send_job(self, job_id: str) -> None:
        """Send preview job state."""
        job = self.gui_state.jobs.get(job_id)
        if job is None:
            self.send_json({"error": "Unknown job"}, HTTPStatus.NOT_FOUND)
            return
        self.send_json(job.__dict__)

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
            session = self.gui_state.preview_service.classify_input(
                input_path,
                sort_workers=gui_worker_count(),
                progress_callback=lambda completed, total, latest: update_preview_job_progress(
                    job,
                    completed,
                    total,
                    latest,
                ),
            )
            session_id = uuid.uuid4().hex
            self.gui_state.sessions[session_id] = session
            job.status = "done"
            job.completed_files = len(session.rows)
            job.total_files = len(session.rows)
            job.message = f"Preview ready: {len(session.rows)} files"
            job.session_id = session_id
        except Exception as exc:
            job.status = "error"
            job.error = str(exc)
            job.message = "Preview failed"

    def export_session(self, payload: dict[str, Any]) -> None:
        """Apply approved folders and export the session."""
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
        try:
            summary = self.gui_state.exporter.export(session, destination, mode=mode)  # type: ignore[arg-type]
            self.send_json(
                {
                    "exported_count": summary.exported_count,
                    "corrected_count": summary.corrected_count,
                    "sorted_root": str(summary.sorted_root),
                    "approved_plan_path": str(summary.approved_plan_path),
                    "corrections_path": str(summary.corrections_path),
                    "errors": summary.errors,
                }
            )
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

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
    if total_files <= 0:
        job.message = "Preparing audio files..."
        return
    if completed_files <= 0:
        job.message = f"Prepared {total_files} files. Starting classification..."
        return
    job.message = f"Classified {completed_files} of {total_files}: {latest_file}"


def create_brain_training_job(import_summary: TrainingImportSummary) -> BrainTrainingJob:
    """Create a background brain-training job from an import summary."""
    trainable_count = len(corrections_from_import_manifest(import_summary.manifest_path))
    return BrainTrainingJob(
        job_id=uuid.uuid4().hex,
        status="running",
        message=f"Imported {trainable_count} trainable correction(s). Updating active brains incrementally...",
        corrected_count=import_summary.staged_count + import_summary.skipped_count,
        staged_count=import_summary.staged_count,
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
    except Exception as exc:
        job.status = "error"
        job.returncode = 1
        job.message = f"Incremental brain update failed before completion: {exc}"


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


def choose_path_with_osascript(kind: str) -> str:
    """Open a native macOS chooser and return the selected path when possible."""
    if os.name != "posix" or not Path("/usr/bin/osascript").exists():
        return ""
    script_by_kind = {
        "folder": 'POSIX path of (choose folder with prompt "Choose sample folder")',
        "destination": 'POSIX path of (choose folder with prompt "Choose destination folder")',
        "brain": 'POSIX path of (choose file with prompt "Choose brain JSON")',
        "file": 'POSIX path of (choose file with prompt "Choose ZIP or audio file")',
    }
    script = script_by_kind.get(kind, script_by_kind["file"])
    try:
        completed = subprocess.run(
            ["/usr/bin/osascript", "-e", script],
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
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    body.busy, body.busy button, body.busy input {{ cursor: progress; }}
    header {{
      padding: 18px 22px 10px;
      border-bottom: 1px solid var(--line);
      background: #fbfaf7;
    }}
    h1 {{ margin: 0; font-size: 24px; }}
    header p {{ margin: 4px 0 0; color: var(--muted); }}
    main {{ padding: 16px 22px 22px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 14px;
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
    .workspace {{
      display: grid;
      grid-template-columns: minmax(420px, 1fr) 390px;
      gap: 14px;
      min-height: 390px;
    }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 6px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
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
    }}
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
    @media (max-width: 980px) {{
      .workspace {{ grid-template-columns: 1fr; }}
      .row, .export-grid, .teach-grid {{ grid-template-columns: 1fr; }}
      button {{ width: 100%; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Aaron Sound Sorter</h1>
    <p>Preview, listen, fix folders, export the approved sort, and teach the brains from your corrections.</p>
  </header>
  <main>
    <section class="panel">
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
    </section>

    <div id="correctionNotice" class="correction-notice" hidden>
      <strong id="correctionCount">0 corrections staged.</strong>
      Changed folders are already accepted for export. To teach the sorter, click Train Brains From Corrections.
    </div>

    <section class="panel workspace">
      <div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <strong>Preview Queue</strong>
          <span id="rowCount" class="small">0 files</span>
        </div>
        <div class="table-wrap">
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
      <aside>
        <strong>Selected File</strong>
        <div class="audio-box">
          <label for="audioPlayer">Listen</label>
          <audio id="audioPlayer" controls preload="none"></audio>
        </div>
        <div id="details" class="details" style="margin-top:8px;">No file selected.</div>
      </aside>
    </section>

    <section class="panel">
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
    </section>

    <section class="panel">
      <div class="teach-grid">
        <div>
          <strong>Teach The Sorter</strong>
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
    </section>
    <div id="status" class="status">Ready.</div>
  </main>

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
  trainingJobId: ""
}};
const statusEl = document.getElementById("status");
const rowsEl = document.getElementById("rows");
const detailsEl = document.getElementById("details");
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

function setStatus(text, isWarn=false) {{
  statusEl.textContent = text;
  statusEl.className = isWarn ? "status warn" : "status";
}}

function setBusy(isBusy) {{
  document.body.classList.toggle("busy", isBusy);
  document.getElementById("previewButton").disabled = isBusy;
  document.getElementById("exportButton").disabled = isBusy;
  trainCorrectionsButton.disabled = isBusy || correctionRows().length === 0 || !state.sessionId || Boolean(state.trainingJobId);
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

async function choosePath(kind, targetId) {{
  setStatus("Opening native chooser...");
  const payload = await jsonFetch(`/api/dialog?kind=${{encodeURIComponent(kind)}}`);
  if (payload.path) document.getElementById(targetId).value = payload.path;
  setStatus(payload.path ? "Path selected." : "No path selected.");
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

async function startPreview() {{
  const inputPath = document.getElementById("inputPath").value.trim();
  if (!inputPath) {{
    setStatus("Choose or paste an input path first.", true);
    return;
  }}
  rowsEl.innerHTML = "";
  detailsEl.textContent = "Preview running...";
  audioPlayer.removeAttribute("src");
  audioPlayer.load();
  rowCountEl.textContent = "0 files";
  state.rows = [];
  state.labels = [];
  state.sessionId = "";
  state.selectedIndex = -1;
  state.lastReportPath = "";
  state.lastSortedPath = "";
  state.lastTrainingReportPath = "";
  state.trainingJobId = "";
  openReportButton.disabled = true;
  openSortedButton.disabled = true;
  openTrainingReportButton.disabled = true;
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
    pollJob(payload.job_id);
  }} catch (error) {{
    showPreviewFailure(error.message || "Could not start preview.");
  }}
}}

async function pollJob(jobId) {{
  let job;
  try {{
    job = await jsonFetch(`/api/job/${{jobId}}`);
  }} catch (error) {{
    showPreviewFailure(error.message || "Could not read preview status.");
    return;
  }}
  updateProgress(job);
  setStatus(job.message || job.status);
  if (job.status === "done") {{
    await loadSession(job.session_id);
    setBusy(false);
    return;
  }}
  if (job.status === "error") {{
    showPreviewFailure(job.error || job.message || "Preview failed.");
    return;
  }}
  window.setTimeout(() => pollJob(jobId), 900);
}}

async function loadSession(sessionId) {{
  const session = await jsonFetch(`/api/session/${{sessionId}}`);
  state.sessionId = sessionId;
  state.rows = session.rows;
  state.labels = session.available_labels || [];
  state.lastReportPath = session.run_dir || "";
  state.lastTrainingReportPath = "";
  state.trainingJobId = "";
  openReportButton.disabled = !state.lastReportPath;
  openTrainingReportButton.disabled = true;
  trainingStatus.hidden = true;
  const labels = document.getElementById("labelOptions");
  labels.innerHTML = "";
  state.labels.forEach(label => {{
    const option = document.createElement("option");
    option.value = label;
    labels.appendChild(option);
  }});
  renderRows();
  updateCorrectionNotice();
  setStatus(`Preview ready: ${{state.rows.length}} files. Run folder: ${{session.run_dir}}`);
}}

function renderRows() {{
  rowsEl.innerHTML = "";
  rowCountEl.textContent = `${{state.rows.length}} files`;
  state.rows.forEach((row, index) => {{
    const tr = document.createElement("tr");
    const classes = [];
    if (index === state.selectedIndex) classes.push("selected");
    if (rowIsCorrected(row)) classes.push("corrected");
    tr.className = classes.join(" ");
    tr.addEventListener("click", () => selectRow(index));
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
    rowsEl.appendChild(tr);
  }});
  rowsEl.querySelectorAll("button[data-category-index]").forEach(button => {{
    button.addEventListener("click", event => {{
      event.stopPropagation();
      openCategoryChooser(Number(event.target.dataset.categoryIndex));
    }});
  }});
  rowsEl.querySelectorAll("button[data-play-index]").forEach(button => {{
    button.addEventListener("click", event => {{
      event.stopPropagation();
      playRow(Number(event.target.dataset.playIndex));
    }});
  }});
  updateCorrectionNotice();
  if (state.rows.length && state.selectedIndex < 0) selectRow(0);
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

function updateCorrectionNotice() {{
  const correctedCount = correctionRows().length;
  correctionNotice.hidden = correctedCount === 0;
  correctionCount.textContent = correctedCount === 1 ? "1 correction staged." : `${{correctedCount}} corrections staged.`;
  trainCorrectionsButton.disabled = correctedCount === 0 || !state.sessionId || Boolean(state.trainingJobId);
}}

function selectRow(index) {{
  state.selectedIndex = index;
  const row = state.rows[index];
  if (state.sessionId) {{
    audioPlayer.src = `/api/audio/${{state.sessionId}}/${{row.index}}`;
  }}
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
    "Reason:",
    row.decision_reason || "(none)"
  ].join("\\n");
  Array.from(rowsEl.children).forEach((tr, rowIndex) => {{
    const candidate = state.rows[rowIndex];
    const classes = [];
    if (rowIndex === index) classes.push("selected");
    if (candidate && rowIsCorrected(candidate)) classes.push("corrected");
    tr.className = classes.join(" ");
  }});
}}

async function playRow(index) {{
  selectRow(index);
  try {{
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
  appendCategoryNodes(categoryTree, tree.children, "");
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

function appendCategoryNodes(parent, children, prefix) {{
  Object.keys(children).sort((left, right) => left.localeCompare(right)).forEach(part => {{
    const node = children[part];
    const childKeys = Object.keys(node.children);
    if (childKeys.length) {{
      const details = document.createElement("details");
      details.open = prefix.split("/").length < 2;
      const summary = document.createElement("summary");
      summary.textContent = part;
      details.appendChild(summary);
      if (node.label) details.appendChild(categoryLeafButton(node.label));
      appendCategoryNodes(details, node.children, prefix ? `${{prefix}}/${{part}}` : part);
      parent.appendChild(details);
      return;
    }}
    parent.appendChild(categoryLeafButton(node.label || (prefix ? `${{prefix}}/${{part}}` : part)));
  }});
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

function applyApprovedCategory() {{
  if (state.categoryEditIndex < 0 || state.categoryEditIndex >= state.rows.length) return;
  const approved = normalizeFolder(customCategory.value || state.selectedCategory);
  if (!approved) {{
    setStatus("Choose or type an approved folder first.", true);
    return;
  }}
  state.rows[state.categoryEditIndex].approved_folder = approved;
  state.lastTrainingReportPath = "";
  openTrainingReportButton.disabled = true;
  const editedIndex = state.categoryEditIndex;
  closeCategoryChooser();
  renderRows();
  selectRow(editedIndex);
  setStatus("Approved folder staged. Export will use it; training can teach the brains from it.");
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
    updateTrainingStatus(payload);
    if (state.trainingJobId) pollTrainingJob(state.trainingJobId);
  }} catch (error) {{
    state.trainingJobId = "";
    trainCorrectionsButton.disabled = correctionRows().length === 0 || !state.sessionId;
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
  setStatus("Exporting approved sort plan...");
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
    state.lastSortedPath = payload.sorted_root || "";
    openSortedButton.disabled = !state.lastSortedPath;
    const errors = payload.errors && payload.errors.length ? ` Errors: ${{payload.errors.join("; ")}}` : "";
    setStatus(`Exported ${{payload.exported_count}} files to ${{payload.sorted_root}}.${{errors}}`, Boolean(errors));
  }} finally {{
    setBusy(false);
  }}
}}

function updateTrainingStatus(job) {{
  trainingStatus.hidden = false;
  trainingText.textContent = job.message || "Updating brains...";
  trainingCount.textContent = `${{Number(job.trainable_count || 0)}} trainable, ${{Number(job.skipped_count || 0)}} skipped`;
  if (job.status === "done") {{
    trainingBar.style.width = "100%";
    state.trainingJobId = "";
    trainCorrectionsButton.disabled = correctionRows().length === 0 || !state.sessionId;
    setStatus(job.message || "Brain update complete.");
    return;
  }}
  if (job.status === "error") {{
    trainingBar.style.width = "100%";
    state.trainingJobId = "";
    trainCorrectionsButton.disabled = correctionRows().length === 0 || !state.sessionId;
    setStatus(job.message || "Brain update failed.", true);
    return;
  }}
  trainingBar.style.width = "55%";
  setStatus(job.message || "Brain update is running...");
}}

async function pollTrainingJob(jobId) {{
  const job = await jsonFetch(`/api/training-job/${{jobId}}`);
  if (job.report_dir) {{
    state.lastTrainingReportPath = job.report_dir;
    openTrainingReportButton.disabled = false;
  }}
  updateTrainingStatus(job);
  if (job.status === "running" || job.status === "queued") {{
    window.setTimeout(() => pollTrainingJob(jobId), 2500);
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
document.getElementById("exportButton").addEventListener("click", () => exportApproved().catch(error => setStatus(error.message, true)));
trainCorrectionsButton.addEventListener("click", () => trainBrainsFromCorrections().catch(error => setStatus(error.message, true)));
openSortedButton.addEventListener("click", () => revealPath(state.lastSortedPath).catch(error => setStatus(error.message, true)));
openReportButton.addEventListener("click", () => revealPath(state.lastReportPath).catch(error => setStatus(error.message, true)));
openTrainingReportButton.addEventListener("click", () => revealPath(state.lastTrainingReportPath).catch(error => setStatus(error.message, true)));
document.getElementById("closeCategoryModal").addEventListener("click", () => closeCategoryChooser());
document.getElementById("applyCategoryButton").addEventListener("click", () => applyApprovedCategory());
document.getElementById("resetApprovedButton").addEventListener("click", () => resetApprovedCategory());
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
