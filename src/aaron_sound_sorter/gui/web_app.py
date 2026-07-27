"""Local browser GUI for previewing and approving sort results."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import subprocess
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

from aaron_sound_sorter.gui.incremental_brain_update import (
    IncrementalBrainUpdater,
    corrections_from_import_manifest,
)
from aaron_sound_sorter.gui.learning_center import LearningCenterService
from aaron_sound_sorter.gui.learning_center_page import render_learning_center_html
from aaron_sound_sorter.gui.models import SortPreviewSession
from aaron_sound_sorter.gui.preview_service import (
    PreviewCancelled,
    SortPlanExporter,
    SortPreviewService,
    TrainingCorrectionImporter,
    gui_worker_count,
    load_available_labels,
    write_preview_manifest,
)
from aaron_sound_sorter.gui.web_jobs import (
    BrainTrainingJob,
    ExportJob,
    PreviewJob,
    add_preview_job_row,
    apply_overrides,
    cancel_preview_job,
    create_brain_training_job,
    create_export_job,
    export_job_to_payload,
    preview_job_to_payload,
    request_stop_preview_job,
    reveal_path_in_finder,
    session_to_payload,
    training_job_to_payload,
    update_preview_job_progress,
)
from aaron_sound_sorter.gui.web_jobs import (
    backup_active_brain_family as backup_active_brain_family,
)
from aaron_sound_sorter.gui.web_jobs import (
    run_brain_training_job as _run_brain_training_job,
)
from aaron_sound_sorter.neural_audio.runtime import run_configured_neural_rebuild

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def run_brain_training_job(job: BrainTrainingJob, project_root: Path) -> None:
    """Run GUI training with inspectable neural and transitional dependencies."""
    _run_brain_training_job(
        job,
        project_root,
        corrections_loader=corrections_from_import_manifest,
        neural_rebuild=run_configured_neural_rebuild,
        updater_class=IncrementalBrainUpdater,
    )


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
    learning_center: LearningCenterService = field(init=False)

    def __post_init__(self) -> None:
        """Attach the human-facing learning service to this local GUI state."""
        self.learning_center = LearningCenterService(self.project_root)


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
            available_labels = load_available_labels(
                brain_config.full_brain_path,
                project_root=self.gui_state.project_root,
            )
            self.send_html(
                render_index_html(
                    brain_config_path=str(brain_config.config_path),
                    brain_summary=brain_config.display_summary(),
                    available_labels=available_labels,
                )
            )
            return
        if parsed.path == "/learning":
            self.send_html(render_learning_center_html())
            return
        if parsed.path == "/api/learning/coverage":
            self.send_learning_coverage()
            return
        if parsed.path == "/api/learning/calibration":
            self.send_json(self.gui_state.learning_center.calibration_status())
            return
        if parsed.path.startswith("/api/learning/rebuild-job/"):
            self.send_learning_rebuild_job(parsed.path.rsplit("/", 1)[-1])
            return
        if parsed.path.startswith("/api/learning/audio/"):
            self.send_learning_audio(parsed.path)
            return
        if parsed.path.startswith("/api/audio/"):
            self.send_audio_from_path(parsed.path)
            return
        if parsed.path.startswith("/api/job-audio/"):
            self.send_job_audio_from_path(parsed.path)
            return
        if parsed.path.startswith("/api/job/"):
            query = parse_qs(parsed.query)
            try:
                after_revision = int(query.get("after_revision", ["0"])[0] or 0)
            except ValueError:
                after_revision = 0
            self.send_job(parsed.path.rsplit("/", 1)[-1], after_revision=after_revision)
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
        if parsed.path == "/api/preview-cancel":
            self.cancel_preview(payload)
            return
        if parsed.path == "/api/preview-stop":
            self.stop_preview(payload)
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
        if parsed.path == "/api/learning/open-pack":
            self.open_learning_pack(payload)
            return
        if parsed.path == "/api/learning/decision":
            self.save_learning_decision(payload)
            return
        if parsed.path == "/api/learning/import":
            self.import_learning_decision(payload)
            return
        if parsed.path == "/api/learning/rebuild-calibration":
            self.rebuild_learning_calibration()
            return
        if parsed.path == "/api/learning/rebuild-prototypes":
            self.start_learning_prototype_rebuild()
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

    def send_learning_coverage(self) -> None:
        """Send canonical category readiness for the Learning Center."""
        try:
            self.send_json(self.gui_state.learning_center.coverage_payload())
        except (OSError, RuntimeError, ValueError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def send_learning_audio(self, request_path: str) -> None:
        """Stream one hash-addressed cluster representative."""
        parts = request_path.removeprefix("/api/learning/audio/").split("/")
        if len(parts) != 2:
            self.send_error(HTTPStatus.BAD_REQUEST, "Cluster audio URL is invalid")
            return
        try:
            audio_path = self.gui_state.learning_center.audio_path(parts[0], parts[1])
            preview_path = browser_preview_audio_path(audio_path, audio_path.parent / ".browser_preview")
            self.send_audio_file(preview_path)
        except (KeyError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            self.send_error(HTTPStatus.NOT_FOUND, str(exc))

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

    def send_job(self, job_id: str, *, after_revision: int = 0) -> None:
        """Send preview job state."""
        job = self.gui_state.jobs.get(job_id)
        if job is None:
            self.send_json({"error": "Unknown job"}, HTTPStatus.NOT_FOUND)
            return
        self.send_json(preview_job_to_payload(job, after_revision=max(0, after_revision)))

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
                cancel_requested=job.cancel_event.is_set,
            )
            if job.cancel_event.is_set():
                if job.keep_completed_on_stop:
                    self.finalize_stopped_preview(job, input_path)
                else:
                    cancel_preview_job(job)
                return
            session_id = uuid.uuid4().hex
            self.gui_state.sessions[session_id] = session
            job.status = "done"
            job.completed_files = len(session.rows)
            job.total_files = len(session.rows)
            job.message = f"Preview ready: {len(session.rows)} files"
            job.session_id = session_id
            job.updated_at = time.time()
        except PreviewCancelled:
            if job.keep_completed_on_stop:
                self.finalize_stopped_preview(job, input_path)
            else:
                cancel_preview_job(job)
        except Exception as exc:
            if job.cancel_event.is_set():
                if job.keep_completed_on_stop:
                    self.finalize_stopped_preview(job, input_path)
                else:
                    cancel_preview_job(job)
                return
            job.status = "error"
            job.error = str(exc)
            job.message = "Preview failed"
            job.updated_at = time.time()

    def cancel_preview(self, payload: dict[str, Any]) -> None:
        """Request cooperative cancellation for a live preview job."""
        job_id = str(payload.get("job_id", "")).strip()
        job = self.gui_state.jobs.get(job_id)
        if job is None:
            self.send_json({"error": "Unknown job"}, HTTPStatus.NOT_FOUND)
            return
        cancel_preview_job(job)
        self.send_json(preview_job_to_payload(job))

    def stop_preview(self, payload: dict[str, Any]) -> None:
        """Stop analysis but turn completed rows into a usable session."""
        job_id = str(payload.get("job_id", "")).strip()
        job = self.gui_state.jobs.get(job_id)
        if job is None:
            self.send_json({"error": "Unknown job"}, HTTPStatus.NOT_FOUND)
            return
        request_stop_preview_job(job)
        self.send_json(preview_job_to_payload(job))

    def finalize_stopped_preview(self, job: PreviewJob, input_path: Path) -> None:
        """Create a normal review/export session from completed live rows."""
        rows = [job.completed_rows[index] for index in sorted(job.completed_rows)]
        for row in rows:
            if row.neural_decision_state == "provisional":
                row.neural_decision_state = "finalized_without_neural"
                row.neural_runtime_status = "stopped_before_neural"
                row.neural_runtime_message = "Preview was stopped before neural analysis finalized this row."
        if not rows:
            job.status = "stopped"
            job.message = "Stopped before any sounds finished. Nothing was discarded because nothing was ready yet."
            job.updated_at = time.time()
            return
        brain_config = self.gui_state.preview_service.load_brain_family_config()
        run_dir = (
            self.gui_state.project_root
            / "_reports"
            / "gui_preview"
            / f"stopped_{time.strftime('%Y%m%d_%H%M%S')}_{job.job_id[:6]}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        session = SortPreviewSession(
            run_dir=run_dir,
            input_path=input_path,
            brain_path=brain_config.full_brain_path,
            available_labels=load_available_labels(
                brain_config.full_brain_path,
                project_root=self.gui_state.project_root,
            ),
            rows=rows,
        )
        write_preview_manifest(run_dir / "Aaron_GUI_Preview.csv", session)
        session_id = uuid.uuid4().hex
        self.gui_state.sessions[session_id] = session
        job.session_id = session_id
        job.status = "stopped"
        job.completed_files = len(rows)
        job.message = f"Stopped and kept {len(rows)} completed sounds for review."
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

    def open_learning_pack(self, payload: dict[str, Any]) -> None:
        """Load a source-name-blind cluster pack for human review."""
        try:
            self.send_json(self.gui_state.learning_center.open_cluster_pack(str(payload.get("path", ""))))
        except (FileNotFoundError, OSError, ValueError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def save_learning_decision(self, payload: dict[str, Any]) -> None:
        """Persist an explicit cluster decision without silently training."""
        try:
            decision = self.gui_state.learning_center.save_cluster_decision(
                pack_id=str(payload.get("pack_id", "")),
                cluster_id=str(payload.get("cluster_id", "")),
                action=str(payload.get("action", "")),
                approved_category=str(payload.get("approved_category", "")),
                reviewed_by=str(payload.get("reviewed_by", "Aaron")),
            )
            self.send_json(decision)
        except (KeyError, OSError, ValueError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def import_learning_decision(self, payload: dict[str, Any]) -> None:
        """Import a separately saved cluster approval into the local inbox."""
        try:
            imported = self.gui_state.learning_center.import_saved_decision(str(payload.get("decision_path", "")))
            self.send_json(imported)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def rebuild_learning_calibration(self) -> None:
        """Recheck whether reviewed outcomes can fit honest calibration."""
        try:
            self.send_json(dict(self.gui_state.learning_center.rebuild_calibration()))
        except (OSError, RuntimeError, ValueError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def start_learning_prototype_rebuild(self) -> None:
        """Start a user-requested background rebuild of approved prototypes."""
        self.send_json(self.gui_state.learning_center.start_prototype_rebuild())

    def send_learning_rebuild_job(self, job_id: str) -> None:
        """Send simple progress for a Learning Center prototype rebuild."""
        try:
            self.send_json(self.gui_state.learning_center.prototype_rebuild_status(job_id))
        except KeyError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)


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


def render_index_html(
    *,
    brain_config_path: str,
    brain_summary: str,
    available_labels: list[str] | None = None,
) -> str:
    """Render the browser GUI shell from its package-local web asset."""
    labels_json = json.dumps(available_labels or []).replace("</", "<\\/")
    template_path = Path(__file__).with_name("static") / "sorter.html"
    template = template_path.read_text(encoding="utf-8")
    return (
        template.replace("@@INITIAL_AVAILABLE_LABELS@@", labels_json)
        .replace("@@BRAIN_SUMMARY@@", escape_html(brain_summary))
        .replace("@@BRAIN_CONFIG_PATH@@", escape_html(brain_config_path))
    )


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
