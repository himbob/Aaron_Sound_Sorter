"""Subprocess boundary for source-name-blind neural GUI inference and training."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .gui_training import DEFAULT_TRAINING_CONFIG


@dataclass(frozen=True)
class NeuralPromptSuggestion:
    """One read-only detailed CLAP taxonomy suggestion for GUI display."""

    path: str
    positive_score: float
    negative_score: float
    prompt_margin: float
    top_positive_prompt: str
    top_positive_similarity: float
    top_negative_prompt: str
    top_negative_similarity: float


@dataclass(frozen=True)
class NeuralEventSuggestion:
    """One raw independent AudioSet event for GUI display."""

    label: str
    score: float


@dataclass(frozen=True)
class NeuralRuntimePrediction:
    """One neural prediction returned to the GUI."""

    row_id: str
    file_sha256: str
    predicted_label: str
    second_label: str
    top_similarity: float
    second_similarity: float
    margin: float
    radius_ratio: float
    known_distribution: bool
    label_example_count: int = 0
    exact_training_match: bool = False
    ownership_ready: bool = False
    ownership_block_reason: str = ""
    semantic_status: str = "unavailable"
    semantic_family: str = ""
    semantic_second_family: str = ""
    semantic_top_score: float = 0.0
    semantic_second_score: float = 0.0
    semantic_margin: float = 0.0
    semantic_family_scores: Mapping[str, float] = field(default_factory=dict)
    prompt_brain_status: str = "unavailable"
    prompt_suggestions: tuple[NeuralPromptSuggestion, ...] = ()
    panns_status: str = "unavailable"
    panns_model_id: str = ""
    panns_events: tuple[NeuralEventSuggestion, ...] = ()
    panns_family_scores: Mapping[str, float] = field(default_factory=dict)
    panns_support_score: float = 0.0
    panns_contradiction_score: float = 0.0
    panns_supporting_events: tuple[NeuralEventSuggestion, ...] = ()
    panns_contradicting_events: tuple[NeuralEventSuggestion, ...] = ()


@dataclass(frozen=True)
class NeuralRuntimeBatch:
    """Result of one isolated neural subprocess invocation."""

    status: str
    message: str
    predictions: tuple[NeuralRuntimePrediction, ...]
    index_path: str = ""
    report_path: Path | None = None
    log_path: Path | None = None
    row_errors: Mapping[str, str] = field(default_factory=dict)


def load_neural_training_config(project_root: Path) -> dict[str, Any]:
    """Read the repository-local neural runtime configuration."""
    root = Path(project_root).expanduser().resolve()
    config_path = root / DEFAULT_TRAINING_CONFIG
    if not config_path.is_file():
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def configured_neural_python(project_root: Path) -> Path:
    """Resolve the isolated Python executable configured for neural work."""
    root = Path(project_root).expanduser().resolve()
    config = load_neural_training_config(root)
    raw_path = Path(str(config.get("python_path", ".venv_neural/bin/python"))).expanduser()
    python_path = raw_path if raw_path.is_absolute() else root / raw_path
    if not python_path.is_file():
        raise FileNotFoundError(f"configured neural Python is missing: {python_path}")
    return python_path




class ConfiguredNeuralPredictionProcess:
    """One long-lived neural worker whose atomic checkpoints can be polled.

    The worker receives the whole list of audio paths once, loads CLAP and PANNs
    once, and checkpoints each completed row. The GUI can therefore combine a
    base sorter result with its neural result immediately instead of waiting for
    the entire folder to finish.
    """

    def __init__(
        self,
        project_root: Path,
        audio_by_row_id: list[tuple[str, Path]],
        report_dir: Path,
        *,
        timeout_seconds: float = 1800.0,
    ) -> None:
        self.root = Path(project_root).expanduser().resolve()
        self.output_dir = Path(report_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.request_path = self.output_dir / "neural_runtime_request.json"
        self.response_path = self.output_dir / "neural_runtime_predictions.json"
        self.log_path = self.output_dir / "neural_runtime.log"
        self._process: subprocess.Popen[str] | None = None
        self._log_handle: Any | None = None
        self._static_batch: NeuralRuntimeBatch | None = None
        self._timed_out = False
        self._started_at = time.monotonic()
        self._effective_timeout = float(timeout_seconds)

        config = load_neural_training_config(self.root)
        if not bool(config.get("enabled", False)):
            self._static_batch = NeuralRuntimeBatch("disabled", "Neural runtime is disabled.", ())
            return
        if not bool(config.get("production_ownership_enabled", False)):
            self._static_batch = NeuralRuntimeBatch(
                "shadow_only",
                "Neural runtime ownership is disabled.",
                (),
            )
            return
        if not audio_by_row_id:
            self._static_batch = NeuralRuntimeBatch("skipped", "No audio files were supplied.", ())
            return

        request_payload = {
            "schema_version": 1,
            "source_name_policy": "paths are I/O metadata only; audio waveforms are the only model input",
            "items": [
                {
                    "row_id": str(row_id),
                    "audio_path": str(Path(audio_path).expanduser().resolve()),
                }
                for row_id, audio_path in audio_by_row_id
            ],
        }
        self.request_path.write_text(
            json.dumps(request_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        base_timeout = max(60.0, float(config.get("runtime_batch_base_timeout_seconds", 300.0)))
        per_file_timeout = max(1.0, float(config.get("runtime_timeout_seconds_per_file", 45.0)))
        self._effective_timeout = max(
            float(timeout_seconds),
            base_timeout + per_file_timeout * len(audio_by_row_id),
        )
        command = [
            str(configured_neural_python(self.root)),
            str(self.root / "tools" / "predict_neural_audio.py"),
            "--project-root",
            str(self.root),
            "--request-json",
            str(self.request_path),
            "--output-json",
            str(self.response_path),
        ]
        try:
            self._log_handle = self.log_path.open("w", encoding="utf-8")
            self._log_handle.write(
                f"effective_timeout={self._effective_timeout:.1f}\n"
                f"command={' '.join(command)}\n\n"
            )
            self._log_handle.flush()
            self._process = subprocess.Popen(
                command,
                cwd=self.root,
                stdin=subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self._close_log_handle()
            self._static_batch = NeuralRuntimeBatch(
                "unavailable",
                str(exc),
                (),
                log_path=self.log_path,
            )

    @property
    def done(self) -> bool:
        """Return whether this worker can produce no additional checkpoints."""
        if self._static_batch is not None or self._timed_out:
            return True
        return self._process is None or self._process.poll() is not None

    def poll(self) -> NeuralRuntimeBatch:
        """Return the newest complete checkpoint without blocking."""
        if self._static_batch is not None:
            return self._static_batch
        if self._process is None:
            return NeuralRuntimeBatch(
                "unavailable",
                "Neural worker was not started.",
                (),
                report_path=self.response_path,
                log_path=self.log_path,
            )

        return_code = self._process.poll()
        elapsed = time.monotonic() - self._started_at
        if return_code is None and elapsed > self._effective_timeout:
            self._timed_out = True
            self._terminate_process()
            return _batch_from_checkpoint(
                self.response_path,
                self.log_path,
                status="partial_timeout",
                fallback_message=(
                    f"Neural runtime timed out after {self._effective_timeout:.0f} seconds; "
                    "completed rows were preserved."
                ),
            )
        if return_code is None:
            return _batch_from_checkpoint(
                self.response_path,
                self.log_path,
                status="processing",
                fallback_message="Neural runtime is loading models or processing its first audio file.",
            )

        self._close_log_handle()
        if return_code == 0:
            return _batch_from_checkpoint(
                self.response_path,
                self.log_path,
                status="predicted",
                fallback_message="Neural prediction process produced no response file.",
            )
        return _batch_from_checkpoint(
            self.response_path,
            self.log_path,
            status="error",
            fallback_message=f"Neural prediction process failed with exit code {return_code}.",
        )

    def wait(self, *, poll_seconds: float = 0.10) -> NeuralRuntimeBatch:
        """Wait for the worker while preserving checkpoint visibility."""
        batch = self.poll()
        while not self.done:
            time.sleep(max(0.01, float(poll_seconds)))
            batch = self.poll()
        return self.poll() if batch.status == "processing" else batch

    def close(self) -> None:
        """Stop a live worker and release its log file."""
        self._terminate_process()
        self._close_log_handle()

    def _terminate_process(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=3.0)

    def _close_log_handle(self) -> None:
        if self._log_handle is None:
            return
        try:
            self._log_handle.flush()
            self._log_handle.close()
        finally:
            self._log_handle = None


def start_configured_neural_prediction_process(
    project_root: Path,
    audio_by_row_id: list[tuple[str, Path]],
    report_dir: Path,
    *,
    timeout_seconds: float = 1800.0,
) -> ConfiguredNeuralPredictionProcess:
    """Start one persistent neural worker for incremental GUI finalization."""
    return ConfiguredNeuralPredictionProcess(
        project_root,
        audio_by_row_id,
        report_dir,
        timeout_seconds=timeout_seconds,
    )


def run_configured_neural_predictions(
    project_root: Path,
    audio_by_row_id: list[tuple[str, Path]],
    report_dir: Path,
    *,
    timeout_seconds: float = 1800.0,
) -> NeuralRuntimeBatch:
    """Embed and classify GUI audio while preserving completed partial rows.

    The neural worker loads the large models once and checkpoints after every
    input file. The parent timeout scales with batch size. If the worker still
    times out or crashes, any completed rows are recovered from the checkpoint
    instead of turning the entire preview into ``No Result``.
    """
    root = Path(project_root).expanduser().resolve()
    config = load_neural_training_config(root)
    if not bool(config.get("enabled", False)):
        return NeuralRuntimeBatch("disabled", "Neural runtime is disabled.", ())
    if not bool(config.get("production_ownership_enabled", False)):
        return NeuralRuntimeBatch("shadow_only", "Neural runtime ownership is disabled.", ())
    if not audio_by_row_id:
        return NeuralRuntimeBatch("skipped", "No audio files were supplied.", ())

    output_dir = Path(report_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "neural_runtime_request.json"
    response_path = output_dir / "neural_runtime_predictions.json"
    log_path = output_dir / "neural_runtime.log"
    request_payload = {
        "schema_version": 1,
        "source_name_policy": "paths are I/O metadata only; audio waveforms are the only model input",
        "items": [
            {"row_id": str(row_id), "audio_path": str(Path(audio_path).expanduser().resolve())}
            for row_id, audio_path in audio_by_row_id
        ],
    }
    request_path.write_text(json.dumps(request_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    base_timeout = max(60.0, float(config.get("runtime_batch_base_timeout_seconds", 300.0)))
    per_file_timeout = max(1.0, float(config.get("runtime_timeout_seconds_per_file", 45.0)))
    effective_timeout = max(float(timeout_seconds), base_timeout + per_file_timeout * len(audio_by_row_id))
    command = [
        str(configured_neural_python(root)),
        str(root / "tools" / "predict_neural_audio.py"),
        "--project-root",
        str(root),
        "--request-json",
        str(request_path),
        "--output-json",
        str(response_path),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _subprocess_text(exc.stdout)
        stderr = _subprocess_text(exc.stderr)
        log_path.write_text(
            f"timeout_after={effective_timeout:.1f}\n\nSTDOUT\n{stdout}\n\nSTDERR\n{stderr}\n",
            encoding="utf-8",
        )
        return _batch_from_checkpoint(
            response_path,
            log_path,
            status="partial_timeout",
            fallback_message=(
                f"Neural runtime timed out after {effective_timeout:.0f} seconds; "
                "completed rows were preserved."
            ),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log_path.write_text(f"Neural prediction launch failed: {exc}\n", encoding="utf-8")
        return NeuralRuntimeBatch("unavailable", str(exc), (), log_path=log_path)

    log_path.write_text(
        f"returncode={completed.returncode}\n"
        f"effective_timeout={effective_timeout:.1f}\n\n"
        f"STDOUT\n{completed.stdout}\n\nSTDERR\n{completed.stderr}\n",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        return _batch_from_checkpoint(
            response_path,
            log_path,
            status="error",
            fallback_message=f"Neural prediction process failed with exit code {completed.returncode}.",
        )
    return _batch_from_checkpoint(
        response_path,
        log_path,
        status="predicted",
        fallback_message="Neural prediction process produced no response file.",
    )


def _batch_from_checkpoint(
    response_path: Path,
    log_path: Path,
    *,
    status: str,
    fallback_message: str,
) -> NeuralRuntimeBatch:
    """Parse a final or partial worker checkpoint without discarding good rows."""
    if not response_path.is_file():
        return NeuralRuntimeBatch(status, fallback_message, (), report_path=response_path, log_path=log_path)
    try:
        payload = json.loads(response_path.read_text(encoding="utf-8"))
        predictions = tuple(_prediction_from_mapping(row) for row in payload.get("predictions", []))
        raw_errors = payload.get("row_errors", {})
        row_errors = (
            {str(row_id): str(message) for row_id, message in raw_errors.items()}
            if isinstance(raw_errors, dict)
            else {}
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return NeuralRuntimeBatch(
            "error",
            f"Neural prediction response was invalid: {exc}",
            (),
            report_path=response_path,
            log_path=log_path,
        )
    payload_status = str(payload.get("status", status))
    if status in {"partial_timeout", "error"}:
        payload_status = status
    return NeuralRuntimeBatch(
        status=payload_status,
        message=str(payload.get("message", "")) or fallback_message,
        predictions=predictions,
        index_path=str(payload.get("index_path", "")),
        report_path=response_path,
        log_path=log_path,
        row_errors=row_errors,
    )


def _subprocess_text(value: str | bytes | None) -> str:
    """Normalize subprocess timeout output for readable logs."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def run_configured_neural_rebuild(
    project_root: Path,
    report_dir: Path,
    *,
    timeout_seconds: float = 3600.0,
) -> dict[str, Any]:
    """Rebuild versioned neural prototypes in the configured environment."""
    root = Path(project_root).expanduser().resolve()
    output_dir = Path(report_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / "neural_rebuild_result.json"
    log_path = output_dir / "neural_rebuild.log"
    command = [
        str(configured_neural_python(root)),
        str(root / "tools" / "train_neural_from_gui_corrections.py"),
        "--project-root",
        str(root),
        "--output-json",
        str(response_path),
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    log_path.write_text(
        f"returncode={completed.returncode}\n\nSTDOUT\n{completed.stdout}\n\nSTDERR\n{completed.stderr}\n",
        encoding="utf-8",
    )
    if completed.returncode != 0 or not response_path.is_file():
        raise RuntimeError(f"neural rebuild failed with exit code {completed.returncode}; log: {log_path}")
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    build = payload.get("build")
    if not isinstance(build, dict):
        raise ValueError(f"neural rebuild response has no build summary: {response_path}")
    build["subprocess_log_path"] = str(log_path)
    return build


def _prediction_from_mapping(payload: Any) -> NeuralRuntimePrediction:
    if not isinstance(payload, dict):
        raise TypeError("prediction row must be an object")
    return NeuralRuntimePrediction(
        row_id=str(payload["row_id"]),
        file_sha256=str(payload["file_sha256"]),
        predicted_label=str(payload["predicted_label"]),
        second_label=str(payload.get("second_label", "")),
        top_similarity=float(payload["top_similarity"]),
        second_similarity=float(payload["second_similarity"]),
        margin=float(payload["margin"]),
        radius_ratio=float(payload["radius_ratio"]),
        known_distribution=bool(payload["known_distribution"]),
        label_example_count=int(payload.get("label_example_count", 0)),
        exact_training_match=bool(payload.get("exact_training_match", False)),
        ownership_ready=bool(payload.get("ownership_ready", False)),
        ownership_block_reason=str(payload.get("ownership_block_reason", "")),
        semantic_status=str(payload.get("semantic_status", "unavailable")),
        semantic_family=str(payload.get("semantic_family", "")),
        semantic_second_family=str(payload.get("semantic_second_family", "")),
        semantic_top_score=float(payload.get("semantic_top_score", 0.0)),
        semantic_second_score=float(payload.get("semantic_second_score", 0.0)),
        semantic_margin=float(payload.get("semantic_margin", 0.0)),
        semantic_family_scores={
            str(family): float(score) for family, score in dict(payload.get("semantic_family_scores", {})).items()
        },
        prompt_brain_status=str(payload.get("prompt_brain_status", "unavailable")),
        prompt_suggestions=tuple(
            _prompt_suggestion_from_mapping(suggestion) for suggestion in payload.get("prompt_suggestions", [])
        ),
        panns_status=str(payload.get("panns_status", "unavailable")),
        panns_model_id=str(payload.get("panns_model_id", "")),
        panns_events=tuple(_event_suggestion_from_mapping(event) for event in payload.get("panns_events", [])),
        panns_family_scores={
            str(family): float(score) for family, score in dict(payload.get("panns_family_scores", {})).items()
        },
        panns_support_score=float(payload.get("panns_support_score", 0.0)),
        panns_contradiction_score=float(payload.get("panns_contradiction_score", 0.0)),
        panns_supporting_events=tuple(
            _event_suggestion_from_mapping(event) for event in payload.get("panns_supporting_events", [])
        ),
        panns_contradicting_events=tuple(
            _event_suggestion_from_mapping(event) for event in payload.get("panns_contradicting_events", [])
        ),
    )


def _prompt_suggestion_from_mapping(payload: Any) -> NeuralPromptSuggestion:
    """Parse one detailed prompt suggestion from subprocess JSON."""
    if not isinstance(payload, dict):
        raise TypeError("prompt suggestion must be an object")
    return NeuralPromptSuggestion(
        path=str(payload["path"]),
        positive_score=float(payload["positive_score"]),
        negative_score=float(payload["negative_score"]),
        prompt_margin=float(payload["prompt_margin"]),
        top_positive_prompt=str(payload.get("top_positive_prompt", "")),
        top_positive_similarity=float(payload.get("top_positive_similarity", 0.0)),
        top_negative_prompt=str(payload.get("top_negative_prompt", "")),
        top_negative_similarity=float(payload.get("top_negative_similarity", 0.0)),
    )


def _event_suggestion_from_mapping(payload: Any) -> NeuralEventSuggestion:
    """Parse one PANNs AudioSet event from subprocess JSON."""
    if not isinstance(payload, dict):
        raise TypeError("PANNs event must be an object")
    return NeuralEventSuggestion(label=str(payload["label"]), score=float(payload["score"]))
