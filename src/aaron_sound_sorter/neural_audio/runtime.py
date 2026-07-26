"""Subprocess boundary for source-name-blind neural GUI inference and training."""

from __future__ import annotations

import json
import subprocess
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


def run_configured_neural_predictions(
    project_root: Path,
    audio_by_row_id: list[tuple[str, Path]],
    report_dir: Path,
    *,
    timeout_seconds: float = 1800.0,
) -> NeuralRuntimeBatch:
    """Embed and classify a GUI batch in the isolated neural environment.

    Paths cross this boundary only so the subprocess can read audio bytes.
    The model receives decoded waveforms, and results are joined by GUI row ID.
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
    try:
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
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log_path.write_text(f"Neural prediction launch failed: {exc}\n", encoding="utf-8")
        return NeuralRuntimeBatch("unavailable", str(exc), (), log_path=log_path)

    log_path.write_text(
        f"returncode={completed.returncode}\n\nSTDOUT\n{completed.stdout}\n\nSTDERR\n{completed.stderr}\n",
        encoding="utf-8",
    )
    if completed.returncode != 0 or not response_path.is_file():
        message = f"Neural prediction process failed with exit code {completed.returncode}."
        return NeuralRuntimeBatch("error", message, (), report_path=response_path, log_path=log_path)
    try:
        payload = json.loads(response_path.read_text(encoding="utf-8"))
        predictions = tuple(_prediction_from_mapping(row) for row in payload.get("predictions", []))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return NeuralRuntimeBatch(
            "error",
            f"Neural prediction response was invalid: {exc}",
            (),
            report_path=response_path,
            log_path=log_path,
        )
    return NeuralRuntimeBatch(
        status=str(payload.get("status", "built")),
        message=str(payload.get("message", "")),
        predictions=predictions,
        index_path=str(payload.get("index_path", "")),
        report_path=response_path,
        log_path=log_path,
    )


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
