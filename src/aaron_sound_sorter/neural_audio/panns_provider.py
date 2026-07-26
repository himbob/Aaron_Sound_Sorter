"""Optional pinned offline PANNs AudioSet broad-event provider."""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import tempfile
import types
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from .hashing import sha256_file
from .panns_mapping import PannsEventScore

DEFAULT_PANNS_CONFIG = Path("config/panns_runtime.json")
PANNS_PREPROCESSING_VERSION = "mono-32khz-three-position-segments-v1"


class PannsBackend(Protocol):
    """Minimal inference backend implemented by the official PANNs package."""

    labels: tuple[str, ...]

    def infer(self, waveform_batch: np.ndarray) -> np.ndarray:
        """Return AudioSet clip scores with shape ``(batch, classes)``."""


@dataclass(frozen=True)
class PannsPrediction:
    """One source-name-blind broad AudioSet prediction."""

    model_id: str
    audio_sha256: str
    top_events: tuple[PannsEventScore, ...]
    segment_count: int


@dataclass
class PannsProvider:
    """Run a checksummed local PANNs Cnn14 checkpoint without network access.

    Args:
        checkpoint_path: Local official Cnn14 checkpoint.
        labels_path: Local official AudioSet label CSV.
        checkpoint_sha256: Required checkpoint digest.
        labels_sha256: Required label digest.
        source_model_id: Human-readable upstream model identity.
        source_revision: Pinned upstream code/checkpoint revision label.
        sample_rate: PANNs model sample rate.
        max_segment_seconds: Maximum duration of one inference segment.
        max_segments: Number of evenly positioned segments for long audio.
        device: ``auto``, ``cpu``, or ``cuda``.

    Side Effects:
        Lazily loads a large local PyTorch checkpoint. It never downloads.
    """

    checkpoint_path: Path
    labels_path: Path
    checkpoint_sha256: str
    labels_sha256: str
    source_model_id: str
    source_revision: str
    sample_rate: int = 32000
    max_segment_seconds: float = 10.0
    max_segments: int = 3
    device: str = "auto"
    _backend: PannsBackend | None = field(default=None, init=False, repr=False)

    @property
    def model_id(self) -> str:
        """Return the full checkpoint and preprocessing identity."""
        return (
            f"{self.source_model_id}@{self.source_revision}"
            f"#checkpoint={self.checkpoint_sha256}"
            f"#preprocess={PANNS_PREPROCESSING_VERSION}:segments={self.max_segments}"
        )

    def predict_file(self, path: Path, *, top_k: int = 10) -> PannsPrediction:
        """Return broad events from decoded audio waveform evidence only.

        Args:
            path: Operational audio locator. Its text is never model input.
            top_k: Maximum number of ranked AudioSet events.

        Returns:
            Pinned model identity, content hash, and raw event scores.

        Raises:
            FileNotFoundError: If required offline model artifacts are absent.
            RuntimeError: If the optional PANNs package cannot load.
            ValueError: If checksums, audio shape, or configuration are invalid.

        Side Effects:
            Lazily loads PANNs and reads the audio file.
        """
        if top_k < 1:
            raise ValueError("top_k must be positive")
        backend = self._load_backend()
        audio_path = Path(path).expanduser().resolve()
        waveform, source_rate = sf.read(audio_path, always_2d=False, dtype="float32")
        mono = self._to_mono(waveform)
        resampled = self._resample(mono, int(source_rate))
        segments = self._segments(resampled)
        segment_scores = backend.infer(np.stack(segments, axis=0))
        if segment_scores.ndim != 2 or segment_scores.shape != (len(segments), len(backend.labels)):
            raise ValueError("PANNs backend returned an invalid score matrix")
        pooled_scores = np.max(segment_scores, axis=0)
        top_indexes = np.argsort(-pooled_scores, kind="stable")[:top_k]
        events = tuple(PannsEventScore(backend.labels[index], float(pooled_scores[index])) for index in top_indexes)
        return PannsPrediction(self.model_id, sha256_file(audio_path), events, len(segments))

    def _load_backend(self) -> PannsBackend:
        if self._backend is not None:
            return self._backend
        self._verify_artifact(self.checkpoint_path, self.checkpoint_sha256, "checkpoint")
        self._verify_artifact(self.labels_path, self.labels_sha256, "labels")
        labels = self._load_labels()
        self._backend = OfficialPannsBackend(
            checkpoint_path=self.checkpoint_path,
            labels=labels,
            device=self._resolved_device(),
        )
        return self._backend

    def _load_labels(self) -> tuple[str, ...]:
        with self.labels_path.open("r", encoding="utf-8", newline="") as handle:
            labels = tuple(str(row.get("display_name", "")).strip() for row in csv.DictReader(handle))
        if len(labels) != 527 or any(not label for label in labels):
            raise ValueError("PANNs AudioSet label file must contain 527 named classes")
        return labels

    def _resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PANNs requires the optional neural dependencies") from exc
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _verify_artifact(self, path: Path, expected_sha256: str, label: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"local PANNs {label} is missing: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"local PANNs {label} checksum mismatch: {path}")

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        waveform = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if waveform.ndim == 2:
            waveform = np.mean(waveform, axis=1)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError(f"unsupported PANNs audio shape: {waveform.shape}")
        return waveform

    def _resample(self, audio: np.ndarray, source_rate: int) -> np.ndarray:
        if source_rate == self.sample_rate:
            return audio.astype(np.float32, copy=False)
        try:
            import librosa
        except ImportError as exc:
            raise RuntimeError("PANNs resampling requires librosa") from exc
        return librosa.resample(audio, orig_sr=source_rate, target_sr=self.sample_rate).astype(np.float32)

    def _segments(self, audio: np.ndarray) -> list[np.ndarray]:
        if self.max_segments < 1 or self.max_segment_seconds <= 0:
            raise ValueError("PANNs segment configuration must be positive")
        segment_samples = max(1, int(round(self.sample_rate * self.max_segment_seconds)))
        if audio.size <= segment_samples:
            return [audio]
        evenly_spaced = np.linspace(0, audio.size - segment_samples, num=self.max_segments, dtype=np.int64)
        return [audio[start : start + segment_samples] for start in sorted(set(evenly_spaced.tolist()))]


class OfficialPannsBackend:
    """Compatibility wrapper around the official ``panns-inference`` wheel."""

    def __init__(self, *, checkpoint_path: Path, labels: tuple[str, ...], device: str) -> None:
        self.labels = labels
        self._install_local_config_shim(labels)
        try:
            from panns_inference import AudioTagging
        except ImportError as exc:
            raise RuntimeError("PANNs requires panns-inference==0.1.1") from exc
        # ``panns-inference==0.1.1`` predates PyTorch 2.6, where ``torch.load``
        # changed its default to ``weights_only=True``. The official PANNs wheel
        # does not pass that argument, so modern PyTorch can reject the pinned
        # legacy checkpoint before inference starts. The provider verifies the
        # exact checkpoint SHA-256 before constructing this backend, so permit
        # the legacy full checkpoint load only for this narrow construction
        # window, then restore the caller's environment unchanged.
        previous_legacy_load = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        try:
            with redirect_stdout(io.StringIO()):
                self._model = AudioTagging(checkpoint_path=str(checkpoint_path), device=device)
        finally:
            if previous_legacy_load is None:
                os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous_legacy_load

    def infer(self, waveform_batch: np.ndarray) -> np.ndarray:
        """Run deterministic clip-level AudioSet tagging."""
        clipwise_output, _embedding = self._model.inference(np.asarray(waveform_batch, dtype=np.float32))
        return np.asarray(clipwise_output, dtype=np.float32)

    @staticmethod
    def _install_local_config_shim(labels: tuple[str, ...]) -> None:
        """Prevent the old wheel from writing/downloading under the user home."""
        config_module = types.ModuleType("panns_inference.config")
        config_module.labels = list(labels)
        config_module.classes_num = len(labels)
        config_module.lb_to_ix = {label: index for index, label in enumerate(labels)}
        config_module.ix_to_lb = {index: label for index, label in enumerate(labels)}
        sys.modules["panns_inference.config"] = config_module
        cache_dir = Path(tempfile.gettempdir()) / "aaron_sound_sorter_matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def configured_panns_provider(project_root: Path) -> PannsProvider:
    """Build the optional checksummed PANNs provider from repository config."""
    root = Path(project_root).expanduser().resolve()
    config = json.loads((root / DEFAULT_PANNS_CONFIG).read_text(encoding="utf-8"))
    if not bool(config.get("enabled", False)):
        raise ValueError("PANNs runtime is disabled")
    if bool(config.get("production_ownership_enabled", False)):
        raise ValueError("PANNs detailed production ownership is not permitted")
    return PannsProvider(
        checkpoint_path=root / str(config["checkpoint_path"]),
        labels_path=root / str(config["labels_path"]),
        checkpoint_sha256=str(config["checkpoint_sha256"]),
        labels_sha256=str(config["labels_sha256"]),
        source_model_id=str(config["source_model_id"]),
        source_revision=str(config["source_revision"]),
        sample_rate=int(config.get("sample_rate", 32000)),
        max_segment_seconds=float(config.get("max_segment_seconds", 10.0)),
        max_segments=int(config.get("max_segments", 3)),
        device=str(config.get("device", "auto")),
    )
