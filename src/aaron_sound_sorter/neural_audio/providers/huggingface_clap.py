"""Optional Hugging Face CLAP embedding provider.

The dependency is imported lazily so the legacy sorter can run without PyTorch
or Transformers.  Network download is disabled by default; production builds
should pin and prefetch a reviewed model snapshot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..contracts import EmbeddingRecord, l2_normalize
from ..hashing import sha256_file
from .identity import ModelIdentity, resolve_model_identity

DEFAULT_CLAP_MODEL = "laion/larger_clap_music_and_speech"
CLAP_PREPROCESSING_VERSION = "clap-mono-processor-rate-three-position-segments-v2"
EMBEDDING_SCHEMA_VERSION = 1


@dataclass
class HuggingFaceClapProvider:
    """Extract deterministic pooled CLAP audio embeddings."""

    model_name_or_path: str = DEFAULT_CLAP_MODEL
    source_model_id: str = DEFAULT_CLAP_MODEL
    model_revision: str = ""
    device: str = "auto"
    allow_network: bool = False
    max_segment_seconds: float = 10.0
    max_segments: int = 3
    text_batch_size: int = 64

    _processor: Any = None
    _model: Any = None
    _resolved_device: str = ""
    _identity: ModelIdentity | None = field(default=None, init=False, repr=False)

    @property
    def provider_id(self) -> str:
        return "hf_clap"

    @property
    def model_id(self) -> str:
        return self._model_identity().model_id

    @property
    def cache_identity(self) -> Mapping[str, str | int]:
        """Return the complete model/preprocessing cache identity."""
        return self._model_identity().cache_identity(self.provider_id)

    def _model_identity(self) -> ModelIdentity:
        if self._identity is None:
            self._identity = resolve_model_identity(
                self.model_name_or_path,
                source_model_id=self.source_model_id,
                model_revision=self.model_revision,
                preprocessing_version=(
                    f"{CLAP_PREPROCESSING_VERSION}"
                    f":max_seconds={self.max_segment_seconds:g}"
                    f":max_segments={self.max_segments}"
                ),
                embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
            )
        return self._identity

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "CLAP support requires the optional 'neural' dependencies: torch and transformers"
            ) from exc

        if self.device == "auto":
            if torch.cuda.is_available():
                resolved = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                resolved = "mps"
            else:
                resolved = "cpu"
        else:
            resolved = self.device

        local_only = not self.allow_network
        model_path = Path(self.model_name_or_path).expanduser()
        revision_kwargs = {"revision": self.model_revision} if self.model_revision and not model_path.is_dir() else {}
        self._processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            local_files_only=local_only,
            **revision_kwargs,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name_or_path,
            local_files_only=local_only,
            **revision_kwargs,
        )
        self._model.eval()
        self._model.to(resolved)
        self._resolved_device = resolved

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 2:
            arr = np.mean(arr, axis=1)
        if arr.ndim != 1:
            raise ValueError(f"unsupported audio shape: {arr.shape}")
        return arr

    @staticmethod
    def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate:
            return audio.astype(np.float32, copy=False)
        try:
            import librosa
        except ImportError as exc:
            raise RuntimeError("CLAP resampling requires librosa") from exc
        return librosa.resample(audio, orig_sr=source_rate, target_sr=target_rate).astype(np.float32)

    def _segments(self, audio: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        max_samples = max(1, int(round(self.max_segment_seconds * sample_rate)))
        if audio.size <= max_samples:
            return [audio]
        candidate_starts = [0, max(0, (audio.size - max_samples) // 2), max(0, audio.size - max_samples)]
        starts = candidate_starts[: max(1, self.max_segments)]
        return [audio[start : start + max_samples] for start in sorted(set(starts))]

    def embed_file(self, path: Path, *, file_sha256: str | None = None) -> EmbeddingRecord:
        self._load()
        import torch

        path = Path(path)
        audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
        mono = self._to_mono(audio)
        target_rate = int(getattr(self._processor.feature_extractor, "sampling_rate", 48000))
        mono = self._resample(mono, int(sample_rate), target_rate)
        segments = self._segments(mono, target_rate)

        vectors: list[np.ndarray] = []
        with torch.inference_mode():
            for segment in segments:
                inputs = self._processor(audio=segment, sampling_rate=target_rate, return_tensors="pt")
                inputs = {name: value.to(self._resolved_device) for name, value in inputs.items()}
                if not hasattr(self._model, "get_audio_features"):
                    raise RuntimeError(f"model does not expose get_audio_features(): {self.model_name_or_path}")
                output = self._model.get_audio_features(**inputs)
                vector = output.detach().float().cpu().numpy().reshape(-1)
                vectors.append(l2_normalize(vector))

        pooled = l2_normalize(np.mean(np.vstack(vectors), axis=0))
        return EmbeddingRecord(
            provider_id=self.provider_id,
            model_id=self.model_id,
            file_sha256=file_sha256 or sha256_file(path),
            vector=pooled,
            segment_count=len(vectors),
            sample_rate=target_rate,
        )

    def embed_texts(self, prompts: Sequence[str]) -> np.ndarray:
        """Return normalized CLAP text embeddings for semantic audio prompts.

        Args:
            prompts: Non-empty natural-language descriptions of audible sound
                content. Paths and filenames must never be included.

        Returns:
            A two-dimensional array containing one normalized embedding per
            prompt, in input order.

        Raises:
            ValueError: If no usable prompt is supplied.
            RuntimeError: If the configured CLAP model lacks text features.

        Side Effects:
            Lazily loads the configured local CLAP model.

        Important Constraints:
            Prompts describe audible concepts only. This method must not encode
            source filenames, folder names, or sample-pack metadata.
        """
        cleaned_prompts = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
        if not cleaned_prompts:
            raise ValueError("at least one semantic audio prompt is required")
        self._load()
        import torch

        if not hasattr(self._model, "get_text_features"):
            raise RuntimeError(f"model does not expose get_text_features(): {self.model_name_or_path}")
        if self.text_batch_size < 1:
            raise ValueError("text_batch_size must be positive")
        normalized_rows: list[np.ndarray] = []
        for start in range(0, len(cleaned_prompts), self.text_batch_size):
            prompt_batch = cleaned_prompts[start : start + self.text_batch_size]
            inputs = self._processor(text=prompt_batch, padding=True, return_tensors="pt")
            inputs = {name: value.to(self._resolved_device) for name, value in inputs.items()}
            with torch.inference_mode():
                output = self._model.get_text_features(**inputs)
            rows = output.detach().float().cpu().numpy()
            normalized_rows.extend(l2_normalize(row) for row in rows)
        return np.vstack(normalized_rows).astype(np.float32)
