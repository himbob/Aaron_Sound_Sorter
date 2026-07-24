"""Optional generic Hugging Face audio encoder for MERT/BEATs experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..contracts import EmbeddingRecord, l2_normalize
from ..hashing import sha256_file
from .identity import ModelIdentity, resolve_model_identity


@dataclass
class HuggingFaceAudioEncoderProvider:
    """Mean-pool hidden states from a frozen audio foundation model.

    This provider is intentionally experimental.  MERT currently requires
    ``trust_remote_code=True`` and its published model card is non-commercial;
    therefore neither setting is enabled silently.
    """

    provider_name: str
    model_name_or_path: str
    source_model_id: str = ""
    model_revision: str = ""
    layer: int = -1
    device: str = "auto"
    allow_network: bool = False
    allow_remote_code: bool = False
    max_segment_seconds: float = 5.0
    max_segments: int = 3

    _processor: Any = None
    _model: Any = None
    _resolved_device: str = ""
    _identity: ModelIdentity | None = field(default=None, init=False, repr=False)

    @property
    def provider_id(self) -> str:
        return self.provider_name

    @property
    def model_id(self) -> str:
        return self._model_identity().model_id

    @property
    def cache_identity(self) -> Mapping[str, str | int]:
        """Return the complete model/preprocessing cache identity."""
        return self._model_identity().cache_identity(self.provider_id)

    def _model_identity(self) -> ModelIdentity:
        if self._identity is None:
            preprocessing_version = (
                "hf-audio-mean-hidden-state-v1"
                f":layer={self.layer}"
                f":max_seconds={self.max_segment_seconds:g}"
                f":max_segments={self.max_segments}"
            )
            self._identity = resolve_model_identity(
                self.model_name_or_path,
                source_model_id=self.source_model_id or self.model_name_or_path,
                model_revision=self.model_revision,
                preprocessing_version=preprocessing_version,
                embedding_schema_version=1,
            )
        return self._identity

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoConfig, AutoFeatureExtractor, AutoModel
        except ImportError as exc:
            raise RuntimeError("audio encoder support requires torch and transformers") from exc

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
        self._processor = AutoFeatureExtractor.from_pretrained(
            self.model_name_or_path,
            local_files_only=local_only,
            trust_remote_code=self.allow_remote_code,
            **revision_kwargs,
        )
        config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            local_files_only=local_only,
            trust_remote_code=self.allow_remote_code,
            **revision_kwargs,
        )
        prepare_audio_model_config(config)
        self._model = AutoModel.from_pretrained(
            self.model_name_or_path,
            config=config,
            local_files_only=local_only,
            trust_remote_code=self.allow_remote_code,
            **revision_kwargs,
        )
        self._model.eval()
        self._model.to(resolved)
        self._resolved_device = resolved

    @staticmethod
    def _mono(audio: np.ndarray) -> np.ndarray:
        arr = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 2:
            arr = np.mean(arr, axis=1)
        if arr.ndim != 1:
            raise ValueError(f"unsupported audio shape: {arr.shape}")
        return arr

    @staticmethod
    def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate:
            return audio
        import librosa

        return librosa.resample(audio, orig_sr=source_rate, target_sr=target_rate).astype(np.float32)

    def _segments(self, audio: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        size = max(1, int(round(self.max_segment_seconds * sample_rate)))
        if audio.size <= size:
            return [audio]
        starts = [0, max(0, (audio.size - size) // 2), max(0, audio.size - size)]
        return [audio[start : start + size] for start in sorted(set(starts[: self.max_segments]))]

    def embed_file(self, path: Path, *, file_sha256: str | None = None) -> EmbeddingRecord:
        self._load()
        import torch

        path = Path(path)
        audio, source_rate = sf.read(path, always_2d=False, dtype="float32")
        mono = self._mono(audio)
        target_rate = int(getattr(self._processor, "sampling_rate", 16000))
        mono = self._resample(mono, int(source_rate), target_rate)

        vectors: list[np.ndarray] = []
        with torch.inference_mode():
            for segment in self._segments(mono, target_rate):
                inputs = self._processor(segment, sampling_rate=target_rate, return_tensors="pt")
                inputs = {name: value.to(self._resolved_device) for name, value in inputs.items()}
                output = self._model(**inputs, output_hidden_states=True)
                hidden_states = getattr(output, "hidden_states", None)
                if not hidden_states:
                    raise RuntimeError(f"model did not return hidden states: {self.model_name_or_path}")
                hidden = hidden_states[self.layer]
                pooled = hidden.mean(dim=1).detach().float().cpu().numpy().reshape(-1)
                vectors.append(l2_normalize(pooled))

        pooled_vector = l2_normalize(np.mean(np.vstack(vectors), axis=0))
        return EmbeddingRecord(
            provider_id=self.provider_id,
            model_id=self.model_id,
            file_sha256=file_sha256 or sha256_file(path),
            vector=pooled_vector,
            segment_count=len(vectors),
            sample_rate=target_rate,
        )


def prepare_audio_model_config(config: Any) -> Any:
    """Add safe compatibility defaults required by newer Transformers.

    Args:
        config: A Hugging Face audio-model configuration loaded from the
            pinned checkpoint.

    Returns:
        The same configuration object after compatibility defaults are added.

    Side Effects:
        Mutates ``config`` only when a newly required HuBERT setting is absent.

    Important Constraints:
        This does not change learned weights or category behavior.
        ``conv_pos_batch_norm=False`` is the historical HuBERT default used by
        MERT checkpoints published before Transformers added the field.
    """
    if str(getattr(config, "model_type", "")) == "mert_model" and not hasattr(config, "conv_pos_batch_norm"):
        config.conv_pos_batch_norm = False
    return config
