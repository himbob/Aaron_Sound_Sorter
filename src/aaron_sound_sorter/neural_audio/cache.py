"""Content-addressed embedding cache with atomic writes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .contracts import EmbeddingRecord
from .hashing import sha256_file
from .providers.base import EmbeddingProvider


def _safe_component(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return value[:120] or "unknown"


class EmbeddingCache:
    """Cache embeddings by model identity and file bytes, never by filename."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _entry_dir(self, provider: EmbeddingProvider, file_sha256: str) -> Path:
        identity_digest = _identity_digest(provider.cache_identity)
        return self.root / _safe_component(provider.provider_id) / identity_digest / file_sha256[:2]

    def _path(self, provider: EmbeddingProvider, file_sha256: str) -> Path:
        folder = self._entry_dir(provider, file_sha256)
        return folder / f"{file_sha256}.npz"

    def load(self, provider: EmbeddingProvider, file_sha256: str) -> EmbeddingRecord | None:
        entry_path = self._path(provider, file_sha256)
        if not entry_path.exists():
            return None
        try:
            with np.load(entry_path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"].item()))
                vector = np.asarray(payload["vector"], dtype=np.float32)
            if metadata.get("cache_identity") != dict(provider.cache_identity):
                return None
            if metadata.get("provider_id") != provider.provider_id or metadata.get("model_id") != provider.model_id:
                return None
            return EmbeddingRecord(
                provider_id=str(metadata["provider_id"]),
                model_id=str(metadata["model_id"]),
                file_sha256=str(metadata["file_sha256"]),
                vector=vector,
                segment_count=int(metadata.get("segment_count", 1)),
                sample_rate=int(metadata.get("sample_rate", 0)),
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return None

    def save(self, record: EmbeddingRecord, provider: EmbeddingProvider) -> None:
        """Atomically save one vector and its complete cache identity."""
        entry_path = self._path(provider, record.file_sha256)
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "cache_file_schema_version": 2,
            "provider_id": record.provider_id,
            "model_id": record.model_id,
            "file_sha256": record.file_sha256,
            "dimension": record.dimension,
            "segment_count": record.segment_count,
            "sample_rate": record.sample_rate,
            "cache_identity": dict(provider.cache_identity),
        }
        with tempfile.NamedTemporaryFile(dir=entry_path.parent, suffix=".npz", delete=False) as handle:
            temp_entry = Path(handle.name)
        try:
            with temp_entry.open("wb") as handle:
                np.savez(
                    handle,
                    vector=record.vector,
                    metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_entry, entry_path)
        finally:
            temp_entry.unlink(missing_ok=True)

    def get_or_compute(self, path: Path, provider: EmbeddingProvider) -> EmbeddingRecord:
        file_sha256 = sha256_file(Path(path))
        cached = self.load(provider, file_sha256)
        if cached is not None:
            return cached
        record = provider.embed_file(Path(path), file_sha256=file_sha256)
        self.save(record, provider)
        return record


def _identity_digest(identity: Mapping[str, str | int]) -> str:
    payload = json.dumps(dict(identity), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
