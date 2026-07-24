"""Embedding-provider interface."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..contracts import EmbeddingRecord


@runtime_checkable
class EmbeddingProvider(Protocol):
    """A frozen audio encoder that produces source-name-blind vectors."""

    @property
    def provider_id(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    @property
    def cache_identity(self) -> Mapping[str, str | int]: ...

    def embed_file(self, path: Path, *, file_sha256: str | None = None) -> EmbeddingRecord: ...
