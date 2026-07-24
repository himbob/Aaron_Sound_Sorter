"""Stable model and preprocessing identities for neural embedding caches."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

MODEL_MANIFEST_NAME = "neural_model_manifest.json"


@dataclass(frozen=True)
class ModelIdentity:
    """Resolved identity of one frozen encoder and preprocessing contract."""

    source_model_id: str
    model_revision: str
    snapshot_sha256: str
    preprocessing_version: str
    embedding_schema_version: int

    @property
    def model_id(self) -> str:
        """Return a stable model identifier suitable for index compatibility."""
        revision = self.model_revision or "local"
        return (
            f"{self.source_model_id}@{revision}"
            f"#snapshot={self.snapshot_sha256}"
            f"#preprocess={self.preprocessing_version}"
            f"#embedding_schema={self.embedding_schema_version}"
        )

    def cache_identity(self, provider_id: str) -> Mapping[str, str | int]:
        """Return explicit cache-key fields required by the architecture."""
        return {
            "provider_id": provider_id,
            "source_model_id": self.source_model_id,
            "model_revision": self.model_revision,
            "model_snapshot_sha256": self.snapshot_sha256,
            "preprocessing_version": self.preprocessing_version,
            "embedding_schema_version": self.embedding_schema_version,
        }


def resolve_model_identity(
    model_name_or_path: str,
    *,
    source_model_id: str,
    model_revision: str,
    preprocessing_version: str,
    embedding_schema_version: int,
) -> ModelIdentity:
    """Resolve a model identity from a local snapshot or pinned remote revision.

    Args:
        model_name_or_path: Local snapshot directory or Hugging Face model ID.
        source_model_id: Stable upstream model identifier.
        model_revision: Pinned upstream commit hash. May be read from a local
            model manifest when omitted.
        preprocessing_version: Versioned audio preparation contract.
        embedding_schema_version: Version of the stored embedding semantics.

    Returns:
        Stable identity containing an actual local snapshot hash or the pinned
        remote revision.

    Raises:
        ValueError: If a remote model is not pinned or identity fields are empty.
    """
    model_path = Path(model_name_or_path).expanduser()
    manifest: Mapping[str, object] = {}
    if model_path.is_dir():
        manifest_path = model_path / MODEL_MANIFEST_NAME
        if manifest_path.is_file():
            raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(raw_manifest, dict):
                manifest = raw_manifest
        resolved_source = str(manifest.get("source_model_id") or source_model_id).strip()
        resolved_revision = str(manifest.get("model_revision") or model_revision).strip()
        snapshot_sha256 = hash_model_snapshot(model_path)
    else:
        resolved_source = source_model_id.strip() or model_name_or_path.strip()
        resolved_revision = model_revision.strip()
        if not resolved_revision:
            raise ValueError("remote neural models require a pinned --model-revision")
        snapshot_sha256 = resolved_revision

    if not resolved_source:
        raise ValueError("source_model_id is required")
    if not preprocessing_version.strip():
        raise ValueError("preprocessing_version is required")
    if embedding_schema_version < 1:
        raise ValueError("embedding_schema_version must be positive")
    return ModelIdentity(
        source_model_id=resolved_source,
        model_revision=resolved_revision,
        snapshot_sha256=snapshot_sha256,
        preprocessing_version=preprocessing_version,
        embedding_schema_version=embedding_schema_version,
    )


def hash_model_snapshot(model_directory: Path) -> str:
    """Hash all non-hidden model snapshot files except the identity manifest."""
    model_directory = Path(model_directory)
    files = sorted(
        path
        for path in model_directory.rglob("*")
        if path.is_file()
        and path.name != MODEL_MANIFEST_NAME
        and not any(part.startswith(".") for part in path.relative_to(model_directory).parts)
    )
    if not files:
        raise ValueError(f"local model snapshot contains no files: {model_directory}")
    digest = hashlib.sha256()
    digest.update(b"aaron-model-snapshot-v1\0")
    for path in files:
        relative = path.relative_to(model_directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, byteorder="little", signed=False))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, byteorder="little", signed=False))
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()
