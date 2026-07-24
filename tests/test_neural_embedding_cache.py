from __future__ import annotations

from pathlib import Path

import numpy as np

from aaron_sound_sorter.neural_audio.cache import EmbeddingCache
from aaron_sound_sorter.neural_audio.contracts import EmbeddingRecord


class FakeProvider:
    provider_id = "fake"
    model_id = "fake-v1"

    def __init__(self, preprocessing_version: str = "fake-preprocess-v1") -> None:
        self.calls = 0
        self.preprocessing_version = preprocessing_version

    @property
    def cache_identity(self):
        return {
            "provider_id": self.provider_id,
            "source_model_id": self.model_id,
            "model_revision": "abc123",
            "model_snapshot_sha256": "f" * 64,
            "preprocessing_version": self.preprocessing_version,
            "embedding_schema_version": 1,
        }

    def embed_file(self, path: Path, *, file_sha256: str | None = None) -> EmbeddingRecord:
        self.calls += 1
        return EmbeddingRecord(self.provider_id, self.model_id, file_sha256 or "0" * 64, np.array([1.0, 0.0]))


def test_cache_keys_by_bytes_not_filename(tmp_path: Path) -> None:
    first = tmp_path / "misleading_vocal_name.wav"
    second = tmp_path / "totally_different_name.wav"
    first.write_bytes(b"same audio bytes")
    second.write_bytes(b"same audio bytes")
    provider = FakeProvider()
    cache = EmbeddingCache(tmp_path / "cache")

    record_a = cache.get_or_compute(first, provider)
    record_b = cache.get_or_compute(second, provider)

    assert provider.calls == 1
    assert record_a.file_sha256 == record_b.file_sha256
    assert np.allclose(record_a.vector, record_b.vector)


def test_cache_invalidates_when_preprocessing_contract_changes(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"same source bytes")
    cache = EmbeddingCache(tmp_path / "cache")
    first_provider = FakeProvider("fake-preprocess-v1")
    second_provider = FakeProvider("fake-preprocess-v2")

    cache.get_or_compute(audio, first_provider)
    cache.get_or_compute(audio, second_provider)

    assert first_provider.calls == 1
    assert second_provider.calls == 1
    assert len(list((tmp_path / "cache" / "fake").glob("*/*/*.npz"))) == 2
