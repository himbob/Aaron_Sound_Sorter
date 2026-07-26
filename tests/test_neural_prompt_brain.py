from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aaron_sound_sorter.neural_audio.contracts import EmbeddingRecord
from aaron_sound_sorter.neural_audio.prompt_brain import ClapPromptIndex
from aaron_sound_sorter.taxonomy_registry import TaxonomyRegistry
from tools.build_neural_taxonomy_prompts import build_prompt_catalog_payload


class FakeTextProvider:
    model_id = "fake_clap_model"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed_texts(self, prompts: tuple[str, ...]) -> np.ndarray:
        return np.asarray([self.vectors[prompt] for prompt in prompts], dtype=np.float32)


def _catalog(path: Path) -> Path:
    payload = {
        "taxonomy_version": "test",
        "prompt_catalog_version": "test-v1",
        "production_ownership_enabled": False,
        "categories": [
            {
                "path": "Instruments/Voice/Vocal Loops/Loops",
                "positive_prompts": ["human singing", "rap vocals"],
                "negative_prompts": ["a saxophone"],
                "broad_family": "Instruments/Voice",
                "structure": "Loops",
                "review_status": "reviewed",
            },
            {
                "path": "Instruments/Woodwinds/Saxophone/Loops",
                "positive_prompts": ["a saxophone", "a reed instrument"],
                "negative_prompts": ["human singing"],
                "broad_family": "Instruments/Woodwinds",
                "structure": "Loops",
                "review_status": "reviewed",
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_prompt_brain_keeps_positive_and_negative_evidence_visible(tmp_path: Path) -> None:
    provider = FakeTextProvider(
        {
            "human singing": [1.0, 0.0],
            "rap vocals": [0.9, 0.1],
            "a saxophone": [0.0, 1.0],
            "a reed instrument": [0.1, 0.9],
        }
    )
    index = ClapPromptIndex.build(_catalog(tmp_path / "prompts.json"), provider)
    record = EmbeddingRecord(
        provider_id="hf_clap",
        model_id=provider.model_id,
        file_sha256="a" * 64,
        vector=np.asarray([1.0, 0.0], dtype=np.float32),
    )

    scores = index.predict(record, top_k=2)

    assert scores[0].path == "Instruments/Voice/Vocal Loops/Loops"
    assert scores[0].top_positive_prompt == "human singing"
    assert scores[0].top_negative_prompt == "a saxophone"
    assert scores[0].prompt_margin > scores[1].prompt_margin
    assert index.metadata.production_ownership_enabled is False


def test_generated_negative_prompt_does_not_override_stronger_positive_support(tmp_path: Path) -> None:
    provider = FakeTextProvider(
        {
            "human singing": [1.0, 0.0],
            "rap vocals": [1.0, 0.0],
            "a saxophone": [0.9, 0.1],
            "a reed instrument": [0.6, 0.8],
        }
    )
    index = ClapPromptIndex.build(_catalog(tmp_path / "prompts.json"), provider)
    record = EmbeddingRecord(
        provider_id="hf_clap",
        model_id=provider.model_id,
        file_sha256="c" * 64,
        vector=np.asarray([1.0, 0.0], dtype=np.float32),
    )

    scores = index.predict(record, top_k=2)

    assert scores[0].path == "Instruments/Voice/Vocal Loops/Loops"
    assert scores[0].positive_score > scores[1].positive_score


def test_prompt_index_round_trip_is_deterministic(tmp_path: Path) -> None:
    provider = FakeTextProvider(
        {
            "human singing": [1.0, 0.0],
            "rap vocals": [0.9, 0.1],
            "a saxophone": [0.0, 1.0],
            "a reed instrument": [0.1, 0.9],
        }
    )
    index = ClapPromptIndex.build(_catalog(tmp_path / "prompts.json"), provider)
    output_dir = index.save(tmp_path / "index")

    loaded = ClapPromptIndex.load(output_dir)

    assert loaded.metadata == index.metadata
    assert loaded.categories == index.categories
    assert np.array_equal(loaded.positive_vectors, index.positive_vectors)
    assert np.array_equal(loaded.negative_vectors, index.negative_vectors)


def test_prompt_brain_rejects_model_identity_mismatch(tmp_path: Path) -> None:
    provider = FakeTextProvider(
        {
            "human singing": [1.0, 0.0],
            "rap vocals": [0.9, 0.1],
            "a saxophone": [0.0, 1.0],
            "a reed instrument": [0.1, 0.9],
        }
    )
    index = ClapPromptIndex.build(_catalog(tmp_path / "prompts.json"), provider)
    record = EmbeddingRecord(
        provider_id="hf_clap",
        model_id="another_model",
        file_sha256="b" * 64,
        vector=np.asarray([1.0, 0.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="model identities differ"):
        index.predict(record)


def test_repository_prompt_catalog_covers_every_canonical_category() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = TaxonomyRegistry.load(
        root / "config" / "canonical_taxonomy.json",
        root / "config" / "taxonomy_aliases.json",
    )
    payload = build_prompt_catalog_payload(registry)

    assert len(payload["categories"]) == len(registry.categories)
    assert payload["production_ownership_enabled"] is False
    assert all(category["positive_prompts"] for category in payload["categories"])
    assert all(category["negative_prompts"] for category in payload["categories"])
