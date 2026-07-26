"""Read-only detailed CLAP taxonomy prompt brain."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .contracts import EmbeddingRecord, l2_normalize
from .hashing import sha256_file


class TextEmbeddingProvider(Protocol):
    """Minimal provider contract required to build a prompt index."""

    @property
    def model_id(self) -> str:
        """Return the pinned model identity."""

    def embed_texts(self, prompts: Sequence[str]) -> np.ndarray:
        """Embed audible prompts in input order."""


@dataclass(frozen=True)
class PromptCategoryDefinition:
    """One canonical category's reviewed or generated prompt ensemble.

    Args:
        path: Canonical taxonomy path.
        positive_prompts: Descriptions supporting this category.
        negative_prompts: Common audible confusions.
        broad_family: Parent source family.
        structure: Objective structure terminal.
        review_status: Human prompt-review state.
        production_ownership_enabled: Must remain false until held-out gates
            are passed.

    Side Effects:
        None.
    """

    path: str
    positive_prompts: tuple[str, ...]
    negative_prompts: tuple[str, ...]
    broad_family: str
    structure: str
    review_status: str
    production_ownership_enabled: bool = False


@dataclass(frozen=True)
class PromptCategoryScore:
    """One uncalibrated detailed category suggestion.

    Args:
        path: Canonical taxonomy path.
        positive_score: Mean of the two strongest positive prompt scores.
        negative_score: Strongest confusion-prompt score.
        prompt_margin: Positive score minus negative score.
        top_positive_prompt: Strongest supporting description.
        top_positive_similarity: Raw similarity for that description.
        top_negative_prompt: Strongest confusion description.
        top_negative_similarity: Raw similarity for that description.

    Side Effects:
        None.
    """

    path: str
    positive_score: float
    negative_score: float
    prompt_margin: float
    top_positive_prompt: str
    top_positive_similarity: float
    top_negative_prompt: str
    top_negative_similarity: float


@dataclass(frozen=True)
class PromptIndexMetadata:
    """Pinned identity and policy for a serialized prompt index."""

    schema_version: int
    model_id: str
    taxonomy_version: str
    prompt_catalog_version: str
    prompt_catalog_sha256: str
    category_count: int
    embedding_dimension: int
    source_name_policy: str
    production_ownership_enabled: bool = False


class ClapPromptIndex:
    """Deterministic positive/negative prompt embeddings by category.

    Args:
        categories: Definitions in canonical path order.
        positive_vectors: Normalized positive prompt matrix.
        positive_offsets: Category offsets into ``positive_vectors``.
        negative_vectors: Normalized negative prompt matrix.
        negative_offsets: Category offsets into ``negative_vectors``.
        metadata: Pinned index identity.

    Side Effects:
        None.
    """

    def __init__(
        self,
        *,
        categories: tuple[PromptCategoryDefinition, ...],
        positive_vectors: np.ndarray,
        positive_offsets: np.ndarray,
        negative_vectors: np.ndarray,
        negative_offsets: np.ndarray,
        metadata: PromptIndexMetadata,
    ) -> None:
        self.categories = categories
        self.positive_vectors = np.asarray(positive_vectors, dtype=np.float32)
        self.positive_offsets = np.asarray(positive_offsets, dtype=np.int64)
        self.negative_vectors = np.asarray(negative_vectors, dtype=np.float32)
        self.negative_offsets = np.asarray(negative_offsets, dtype=np.int64)
        self.metadata = metadata
        self._validate()

    @classmethod
    def build(
        cls,
        catalog_path: Path,
        provider: TextEmbeddingProvider,
    ) -> ClapPromptIndex:
        """Embed a prompt catalog with one pinned CLAP text encoder.

        Args:
            catalog_path: Versioned positive/negative prompt catalog.
            provider: Local text embedding provider.

        Returns:
            Deterministic read-only prompt index.

        Raises:
            ValueError: If the catalog is empty or enables prompt ownership.

        Side Effects:
            Reads the catalog and may lazily load the provider model.
        """
        resolved_catalog = Path(catalog_path).expanduser().resolve()
        payload = json.loads(resolved_catalog.read_text(encoding="utf-8"))
        if bool(payload.get("production_ownership_enabled", False)):
            raise ValueError("prompt brain production ownership must remain disabled")
        categories = _category_definitions(payload.get("categories", []))
        if not categories:
            raise ValueError("prompt catalog has no categories")
        all_prompts = [
            prompt for category in categories for prompt in (*category.positive_prompts, *category.negative_prompts)
        ]
        unique_prompts = tuple(dict.fromkeys(all_prompts))
        unique_vectors = np.asarray(provider.embed_texts(unique_prompts), dtype=np.float32)
        if unique_vectors.ndim != 2 or unique_vectors.shape[0] != len(unique_prompts):
            raise ValueError("prompt embedding provider returned an invalid matrix")
        vector_by_prompt = {prompt: unique_vectors[index] for index, prompt in enumerate(unique_prompts)}
        positive_vectors, positive_offsets = _prompt_matrix(categories, vector_by_prompt, positive=True)
        negative_vectors, negative_offsets = _prompt_matrix(categories, vector_by_prompt, positive=False)
        metadata = PromptIndexMetadata(
            schema_version=1,
            model_id=provider.model_id,
            taxonomy_version=str(payload.get("taxonomy_version", "")),
            prompt_catalog_version=str(payload.get("prompt_catalog_version", "")),
            prompt_catalog_sha256=sha256_file(resolved_catalog),
            category_count=len(categories),
            embedding_dimension=int(unique_vectors.shape[1]),
            source_name_policy="taxonomy prompts and audio embeddings only; source names forbidden",
            production_ownership_enabled=False,
        )
        return cls(
            categories=categories,
            positive_vectors=positive_vectors,
            positive_offsets=positive_offsets,
            negative_vectors=negative_vectors,
            negative_offsets=negative_offsets,
            metadata=metadata,
        )

    def predict(self, record: EmbeddingRecord, *, top_k: int = 5) -> tuple[PromptCategoryScore, ...]:
        """Return detailed read-only suggestions for one audio embedding.

        Args:
            record: Source-name-blind audio embedding from the same model.
            top_k: Maximum ranked categories returned.

        Returns:
        Scores ranked by positive prompt support. Confusion-prompt evidence is
            preserved for display and later calibration, but generated
            negatives do not control rank. Raw values are not probabilities.

        Raises:
            ValueError: If model identity, dimensions, or ``top_k`` are invalid.

        Side Effects:
            None.
        """
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if record.model_id != self.metadata.model_id:
            raise ValueError("audio and prompt index model identities differ")
        if record.dimension != self.metadata.embedding_dimension:
            raise ValueError("audio and prompt embedding dimensions differ")
        scores = [self._category_score(index, record.vector) for index in range(len(self.categories))]
        scores.sort(key=lambda row: (-row.positive_score, -row.prompt_margin, row.path))
        return tuple(scores[:top_k])

    def save(self, output_dir: Path) -> Path:
        """Serialize this prompt index to a new or empty directory."""
        destination = Path(output_dir).expanduser().resolve()
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"prompt index destination is not empty: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination / "prompt_vectors.npz",
            positive_vectors=self.positive_vectors,
            positive_offsets=self.positive_offsets,
            negative_vectors=self.negative_vectors,
            negative_offsets=self.negative_offsets,
        )
        payload = {
            "metadata": asdict(self.metadata),
            "categories": [asdict(category) for category in self.categories],
        }
        (destination / "prompt_index.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, index_dir: Path) -> ClapPromptIndex:
        """Load a serialized prompt index from disk."""
        source = Path(index_dir).expanduser().resolve()
        payload = json.loads((source / "prompt_index.json").read_text(encoding="utf-8"))
        arrays = np.load(source / "prompt_vectors.npz", allow_pickle=False)
        return cls(
            categories=_category_definitions(payload.get("categories", [])),
            positive_vectors=arrays["positive_vectors"],
            positive_offsets=arrays["positive_offsets"],
            negative_vectors=arrays["negative_vectors"],
            negative_offsets=arrays["negative_offsets"],
            metadata=PromptIndexMetadata(**payload["metadata"]),
        )

    def _category_score(self, category_index: int, audio_vector: np.ndarray) -> PromptCategoryScore:
        category = self.categories[category_index]
        positive_start, positive_end = self.positive_offsets[category_index : category_index + 2]
        positive_similarities = self.positive_vectors[positive_start:positive_end] @ audio_vector
        positive_order = np.argsort(-positive_similarities, kind="stable")
        positive_top_count = min(2, len(positive_order))
        positive_score = float(np.mean(positive_similarities[positive_order[:positive_top_count]]))
        top_positive_index = int(positive_order[0])

        negative_start, negative_end = self.negative_offsets[category_index : category_index + 2]
        negative_similarities = self.negative_vectors[negative_start:negative_end] @ audio_vector
        if negative_similarities.size:
            top_negative_index = int(np.argmax(negative_similarities))
            negative_score = float(negative_similarities[top_negative_index])
            top_negative_prompt = category.negative_prompts[top_negative_index]
        else:
            top_negative_index = 0
            negative_score = -1.0
            top_negative_prompt = ""
        return PromptCategoryScore(
            path=category.path,
            positive_score=positive_score,
            negative_score=negative_score,
            prompt_margin=positive_score - negative_score,
            top_positive_prompt=category.positive_prompts[top_positive_index],
            top_positive_similarity=float(positive_similarities[top_positive_index]),
            top_negative_prompt=top_negative_prompt,
            top_negative_similarity=negative_score,
        )

    def _validate(self) -> None:
        category_count = len(self.categories)
        if category_count != self.metadata.category_count:
            raise ValueError("prompt index category count differs from metadata")
        if len(self.positive_offsets) != category_count + 1 or len(self.negative_offsets) != category_count + 1:
            raise ValueError("prompt index offsets do not align with categories")
        for matrix in (self.positive_vectors, self.negative_vectors):
            if matrix.ndim != 2 or matrix.shape[1] != self.metadata.embedding_dimension:
                raise ValueError("prompt vector matrix has invalid dimensions")
        if int(self.positive_offsets[-1]) != len(self.positive_vectors):
            raise ValueError("positive prompt offsets do not cover the matrix")
        if int(self.negative_offsets[-1]) != len(self.negative_vectors):
            raise ValueError("negative prompt offsets do not cover the matrix")


def _category_definitions(payload: Any) -> tuple[PromptCategoryDefinition, ...]:
    if not isinstance(payload, list):
        raise ValueError("prompt catalog categories must be a list")
    categories: list[PromptCategoryDefinition] = []
    for raw_category in payload:
        if not isinstance(raw_category, dict):
            raise ValueError("prompt category must be an object")
        category = PromptCategoryDefinition(
            path=str(raw_category.get("path", "")).strip(),
            positive_prompts=_clean_prompts(raw_category.get("positive_prompts", [])),
            negative_prompts=_clean_prompts(raw_category.get("negative_prompts", [])),
            broad_family=str(raw_category.get("broad_family", "")).strip(),
            structure=str(raw_category.get("structure", "")).strip(),
            review_status=str(raw_category.get("review_status", "")).strip(),
            production_ownership_enabled=bool(raw_category.get("production_ownership_enabled", False)),
        )
        if not category.path or not category.positive_prompts:
            raise ValueError("prompt categories require path and positive prompts")
        if category.production_ownership_enabled:
            raise ValueError(f"prompt category ownership is not permitted yet: {category.path}")
        categories.append(category)
    categories.sort(key=lambda row: row.path)
    if len({category.path for category in categories}) != len(categories):
        raise ValueError("prompt catalog contains duplicate category paths")
    return tuple(categories)


def _clean_prompts(payload: Any) -> tuple[str, ...]:
    if not isinstance(payload, list):
        raise ValueError("category prompts must be a list")
    return tuple(dict.fromkeys(str(prompt).strip() for prompt in payload if str(prompt).strip()))


def _prompt_matrix(
    categories: tuple[PromptCategoryDefinition, ...],
    vector_by_prompt: Mapping[str, np.ndarray],
    *,
    positive: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    offsets = [0]
    for category in categories:
        prompts = category.positive_prompts if positive else category.negative_prompts
        rows.extend(l2_normalize(vector_by_prompt[prompt]) for prompt in prompts)
        offsets.append(len(rows))
    dimension = len(next(iter(vector_by_prompt.values())))
    matrix = np.vstack(rows).astype(np.float32) if rows else np.empty((0, dimension), dtype=np.float32)
    return matrix, np.asarray(offsets, dtype=np.int64)
