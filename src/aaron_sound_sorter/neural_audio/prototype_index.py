"""Deterministic multi-prototype learning and exact cosine search."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .contracts import EmbeddingRecord, LabelPrediction, Prototype, PrototypeIndexMetadata, l2_normalize


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(max(0.0, 1.0 - float(np.dot(l2_normalize(a), l2_normalize(b)))))


def _spherical_centroid(rows: np.ndarray) -> np.ndarray:
    return l2_normalize(np.mean(rows, axis=0))


def _deterministic_seed_indices(rows: np.ndarray, k: int) -> list[int]:
    centroid = _spherical_centroid(rows)
    distances = 1.0 - rows @ centroid
    first = int(np.argmax(distances))
    seeds = [first]
    while len(seeds) < k:
        best_idx = -1
        best_distance = -1.0
        for idx in range(rows.shape[0]):
            if idx in seeds:
                continue
            nearest = min(1.0 - float(np.dot(rows[idx], rows[seed])) for seed in seeds)
            if nearest > best_distance + 1e-12:
                best_idx, best_distance = idx, nearest
        if best_idx < 0:
            break
        seeds.append(best_idx)
    return seeds


def _spherical_kmeans(rows: np.ndarray, k: int, max_iterations: int = 50) -> tuple[np.ndarray, np.ndarray]:
    if k <= 1 or rows.shape[0] <= 1:
        return _spherical_centroid(rows).reshape(1, -1), np.zeros(rows.shape[0], dtype=np.int32)
    seeds = _deterministic_seed_indices(rows, k)
    centroids = rows[seeds].copy()
    assignments = np.full(rows.shape[0], -1, dtype=np.int32)
    for _ in range(max_iterations):
        similarities = rows @ centroids.T
        new_assignments = np.argmax(similarities, axis=1).astype(np.int32)
        if np.array_equal(assignments, new_assignments):
            break
        assignments = new_assignments
        new_centroids: list[np.ndarray] = []
        for cluster_idx in range(k):
            members = rows[assignments == cluster_idx]
            if members.size == 0:
                farthest = int(np.argmin(np.max(similarities, axis=1)))
                new_centroids.append(rows[farthest])
            else:
                new_centroids.append(_spherical_centroid(members))
        centroids = np.vstack(new_centroids).astype(np.float32)
    return centroids, assignments


def _prototype_count(example_count: int, max_prototypes: int) -> int:
    if example_count < 4:
        return 1
    if example_count < 10:
        return min(2, max_prototypes)
    if example_count < 25:
        return min(3, max_prototypes)
    return min(max_prototypes, max(3, int(round(math.sqrt(example_count / 2.0)))))


def _loose_trim(rows: np.ndarray, hashes: Sequence[str], keep_fraction: float) -> tuple[np.ndarray, tuple[str, ...]]:
    """Drop only extreme tails while preserving subtype variety."""
    if rows.shape[0] < 20 or keep_fraction >= 1.0:
        return rows, tuple(hashes)
    centroid = _spherical_centroid(rows)
    distances = 1.0 - rows @ centroid
    keep_count = max(2, int(math.ceil(rows.shape[0] * keep_fraction)))
    keep_indices = np.argsort(distances, kind="stable")[:keep_count]
    return rows[keep_indices], tuple(hashes[idx] for idx in keep_indices)


class PrototypeIndexBuilder:
    """Build one provider-specific index from preview-only embeddings."""

    def __init__(self, *, max_prototypes_per_label: int = 6, keep_fraction: float = 0.95) -> None:
        if max_prototypes_per_label < 1:
            raise ValueError("max_prototypes_per_label must be positive")
        if not 0.5 <= keep_fraction <= 1.0:
            raise ValueError("keep_fraction must be between 0.5 and 1.0")
        self.max_prototypes_per_label = max_prototypes_per_label
        self.keep_fraction = keep_fraction

    def build(self, labeled_embeddings: Mapping[str, Sequence[EmbeddingRecord]]) -> PrototypeIndex:
        if not labeled_embeddings:
            raise ValueError("no labeled embeddings supplied")
        provider_pairs = {
            (record.provider_id, record.model_id) for rows in labeled_embeddings.values() for record in rows
        }
        if len(provider_pairs) != 1:
            raise ValueError("one prototype index may contain exactly one provider/model pair")
        provider_id, model_id = next(iter(provider_pairs))
        dimensions = {record.dimension for rows in labeled_embeddings.values() for record in rows}
        if len(dimensions) != 1:
            raise ValueError("all embeddings must have the same dimension")
        dimension = next(iter(dimensions))

        prototypes: list[Prototype] = []
        label_example_counts: dict[str, int] = {}
        label_prototype_counts: dict[str, int] = {}
        label_spread_p95: dict[str, float] = {}
        training_hashes: list[str] = []

        for label, records in sorted(labeled_embeddings.items()):
            unique: dict[str, EmbeddingRecord] = {record.file_sha256: record for record in records}
            if not unique:
                continue
            ordered = [unique[digest] for digest in sorted(unique)]
            rows = np.vstack([record.vector for record in ordered]).astype(np.float32)
            hashes = [record.file_sha256 for record in ordered]
            rows, retained_hashes = _loose_trim(rows, hashes, self.keep_fraction)
            k = min(rows.shape[0], _prototype_count(rows.shape[0], self.max_prototypes_per_label))
            centroids, assignments = _spherical_kmeans(rows, k)
            assigned_similarities = np.asarray(
                [
                    float(np.dot(rows[index], centroids[cluster_index]))
                    for index, cluster_index in enumerate(assignments)
                ],
                dtype=np.float64,
            )
            assigned_distances = np.maximum(0.0, 1.0 - assigned_similarities)

            label_example_counts[label] = len(unique)
            label_prototype_counts[label] = int(centroids.shape[0])
            label_spread_p95[label] = float(np.percentile(assigned_distances, 95)) if assigned_distances.size else 0.0
            training_hashes.extend(hashes)
            for cluster_idx, centroid in enumerate(centroids):
                member_indices = np.where(assignments == cluster_idx)[0]
                if member_indices.size == 0:
                    member_indices = np.array([int(np.argmax(rows @ centroid))])
                member_vectors = rows[member_indices]
                member_hashes = tuple(retained_hashes[idx] for idx in member_indices)
                distances = np.maximum(0.0, 1.0 - member_vectors @ centroid)
                radius = float(np.percentile(distances, 95)) if distances.size else 0.0
                mean_distance = float(np.mean(distances)) if distances.size else 0.0
                prototypes.append(
                    Prototype(
                        label=label,
                        prototype_id=f"{label}::p{cluster_idx + 1}",
                        vector=centroid,
                        member_hashes=member_hashes,
                        radius_p95=radius,
                        mean_distance=mean_distance,
                    )
                )

        metadata = PrototypeIndexMetadata(
            schema_version=1,
            provider_id=provider_id,
            model_id=model_id,
            dimension=dimension,
            created_utc=datetime.now(timezone.utc).isoformat(),
            training_hashes=tuple(sorted(set(training_hashes))),
            label_example_counts=label_example_counts,
            label_prototype_counts=label_prototype_counts,
            build_settings={
                "max_prototypes_per_label": self.max_prototypes_per_label,
                "keep_fraction": self.keep_fraction,
            },
            label_spread_p95=label_spread_p95,
        )
        return PrototypeIndex(metadata=metadata, prototypes=tuple(prototypes))


class PrototypeIndex:
    """Exact cosine classifier over a small set of learned prototypes."""

    def __init__(self, *, metadata: PrototypeIndexMetadata, prototypes: Sequence[Prototype]) -> None:
        if not prototypes:
            raise ValueError("prototype index cannot be empty")
        if any(proto.vector.shape[0] != metadata.dimension for proto in prototypes):
            raise ValueError("prototype dimension does not match metadata")
        self.metadata = metadata
        self.prototypes = tuple(prototypes)
        self._matrix = np.vstack([proto.vector for proto in self.prototypes]).astype(np.float32)

    def predict(self, record: EmbeddingRecord, *, radius_slack: float = 0.08) -> LabelPrediction:
        """Return the closest learned label and its neighborhood evidence."""
        ranking = self._rank_labels(record)
        best_label, (best_score, best_idx) = ranking[0]
        if len(ranking) > 1:
            second_label, (second_score, _) = ranking[1]
        else:
            second_label, second_score = "", -1.0
        proto = self.prototypes[best_idx]
        distance = max(0.0, 1.0 - best_score)
        category_spread = float(self.metadata.label_spread_p95.get(best_label, 0.0))
        allowed_radius = max(radius_slack, proto.radius_p95 + radius_slack, category_spread + radius_slack)
        radius_ratio = distance / max(allowed_radius, 1e-9)
        known = bool(distance <= allowed_radius)
        return LabelPrediction(
            provider_id=record.provider_id,
            model_id=record.model_id,
            predicted_label=best_label,
            top_similarity=best_score,
            second_label=second_label,
            second_similarity=second_score,
            margin=best_score - second_score,
            prototype_id=proto.prototype_id,
            prototype_radius_p95=proto.radius_p95,
            distance_to_prototype=distance,
            radius_ratio=radius_ratio,
            known_distribution=known,
            evidence={
                "label_example_count": int(self.metadata.label_example_counts.get(best_label, 0)),
                "label_prototype_count": int(self.metadata.label_prototype_counts.get(best_label, 0)),
                "prototype_support_count": len(proto.member_hashes),
                "category_spread_p95": category_spread,
                "encoder_reliability": "unavailable",
                "objective_structure_agreement": "unavailable",
                "trainer_warning_count": "unavailable",
            },
        )

    def label_similarities(self, record: EmbeddingRecord) -> dict[str, float]:
        """Return the best cosine similarity for every learned label.

        Args:
            record: Source-name-blind audio embedding from the index provider.

        Returns:
            Label-to-similarity mapping sorted from strongest to weakest.

        Raises:
            ValueError: If provider, model, or embedding dimensions differ from
                the active index.

        Side Effects:
            None.
        """
        return {label: score for label, (score, _) in self._rank_labels(record)}

    def _rank_labels(self, record: EmbeddingRecord) -> list[tuple[str, tuple[float, int]]]:
        """Return labels ranked by their best prototype similarity."""
        if record.provider_id != self.metadata.provider_id or record.model_id != self.metadata.model_id:
            raise ValueError("embedding provider/model does not match prototype index")
        if record.dimension != self.metadata.dimension:
            raise ValueError("embedding dimension does not match prototype index")

        # NumPy's accelerated matmul can emit spurious floating-point warnings
        # for finite, normalized CLAP vectors on some macOS Accelerate builds.
        # Element-wise reduction is equally exact for this small index and
        # avoids that unstable BLAS path.
        similarities = np.sum(
            self._matrix.astype(np.float64) * record.vector.astype(np.float64)[None, :],
            axis=1,
            dtype=np.float64,
        )
        best_per_label: dict[str, tuple[float, int]] = {}
        for idx, proto in enumerate(self.prototypes):
            score = float(similarities[idx])
            current = best_per_label.get(proto.label)
            if current is None or score > current[0]:
                best_per_label[proto.label] = (score, idx)
        return sorted(best_per_label.items(), key=lambda item: (-item[1][0], item[0]))

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{directory.name}.tmp-", dir=directory.parent))
        try:
            matrix = np.vstack([proto.vector for proto in self.prototypes]).astype(np.float32)
            np.save(temp_dir / "prototype_vectors.npy", matrix, allow_pickle=False)
            payload = {
                "metadata": {
                    "schema_version": self.metadata.schema_version,
                    "provider_id": self.metadata.provider_id,
                    "model_id": self.metadata.model_id,
                    "dimension": self.metadata.dimension,
                    "created_utc": self.metadata.created_utc,
                    "training_hashes": list(self.metadata.training_hashes),
                    "label_example_counts": dict(self.metadata.label_example_counts),
                    "label_prototype_counts": dict(self.metadata.label_prototype_counts),
                    "label_spread_p95": dict(self.metadata.label_spread_p95),
                    "build_settings": dict(self.metadata.build_settings),
                },
                "prototypes": [
                    {
                        "label": proto.label,
                        "prototype_id": proto.prototype_id,
                        "member_hashes": list(proto.member_hashes),
                        "radius_p95": proto.radius_p95,
                        "mean_distance": proto.mean_distance,
                    }
                    for proto in self.prototypes
                ],
            }
            (temp_dir / "index.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            backup = directory.with_name(f".{directory.name}.old")
            if backup.exists():
                shutil.rmtree(backup)
            backup_moved = False
            if directory.exists():
                os.replace(directory, backup)
                backup_moved = True
            try:
                os.replace(temp_dir, directory)
            except Exception:
                if backup_moved and backup.exists() and not directory.exists():
                    os.replace(backup, directory)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    @classmethod
    def load(cls, directory: Path) -> PrototypeIndex:
        directory = Path(directory)
        payload = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        vectors = np.load(directory / "prototype_vectors.npy", allow_pickle=False)
        raw_meta = payload["metadata"]
        metadata = PrototypeIndexMetadata(
            schema_version=int(raw_meta["schema_version"]),
            provider_id=str(raw_meta["provider_id"]),
            model_id=str(raw_meta["model_id"]),
            dimension=int(raw_meta["dimension"]),
            created_utc=str(raw_meta["created_utc"]),
            training_hashes=tuple(str(value) for value in raw_meta.get("training_hashes", [])),
            label_example_counts={str(k): int(v) for k, v in raw_meta.get("label_example_counts", {}).items()},
            label_prototype_counts={str(k): int(v) for k, v in raw_meta.get("label_prototype_counts", {}).items()},
            build_settings=dict(raw_meta.get("build_settings", {})),
            label_spread_p95={str(k): float(v) for k, v in raw_meta.get("label_spread_p95", {}).items()},
        )
        raw_prototypes = payload["prototypes"]
        if len(raw_prototypes) != vectors.shape[0]:
            raise ValueError("prototype metadata/vector count mismatch")
        prototypes = [
            Prototype(
                label=str(raw["label"]),
                prototype_id=str(raw["prototype_id"]),
                vector=vectors[idx],
                member_hashes=tuple(str(value) for value in raw.get("member_hashes", [])),
                radius_p95=float(raw.get("radius_p95", 0.0)),
                mean_distance=float(raw.get("mean_distance", 0.0)),
            )
            for idx, raw in enumerate(raw_prototypes)
        ]
        return cls(metadata=metadata, prototypes=prototypes)
