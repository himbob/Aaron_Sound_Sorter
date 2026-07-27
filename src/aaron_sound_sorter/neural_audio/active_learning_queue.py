"""Deterministic cluster-first review queues that retain every audio member."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ActiveLearningMember:
    """One source-name-blind member eligible for cluster review.

    Args:
        audio_path: Operational locator used only when copying the pack.
        file_sha256: Raw content identity.
        vector: Normalized audio embedding.
        structure_bucket: Objective measured structure partition.
        broad_family: Independent broad audible family partition.
        predicted_label: Aaron prototype suggestion, never training truth.
        second_label: Runner-up prototype suggestion.
        top_similarity: Raw winning prototype similarity.
        margin: Lead over the second prototype label.
        known_distribution: Whether the audio is inside a learned radius.

    Important Constraints:
        ``audio_path`` text must never influence clustering or suggestions.
    """

    audio_path: Path = field(repr=False)
    file_sha256: str
    vector: np.ndarray = field(repr=False, compare=False)
    structure_bucket: str
    broad_family: str
    predicted_label: str
    second_label: str
    top_similarity: float
    margin: float
    known_distribution: bool


@dataclass(frozen=True)
class ActiveLearningCluster:
    """One complete cluster split into safe core, boundary, and outliers."""

    cluster_id: str
    model_id: str
    structure_bucket: str
    broad_family: str
    member_hashes: tuple[str, ...]
    safe_core_hashes: tuple[str, ...]
    boundary_hashes: tuple[str, ...]
    outlier_hashes: tuple[str, ...]
    center_hash: str
    typical_hashes: tuple[str, ...]
    boundary_representative_hash: str
    outlier_representative_hash: str
    suggested_parent_category: str
    suggested_detailed_categories: tuple[str, ...]
    mean_similarity: float
    radius_p95: float
    novelty_score: float
    conflict_score: float
    review_priority: float
    source_pack_id: str
    training_run_id: str
    members: tuple[ActiveLearningMember, ...] = field(repr=False, compare=False)


def create_active_learning_clusters(
    members: list[ActiveLearningMember],
    *,
    model_id: str,
    training_run_id: str,
    similarity_threshold: float = 0.84,
) -> tuple[ActiveLearningCluster, ...]:
    """Partition by structure/family, cluster embeddings, and retain all members.

    Args:
        members: Deduplicated audio records with measured partitions.
        model_id: Pinned embedding model identity.
        training_run_id: Active prototype run identifier.
        similarity_threshold: Minimum cosine similarity connecting members.

    Returns:
        Deterministically ordered clusters. Every input hash appears exactly
        once and no similar member is discarded as redundant.

    Raises:
        ValueError: If thresholds, hashes, vectors, or duplicate content are
            invalid.

    Side Effects:
        Lazily imports scikit-learn for density clustering.
    """
    if not 0.0 < similarity_threshold < 1.0:
        raise ValueError("similarity_threshold must be in (0, 1)")
    ordered = sorted(members, key=lambda member: member.file_sha256)
    hashes = [member.file_sha256 for member in ordered]
    if len(hashes) != len(set(hashes)):
        raise ValueError("active-learning members must be unique by content hash")
    partitions: dict[tuple[str, str], list[ActiveLearningMember]] = defaultdict(list)
    for member in ordered:
        vector = np.asarray(member.vector, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError(f"invalid embedding vector for {member.file_sha256}")
        partitions[(member.structure_bucket, member.broad_family)].append(member)

    raw_clusters: list[tuple[tuple[str, str], list[ActiveLearningMember]]] = []
    for partition_key in sorted(partitions):
        partition_members = partitions[partition_key]
        labels = _density_cluster_labels(partition_members, similarity_threshold)
        grouped: dict[int, list[ActiveLearningMember]] = defaultdict(list)
        for member, cluster_label in zip(partition_members, labels):
            grouped[int(cluster_label)].append(member)
        for cluster_members in grouped.values():
            raw_clusters.append((partition_key, sorted(cluster_members, key=lambda member: member.file_sha256)))
    raw_clusters.sort(key=lambda row: (row[0], row[1][0].file_sha256))
    clusters = tuple(
        _summarize_cluster(
            cluster_members,
            cluster_id=f"Cluster_{position:04d}",
            model_id=model_id,
            training_run_id=training_run_id,
        )
        for position, (_partition, cluster_members) in enumerate(raw_clusters, start=1)
    )
    retained_hashes = [file_hash for cluster in clusters for file_hash in cluster.member_hashes]
    if sorted(retained_hashes) != sorted(hashes):
        raise RuntimeError("cluster construction did not retain every member exactly once")
    return clusters


def measured_structure_bucket(facts: object) -> str:
    """Return a coarse objective structure bucket from shared audio facts."""
    if bool(getattr(facts, "is_broken_or_tiny", False)):
        return "Broken or Invalid"
    if bool(getattr(facts, "is_loop_like", False)):
        return "Loops"
    if bool(getattr(facts, "is_single_event_like", False)) or bool(getattr(facts, "is_short_hit_like", False)):
        return "One Shots"
    evidence = getattr(facts, "evidence", {})
    structure = evidence.get("structure_facts", {}) if isinstance(evidence, dict) else {}
    event_count = float(structure.get("event_count_estimate", 0.0)) if isinstance(structure, dict) else 0.0
    if bool(getattr(facts, "is_long", False)) and event_count <= 2.0:
        return "Long Running"
    return "Phrases"


def semantic_partition_family(semantic_family: str) -> str:
    """Map a CLAP broad semantic family into an active-learning partition."""
    family = str(semantic_family)
    if family.startswith("drum") or family == "drums":
        return "Percussive"
    if family in {
        "bass",
        "keys",
        "guitar_plucked",
        "strings_bowed",
        "woodwind_reed",
        "brass",
        "mallet_bell",
        "synth",
    }:
        return "Tonal Instrument"
    if family in {"human_voice", "animal_creature"}:
        return "Vocal or Biological"
    if family in {"fx_nature", "fx_texture", "fx_foley"}:
        return "Environmental or Texture"
    if family.startswith("fx_"):
        return "Designed FX"
    return "Mixed or Unknown"


def _density_cluster_labels(members: list[ActiveLearningMember], similarity_threshold: float) -> np.ndarray:
    if len(members) == 1:
        return np.zeros(1, dtype=np.int64)
    matrix = np.vstack([np.asarray(member.vector, dtype=np.float64) for member in members])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("density clustering requires finite nonzero embeddings")
    matrix /= norms
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        return _connected_component_cluster_labels(matrix, similarity_threshold)
    return DBSCAN(
        eps=1.0 - similarity_threshold,
        min_samples=1,
        metric="cosine",
        algorithm="brute",
    ).fit_predict(matrix)


def _connected_component_cluster_labels(matrix: np.ndarray, similarity_threshold: float) -> np.ndarray:
    """Match DBSCAN(min_samples=1) with a deterministic NumPy fallback."""
    similarities = matrix @ matrix.T
    labels = np.full(matrix.shape[0], -1, dtype=np.int64)
    next_label = 0
    for start in range(matrix.shape[0]):
        if labels[start] >= 0:
            continue
        labels[start] = next_label
        stack = [start]
        while stack:
            current = stack.pop()
            neighbors = np.flatnonzero(similarities[current] >= similarity_threshold - 1e-12)
            for neighbor in neighbors.tolist():
                if labels[neighbor] < 0:
                    labels[neighbor] = next_label
                    stack.append(neighbor)
        next_label += 1
    return labels


def _summarize_cluster(
    members: list[ActiveLearningMember],
    *,
    cluster_id: str,
    model_id: str,
    training_run_id: str,
) -> ActiveLearningCluster:
    matrix = np.vstack([np.asarray(member.vector, dtype=np.float64) for member in members])
    center = np.mean(matrix, axis=0)
    center_norm = float(np.linalg.norm(center))
    if center_norm <= 0.0:
        raise ValueError(f"cluster center has zero norm: {cluster_id}")
    center /= center_norm
    similarities = matrix @ center
    distances = 1.0 - similarities
    order = sorted(range(len(members)), key=lambda index: (float(distances[index]), members[index].file_sha256))
    core_cutoff = min(0.12, max(float(np.quantile(distances, 0.60)), float(distances[order[0]]) + 1e-7))
    outlier_cutoff = max(float(np.quantile(distances, 0.95)), core_cutoff + 0.04)
    safe_indexes = [index for index in order if float(distances[index]) <= core_cutoff]
    outlier_indexes = [index for index in order if float(distances[index]) > outlier_cutoff]
    boundary_indexes = [index for index in order if index not in safe_indexes and index not in outlier_indexes]
    center_index = order[0]
    typical_indexes = [index for index in order[1:] if index in safe_indexes][:2]
    boundary_representative = max(boundary_indexes, key=lambda index: distances[index]) if boundary_indexes else None
    outlier_representative = max(outlier_indexes, key=lambda index: distances[index]) if outlier_indexes else None
    label_counts = Counter(member.predicted_label for member in members if member.predicted_label)
    detailed_suggestions = tuple(
        label for label, _count in sorted(label_counts.items(), key=lambda row: (-row[1], row[0]))[:3]
    )
    majority_count = max(label_counts.values(), default=0)
    conflict_score = 1.0 - majority_count / len(members) if members else 0.0
    novelty_score = float(np.mean([max(0.0, 1.0 - member.top_similarity) for member in members]))
    boundary_fraction = (len(boundary_indexes) + len(outlier_indexes)) / len(members)
    review_priority = float(
        math.log2(len(members) + 1.0) + 2.0 * conflict_score + 1.5 * novelty_score + boundary_fraction
    )
    hashes = tuple(member.file_sha256 for member in members)
    source_pack_id = hashlib.sha256("\n".join(sorted(hashes)).encode("ascii")).hexdigest()[:16]
    suggested_parent = _suggested_parent(detailed_suggestions[0]) if detailed_suggestions else ""
    return ActiveLearningCluster(
        cluster_id=cluster_id,
        model_id=model_id,
        structure_bucket=members[0].structure_bucket,
        broad_family=members[0].broad_family,
        member_hashes=hashes,
        safe_core_hashes=tuple(members[index].file_sha256 for index in safe_indexes),
        boundary_hashes=tuple(members[index].file_sha256 for index in boundary_indexes),
        outlier_hashes=tuple(members[index].file_sha256 for index in outlier_indexes),
        center_hash=members[center_index].file_sha256,
        typical_hashes=tuple(members[index].file_sha256 for index in typical_indexes),
        boundary_representative_hash=(
            members[boundary_representative].file_sha256 if boundary_representative is not None else ""
        ),
        outlier_representative_hash=(
            members[outlier_representative].file_sha256 if outlier_representative is not None else ""
        ),
        suggested_parent_category=suggested_parent,
        suggested_detailed_categories=detailed_suggestions,
        mean_similarity=float(np.mean(similarities)),
        radius_p95=float(np.quantile(distances, 0.95)),
        novelty_score=novelty_score,
        conflict_score=float(conflict_score),
        review_priority=review_priority,
        source_pack_id=source_pack_id,
        training_run_id=training_run_id,
        members=tuple(members),
    )


def _suggested_parent(label: str) -> str:
    parts = str(label).split("/")
    return "/".join(parts[:-2]) if len(parts) >= 4 else "/".join(parts[:-1])
