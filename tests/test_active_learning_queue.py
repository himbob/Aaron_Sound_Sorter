from __future__ import annotations

from pathlib import Path

import numpy as np

from aaron_sound_sorter.neural_audio.active_learning_queue import (
    ActiveLearningMember,
    create_active_learning_clusters,
    semantic_partition_family,
)


def _member(number: int, vector: list[float], *, structure: str = "Loops") -> ActiveLearningMember:
    normalized = np.asarray(vector, dtype=np.float32)
    normalized /= np.linalg.norm(normalized)
    return ActiveLearningMember(
        audio_path=Path(f"/operational/{number}.wav"),
        file_sha256=f"{number:064x}",
        vector=normalized,
        structure_bucket=structure,
        broad_family="Tonal Instrument",
        predicted_label="Instruments/Synths/Synth Pad/Loops",
        second_label="Instruments/Keys/Keys Loops/Loops",
        top_similarity=0.8,
        margin=0.2,
        known_distribution=True,
    )


def test_cluster_queue_retains_similar_members_instead_of_suppressing_them() -> None:
    members = [
        _member(1, [1.0, 0.0]),
        _member(2, [0.99, 0.05]),
        _member(3, [0.97, 0.10]),
        _member(4, [0.0, 1.0]),
    ]

    clusters = create_active_learning_clusters(
        members,
        model_id="test-model",
        training_run_id="test-run",
        similarity_threshold=0.90,
    )

    retained = {file_hash for cluster in clusters for file_hash in cluster.member_hashes}
    assert retained == {member.file_sha256 for member in members}
    assert sorted(len(cluster.member_hashes) for cluster in clusters) == [1, 3]
    assert all(cluster.safe_core_hashes for cluster in clusters)


def test_cluster_queue_partitions_structure_before_embedding_similarity() -> None:
    loop = _member(1, [1.0, 0.0], structure="Loops")
    one_shot = _member(2, [1.0, 0.0], structure="One Shots")

    clusters = create_active_learning_clusters(
        [loop, one_shot],
        model_id="test-model",
        training_run_id="test-run",
    )

    assert len(clusters) == 2
    assert {cluster.structure_bucket for cluster in clusters} == {"Loops", "One Shots"}


def test_semantic_partition_keeps_fx_and_instruments_separate() -> None:
    assert semantic_partition_family("woodwind_reed") == "Tonal Instrument"
    assert semantic_partition_family("fx_impact") == "Designed FX"
    assert semantic_partition_family("fx_nature") == "Environmental or Texture"
