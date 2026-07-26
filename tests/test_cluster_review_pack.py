from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from aaron_sound_sorter.neural_audio.active_learning_queue import (
    ActiveLearningMember,
    create_active_learning_clusters,
)
from tools.build_cluster_review_pack import copy_review_pack


def _member(tmp_path: Path, number: int, vector: list[float]) -> ActiveLearningMember:
    source = tmp_path / f"locator_{number}.wav"
    source.write_bytes(f"audio-{number}".encode())
    normalized = np.asarray(vector, dtype=np.float32)
    normalized /= np.linalg.norm(normalized)
    return ActiveLearningMember(
        audio_path=source,
        file_sha256=f"{number:064x}",
        vector=normalized,
        structure_bucket="Loops",
        broad_family="Tonal Instrument",
        predicted_label="Instruments/Synths/Synth Pad/Loops",
        second_label="Instruments/Keys/Keys Loops/Loops",
        top_similarity=0.8,
        margin=0.2,
        known_distribution=True,
    )


def test_cluster_review_pack_copies_every_member_under_hash_only_names(tmp_path: Path) -> None:
    members = [_member(tmp_path, 1, [1.0, 0.0]), _member(tmp_path, 2, [0.99, 0.05])]
    clusters = create_active_learning_clusters(members, model_id="test", training_run_id="run")
    destination = tmp_path / "review"

    copy_review_pack(destination, clusters)

    rows = list(csv.DictReader((destination / "cluster_manifest.csv").open(encoding="utf-8")))
    assert len(rows) == 2
    assert all("locator" not in row["review_file"] for row in rows)
    assert all((destination / row["review_file"]).is_file() for row in rows)
    assert (destination / "cluster_manifest.json").is_file()
    assert "does not modify" in (destination / "README.txt").read_text(encoding="utf-8")
