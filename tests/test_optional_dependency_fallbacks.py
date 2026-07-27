from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np

from aaron_sound_sorter.neural_audio.active_learning_queue import (
    ActiveLearningMember,
    create_active_learning_clusters,
)
from aaron_sound_sorter.neural_audio.panns_provider import PannsProvider


def _block_import(monkeypatch, blocked: str) -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == blocked or name.startswith(blocked + "."):
            raise ImportError(f"blocked {blocked}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def _member(number: int, vector: list[float]) -> ActiveLearningMember:
    array = np.asarray(vector, dtype=np.float32)
    array /= np.linalg.norm(array)
    return ActiveLearningMember(
        audio_path=Path(f"/{number}.wav"),
        file_sha256=f"{number:064x}",
        vector=array,
        structure_bucket="Loops",
        broad_family="Tonal Instrument",
        predicted_label="Instruments/Synths/Synth Pad/Loops",
        second_label="Instruments/Keys/Keys Loops/Loops",
        top_similarity=0.8,
        margin=0.2,
        known_distribution=True,
    )


def test_active_learning_falls_back_without_sklearn(monkeypatch) -> None:
    _block_import(monkeypatch, "sklearn")
    clusters = create_active_learning_clusters(
        [
            _member(1, [1.0, 0.0]),
            _member(2, [0.99, 0.05]),
            _member(3, [0.0, 1.0]),
        ],
        model_id="test",
        training_run_id="run",
        similarity_threshold=0.90,
    )
    assert sorted(len(cluster.member_hashes) for cluster in clusters) == [1, 2]


def test_panns_resampling_falls_back_without_librosa(monkeypatch, tmp_path: Path) -> None:
    _block_import(monkeypatch, "librosa")
    provider = PannsProvider(
        checkpoint_path=tmp_path / "unused.pth",
        labels_path=tmp_path / "unused.csv",
        checkpoint_sha256="a" * 64,
        labels_sha256="b" * 64,
        source_model_id="test/panns",
        source_revision="test",
        sample_rate=32000,
    )
    source = np.linspace(-1.0, 1.0, 16000, dtype=np.float32)
    output = provider._resample(source, 16000)
    assert output.dtype == np.float32
    assert output.shape == (32000,)
    assert np.isfinite(output).all()
