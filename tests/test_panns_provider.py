from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from aaron_sound_sorter.neural_audio.panns_provider import PannsProvider


class FakePannsBackend:
    labels = ("Speech", "Saxophone", "Drum")

    def infer(self, waveform_batch: np.ndarray) -> np.ndarray:
        assert waveform_batch.ndim == 2
        return np.tile(np.asarray([[0.1, 0.8, 0.2]], dtype=np.float32), (len(waveform_batch), 1))


def _provider(tmp_path: Path) -> PannsProvider:
    provider = PannsProvider(
        checkpoint_path=tmp_path / "unused.pth",
        labels_path=tmp_path / "unused.csv",
        checkpoint_sha256="a" * 64,
        labels_sha256="b" * 64,
        source_model_id="test/panns",
        source_revision="test",
        sample_rate=32000,
    )
    provider._backend = FakePannsBackend()
    return provider


def test_panns_provider_ranks_raw_audio_events(tmp_path: Path) -> None:
    audio_path = tmp_path / "hash-only.wav"
    sf.write(audio_path, np.sin(np.linspace(0, 40, 16000)).astype(np.float32), 16000)

    prediction = _provider(tmp_path).predict_file(audio_path, top_k=2)

    assert [event.label for event in prediction.top_events] == ["Saxophone", "Drum"]
    assert prediction.top_events[0].score == np.float32(0.8)
    assert prediction.segment_count == 1
    assert len(prediction.audio_sha256) == 64


def test_panns_provider_requires_positive_segment_configuration(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    provider.max_segments = 0

    try:
        provider._segments(np.ones(100, dtype=np.float32))
    except ValueError as exc:
        assert "segment configuration" in str(exc)
    else:
        raise AssertionError("invalid segment configuration should fail")
