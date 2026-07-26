from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf

from aaron_sound_sorter.neural_audio.panns_provider import OfficialPannsBackend, PannsProvider


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


def test_official_panns_backend_allows_pinned_legacy_checkpoint_on_modern_torch(
    monkeypatch, tmp_path: Path
) -> None:
    seen = {}

    class FakeAudioTagging:
        def __init__(self, *, checkpoint_path: str, device: str) -> None:
            seen["checkpoint_path"] = checkpoint_path
            seen["device"] = device
            seen["legacy_load"] = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")

        def inference(self, waveform_batch: np.ndarray):
            return np.zeros((len(waveform_batch), 2), dtype=np.float32), None

    fake_module = types.ModuleType("panns_inference")
    fake_module.AudioTagging = FakeAudioTagging
    monkeypatch.setitem(sys.modules, "panns_inference", fake_module)
    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)

    checkpoint = tmp_path / "panns.pth"
    checkpoint.write_bytes(b"pinned-test-checkpoint")
    backend = OfficialPannsBackend(
        checkpoint_path=checkpoint,
        labels=("Speech", "Music"),
        device="cpu",
    )

    assert backend.labels == ("Speech", "Music")
    assert seen == {
        "checkpoint_path": str(checkpoint),
        "device": "cpu",
        "legacy_load": "1",
    }
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in os.environ


def test_official_panns_backend_restores_existing_torch_load_setting(
    monkeypatch, tmp_path: Path
) -> None:
    seen = []

    class FakeAudioTagging:
        def __init__(self, *, checkpoint_path: str, device: str) -> None:
            seen.append(os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"))

    fake_module = types.ModuleType("panns_inference")
    fake_module.AudioTagging = FakeAudioTagging
    monkeypatch.setitem(sys.modules, "panns_inference", fake_module)
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "existing-value")

    OfficialPannsBackend(
        checkpoint_path=tmp_path / "panns.pth",
        labels=("Speech",),
        device="cpu",
    )

    assert seen == ["1"]
    assert os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "existing-value"
