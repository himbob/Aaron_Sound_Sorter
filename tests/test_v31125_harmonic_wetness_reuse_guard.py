from __future__ import annotations

from pathlib import Path

import numpy as np

import aaron_sound_sorter.features as features
from aaron_sound_sorter.engine.sorter import analyze_harmonic_core_audio_file


def test_harmonic_core_reuses_precomputed_wetness_without_remeasuring(monkeypatch) -> None:
    calls = {"wetness": 0}

    def forbidden_wetness_profile(path: Path) -> dict[str, float | str]:
        calls["wetness"] += 1
        raise AssertionError("wetness_profile should not be called when precomputed profile is provided")

    monkeypatch.setattr(features, "wetness_profile", forbidden_wetness_profile)
    monkeypatch.setattr(features, "read_audio", lambda path: (np.ones((128, 1), dtype=np.float32), 22050))
    monkeypatch.setattr(features, "resample_linear", lambda audio, sample_rate, target_rate: (audio, target_rate))
    monkeypatch.setattr(features, "trim_and_normalize", lambda audio: (audio, "ok"))
    monkeypatch.setattr(
        features,
        "harmonic_core_signal",
        lambda mono, harmonic_margin=8.0: (mono, {"harmonic_energy_ratio": 0.75}),
    )
    monkeypatch.setattr(
        features,
        "make_fingerprint_from_preprocessed_audio",
        lambda audio, sample_rate, status: (np.arange(features.FP_SIZE, dtype=np.float32), 1.0, "ok"),
    )

    fingerprint, duration, status, wetness = features.make_harmonic_core_fingerprint(
        Path("sample.wav"),
        wet_profile={"status": "ok", "wetness_score": 0.9},
    )

    assert calls["wetness"] == 0
    assert status == "ok"
    assert duration == 1.0
    assert float(fingerprint[3]) == 3.0
    assert wetness["wetness_score"] == 0.9
    assert wetness["harmonic_core_harmonic_energy_ratio"] == 0.75


def test_sorter_harmonic_analyzer_accepts_precomputed_wetness(monkeypatch) -> None:
    monkeypatch.setattr(
        "aaron_sound_sorter.engine.sorter.make_harmonic_core_fingerprint_safe",
        lambda path, wet_profile=None: (np.zeros(4, dtype=np.float32), 2.0, "ok", dict(wet_profile or {})),
    )

    physics = analyze_harmonic_core_audio_file(
        Path("sample.wav"),
        precomputed_wet_profile={"status": "ok", "wetness_score": 0.66},
    )

    assert physics.read_status == "ok"
    assert physics.duration_sec == 2.0
    assert physics.wetness_profile["wetness_score"] == 0.66
