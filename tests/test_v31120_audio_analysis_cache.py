from __future__ import annotations

from pathlib import Path

import numpy as np

from aaron_sound_sorter.domain.models import AudioPhysics
from aaron_sound_sorter.engine.audio_analysis_cache import AudioAnalysisCache


def _physics(path: Path, value: float = 1.0) -> AudioPhysics:
    return AudioPhysics(
        source_path=path,
        fingerprint=np.asarray([value, value + 1.0], dtype=np.float32),
        duration_sec=1.25,
        read_status="ok",
        direct_body_fingerprint=np.asarray([value + 2.0], dtype=np.float32),
        direct_body_duration_sec=0.5,
        direct_body_status="ok",
    )


def test_audio_physics_cache_reuses_measurement_but_returns_independent_arrays(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    calls = {"count": 0}

    def factory(path: Path) -> AudioPhysics:
        calls["count"] += 1
        return _physics(path, value=float(calls["count"]))

    cache = AudioAnalysisCache(max_items=8)
    first = cache.audio_physics(audio, factory)
    first.fingerprint[0] = 99.0
    second = cache.audio_physics(audio, factory)

    assert calls["count"] == 1
    assert float(second.fingerprint[0]) == 1.0
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1


def test_wetness_cache_reuses_measurement_and_returns_copy(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    calls = {"count": 0}

    def factory(path: Path) -> dict[str, object]:
        calls["count"] += 1
        return {"wetness_score": 0.75, "nested": {"kept": True}}

    cache = AudioAnalysisCache(max_items=8)
    first = cache.wetness_profile(audio, factory)
    first["wetness_score"] = 0.1
    second = cache.wetness_profile(audio, factory)

    assert calls["count"] == 1
    assert second["wetness_score"] == 0.75


def test_harmonic_cache_uses_separate_namespace(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    calls = {"audio": 0, "harmonic": 0}

    def audio_factory(path: Path) -> AudioPhysics:
        calls["audio"] += 1
        return _physics(path, value=1.0)

    def harmonic_factory(path: Path) -> AudioPhysics:
        calls["harmonic"] += 1
        return _physics(path, value=10.0)

    cache = AudioAnalysisCache(max_items=8)
    full = cache.audio_physics(audio, audio_factory)
    harmonic = cache.harmonic_physics(audio, harmonic_factory)
    harmonic_again = cache.harmonic_physics(audio, harmonic_factory)

    assert calls == {"audio": 1, "harmonic": 1}
    assert float(full.fingerprint[0]) == 1.0
    assert float(harmonic.fingerprint[0]) == 10.0
    assert float(harmonic_again.fingerprint[0]) == 10.0


def test_persistent_audio_cache_reuses_measurement_across_instances(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    cache_dir = tmp_path / "analysis_cache"
    calls = {"first": 0, "second": 0}

    def first_factory(path: Path) -> AudioPhysics:
        calls["first"] += 1
        physics = _physics(path, value=7.0)
        object.__setattr__(physics, "direct_body_profile", {"body": 0.75})
        object.__setattr__(physics, "third_party_feature_profile", {"flat": {"tone": 0.5}})
        return physics

    first_cache = AudioAnalysisCache(max_items=8, persistent_dir=cache_dir)
    first = first_cache.audio_physics(audio, first_factory)

    def second_factory(path: Path) -> AudioPhysics:
        calls["second"] += 1
        return _physics(path, value=99.0)

    second_cache = AudioAnalysisCache(max_items=8, persistent_dir=cache_dir)
    second = second_cache.audio_physics(audio, second_factory)

    assert calls == {"first": 1, "second": 0}
    assert float(first.fingerprint[0]) == 7.0
    assert float(second.fingerprint[0]) == 7.0
    assert second.direct_body_profile == {"body": 0.75}
    assert second.third_party_feature_profile == {"flat": {"tone": 0.5}}
    assert second_cache.stats()["disk_hits"] == 1
    assert second_cache.stats()["misses"] == 0


def test_persistent_cache_invalidates_when_file_state_changes(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    cache_dir = tmp_path / "analysis_cache"

    first_cache = AudioAnalysisCache(max_items=8, persistent_dir=cache_dir)
    first_cache.audio_physics(audio, lambda path: _physics(path, value=2.0))

    audio.write_bytes(b"changed audio")
    calls = {"count": 0}

    def changed_factory(path: Path) -> AudioPhysics:
        calls["count"] += 1
        return _physics(path, value=3.0)

    changed_cache = AudioAnalysisCache(max_items=8, persistent_dir=cache_dir)
    changed = changed_cache.audio_physics(audio, changed_factory)

    assert calls["count"] == 1
    assert float(changed.fingerprint[0]) == 3.0
    assert changed_cache.stats()["disk_hits"] == 0
