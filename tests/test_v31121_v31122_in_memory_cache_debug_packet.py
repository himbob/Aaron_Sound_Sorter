from __future__ import annotations

from pathlib import Path

import numpy as np

import aaron_sound_sorter.engine.sorter as sorter_module
from aaron_sound_sorter.domain.models import AudioPhysics, SharedAudioFacts
from aaron_sound_sorter.engine.audio_analysis_cache import AudioAnalysisCache, AudioAnalysisPacket
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path
from aaron_sound_sorter.engine.sorter import SortSamplesUseCase


def _physics(path: Path, value: float = 1.0, status: str = "ok") -> AudioPhysics:
    return AudioPhysics(
        source_path=path,
        fingerprint=np.asarray([value, value + 1.0], dtype=np.float32),
        duration_sec=1.25,
        read_status=status,
        direct_body_fingerprint=np.asarray([value + 2.0], dtype=np.float32),
        direct_body_duration_sec=0.5,
        direct_body_status=status,
    )


def _claim(folder_path: str, *, source: str = "test", strength: float = 0.90):
    return claim_from_folder_path(
        folder_path=folder_path,
        source=source,
        reason="cache/debug packet test",
        shared=[],
        raw_candidate_score=4.0,
        brain_rank=1,
        physics_rank=1,
        shared_winner=folder_path,
        can_override=True,
        strength=strength,
        is_real_candidate=True,
    )


def _facts(evidence: dict[str, object]) -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=False,
        evidence=evidence,
        feature_values_by_name={},
    )


def test_audio_analysis_packet_summary_is_memory_only_and_decision_free(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    cache = AudioAnalysisCache(max_items=8)
    physics = cache.audio_physics(audio, lambda path: _physics(path, 1.0))
    wetness = cache.wetness_profile(audio, lambda path: {"status": "ok", "wetness_score": 0.66})
    harmonic = cache.harmonic_physics(audio, lambda path: _physics(path, 10.0))
    packet = AudioAnalysisPacket(
        audio_physics=physics,
        wetness_profile=wetness,
        harmonic_physics=harmonic,
        cache_stats_before={"hits": 0, "misses": 0},
        cache_stats_after=cache.stats(),
    )

    summary = packet.summary()

    assert summary["cache_policy"] == "per_run_memory_only"
    assert summary["persistent_disk_cache"] is False
    assert summary["stores_final_decisions"] is False
    assert summary["stores_training_labels"] is False
    assert "final_label" not in summary
    assert "folder_path" not in summary


def test_cache_reset_clears_per_run_state_without_disk_persistence(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    before_paths = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    cache = AudioAnalysisCache(max_items=8)

    cache.audio_physics(audio, lambda path: _physics(path, 1.0))
    cache.wetness_profile(audio, lambda path: {"status": "ok", "wetness_score": 0.25})
    assert cache.stats()["total_items"] == 2
    assert cache.stats()["persistent_disk_cache"] is False

    cache.reset()

    assert cache.stats()["total_items"] == 0
    assert cache.stats()["hits"] == 0
    assert cache.stats()["misses"] == 0
    after_paths = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after_paths == before_paths


def test_sort_use_case_builds_shared_packet_through_in_memory_cache(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    calls = {"audio": 0, "wetness": 0, "harmonic": 0}

    def fake_audio(path: Path) -> AudioPhysics:
        calls["audio"] += 1
        return _physics(path, 1.0)

    def fake_wetness(path: Path) -> dict[str, object]:
        calls["wetness"] += 1
        return {"status": "ok", "wetness_score": 0.60}

    def fake_harmonic(path: Path) -> AudioPhysics:
        calls["harmonic"] += 1
        return _physics(path, 10.0)

    monkeypatch.setattr(sorter_module, "analyze_audio_file", fake_audio)
    monkeypatch.setattr(sorter_module, "analyze_wetness_profile", fake_wetness)
    monkeypatch.setattr(sorter_module, "analyze_harmonic_core_audio_file", fake_harmonic)
    use_case = object.__new__(SortSamplesUseCase)
    use_case.analysis_cache = AudioAnalysisCache(max_items=8)

    first = use_case.build_audio_analysis_packet(audio_file=audio, harmonic_baby_brains={})
    second = use_case.build_audio_analysis_packet(audio_file=audio, harmonic_baby_brains={})

    assert calls == {"audio": 1, "wetness": 1, "harmonic": 1}
    assert first.summary()["cache_policy"] == "per_run_memory_only"
    assert second.cache_stats_after["hits"] >= 3
    assert second.summary()["persistent_disk_cache"] is False


def test_claim_debug_trace_includes_unified_packet_and_cache_stats(monkeypatch, tmp_path: Path) -> None:
    debug_file = tmp_path / "claims_debug.txt"
    raw = _claim("Instruments/Synths/Pads/Loops", source="raw_synth_pad")
    candidate = _claim("Instruments/Synths/Pads/Loops", source="profile_candidate_synth_pad")
    facts = _facts(
        {
            "audio_analysis_packet_summary": {
                "cache_policy": "per_run_memory_only",
                "persistent_disk_cache": False,
                "stores_final_decisions": False,
                "audio_status": "ok",
                "direct_body_status": "ok",
                "wetness_score": 0.25,
                "harmonic_status": "not_computed",
            },
            "audio_analysis_cache_stats": {
                "policy": "per_run_memory_only",
                "persistent_disk_cache": False,
                "hits": 2,
                "misses": 3,
                "total_items": 3,
            },
        }
    )
    monkeypatch.setenv("AARON_DEBUG_CLAIMS_FILE", str(debug_file))

    FamilyClaimArbiter().adjudicate(
        raw_claim=raw,
        consensus_claims=[],
        eligibility_claims=[candidate],
        facts=facts,
    )

    text = debug_file.read_text(encoding="utf-8")
    assert "EVIDENCE_PACKET policy=per_run_memory_only persistent=False stores_final=False" in text
    assert "CACHE_STATS policy=per_run_memory_only persistent=False hits=2 misses=3 items=3" in text
