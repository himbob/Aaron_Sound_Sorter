from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aaron_sound_sorter.third_party_audio_features import (
    librosa_is_available,
    third_party_feature_profile_from_audio,
)


def _require_librosa() -> None:
    if not librosa_is_available():
        pytest.skip("librosa is not installed in the active pytest interpreter")


def test_librosa_is_required_runtime_dependency_in_project_metadata() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    requirements = Path("requirements.txt").read_text(encoding="utf-8")

    assert "librosa>=0.10.2,<0.12" in pyproject
    assert "librosa>=0.10.2,<0.12" in requirements


def test_librosa_adapter_always_emits_measured_flat_features() -> None:
    _require_librosa()
    sr = 22050
    seconds = 2.0
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    tone = 0.4 * np.sin(2.0 * np.pi * 220.0 * t)
    clicks = np.zeros_like(tone)
    clicks[:: sr // 4] = 1.0
    audio = (tone + clicks).astype(np.float32)

    profile = third_party_feature_profile_from_audio(audio, sr)
    flat = profile["flat"]

    assert profile["dependency_policy"] == "librosa_required_runtime"
    assert profile["adapters"]["librosa"]["status"] == "ok"
    assert profile["adapters"]["essentia"]["owns_final_folder"] is False
    assert profile["adapters"]["yamnet"]["owns_final_folder"] is False
    assert profile["adapters"]["mediapipe_audio_classifier"]["owns_final_folder"] is False
    assert flat["librosa_onset_event_count"] >= 2.0
    assert flat["librosa_loop_confidence"] > 0.0
    assert flat["librosa_tonal_confidence"] > 0.0
    assert "librosa_harmonic_energy_ratio" in flat
    assert "librosa_percussive_energy_ratio" in flat
    assert "librosa_mfcc_delta_std" in flat
    assert "librosa_chroma_frame_change_mean" in flat
    assert "librosa_spectral_centroid_std" in flat
    assert "librosa_yin_f0_median_hz" in flat
    assert "librosa_yin_voiced_ratio" in flat


from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.claim_producers.instrument_loop_conflicts import InstrumentLoopConflictMixin
from aaron_sound_sorter.engine.claim_producers.instrument_loop_one_shot import InstrumentOneShotLoopMixin
from aaron_sound_sorter.engine.claim_producers.measured_transition_fx import MeasuredTransitionFxClaimProducer


def _facts(evidence: dict[str, object]) -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence=evidence,
        feature_values_by_name={},
    )


def test_librosa_adapter_emits_spectral_motion_and_timbre_features() -> None:
    _require_librosa()
    sr = 22050
    seconds = 2.0
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    # Manual upward chirp: enough spectral motion to verify the adapter exposes
    # slope evidence, without relying on producer filenames or labels.
    f0 = 180.0
    f1 = 1200.0
    phase = 2.0 * np.pi * (f0 * t + ((f1 - f0) / (2.0 * seconds)) * t * t)
    audio = (0.4 * np.sin(phase)).astype(np.float32)

    profile = third_party_feature_profile_from_audio(audio, sr)
    flat = profile["flat"]

    assert flat["librosa_spectral_centroid_mean"] > 0.0
    assert flat["librosa_spectral_rolloff85_mean"] > 0.0
    assert flat["librosa_spectral_flatness_mean"] >= 0.0
    assert flat["librosa_spectral_centroid_slope_norm"] > 0.02


def test_loop_claim_helpers_consume_librosa_loop_evidence() -> None:
    facts = _facts(
        {
            "shape_vote": {"primary_shape": "ambiguous_phrase", "confidence": 0.20},
            "librosa_loop_confidence": 0.72,
            "librosa_tonal_confidence": 0.58,
            "librosa_percussive_confidence": 0.18,
            "librosa_harmonic_energy_ratio": 0.66,
            "librosa_onset_event_count": 4.0,
        }
    )

    assert InstrumentOneShotLoopMixin._measured_loop_structure_is_strong(facts)
    assert InstrumentLoopConflictMixin._has_music_loop_structure(facts)


def test_transition_body_uses_librosa_spectral_motion_as_supporting_slope() -> None:
    facts = _facts(
        {
            "shape_vote": {"primary_shape": "transition_riser", "confidence": 0.86},
            "centroid_slope_norm": 0.0,
            "librosa_spectral_centroid_slope_norm": 0.11,
            "onset_span_ratio": 0.12,
            "sustained_tonal_frame_ratio": 0.10,
            "non_event_tonal_ratio": 0.10,
            "physics_subpanels": {
                "flat": {
                    "fx_transition_authority_score": 0.58,
                    "fx_riser_build_score": 0.42,
                    "fx_whoosh_sweep_score": 0.20,
                    "fx_reverse_score": 0.10,
                    "fx_impact_score": 0.10,
                    "fx_motion_score": 0.40,
                }
            },
        }
    )

    assert MeasuredTransitionFxClaimProducer()._facts_support_measured_transition_body(facts)


from aaron_sound_sorter.voters.shape_voter import classify_shape


def test_shape_voter_treats_librosa_as_supporting_evidence_not_primary_owner() -> None:
    facts = _facts(
        {
            "duration_sec": 2.4,
            "librosa_loop_confidence": 0.82,
            "librosa_onset_event_count": 7.0,
            "librosa_onset_density_hz": 2.9,
            "librosa_tonal_confidence": 0.76,
            "librosa_harmonic_energy_ratio": 0.78,
            "librosa_percussive_confidence": 0.12,
            "librosa_percussive_energy_ratio": 0.10,
        }
    )
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence=facts.evidence,
        feature_values_by_name={
            "pitch_confidence": 0.30,
            "f0_voiced_ratio": 0.32,
            "loop_pitched_event_ratio": 0.52,
            "loop_sustained_tonal_frame_ratio": 0.45,
            "loop_non_event_tonal_ratio": 0.40,
            "spectral_flatness_mean": 0.04,
            "spectral_entropy_mean": 0.20,
            "onset_span_ratio": 0.58,
            "temporal_centroid_ratio": 0.48,
            "attack_rise_time_norm": 0.12,
            "tail_energy_ratio": 0.25,
            "loop_mean_event_low_ratio": 0.10,
            "loop_mean_event_high_ratio": 0.10,
            "loop_drumlike_frame_ratio": 0.05,
            "loop_percussive_event_ratio": 0.05,
            "onset_interval_regularity": 1.0,
            "event_rate_hz": 0.0,
            "log_transient_count": 0.0,
        },
    )

    evidence = classify_shape(facts.feature_values_by_name, facts=facts)

    assert evidence.primary_shape == "single_hit"
    assert evidence.true_repetition_score < 0.35
    assert InstrumentOneShotLoopMixin._measured_loop_structure_is_strong(facts)
