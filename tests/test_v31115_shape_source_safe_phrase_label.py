from __future__ import annotations

from pathlib import Path

import numpy as np

from aaron_sound_sorter.core import FEATURE_NAMES, FP_SIZE
from aaron_sound_sorter.domain.facts import build_shared_audio_facts
from aaron_sound_sorter.domain.models import AudioPhysics
from aaron_sound_sorter.voters.shape_voter import ShapeVoter, shape_compatible_tops


def _fp(**values: float) -> np.ndarray:
    fp = np.zeros((FP_SIZE,), dtype=np.float32)
    for name, value in values.items():
        fp[FEATURE_NAMES.index(name)] = float(value)
    return fp


def _voiced_formant_phrase_physics() -> AudioPhysics:
    return AudioPhysics(
        Path("source_safe_shape.wav"),
        _fp(
            log_transient_count=np.log1p(5.0),
            event_rate_hz=0.9,
            onset_span_ratio=0.62,
            onset_interval_regularity=0.30,
            temporal_centroid_ratio=0.48,
            attack_rise_time_norm=0.18,
            tail_energy_ratio=0.48,
            pitch_confidence=0.88,
            f0_voiced_ratio=0.86,
            formant_like_peak_spacing=2.2,
            loop_pitched_event_ratio=0.82,
            loop_sustained_tonal_frame_ratio=0.70,
            loop_non_event_tonal_ratio=0.66,
            loop_percussive_event_ratio=0.06,
            loop_drumlike_frame_ratio=0.03,
            spectral_flatness_mean=0.06,
            harmonic_energy_ratio=0.65,
            harmonic_to_noise_ratio=0.60,
            sub_bass_ratio_lt_150hz=0.03,
            bass_ratio_150_500hz=0.10,
            mid_ratio_500_2000hz=0.55,
            presence_ratio_2000_8000hz=0.18,
            air_ratio_gt_8000hz=0.05,
            stereo_width=0.18,
        ),
        2.4,
        "ok",
    )


def test_shape_voter_uses_source_safe_voiced_phrase_label() -> None:
    physics = _voiced_formant_phrase_physics()
    facts = build_shared_audio_facts(physics)

    shape = ShapeVoter().vote(physics, facts, {}).diagnostics["shape_vote"]
    shape_score_names = {str(name) for name, _score in shape["shape_scores"]}

    assert shape["primary_shape"] == "pitched_phrase_shape"
    assert shape["secondary_shape"] != "vocal_phrase"
    assert "vocal_phrase" not in shape_score_names


def test_source_safe_voiced_phrase_shape_does_not_claim_fx_compatibility() -> None:
    assert shape_compatible_tops("pitched_phrase_shape") == ["Instruments", "Textures"]


def test_legacy_vocal_phrase_shape_alias_remains_for_old_synthetic_panels() -> None:
    # Existing regression tests still inject the older shape name directly.
    # ShapeVoter no longer emits it, but downstream compatibility is kept so old
    # panels do not become meaningless while they are migrated case by case.
    assert shape_compatible_tops("vocal_phrase") == ["Instruments", "FX"]
