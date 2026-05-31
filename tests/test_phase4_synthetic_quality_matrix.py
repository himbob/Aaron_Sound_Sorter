#!/usr/bin/env python3
"""Synthetic quality matrix for Phase 4 broad/deep category smoke tests.

These tests do not try to generate audio. They generate 100-feature fingerprints
that obey the same physical shapes the real extractor writes: duration,
transient count, temporal centroid, attack, tail, band ratios, pitch confidence,
flatness/entropy, and motion. The goal is to protect category physics before
running slow real-ZIP reviews.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "Aaron_Sound_Sorter.py"


def load_module():
    from aaron_sound_sorter import api

    return api


@pytest.fixture(scope="session")
def m():
    return load_module()


class FP:
    def __init__(self, module):
        self.m = module
        self.idx = {name: i for i, name in enumerate(module.FEATURE_NAMES)}

    def make(self, values: dict[str, float]) -> list[float]:
        x = np.zeros(self.m.FP_SIZE, dtype=np.float32)
        for name, value in values.items():
            assert name in self.idx, f"unknown feature: {name}"
            x[self.idx[name]] = float(value)
        return x.tolist()

    def row(self, label: str, values: dict[str, float], *, duration: float, structure: str = "one_shot", i: int = 0):
        return self.m.FeatureRow(
            path=f"/synthetic/quality_matrix/{label.replace('/', '_')}_{i:03d}.wav",
            group_key=label,
            label=label,
            top=label.split("/", 1)[0],
            structure=structure,
            duration_sec=float(duration),
            fingerprint=self.make(values),
            read_status="ok",
        )


def jitter(base: dict[str, float], i: int, amount: float = 0.012) -> dict[str, float]:
    out = dict(base)
    for offset, key in enumerate(sorted(out)):
        # deterministic, small, bounded jitter so each category has a range.
        val = float(out[key]) + (((i + offset) % 5) - 2) * amount
        # Keep normalized ratio-like features sane. Hz features are left positive.
        val = max(0.0, min(1.5, val)) if not key.endswith("_hz") and not key.endswith("_ms") else max(0.0, val)
        out[key] = val
    return out


def train_rows(
    f: FP, label: str, base: dict[str, float], *, count: int = 5, duration: float = 0.7, structure: str = "one_shot"
):
    return [f.row(label, jitter(base, i), duration=duration, structure=structure, i=i) for i in range(count)]


def predict_label(
    module, brain: dict, f: FP, base: dict[str, float], *, duration: float, structure: str = "one_shot"
) -> str:
    label, _top, _sim, _margin, _top5 = module.predict(brain, f.make(base), structure, duration)
    return label


# Canonical labels that exist in the Stage 4 brain tree and hit level 2/3 folders.
KICK = "Drums/Kick Drums/Generic Kick/One Shots"
SNARE = "Drums/Snares/Generic Snare/One Shots"
TOM = "Drums/Toms/Generic Tom/One Shots"
HAT = "Drums/Hi Hats/Closed Hat/One Shots"
CYMBAL = "Drums/Cymbals/Crash Cymbal/One Shots"
METAL_PERC = "Drums/Percussion/Bells and Metallic Percussion/One Shots"
DRUM_LOOP = "Drums/Drum Loops/Loops"
GUITAR = "Instruments/Guitar/Acoustic Guitar/One Shots"
TRUMPET = "Instruments/Brass/Trumpet/One Shots"
SYNTH_PAD = "Instruments/Synths/Synth Pad/One Shots"
SYNTH_LEAD = "Instruments/Synths/Synth Lead/One Shots"
INSTR_LOOP = "Instruments/Instrument Loops/Loops"
BIRD = "FX/Animals and Creatures/Bird/One Shots"
IMPACT = "FX/Impacts and Hits/Generic Impact/One Shots"
RISER = "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/One Shots"
WHOOSH = "FX/Structural and Transitional FX/Sweeps and Whooshes/Generic Whoosh or Sweep/One Shots"
GLITCH = "FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/One Shots"
WATER = "FX/Textures/Natural Ambience/Water/One Shots"


BASES = {
    KICK: dict(
        log_transient_count=np.log1p(1),
        sub_bass_ratio_lt_150hz=0.80,
        bass_ratio_150_500hz=0.18,
        presence_ratio_2000_8000hz=0.08,
        air_ratio_gt_8000hz=0.02,
        spectral_flatness_mean=0.12,
        spectral_entropy_mean=0.35,
        log_crest=1.20,
        temporal_centroid_ratio=0.07,
        attack_rise_time_norm=0.025,
        tail_energy_ratio=0.06,
        pitch_confidence=0.10,
    ),
    SNARE: dict(
        log_transient_count=np.log1p(1),
        bass_ratio_150_500hz=0.22,
        mid_ratio_500_2000hz=0.28,
        presence_ratio_2000_8000hz=0.35,
        air_ratio_gt_8000hz=0.12,
        spectral_flatness_mean=0.42,
        spectral_entropy_mean=0.62,
        log_crest=1.00,
        temporal_centroid_ratio=0.08,
        attack_rise_time_norm=0.025,
        tail_energy_ratio=0.08,
        pitch_confidence=0.22,
    ),
    TOM: dict(
        log_transient_count=np.log1p(1),
        sub_bass_ratio_lt_150hz=0.16,
        bass_ratio_150_500hz=0.58,
        mid_ratio_500_2000hz=0.23,
        presence_ratio_2000_8000hz=0.10,
        spectral_flatness_mean=0.18,
        spectral_entropy_mean=0.38,
        log_crest=0.92,
        temporal_centroid_ratio=0.14,
        attack_rise_time_norm=0.045,
        tail_energy_ratio=0.18,
        pitch_confidence=0.62,
        harmonic_energy_ratio=0.42,
        low_peak_frequency_hz=190.0,
    ),
    HAT: dict(
        log_transient_count=np.log1p(1),
        presence_ratio_2000_8000hz=0.50,
        air_ratio_gt_8000hz=0.30,
        spectral_flatness_mean=0.58,
        spectral_entropy_mean=0.78,
        log_crest=1.05,
        temporal_centroid_ratio=0.06,
        attack_rise_time_norm=0.015,
        tail_energy_ratio=0.04,
        pitch_confidence=0.08,
        attack_high_ratio=0.70,
    ),
    CYMBAL: dict(
        log_transient_count=np.log1p(1),
        presence_ratio_2000_8000hz=0.42,
        air_ratio_gt_8000hz=0.38,
        spectral_flatness_mean=0.45,
        spectral_entropy_mean=0.72,
        log_crest=0.95,
        temporal_centroid_ratio=0.16,
        attack_rise_time_norm=0.02,
        tail_energy_ratio=0.34,
        pitch_confidence=0.12,
        attack_high_ratio=0.68,
        tail_high_ratio=0.42,
    ),
    METAL_PERC: dict(
        log_transient_count=np.log1p(1),
        mid_ratio_500_2000hz=0.20,
        presence_ratio_2000_8000hz=0.45,
        air_ratio_gt_8000hz=0.18,
        spectral_flatness_mean=0.16,
        spectral_entropy_mean=0.32,
        temporal_centroid_ratio=0.10,
        attack_rise_time_norm=0.02,
        tail_energy_ratio=0.12,
        pitch_confidence=0.66,
        inharmonicity=0.55,
        spectral_peak_count=7.0,
        top1_peak_frequency_hz=1250.0,
    ),
    DRUM_LOOP: dict(
        log_transient_count=np.log1p(10),
        spectral_flatness_mean=0.36,
        spectral_entropy_mean=0.62,
        temporal_centroid_ratio=0.48,
        onset_interval_regularity=0.78,
        onset_span_ratio=0.92,
        event_rate_hz=4.0,
        attack_rise_time_norm=0.03,
        tail_energy_ratio=0.36,
        pitch_confidence=0.18,
        presence_ratio_2000_8000hz=0.33,
    ),
    GUITAR: dict(
        log_transient_count=np.log1p(1),
        spectral_flatness_mean=0.05,
        spectral_entropy_mean=0.42,
        temporal_centroid_ratio=0.18,
        attack_rise_time_norm=0.055,
        tail_energy_ratio=0.28,
        pitch_confidence=0.82,
        harmonic_energy_ratio=0.70,
        fundamental_dominance_ratio=0.48,
        mid_ratio_500_2000hz=0.46,
        presence_ratio_2000_8000hz=0.20,
        attack_pitch_confidence=0.70,
        body_pitch_confidence=0.76,
        spectral_peak_count=8.0,
    ),
    TRUMPET: dict(
        log_transient_count=np.log1p(1),
        spectral_flatness_mean=0.04,
        spectral_entropy_mean=0.34,
        temporal_centroid_ratio=0.34,
        attack_rise_time_norm=0.12,
        tail_energy_ratio=0.42,
        pitch_confidence=0.88,
        harmonic_energy_ratio=0.78,
        body_pitch_confidence=0.84,
        mid_ratio_500_2000hz=0.54,
        presence_ratio_2000_8000hz=0.28,
        spectral_peak_count=10.0,
        formant_like_peak_spacing=0.35,
    ),
    SYNTH_PAD: dict(
        log_transient_count=np.log1p(0),
        spectral_flatness_mean=0.06,
        spectral_entropy_mean=0.42,
        temporal_centroid_ratio=0.52,
        attack_rise_time_norm=0.36,
        tail_energy_ratio=0.74,
        stereo_width=0.42,
        pitch_confidence=0.72,
        harmonic_energy_ratio=0.66,
        body_pitch_confidence=0.70,
        tail_pitch_confidence=0.64,
    ),
    SYNTH_LEAD: dict(
        log_transient_count=np.log1p(1),
        spectral_flatness_mean=0.07,
        spectral_entropy_mean=0.38,
        temporal_centroid_ratio=0.24,
        attack_rise_time_norm=0.07,
        tail_energy_ratio=0.34,
        pitch_confidence=0.85,
        harmonic_energy_ratio=0.70,
        mid_ratio_500_2000hz=0.42,
        presence_ratio_2000_8000hz=0.32,
    ),
    INSTR_LOOP: dict(
        log_transient_count=np.log1p(8),
        spectral_flatness_mean=0.08,
        spectral_entropy_mean=0.38,
        temporal_centroid_ratio=0.48,
        onset_interval_regularity=0.72,
        onset_span_ratio=0.90,
        event_rate_hz=4.5,
        pitch_confidence=0.82,
        harmonic_energy_ratio=0.68,
        mid_ratio_500_2000hz=0.45,
        presence_ratio_2000_8000hz=0.24,
        attack_rise_time_norm=0.04,
        tail_energy_ratio=0.48,
    ),
    BIRD: dict(
        log_transient_count=np.log1p(4),
        spectral_flatness_mean=0.08,
        spectral_entropy_mean=0.34,
        temporal_centroid_ratio=0.35,
        onset_span_ratio=0.38,
        event_rate_hz=5.0,
        pitch_confidence=0.78,
        f0_slope_cents_per_sec=480.0,
        top1_peak_frequency_hz=4200.0,
        presence_ratio_2000_8000hz=0.55,
        air_ratio_gt_8000hz=0.18,
        tail_energy_ratio=0.18,
    ),
    IMPACT: dict(
        log_transient_count=np.log1p(1),
        sub_bass_ratio_lt_150hz=0.34,
        bass_ratio_150_500hz=0.34,
        spectral_flatness_mean=0.34,
        spectral_entropy_mean=0.64,
        log_crest=1.10,
        temporal_centroid_ratio=0.15,
        attack_rise_time_norm=0.025,
        tail_energy_ratio=0.55,
        stereo_width=0.36,
        pitch_confidence=0.12,
    ),
    RISER: dict(
        log_transient_count=np.log1p(4),
        spectral_flux_mean=0.42,
        centroid_slope_norm=0.32,
        spectral_flatness_mean=0.30,
        spectral_entropy_mean=0.66,
        temporal_centroid_ratio=0.58,
        onset_span_ratio=0.80,
        event_rate_hz=2.2,
        tail_energy_ratio=0.72,
        stereo_width=0.45,
        pitch_confidence=0.36,
        f0_slope_cents_per_sec=720.0,
    ),
    WHOOSH: dict(
        log_transient_count=np.log1p(1),
        spectral_flux_mean=0.35,
        centroid_slope_norm=-0.18,
        spectral_flatness_mean=0.62,
        spectral_entropy_mean=0.78,
        temporal_centroid_ratio=0.45,
        tail_energy_ratio=0.50,
        stereo_width=0.55,
        pitch_confidence=0.10,
        air_ratio_gt_8000hz=0.35,
    ),
    GLITCH: dict(
        log_transient_count=np.log1p(7),
        spectral_flux_mean=0.55,
        spectral_flux_variance=0.40,
        spectral_flatness_mean=0.46,
        spectral_entropy_mean=0.70,
        temporal_centroid_ratio=0.42,
        onset_interval_regularity=0.22,
        onset_span_ratio=0.55,
        event_rate_hz=9.0,
        attack_rise_time_norm=0.018,
        tail_energy_ratio=0.16,
        pitch_confidence=0.18,
    ),
    WATER: dict(
        log_transient_count=np.log1p(5),
        spectral_flatness_mean=0.70,
        spectral_entropy_mean=0.82,
        temporal_centroid_ratio=0.55,
        onset_span_ratio=0.85,
        event_rate_hz=3.0,
        tail_energy_ratio=0.75,
        stereo_width=0.48,
        pitch_confidence=0.08,
        spectral_flux_mean=0.20,
        air_ratio_gt_8000hz=0.18,
    ),
}


def build_quality_brain(m):
    f = FP(m)
    rows = []
    for label, base in BASES.items():
        duration = 3.0 if label in {DRUM_LOOP, INSTR_LOOP, RISER, WATER, SYNTH_PAD} else 0.7
        if label == IMPACT:
            duration = 1.4
        if label == WHOOSH:
            duration = 1.8
        if label == GLITCH:
            duration = 0.9
        structure = "loop" if label in {DRUM_LOOP, INSTR_LOOP} else "one_shot"
        rows.extend(train_rows(f, label, base, count=5, duration=duration, structure=structure))
    return f, m.build_brain(rows, max_centroids=3)


@pytest.mark.parametrize(
    "label",
    [
        KICK,
        SNARE,
        TOM,
        HAT,
        CYMBAL,
        METAL_PERC,
        DRUM_LOOP,
        GUITAR,
        TRUMPET,
        SYNTH_PAD,
        SYNTH_LEAD,
        INSTR_LOOP,
        BIRD,
        IMPACT,
        RISER,
        WHOOSH,
        GLITCH,
        WATER,
    ],
)
def test_synthetic_quality_matrix_predicts_major_level2_level3_shapes(m, label):
    f, brain = build_quality_brain(m)
    duration = 3.0 if label in {DRUM_LOOP, INSTR_LOOP, RISER, WATER, SYNTH_PAD} else 0.7
    if label == IMPACT:
        duration = 1.4
    if label == WHOOSH:
        duration = 1.8
    if label == GLITCH:
        duration = 0.9
    structure = "loop" if label in {DRUM_LOOP, INSTR_LOOP} else "one_shot"
    predicted = predict_label(
        m, brain, f, jitter(BASES[label], 100, amount=0.01), duration=duration, structure=structure
    )
    assert predicted == label


def test_pitched_percussion_and_clean_trumpet_stay_separate_in_quality_matrix(m):
    f, brain = build_quality_brain(m)
    metal = predict_label(m, brain, f, jitter(BASES[METAL_PERC], 201), duration=0.62)
    trumpet = predict_label(m, brain, f, jitter(BASES[TRUMPET], 202), duration=1.05)
    assert metal == METAL_PERC
    assert trumpet == TRUMPET


def test_tonal_instrument_loop_beats_drum_loop_when_loop_is_pitched_and_harmonic(m):
    f, brain = build_quality_brain(m)
    predicted = predict_label(m, brain, f, jitter(BASES[INSTR_LOOP], 300), duration=3.0, structure="loop")
    assert predicted == INSTR_LOOP


def test_noisy_nonpitched_drum_loop_beats_instrument_loop(m):
    f, brain = build_quality_brain(m)
    predicted = predict_label(m, brain, f, jitter(BASES[DRUM_LOOP], 301), duration=3.0, structure="loop")
    assert predicted == DRUM_LOOP
