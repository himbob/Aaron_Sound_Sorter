#!/usr/bin/env python3
"""Regression tests for voice-like audio that raw ranking mistakes for drums.

These tests use synthetic fingerprints only. They do not read filenames or
source folders, and they do not require a fixed vocal category to exist.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "Aaron_Sound_Sorter.py"


def load_module():
    from aaron_sound_sorter import api

    return api


def fp(m, **values):
    idx = {name: i for i, name in enumerate(m.FEATURE_NAMES)}
    x = np.zeros(m.FP_SIZE, dtype=np.float32)
    for name, value in values.items():
        assert name in idx, name
        x[idx[name]] = float(value)
    return x.tolist()


def minimal_brain():
    return {
        "labels": [
            "Drums/Drum Loops/Loops",
            "Drums/Claps Snaps Slaps/Generic Clap/One Shots",
            "FX/Human and Voice FX/Spoken Voice/One Shots",
            "Instruments/Instrument Loops/Loops",
        ],
        "top_by_label": {
            "Drums/Drum Loops/Loops": "Drums",
            "Drums/Claps Snaps Slaps/Generic Clap/One Shots": "Drums",
            "FX/Human and Voice FX/Spoken Voice/One Shots": "FX",
            "Instruments/Instrument Loops/Loops": "Instruments",
        },
        "membership_gate_enabled": True,
    }


def test_voiced_rap_phrase_raw_ranked_as_drum_loop_goes_to_review_not_drums():
    m = load_module()
    brain = minimal_brain()
    x = fp(
        m,
        log_transient_count=np.log1p(33.0),
        sub_bass_ratio_lt_150hz=0.026,
        bass_ratio_150_500hz=0.269,
        mid_ratio_500_2000hz=0.546,
        presence_ratio_2000_8000hz=0.078,
        air_ratio_gt_8000hz=0.080,
        spectral_flatness_mean=0.425,
        spectral_entropy_mean=0.563,
        temporal_centroid_ratio=0.521,
        attack_rise_time_norm=0.222,
        onset_span_ratio=0.979,
        event_rate_hz=4.867,
        tail_energy_ratio=0.693,
        pitch_confidence=0.652,
        f0_voiced_ratio=0.660,
        harmonic_energy_ratio=0.168,
        attack_pitch_confidence=0.603,
        body_pitch_confidence=0.558,
        attack_noise_ratio=0.421,
    )
    label, top, action, reason = m.physical_family_guard_decision(
        brain,
        x,
        "Drums/Drum Loops/Loops",
        "Drums",
        "Drums",
        [("Drums/Drum Loops/Loops", 0.10)],
        6.855,
    )
    assert action == "review"
    assert top == "_TO_REVIEW"
    assert not label.startswith("Drums/")
    assert "voiced_formant_non_drum" in reason


def test_short_voiced_shout_raw_ranked_as_clap_goes_to_review_not_drums():
    m = load_module()
    brain = minimal_brain()
    x = fp(
        m,
        log_transient_count=np.log1p(1.0),
        sub_bass_ratio_lt_150hz=0.0002,
        bass_ratio_150_500hz=0.019,
        mid_ratio_500_2000hz=0.453,
        presence_ratio_2000_8000hz=0.524,
        air_ratio_gt_8000hz=0.004,
        zcr_mean=0.209,
        spectral_flatness_mean=0.416,
        spectral_entropy_mean=0.696,
        temporal_centroid_ratio=0.144,
        attack_rise_time_norm=0.135,
        event_rate_hz=1.436,
        tail_energy_ratio=0.008,
        pitch_confidence=0.422,
        f0_voiced_ratio=0.906,
        harmonic_energy_ratio=0.172,
        inharmonicity=0.188,
        attack_pitch_confidence=0.478,
        body_pitch_confidence=0.414,
        attack_high_ratio=0.701,
        body_high_ratio=0.354,
        attack_noise_ratio=0.536,
        noise_burst_duration_ms=232.2,
    )
    label, top, action, reason = m.physical_family_guard_decision(
        brain,
        x,
        "Drums/Claps Snaps Slaps/Generic Clap/One Shots",
        "Drums",
        "Drums",
        [("Drums/Claps Snaps Slaps/Generic Clap/One Shots", 0.10)],
        0.767,
    )
    assert action == "review"
    assert top == "_TO_REVIEW"
    assert not label.startswith("Drums/")
    assert "voiced_formant_non_drum" in reason


def test_short_unvoiced_snare_shape_stays_available_to_drums():
    m = load_module()
    brain = minimal_brain()
    x = fp(
        m,
        log_transient_count=0.0,
        sub_bass_ratio_lt_150hz=0.0003,
        bass_ratio_150_500hz=0.047,
        mid_ratio_500_2000hz=0.389,
        presence_ratio_2000_8000hz=0.561,
        air_ratio_gt_8000hz=0.003,
        spectral_flatness_mean=0.454,
        spectral_entropy_mean=0.795,
        temporal_centroid_ratio=0.196,
        attack_rise_time_norm=0.055,
        tail_energy_ratio=0.179,
        pitch_confidence=0.347,
        f0_voiced_ratio=0.333,
        harmonic_energy_ratio=0.132,
        attack_pitch_confidence=0.542,
        body_pitch_confidence=0.180,
        attack_high_ratio=0.648,
        body_high_ratio=0.522,
    )
    label, top, action, reason = m.physical_family_guard_decision(
        brain,
        x,
        "Drums/Claps Snaps Slaps/Generic Clap/One Shots",
        "Drums",
        "Drums",
        [("Drums/Claps Snaps Slaps/Generic Clap/One Shots", 0.10)],
        0.070,
    )
    assert action in {"pass", ""}
    assert label == ""
    assert top == ""
    assert "voiced_formant_non_drum" not in reason
