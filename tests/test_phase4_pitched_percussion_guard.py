#!/usr/bin/env python3
"""Regression tests for pitched/tiny percussion guards.

These tests lock the bug Aaron found in real files:
- pitched bell/metallic percussion must not be laundered into trumpet/brass
- tiny low-body percussion hits must not stay in guitar just because onset count is 0
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
            "Drums/Percussion/Generic Percussion/One Shots",
            "Drums/Toms/Generic Tom/One Shots",
            "Drums/Percussion/Bells and Metallic Percussion/One Shots",
            "Instruments/Brass/Trumpet/One Shots",
            "Instruments/Guitar/Acoustic Guitar/One Shots",
        ],
        "top_by_label": {
            "Drums/Percussion/Generic Percussion/One Shots": "Drums",
            "Drums/Toms/Generic Tom/One Shots": "Drums",
            "Drums/Percussion/Bells and Metallic Percussion/One Shots": "Drums",
            "Instruments/Brass/Trumpet/One Shots": "Instruments",
            "Instruments/Guitar/Acoustic Guitar/One Shots": "Instruments",
        },
    }


def test_pitched_short_percussion_does_not_switch_from_drums_to_trumpet():
    m = load_module()
    brain = minimal_brain()
    x = fp(
        m,
        log_transient_count=np.log1p(1.0),
        spectral_flatness_mean=0.14,
        spectral_entropy_mean=0.25,
        temporal_centroid_ratio=0.096,
        attack_rise_time_norm=0.0004,
        tail_energy_ratio=0.052,
        pitch_confidence=0.64,
        mid_ratio_500_2000hz=0.59,
        presence_ratio_2000_8000hz=0.40,
    )
    top5 = [
        ("Instruments/Brass/Trumpet/One Shots", 0.10),
        ("Drums/Percussion/Generic Percussion/One Shots", 0.12),
        ("Drums/Percussion/Bells and Metallic Percussion/One Shots", 0.20),
    ]
    label, top, action, reason = m.physical_family_guard_decision(
        brain,
        x,
        "Drums/Percussion/Generic Percussion/One Shots",
        "Drums",
        "Drums",
        top5,
        0.789,
    )
    # No guard action is the desired result here: the already-Drums candidate
    # must not be laundered into Trumpet just because it is pitched/tonal.
    assert action in {"pass", "", "review"}
    assert not label.startswith("Instruments/")
    assert "tonal_non_drum_to_instruments" not in reason


def test_tiny_low_body_hit_switches_from_guitar_to_drums_even_when_onset_count_is_zero():
    m = load_module()
    brain = minimal_brain()
    x = fp(
        m,
        log_transient_count=0.0,  # exact real failure mode: onset detector missed the tiny hit
        spectral_flatness_mean=0.34,
        spectral_entropy_mean=0.41,
        sub_bass_ratio_lt_150hz=0.005,
        bass_ratio_150_500hz=0.87,
        temporal_centroid_ratio=0.17,
        attack_rise_time_norm=0.016,
        tail_energy_ratio=0.151,
        pitch_confidence=0.65,
    )
    top5 = [
        ("Instruments/Guitar/Acoustic Guitar/One Shots", 0.10),
        ("Drums/Toms/Generic Tom/One Shots", 0.20),
        ("Drums/Percussion/Bells and Metallic Percussion/One Shots", 0.30),
    ]
    label, top, action, reason = m.physical_family_guard_decision(
        brain,
        x,
        "Instruments/Guitar/Acoustic Guitar/One Shots",
        "Instruments",
        "Instruments",
        top5,
        0.068,
    )
    assert action == "switch"
    assert top == "Drums"
    assert label.startswith("Drums/")
    assert ("tiny_real_hit=True" in reason) or ("low_body_pitched_ring" in reason)


def test_short_tonal_drum_candidate_does_not_get_laundered_to_guitar():
    m = load_module()
    brain = minimal_brain()
    x = fp(
        m,
        log_transient_count=np.log1p(1.0),
        spectral_flatness_mean=0.13,
        spectral_entropy_mean=0.25,
        bass_ratio_150_500hz=0.20,
        mid_ratio_500_2000hz=0.53,
        presence_ratio_2000_8000hz=0.12,
        temporal_centroid_ratio=0.075,
        attack_rise_time_norm=0.020,
        tail_energy_ratio=0.031,
        pitch_confidence=0.71,
    )
    top5 = [
        ("Instruments/Guitar/Acoustic Guitar/One Shots", 0.10),
        ("Drums/Percussion/Generic Percussion/One Shots", 0.12),
        ("Drums/Toms/Generic Tom/One Shots", 0.20),
    ]
    label, top, action, reason = m.physical_family_guard_decision(
        brain,
        x,
        "Drums/Percussion/Generic Percussion/One Shots",
        "Drums",
        "Drums",
        top5,
        0.805,
    )
    assert not label.startswith("Instruments/")
    assert "tonal_non_drum_to_instruments" not in reason


def test_tiny_low_body_hit_can_rescue_from_fx_blip_to_drums_candidate():
    m = load_module()
    brain = minimal_brain()
    # Add a generic FX one-shot to this local brain.
    brain["labels"].append("FX/Designed Noise FX/Blip/One Shots")
    brain["top_by_label"]["FX/Designed Noise FX/Blip/One Shots"] = "FX"
    x = fp(
        m,
        log_transient_count=0.0,
        spectral_flatness_mean=0.22,
        spectral_entropy_mean=0.36,
        bass_ratio_150_500hz=0.32,
        temporal_centroid_ratio=0.10,
        attack_rise_time_norm=0.025,
        tail_energy_ratio=0.12,
        pitch_confidence=0.35,
    )
    top5 = [
        ("FX/Designed Noise FX/Blip/One Shots", 0.10),
        ("Drums/Percussion/Generic Percussion/One Shots", 0.12),
        ("Drums/Toms/Generic Tom/One Shots", 0.20),
    ]
    label, top, action, reason = m.physical_family_guard_decision(
        brain,
        x,
        "FX/Designed Noise FX/Blip/One Shots",
        "FX",
        "FX",
        top5,
        0.095,
    )
    assert action == "switch"
    assert top == "Drums"
    assert label.startswith("Drums/")


def test_voice_like_shout_candidate_is_not_swallowed_by_drums():
    m = load_module()
    brain = minimal_brain()
    brain["labels"].append("FX/Human and Voice FX/Scream/One Shots")
    brain["top_by_label"]["FX/Human and Voice FX/Scream/One Shots"] = "FX"
    x = fp(
        m,
        log_transient_count=np.log1p(2.0),
        spectral_flatness_mean=0.21,
        spectral_entropy_mean=0.46,
        sub_bass_ratio_lt_150hz=0.02,
        bass_ratio_150_500hz=0.10,
        mid_ratio_500_2000hz=0.52,
        presence_ratio_2000_8000hz=0.30,
        air_ratio_gt_8000hz=0.05,
        temporal_centroid_ratio=0.28,
        onset_span_ratio=0.20,
        attack_rise_time_norm=0.055,
        tail_energy_ratio=0.16,
        pitch_confidence=0.52,
        formant_like_peak_spacing=0.32,
    )
    top5 = [
        ("Drums/Claps Snaps Slaps/Generic Clap/One Shots", 0.10),
        ("FX/Human and Voice FX/Scream/One Shots", 0.12),
        ("Drums/Rims and Sticks/Rimshot/One Shots", 0.18),
    ]
    label, top, action, reason = m.physical_family_guard_decision(
        brain,
        x,
        "Drums/Claps Snaps Slaps/Generic Clap/One Shots",
        "Drums",
        "Drums",
        top5,
        0.72,
    )
    assert action == "switch"
    assert top == "FX"
    assert label == "FX/Human and Voice FX/Scream/One Shots"
    assert "voice_human" in reason


def test_plain_snare_like_hit_is_not_stolen_by_voice_guard_without_voice_candidate():
    m = load_module()
    brain = minimal_brain()
    x = fp(
        m,
        log_transient_count=np.log1p(1.0),
        spectral_flatness_mean=0.40,
        spectral_entropy_mean=0.62,
        sub_bass_ratio_lt_150hz=0.05,
        bass_ratio_150_500hz=0.22,
        mid_ratio_500_2000hz=0.28,
        presence_ratio_2000_8000hz=0.34,
        air_ratio_gt_8000hz=0.13,
        temporal_centroid_ratio=0.08,
        attack_rise_time_norm=0.025,
        tail_energy_ratio=0.08,
        pitch_confidence=0.34,
    )
    top5 = [
        ("Drums/Percussion/Generic Percussion/One Shots", 0.10),
        ("Drums/Toms/Generic Tom/One Shots", 0.18),
    ]
    label, top, action, reason = m.physical_family_guard_decision(
        brain,
        x,
        "Drums/Percussion/Generic Percussion/One Shots",
        "Drums",
        "Drums",
        top5,
        0.48,
    )
    assert label in {"", "Drums/Percussion/Generic Percussion/One Shots"}
    assert action in {"pass", "", "review"}
    assert "voice_human" not in reason


def test_low_body_pitched_ring_with_drum_candidate_does_not_escape_to_guitar():
    m = load_module()
    brain = minimal_brain()
    # This synthetic shape mirrors the real failure class: very low/body-heavy,
    # strongly pitched, short, fast attack, but with a ringing tail.  It can be
    # a tom/bongo/tabla-like hit. If the learned top set already nominated Drums
    # above Guitar, the guard must not let Guitar win.
    x = fp(
        m,
        log_transient_count=np.log1p(1.0),
        spectral_flatness_mean=0.04,
        spectral_entropy_mean=0.22,
        sub_bass_ratio_lt_150hz=0.02,
        bass_ratio_150_500hz=0.88,
        mid_ratio_500_2000hz=0.08,
        presence_ratio_2000_8000hz=0.01,
        air_ratio_gt_8000hz=0.00,
        temporal_centroid_ratio=0.27,
        attack_rise_time_norm=0.007,
        tail_energy_ratio=0.44,
        pitch_confidence=0.78,
    )
    top5 = [
        ("Drums/World Percussion/Latin Percussion/Conga/One Shots", 0.10),
        ("Drums/Toms/Generic Tom/One Shots", 0.12),
        ("Instruments/Guitar/Acoustic Guitar/One Shots", 0.20),
    ]
    brain["labels"].append("Drums/World Percussion/Latin Percussion/Conga/One Shots")
    brain["top_by_label"]["Drums/World Percussion/Latin Percussion/Conga/One Shots"] = "Drums"
    label, top, action, reason = m.physical_family_guard_decision(
        brain,
        x,
        "Instruments/Guitar/Acoustic Guitar/One Shots",
        "Instruments",
        "Instruments",
        top5,
        0.35,
    )
    assert action == "switch"
    assert top == "Drums"
    assert label.startswith("Drums/")
    assert "low_body_pitched_ring" in reason


def test_low_body_short_bass_like_instrument_is_not_forced_to_drums_without_drum_candidate():
    m = load_module()
    brain = minimal_brain()
    x = fp(
        m,
        log_transient_count=np.log1p(1.0),
        spectral_flatness_mean=0.035,
        spectral_entropy_mean=0.25,
        sub_bass_ratio_lt_150hz=0.06,
        bass_ratio_150_500hz=0.82,
        mid_ratio_500_2000hz=0.10,
        temporal_centroid_ratio=0.24,
        attack_rise_time_norm=0.045,
        tail_energy_ratio=0.46,
        pitch_confidence=0.82,
    )
    top5 = [
        ("Instruments/Guitar/Acoustic Guitar/One Shots", 0.10),
        ("Instruments/Brass/Trumpet/One Shots", 0.20),
    ]
    label, top, action, reason = m.physical_family_guard_decision(
        brain,
        x,
        "Instruments/Guitar/Acoustic Guitar/One Shots",
        "Instruments",
        "Instruments",
        top5,
        0.35,
    )
    assert action in {"pass", "", "review"}
    assert "low_body_pitched_ring" not in reason
