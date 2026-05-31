"""Regression tests for FX transition labels.

These tests use synthetic measured fingerprints only. They do not route from
filenames or source folders; transition labels must earn auto-place from
envelope/motion physics or move to an already nominated safer FX candidate.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from aaron_sound_sorter import api as mod


def fp(**vals):
    x = [0.0] * mod.FP_SIZE
    for name, value in vals.items():
        x[mod.FEATURE_NAMES.index(name)] = value
    return x


def brain():
    labels = [
        "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/One Shots",
        "FX/Structural and Transitional FX/Sweeps and Whooshes/Generic Whoosh or Sweep/One Shots",
        "FX/Textures/Natural Ambience/Water/One Shots",
        "FX/Human and Voice FX/Spoken Voice/One Shots",
    ]
    return {
        "labels": labels,
        "top_by_label": {label: "FX" for label in labels},
        "category_fact_profiles": {},
        "membership_gate_enabled": False,
        "model_ensemble_enabled": False,
        "committee_agreement_enabled": False,
        "adaptive_depth_placement_enabled": False,
    }


def test_short_stationary_foley_does_not_auto_place_as_riser_when_fx_alt_exists():
    b = brain()
    x = fp(
        log_transient_count=mod.math.log1p(2.0),
        temporal_centroid_ratio=0.24,
        onset_span_ratio=0.08,
        event_rate_hz=2.1,
        attack_rise_time_norm=0.03,
        tail_energy_ratio=0.18,
        centroid_slope_norm=0.01,
        spectral_flatness_mean=0.32,
        spectral_entropy_mean=0.52,
        pitch_confidence=0.38,
    )
    label, top, action, reason = mod.physical_family_guard_decision(
        b,
        x,
        "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/One Shots",
        "FX",
        "FX",
        [
            ("FX/Structural and Transitional FX/Risers and Builds/Generic Riser/One Shots", 0.1),
            ("FX/Textures/Natural Ambience/Water/One Shots", 0.2),
        ],
        0.72,
    )
    assert action == "switch"
    assert top == "FX"
    assert label == "FX/Textures/Natural Ambience/Water/One Shots"
    assert "unsupported_fx_transition" in reason


def test_unsupported_transition_without_safe_fx_alt_goes_to_review():
    b = brain()
    x = fp(
        log_transient_count=mod.math.log1p(3.0),
        temporal_centroid_ratio=0.35,
        onset_span_ratio=0.12,
        event_rate_hz=4.0,
        attack_rise_time_norm=0.05,
        tail_energy_ratio=0.28,
        centroid_slope_norm=0.02,
        spectral_flatness_mean=0.40,
        spectral_entropy_mean=0.55,
        pitch_confidence=0.50,
    )
    label, top, action, reason = mod.physical_family_guard_decision(
        b,
        x,
        "FX/Structural and Transitional FX/Sweeps and Whooshes/Generic Whoosh or Sweep/One Shots",
        "FX",
        "FX",
        [("FX/Structural and Transitional FX/Sweeps and Whooshes/Generic Whoosh or Sweep/One Shots", 0.1)],
        0.85,
    )
    assert action == "review"
    assert top == "_TO_REVIEW"
    assert label == "_TO_REVIEW/Conflicting Evidence/Unsupported FX Transition Physics"
    assert "unsupported_fx_transition" in reason


def test_real_transition_envelope_can_remain_fx_transition():
    b = brain()
    x = fp(
        log_transient_count=mod.math.log1p(4.0),
        temporal_centroid_ratio=0.76,
        onset_span_ratio=0.50,
        event_rate_hz=2.3,
        attack_rise_time_norm=0.42,
        tail_energy_ratio=0.96,
        centroid_slope_norm=0.14,
        spectral_flatness_mean=0.34,
        spectral_entropy_mean=0.62,
        pitch_confidence=0.46,
    )
    label, top, action, reason = mod.physical_family_guard_decision(
        b,
        x,
        "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/One Shots",
        "FX",
        "FX",
        [("FX/Structural and Transitional FX/Risers and Builds/Generic Riser/One Shots", 0.1)],
        2.4,
    )
    assert action == "pass"
    assert top == ""
    assert label == ""
    assert "no_action" in reason
