"""Shared structural fact builder."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from aaron_sound_sorter.core import FEATURE_NAMES, FP_SIZE
from aaron_sound_sorter.domain.models import AudioPhysics, SharedAudioFacts
from aaron_sound_sorter.domain.physics_subpanels import build_low_level_physics_subpanels
from aaron_sound_sorter.domain.policies import SharedFactsPolicy
from aaron_sound_sorter.domain.roles import measured_roles_from_features
from aaron_sound_sorter.training_labels import safe_feature_value


def fingerprint_array(fingerprint: np.ndarray) -> np.ndarray:
    """Return a finite FP_SIZE fingerprint vector."""
    values = np.asarray(fingerprint, dtype=np.float32).reshape(-1)
    if values.size < FP_SIZE:
        padded = np.zeros((FP_SIZE,), dtype=np.float32)
        padded[: values.size] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return padded
    return np.nan_to_num(values[:FP_SIZE], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def finite_float(value: Any, default: float = 0.0) -> float:
    """Return a finite float or a default."""
    try:
        number = float(value)
        if math.isfinite(number):
            return number
    except Exception:
        pass
    return float(default)


def named_feature_values(fingerprint: np.ndarray) -> dict[str, float]:
    """Return every fingerprint value keyed by its stable feature name.

    This keeps all measured audio evidence available from the first shared
    fact object instead of reducing the file to a small structure-only summary.
    """
    fp = fingerprint_array(fingerprint)
    values: dict[str, float] = {}
    for index, name in enumerate(FEATURE_NAMES[:FP_SIZE]):
        values[str(name)] = round(finite_float(fp[index], 0.0), 6)
    return values


def grouped_feature_values(feature_values: dict[str, float]) -> dict[str, dict[str, float]]:
    """Group the complete named feature set for easier diagnostics.

    Groups are intentionally broad and non-exclusive in spirit.  They do not
    choose a category; they make the 100 facts readable to voters and reports.
    """
    groups: dict[str, dict[str, float]] = {
        "mfcc_timbre_mean": {},
        "mfcc_timbre_movement": {},
        "spectral_timbre": {},
        "envelope_transient_structure": {},
        "rhythm_loop_structure": {},
        "stereo_spatial": {},
        "pitch_harmonic": {},
        "attack_body_tail": {},
        "low_end_identity": {},
        "loop_long_population": {},
    }
    for name, value in feature_values.items():
        if name.startswith("mfcc_mu_"):
            groups["mfcc_timbre_mean"][name] = value
        elif name.startswith("mfcc_std_"):
            groups["mfcc_timbre_movement"][name] = value
        elif name in {
            "log_rolloff85_hz",
            "spectral_flux_mean",
            "spectral_flatness_mean",
            "spectral_entropy_mean",
            "centroid_slope_norm",
            "presence_ratio_2000_8000hz",
            "air_ratio_gt_8000hz",
            "zcr_mean",
            "spectral_flux_variance",
            "spectral_peak_count",
            "top1_peak_frequency_hz",
            "top2_peak_frequency_hz",
            "top3_peak_frequency_hz",
            "peak_bandwidth_mean_hz",
            "spectral_peak_stability",
            "formant_like_peak_spacing",
            "spectral_envelope_slope",
        }:
            groups["spectral_timbre"][name] = value
        elif name in {
            "log_crest",
            "log_decay_ratio",
            "log_transient_count",
            "attack_rise_time_norm",
            "tail_energy_ratio",
            "noise_burst_duration_ms",
            "high_band_decay_slope",
            "spectral_centroid_decay_slope",
            "noise_tail_decay_slope",
        }:
            groups["envelope_transient_structure"][name] = value
        elif name in {
            "temporal_centroid_ratio",
            "onset_interval_regularity",
            "onset_span_ratio",
            "event_rate_hz",
        }:
            groups["rhythm_loop_structure"][name] = value
        elif name in {"stereo_width", "mid_side_ratio"}:
            groups["stereo_spatial"][name] = value
        elif name in {
            "pitch_confidence",
            "f0_median_hz",
            "f0_voiced_ratio",
            "f0_stability_cents",
            "f0_slope_cents_per_sec",
            "harmonic_to_noise_ratio",
            "harmonic_peak_count",
            "harmonic_energy_ratio",
            "inharmonicity",
            "fundamental_dominance_ratio",
            "overtone_slope",
        }:
            groups["pitch_harmonic"][name] = value
        elif name.startswith("attack_") or name.startswith("body_") or name.startswith("tail_"):
            groups["attack_body_tail"][name] = value
        elif name in {
            "sub_bass_ratio_lt_150hz",
            "bass_ratio_150_500hz",
            "mid_ratio_500_2000hz",
            "low_peak_frequency_hz",
            "low_peak_bandwidth_hz",
            "sub_attack_time_ms",
            "sub_decay_time_ms",
            "sub_to_click_offset_ms",
            "kick_pitch_drop_cents",
            "sub_sustain_ratio",
        }:
            groups["low_end_identity"][name] = value
        elif name.startswith("loop_"):
            groups["loop_long_population"][name] = value
        else:
            groups.setdefault("other", {})[name] = value
    return {name: values for name, values in groups.items() if values}


def build_shared_audio_facts(
    physics: AudioPhysics,
    policy: SharedFactsPolicy | None = None,
) -> SharedAudioFacts:
    """Build broad structure facts from measured audio physics."""
    policy = policy or SharedFactsPolicy()
    fp = fingerprint_array(physics.fingerprint)
    duration = max(0.0, finite_float(physics.duration_sec, 0.0))
    read_status = str(physics.read_status or "")

    log_transients = safe_feature_value(fp, 35, 0.0)
    event_count = float(np.expm1(max(0.0, log_transients)))
    onset_span = safe_feature_value(fp, 45, 0.0)
    temporal_centroid = safe_feature_value(fp, 42, 0.5)
    event_rate = safe_feature_value(fp, 46, 0.0)
    attack_rise = safe_feature_value(fp, 44, 1.0)
    tail_energy = safe_feature_value(fp, 47, 0.0)
    regularity = safe_feature_value(fp, 43, 1.0)

    is_broken_or_tiny = (
        read_status != "ok" or duration <= policy.tiny_duration_sec or not math.isfinite(duration) or fp.size <= 0
    )
    is_long = duration >= policy.long_duration_sec
    loop_like = is_loop_like(
        duration=duration,
        event_count=event_count,
        onset_span=onset_span,
        temporal_centroid=temporal_centroid,
        event_rate=event_rate,
        regularity=regularity,
        policy=policy,
    )
    single_event_like = is_single_event_like(
        duration=duration,
        event_count=event_count,
        onset_span=onset_span,
        event_rate=event_rate,
        temporal_centroid=temporal_centroid,
        attack_rise=attack_rise,
        loop_like=loop_like,
        policy=policy,
    )
    short_hit_like = (
        not is_broken_or_tiny
        and duration <= policy.short_hit_max_duration_sec
        and single_event_like
        and temporal_centroid <= 0.42
        and attack_rise <= 0.24
    )

    feature_values = named_feature_values(fp)
    feature_groups = grouped_feature_values(feature_values)
    measured_roles = measured_roles_from_features(
        is_loop_like=bool((not is_broken_or_tiny) and loop_like),
        is_single_event_like=bool((not is_broken_or_tiny) and single_event_like),
        is_short_hit_like=bool(short_hit_like),
        is_long=bool(is_long),
        feature_values=feature_values,
    )

    direct_body_view = build_direct_body_view_evidence(physics, policy)
    physics_subpanels = build_low_level_physics_subpanels(feature_values)
    physics_subpanel_flat = physics_subpanels.get("flat", {}) if isinstance(physics_subpanels, dict) else {}
    structure_evidence = {
        "duration_sec": round(duration, 6),
        "read_status": read_status,
        "event_count_estimate": round(event_count, 6),
        "onset_span_ratio": round(float(onset_span), 6),
        "temporal_centroid_ratio": round(float(temporal_centroid), 6),
        "event_rate_hz": round(float(event_rate), 6),
        "attack_rise_time_norm": round(float(attack_rise), 6),
        "tail_energy_ratio": round(float(tail_energy), 6),
        "onset_interval_regularity": round(float(regularity), 6),
    }

    return SharedAudioFacts(
        is_broken_or_tiny=bool(is_broken_or_tiny),
        is_loop_like=bool((not is_broken_or_tiny) and loop_like),
        is_single_event_like=bool((not is_broken_or_tiny) and single_event_like),
        is_short_hit_like=bool(short_hit_like),
        is_long=bool(is_long),
        evidence={
            **structure_evidence,
            "feature_count": len(feature_values),
            "structure_facts": structure_evidence,
            "feature_values_by_name": feature_values,
            "feature_groups": feature_groups,
            "measured_roles": measured_roles.to_evidence(),
            "physics_subpanels": physics_subpanels,
            **physics_subpanel_flat,
            "direct_body_view": direct_body_view,
        },
        feature_values_by_name=feature_values,
        feature_count=len(feature_values),
        feature_vector=tuple(float(fp[index]) for index in range(FP_SIZE)),
        feature_groups=feature_groups,
    )


def build_direct_body_view_evidence(
    physics: AudioPhysics,
    policy: SharedFactsPolicy,
) -> dict[str, Any]:
    """Return auxiliary tail-reduced direct/body evidence for one file.

    The view is computed from onset/peak-centered audio windows. It supports
    voter comparison between the full file and the source body before long tails,
    reverb, or delay dominate the global fingerprint.
    """
    direct_fp = getattr(physics, "direct_body_fingerprint", None)
    if direct_fp is None:
        return {"status": "not_computed", "available": False}
    direct_duration = max(0.0, finite_float(getattr(physics, "direct_body_duration_sec", 0.0), 0.0))
    direct_status = str(getattr(physics, "direct_body_status", "not_computed") or "not_computed")
    direct_values = named_feature_values(fingerprint_array(direct_fp))

    log_transients = finite_float(direct_values.get("log_transient_count"), 0.0)
    event_count = float(np.expm1(max(0.0, log_transients)))
    onset_span = finite_float(direct_values.get("onset_span_ratio"), 0.0)
    temporal_centroid = finite_float(direct_values.get("temporal_centroid_ratio"), 0.5)
    event_rate = finite_float(direct_values.get("event_rate_hz"), 0.0)
    attack_rise = finite_float(direct_values.get("attack_rise_time_norm"), 1.0)
    regularity = finite_float(direct_values.get("onset_interval_regularity"), 1.0)
    direct_broken = direct_status != "ok" or direct_duration <= policy.tiny_duration_sec
    direct_loop_like = is_loop_like(
        duration=direct_duration,
        event_count=event_count,
        onset_span=onset_span,
        temporal_centroid=temporal_centroid,
        event_rate=event_rate,
        regularity=regularity,
        policy=policy,
    )
    direct_single_event = is_single_event_like(
        duration=direct_duration,
        event_count=event_count,
        onset_span=onset_span,
        event_rate=event_rate,
        temporal_centroid=temporal_centroid,
        attack_rise=attack_rise,
        loop_like=direct_loop_like,
        policy=policy,
    )
    direct_short_hit = (
        not direct_broken
        and direct_duration <= policy.short_hit_max_duration_sec
        and direct_single_event
        and temporal_centroid <= 0.42
        and attack_rise <= 0.24
    )
    direct_roles = measured_roles_from_features(
        is_loop_like=bool((not direct_broken) and direct_loop_like),
        is_single_event_like=bool((not direct_broken) and direct_single_event),
        is_short_hit_like=bool(direct_short_hit),
        is_long=bool(direct_duration >= policy.long_duration_sec),
        feature_values=direct_values,
    ).to_evidence()
    meta = getattr(physics, "direct_body_profile", {})
    if not isinstance(meta, dict):
        meta = {}
    return {
        "available": bool(not direct_broken),
        "status": direct_status,
        "duration_sec": round(float(direct_duration), 6),
        "event_count_estimate": round(float(event_count), 6),
        "is_loop_like": bool((not direct_broken) and direct_loop_like),
        "is_single_event_like": bool((not direct_broken) and direct_single_event),
        "is_short_hit_like": bool(direct_short_hit),
        "feature_values_by_name": direct_values,
        "feature_groups": grouped_feature_values(direct_values),
        "measured_roles": direct_roles,
        "profile": meta,
    }


def is_loop_like(
    *,
    duration: float,
    event_count: float,
    onset_span: float,
    temporal_centroid: float,
    event_rate: float,
    regularity: float,
    policy: SharedFactsPolicy,
) -> bool:
    """Return whether physics looks like repeated material."""
    event_proof = (
        event_count >= policy.loop_min_event_count
        and onset_span >= policy.loop_min_onset_span
        and event_rate >= policy.loop_min_event_rate_hz
    )
    time_proof = duration >= policy.loop_min_duration_sec and 0.18 <= temporal_centroid <= 0.86 and onset_span >= 0.45
    regular_proof = (
        duration >= policy.loop_min_duration_sec and event_count >= 3.0 and regularity <= 0.82 and onset_span >= 0.30
    )
    return bool(event_proof or time_proof or regular_proof)


def is_single_event_like(
    *,
    duration: float,
    event_count: float,
    onset_span: float,
    event_rate: float,
    temporal_centroid: float,
    attack_rise: float,
    loop_like: bool,
    policy: SharedFactsPolicy,
) -> bool:
    """Return whether physics looks like one main event."""
    count_proof = event_count <= policy.single_event_max_event_count and onset_span <= 0.32
    rate_proof = duration <= 1.70 and event_rate <= 0.45 and onset_span <= policy.single_event_max_onset_span
    shape_proof = temporal_centroid <= 0.40 and attack_rise <= 0.20 and onset_span <= policy.single_event_max_onset_span
    return bool((not loop_like) and (count_proof or rate_proof or shape_proof))
