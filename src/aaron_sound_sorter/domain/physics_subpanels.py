"""Source-name-blind low-level physics subpanels.

This module owns measured, category-oriented audio evidence.  The domain
``facts`` builder creates the neutral ``SharedAudioFacts`` contract; this
module translates the complete named feature dictionary into reusable physics
subpanels for voters and reports.  No function here may read producer file
names, source folders, ZIP member names, or sample-pack labels.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from aaron_sound_sorter.domain.physics_category_panels import build_category_panel_scores


def finite_float(value: Any, default: float = 0.0) -> float:
    """Return a finite float or a default."""
    try:
        number = float(value)
        if math.isfinite(number):
            return number
    except Exception:
        pass
    return float(default)


def clamp01(value: float) -> float:
    """Clamp a numeric score to the 0..1 evidence range."""
    try:
        number = float(value)
        if math.isfinite(number):
            return max(0.0, min(1.0, number))
    except Exception:
        pass
    return 0.0


def ramp(value: float, start: float, full: float) -> float:
    """Return 0 at start and 1 at full using a linear ramp."""
    if full <= start:
        return 0.0
    return clamp01((finite_float(value, 0.0) - float(start)) / (float(full) - float(start)))


def inverse_ramp(value: float, good_at_or_below: float, bad_at_or_above: float) -> float:
    """Return 1 below a good value and 0 at or above a bad value."""
    if bad_at_or_above <= good_at_or_below:
        return 0.0
    return clamp01(
        1.0
        - ((finite_float(value, 0.0) - float(good_at_or_below)) / (float(bad_at_or_above) - float(good_at_or_below)))
    )


def named_value(values: dict[str, float], name: str, default: float = 0.0) -> float:
    """Read one finite named feature value."""
    return finite_float(values.get(name, default), default)


def logged_event_count(values: dict[str, float]) -> float:
    """Return estimated event count from the stable log-count feature."""
    try:
        return float(np.expm1(max(0.0, named_value(values, "log_transient_count", 0.0))))
    except Exception:
        return 0.0


def round_score_map(scores: dict[str, Any]) -> dict[str, Any]:
    """Round float scores for compact manifests while preserving labels/bools."""
    rounded: dict[str, Any] = {}
    for key, value in scores.items():
        if isinstance(value, bool):
            rounded[key] = bool(value)
        elif isinstance(value, (int, np.integer)):
            rounded[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            rounded[key] = round(float(value), 6)
        else:
            rounded[key] = value
    return rounded


def build_low_level_physics_subpanels(feature_values: dict[str, float]) -> dict[str, Any]:
    """Build source-name-blind low-level physics subpanel evidence.

    These scores are not final folder decisions. They preserve the measured
    feature-family evidence that separates role, source family, and character:
    onset type, loop role, plucked strings, reeds/winds, struck keys, metallic
    percussion, FX motion, textures, voice/formants, and low-end source shape.
    """
    values = feature_values or {}
    duration_event_count = logged_event_count(values)
    event_count = duration_event_count
    event_rate = named_value(values, "event_rate_hz", 0.0)
    regularity = named_value(values, "onset_interval_regularity", 1.0)
    onset_span = named_value(values, "onset_span_ratio", 0.0)
    temporal = named_value(values, "temporal_centroid_ratio", 0.5)
    attack = named_value(values, "attack_rise_time_norm", 1.0)
    tail = named_value(values, "tail_energy_ratio", 0.0)
    decay = named_value(values, "log_decay_ratio", 0.0)

    pitch_conf = named_value(values, "pitch_confidence", 0.0)
    f0_voiced = named_value(values, "f0_voiced_ratio", 0.0)
    f0_stability = named_value(values, "f0_stability_cents", 9999.0)
    f0_slope = abs(named_value(values, "f0_slope_cents_per_sec", 0.0))
    harmonic = named_value(values, "harmonic_energy_ratio", 0.0)
    harmonic_to_noise = named_value(values, "harmonic_to_noise_ratio", 0.0)
    harmonic_peak_count = named_value(values, "harmonic_peak_count", 0.0)
    inharmonicity = named_value(values, "inharmonicity", 0.0)
    fundamental = named_value(values, "fundamental_dominance_ratio", 0.0)
    overtone_slope = named_value(values, "overtone_slope", 0.0)

    flatness = named_value(values, "spectral_flatness_mean", 0.0)
    entropy = named_value(values, "spectral_entropy_mean", 0.0)
    flux = named_value(values, "spectral_flux_mean", 0.0)
    flux_var = named_value(values, "spectral_flux_variance", 0.0)
    zcr = named_value(values, "zcr_mean", 0.0)
    centroid_slope = named_value(values, "centroid_slope_norm", 0.0)
    spectral_peak_count = named_value(values, "spectral_peak_count", 0.0)
    peak_stability = named_value(values, "spectral_peak_stability", 0.0)
    peak_bandwidth = named_value(values, "peak_bandwidth_mean_hz", 0.0)
    formant_spacing = named_value(values, "formant_like_peak_spacing", 0.0)
    spectral_env_slope = named_value(values, "spectral_envelope_slope", 0.0)

    sub = named_value(values, "sub_bass_ratio_lt_150hz", 0.0)
    bass = named_value(values, "bass_ratio_150_500hz", 0.0)
    mid = named_value(values, "mid_ratio_500_2000hz", 0.0)
    presence = named_value(values, "presence_ratio_2000_8000hz", 0.0)
    air = named_value(values, "air_ratio_gt_8000hz", 0.0)
    high = presence + air
    low_total = sub + bass

    attack_pitch = named_value(values, "attack_pitch_confidence", pitch_conf)
    body_pitch = named_value(values, "body_pitch_confidence", pitch_conf)
    tail_pitch = named_value(values, "tail_pitch_confidence", pitch_conf)
    attack_flatness = named_value(values, "attack_flatness", flatness)
    body_flatness = named_value(values, "body_flatness", flatness)
    tail_flatness = named_value(values, "tail_flatness", flatness)
    attack_entropy = named_value(values, "attack_entropy", entropy)
    body_entropy = named_value(values, "body_entropy", entropy)
    tail_entropy = named_value(values, "tail_entropy", entropy)
    attack_zcr = named_value(values, "attack_zcr", zcr)
    body_zcr = named_value(values, "body_zcr", zcr)
    tail_zcr = named_value(values, "tail_zcr", zcr)
    attack_high = named_value(values, "attack_high_ratio", high)
    body_high = named_value(values, "body_high_ratio", high)
    tail_high = named_value(values, "tail_high_ratio", high)
    attack_low = named_value(values, "attack_low_ratio", low_total)
    body_low = named_value(values, "body_low_ratio", low_total)
    tail_low = named_value(values, "tail_low_ratio", low_total)
    attack_noise = named_value(values, "attack_noise_ratio", attack_flatness)
    body_noise = named_value(values, "body_noise_ratio", body_flatness)
    tail_noise = named_value(values, "tail_noise_ratio", tail_flatness)
    noise_burst_ms = named_value(values, "noise_burst_duration_ms", 0.0)
    high_decay = named_value(values, "high_band_decay_slope", 0.0)
    centroid_decay = named_value(values, "spectral_centroid_decay_slope", 0.0)
    noise_tail_decay = named_value(values, "noise_tail_decay_slope", 0.0)

    loop_pitched = named_value(values, "loop_pitched_event_ratio", 0.0)
    loop_percussive = named_value(values, "loop_percussive_event_ratio", 0.0)
    loop_noisy = named_value(values, "loop_noisy_event_ratio", 0.0)
    loop_diversity = named_value(values, "loop_event_timbre_diversity", 0.0)
    loop_sustained = named_value(values, "loop_sustained_tonal_frame_ratio", 0.0)
    loop_drumlike = named_value(values, "loop_drumlike_frame_ratio", 0.0)
    loop_tonal_balance = named_value(values, "loop_tonal_to_percussive_balance", 0.0)
    loop_event_pitch = named_value(values, "loop_mean_event_pitch_confidence", 0.0)
    loop_event_noise = named_value(values, "loop_mean_event_noise_ratio", 0.0)
    loop_event_low = named_value(values, "loop_mean_event_low_ratio", 0.0)
    loop_event_high = named_value(values, "loop_mean_event_high_ratio", 0.0)
    loop_non_event_tonal = named_value(values, "loop_non_event_tonal_ratio", 0.0)

    low_peak_hz = named_value(values, "low_peak_frequency_hz", 0.0)
    low_peak_bw = named_value(values, "low_peak_bandwidth_hz", 0.0)
    sub_attack_ms = named_value(values, "sub_attack_time_ms", 0.0)
    sub_decay_ms = named_value(values, "sub_decay_time_ms", 0.0)
    sub_click_offset = named_value(values, "sub_to_click_offset_ms", 0.0)
    kick_drop = named_value(values, "kick_pitch_drop_cents", 0.0)
    sub_sustain = named_value(values, "sub_sustain_ratio", 0.0)
    stereo_width = named_value(values, "stereo_width", 0.0)
    mid_side = named_value(values, "mid_side_ratio", 1.0)

    fast_attack = inverse_ramp(attack, 0.006, 0.18)
    slow_attack = ramp(attack, 0.08, 0.42)
    early_centroid = inverse_ramp(temporal, 0.18, 0.58)
    eventful = max(ramp(event_count, 2.0, 14.0), ramp(event_rate, 1.2, 7.0))
    repeated_events = clamp01(
        0.38 * ramp(event_count, 4.0, 18.0)
        + 0.32 * ramp(onset_span, 0.32, 0.86)
        + 0.30 * inverse_ramp(regularity, 0.06, 0.55)
    )
    clean_tone = clamp01(
        0.34 * ramp(max(pitch_conf, body_pitch, loop_event_pitch), 0.35, 0.92)
        + 0.30 * ramp(max(harmonic, harmonic_to_noise, fundamental), 0.18, 0.86)
        + 0.20 * inverse_ramp(flatness, 0.035, 0.42)
        + 0.16 * inverse_ramp(entropy, 0.30, 0.78)
    )
    noisy_air = clamp01(
        0.34 * ramp(max(attack_noise, body_noise, tail_noise, flatness), 0.14, 0.55)
        + 0.24 * ramp(max(zcr, attack_zcr, body_zcr), 0.05, 0.26)
        + 0.22 * ramp(high, 0.035, 0.34)
        + 0.20 * ramp(entropy, 0.36, 0.82)
    )

    onset_panel = {
        "percussive_onset_score": clamp01(
            0.30 * fast_attack
            + 0.24 * early_centroid
            + 0.18 * ramp(max(attack_noise, attack_flatness), 0.10, 0.46)
            + 0.16 * ramp(max(attack_high, loop_event_high), 0.08, 0.48)
            + 0.12 * ramp(loop_percussive, 0.12, 0.90)
        ),
        "pitched_onset_score": clamp01(
            0.26 * fast_attack
            + 0.26 * ramp(max(attack_pitch, body_pitch, loop_event_pitch), 0.35, 0.92)
            + 0.20 * ramp(max(harmonic, fundamental), 0.20, 0.86)
            + 0.16 * inverse_ramp(max(attack_noise, attack_flatness), 0.04, 0.38)
            + 0.12 * ramp(loop_pitched, 0.35, 1.0)
        ),
        "scrape_onset_score": clamp01(
            0.30 * ramp(noise_burst_ms, 70.0, 420.0)
            + 0.24 * ramp(loop_noisy, 0.18, 0.85)
            + 0.18 * ramp(max(flux_var, event_rate), 0.20, 2.8)
            + 0.16 * ramp(max(attack_zcr, body_zcr, zcr), 0.10, 0.36)
            + 0.12 * ramp(entropy, 0.50, 0.90)
        ),
        "swell_onset_score": clamp01(
            0.36 * slow_attack
            + 0.24 * ramp(tail, 0.28, 0.92)
            + 0.18 * ramp(abs(centroid_slope), 0.06, 0.36)
            + 0.12 * ramp(flux, 0.12, 0.58)
            + 0.10 * inverse_ramp(event_count, 1.0, 8.0)
        ),
        "echo_tail_likelihood": clamp01(
            0.32 * ramp(tail, 0.26, 0.92)
            + 0.24 * ramp(event_count, 2.0, 8.0)
            + 0.18 * inverse_ramp(onset_span, 0.12, 0.55)
            + 0.14 * ramp(temporal, 0.34, 0.84)
            + 0.12 * inverse_ramp(loop_diversity, 0.05, 0.36)
        ),
        "true_repetition_likelihood": repeated_events,
    }

    plucked_panel = {
        "pluck_attack_score": clamp01(
            0.38 * onset_panel["pitched_onset_score"]
            + 0.24 * fast_attack
            + 0.16 * ramp(attack_high, 0.03, 0.22)
            + 0.12 * inverse_ramp(attack_noise, 0.03, 0.35)
            + 0.10 * ramp(loop_pitched, 0.40, 1.0)
        ),
        "string_decay_score": clamp01(
            0.28 * inverse_ramp(tail, 0.10, 0.78)
            + 0.22 * ramp(max(body_pitch, tail_pitch, loop_sustained), 0.45, 1.0)
            + 0.20 * inverse_ramp(abs(high_decay), 0.02, 0.48)
            + 0.16 * inverse_ramp(body_noise, 0.06, 0.38)
            + 0.14 * ramp(max(harmonic, fundamental), 0.28, 0.90)
        ),
        "harmonic_stack_score": clamp01(
            0.34 * ramp(harmonic, 0.25, 0.92)
            + 0.22 * ramp(harmonic_peak_count, 3.0, 12.0)
            + 0.18 * ramp(fundamental, 0.22, 0.82)
            + 0.14 * ramp(peak_stability, 0.08, 0.52)
            + 0.12 * inverse_ramp(inharmonicity, 0.05, 0.50)
        ),
        "inharmonicity_score": clamp01(
            0.42 * ramp(inharmonicity, 0.06, 0.48)
            + 0.22 * ramp(spectral_peak_count, 4.0, 18.0)
            + 0.18 * inverse_ramp(inharmonicity, 0.45, 0.90)
            + 0.18 * ramp(peak_stability, 0.06, 0.40)
        ),
        "pick_noise_score": clamp01(
            0.30 * ramp(attack_high, 0.045, 0.30)
            + 0.24 * ramp(attack_zcr, 0.06, 0.28)
            + 0.20 * fast_attack
            + 0.14 * ramp(max(presence, body_high), 0.08, 0.38)
            + 0.12 * inverse_ramp(air, 0.04, 0.24)
        ),
        "sustain_spectral_stability": clamp01(
            0.30 * ramp(peak_stability, 0.08, 0.56)
            + 0.24 * inverse_ramp(abs(spectral_env_slope), 0.02, 0.52)
            + 0.18 * inverse_ramp(abs(centroid_decay), 0.02, 0.52)
            + 0.16 * ramp(max(loop_sustained, loop_non_event_tonal), 0.50, 1.0)
            + 0.12 * inverse_ramp(body_entropy, 0.25, 0.78)
        ),
        "note_event_consistency": clamp01(
            0.30 * ramp(loop_pitched, 0.45, 1.0)
            + 0.24 * ramp(loop_event_pitch, 0.35, 0.92)
            + 0.22 * repeated_events
            + 0.14 * inverse_ramp(regularity, 0.04, 0.55)
            + 0.10 * inverse_ramp(loop_diversity, 0.04, 0.55)
        ),
        "guitar_body_band_score": clamp01(
            0.36 * ramp(mid + presence, 0.32, 0.96)
            + 0.24 * inverse_ramp(low_total, 0.08, 0.62)
            + 0.18 * inverse_ramp(air, 0.02, 0.24)
            + 0.14 * ramp(bass + mid, 0.22, 0.78)
            + 0.08 * inverse_ramp(body_noise, 0.04, 0.42)
        ),
    }
    plucked_panel["plucked_string_score"] = clamp01(
        0.18 * plucked_panel["pluck_attack_score"]
        + 0.16 * plucked_panel["string_decay_score"]
        + 0.16 * plucked_panel["harmonic_stack_score"]
        + 0.10 * plucked_panel["inharmonicity_score"]
        + 0.10 * plucked_panel["pick_noise_score"]
        + 0.12 * plucked_panel["sustain_spectral_stability"]
        + 0.10 * plucked_panel["note_event_consistency"]
        + 0.08 * plucked_panel["guitar_body_band_score"]
    )

    reed_panel = {
        "reed_noise_score": clamp01(
            0.30 * noisy_air
            + 0.22 * ramp(max(body_noise, tail_noise), 0.12, 0.45)
            + 0.18 * ramp(max(presence, high), 0.035, 0.26)
            + 0.16 * inverse_ramp(air, 0.04, 0.32)
            + 0.14 * ramp(formant_spacing, 0.15, 0.75)
        ),
        "vibrato_score": clamp01(
            0.30 * ramp(f0_slope, 5.0, 160.0)
            + 0.24 * ramp(f0_voiced, 0.52, 0.96)
            + 0.18 * inverse_ramp(f0_stability, 18.0, 220.0)
            + 0.16 * ramp(flux_var, 0.04, 0.34)
            + 0.12 * ramp(peak_stability, 0.08, 0.46)
        ),
        "legato_score": clamp01(
            0.28 * ramp(max(loop_sustained, loop_non_event_tonal), 0.58, 1.0)
            + 0.22 * ramp(f0_voiced, 0.62, 0.98)
            + 0.20 * inverse_ramp(onset_panel["percussive_onset_score"], 0.04, 0.42)
            + 0.16 * slow_attack
            + 0.14 * inverse_ramp(loop_percussive, 0.02, 0.22)
        ),
        "formant_envelope_score": clamp01(
            0.32 * ramp(formant_spacing, 0.18, 0.90)
            + 0.24 * ramp(spectral_peak_count, 3.0, 14.0)
            + 0.18 * ramp(peak_stability, 0.06, 0.46)
            + 0.14 * ramp(mid + presence, 0.30, 0.92)
            + 0.12 * inverse_ramp(peak_bandwidth, 220.0, 1800.0)
        ),
        "breath_attack_score": clamp01(
            0.30 * ramp(max(attack_noise, attack_zcr), 0.08, 0.36)
            + 0.22 * ramp(attack_high, 0.04, 0.28)
            + 0.20 * slow_attack
            + 0.16 * ramp(max(body_noise, tail_noise), 0.12, 0.46)
            + 0.12 * inverse_ramp(sub, 0.04, 0.30)
        ),
        "pitch_stability_score": clamp01(
            0.34 * ramp(f0_voiced, 0.55, 1.0)
            + 0.24 * inverse_ramp(f0_stability, 20.0, 260.0)
            + 0.20 * ramp(max(pitch_conf, body_pitch), 0.42, 0.94)
            + 0.12 * ramp(max(harmonic, harmonic_to_noise), 0.20, 0.78)
            + 0.10 * inverse_ramp(inharmonicity, 0.08, 0.46)
        ),
        "non_plucked_attack_score": clamp01(
            0.34 * inverse_ramp(plucked_panel["pluck_attack_score"], 0.16, 0.72)
            + 0.24 * slow_attack
            + 0.18 * inverse_ramp(plucked_panel["pick_noise_score"], 0.06, 0.48)
            + 0.14 * inverse_ramp(abs(high_decay), 0.02, 0.45)
            + 0.10 * ramp(max(loop_sustained, loop_non_event_tonal), 0.52, 1.0)
        ),
    }
    reed_panel["reed_wind_score"] = clamp01(
        0.16 * reed_panel["reed_noise_score"]
        + 0.12 * reed_panel["vibrato_score"]
        + 0.16 * reed_panel["legato_score"]
        + 0.16 * reed_panel["formant_envelope_score"]
        + 0.12 * reed_panel["breath_attack_score"]
        + 0.18 * reed_panel["pitch_stability_score"]
        + 0.10 * reed_panel["non_plucked_attack_score"]
    )

    keys_panel = {
        "hammer_attack_score": clamp01(
            0.30 * fast_attack
            + 0.24 * ramp(max(attack_flatness - body_flatness, attack_high - body_high), 0.04, 0.34)
            + 0.18 * ramp(inharmonicity, 0.08, 0.55)
            + 0.16 * clean_tone
            + 0.12 * ramp(mid, 0.30, 0.88)
        ),
        "tonal_decay_score": clamp01(
            0.28 * ramp(max(body_pitch, tail_pitch, loop_sustained), 0.48, 1.0)
            + 0.22 * inverse_ramp(tail_noise, 0.04, 0.36)
            + 0.18 * ramp(tail, 0.12, 0.78)
            + 0.16 * inverse_ramp(abs(centroid_decay), 0.02, 0.46)
            + 0.16 * clean_tone
        ),
        "partial_inharmonicity_score": clamp01(
            0.36 * ramp(inharmonicity, 0.08, 0.62)
            + 0.22 * ramp(harmonic_peak_count, 3.0, 14.0)
            + 0.18 * ramp(spectral_peak_count, 4.0, 20.0)
            + 0.14 * ramp(peak_stability, 0.05, 0.42)
            + 0.10 * inverse_ramp(air, 0.04, 0.26)
        ),
        "chord_density_score": clamp01(
            0.30 * ramp(spectral_peak_count, 6.0, 28.0)
            + 0.22 * ramp(harmonic_peak_count, 5.0, 18.0)
            + 0.18 * ramp(mid, 0.38, 0.92)
            + 0.16 * ramp(event_count, 2.0, 12.0)
            + 0.14 * inverse_ramp(fundamental, 0.18, 0.76)
        ),
        "sustain_tail_score": clamp01(
            0.34 * ramp(tail, 0.18, 0.86)
            + 0.22 * ramp(max(loop_sustained, loop_non_event_tonal), 0.55, 1.0)
            + 0.18 * ramp(tail_pitch, 0.38, 0.92)
            + 0.14 * inverse_ramp(tail_noise, 0.04, 0.38)
            + 0.12 * inverse_ramp(tail_entropy, 0.28, 0.82)
        ),
        "electric_piano_tine_score": clamp01(
            0.28 * ramp(mid, 0.45, 0.95)
            + 0.22 * inverse_ramp(flatness, 0.006, 0.12)
            + 0.18 * ramp(max(loop_sustained, body_pitch), 0.58, 1.0)
            + 0.16 * inverse_ramp(high, 0.02, 0.18)
            + 0.16 * ramp(peak_stability, 0.06, 0.46)
        ),
        "organ_steady_tone_score": clamp01(
            0.34 * ramp(max(loop_sustained, loop_non_event_tonal), 0.72, 1.0)
            + 0.24 * slow_attack
            + 0.18 * inverse_ramp(event_count, 1.0, 8.0)
            + 0.14 * ramp(max(harmonic, fundamental), 0.34, 0.92)
            + 0.10 * inverse_ramp(inharmonicity, 0.04, 0.32)
        ),
    }
    keys_panel["struck_keys_score"] = clamp01(
        0.18 * keys_panel["hammer_attack_score"]
        + 0.16 * keys_panel["tonal_decay_score"]
        + 0.14 * keys_panel["partial_inharmonicity_score"]
        + 0.14 * keys_panel["chord_density_score"]
        + 0.12 * keys_panel["sustain_tail_score"]
        + 0.14 * keys_panel["electric_piano_tine_score"]
        + 0.12 * keys_panel["organ_steady_tone_score"]
    )

    loop_panel = {
        "tempo_confidence": repeated_events,
        "pulse_clarity": clamp01(
            0.36 * repeated_events
            + 0.24 * inverse_ramp(regularity, 0.04, 0.45)
            + 0.20 * ramp(event_count, 4.0, 22.0)
            + 0.12 * ramp(onset_span, 0.38, 0.90)
            + 0.08 * inverse_ramp(loop_diversity, 0.04, 0.58)
        ),
        "onset_periodicity": clamp01(
            0.42 * inverse_ramp(regularity, 0.03, 0.45)
            + 0.26 * ramp(event_count, 4.0, 24.0)
            + 0.20 * ramp(event_rate, 1.0, 8.0)
            + 0.12 * ramp(onset_span, 0.34, 0.88)
        ),
        "bar_length_likelihood": clamp01(
            0.34 * ramp(event_count, 6.0, 32.0)
            + 0.24 * inverse_ramp(regularity, 0.04, 0.42)
            + 0.22 * ramp(onset_span, 0.50, 0.96)
            + 0.20 * ramp(max(loop_pitched, loop_percussive, loop_noisy), 0.35, 1.0)
        ),
        "loop_boundary_likelihood": clamp01(
            0.32 * ramp(onset_span, 0.50, 0.96)
            + 0.28 * inverse_ramp(temporal, 0.20, 0.62)
            + 0.20 * ramp(event_count, 4.0, 20.0)
            + 0.20 * inverse_ramp(tail, 0.10, 0.74)
        ),
        "delay_tail_likelihood": onset_panel["echo_tail_likelihood"],
        "phrase_likelihood": clamp01(
            0.26 * clean_tone
            + 0.22 * ramp(max(loop_pitched, loop_event_pitch), 0.45, 1.0)
            + 0.18 * ramp(event_count, 2.0, 12.0)
            + 0.16 * inverse_ramp(regularity, 0.10, 0.70)
            + 0.10 * ramp(onset_span, 0.28, 0.82)
            + 0.08 * ramp(tail, 0.14, 0.72)
        ),
        "one_shot_likelihood": clamp01(
            0.34 * fast_attack
            + 0.24 * inverse_ramp(event_count, 1.0, 4.0)
            + 0.20 * inverse_ramp(onset_span, 0.05, 0.42)
            + 0.14 * early_centroid
            + 0.08 * inverse_ramp(tail, 0.10, 0.68)
        ),
    }
    loop_panel["role_loop_score"] = clamp01(
        0.26 * loop_panel["tempo_confidence"]
        + 0.22 * loop_panel["pulse_clarity"]
        + 0.20 * loop_panel["onset_periodicity"]
        + 0.16 * loop_panel["bar_length_likelihood"]
        + 0.16 * loop_panel["loop_boundary_likelihood"]
        - 0.18 * loop_panel["delay_tail_likelihood"]
    )
    loop_panel["role_one_shot_score"] = loop_panel["one_shot_likelihood"]
    loop_panel["role_phrase_score"] = clamp01(
        0.42 * loop_panel["phrase_likelihood"]
        + 0.24 * clean_tone
        + 0.18 * ramp(event_count, 2.0, 10.0)
        + 0.16 * inverse_ramp(loop_panel["pulse_clarity"], 0.14, 0.86)
    )

    metallic_noise_score = clamp01(
        0.28 * ramp(max(inharmonicity, spectral_peak_count / 24.0), 0.18, 0.78)
        + 0.24 * ramp(high, 0.14, 0.58)
        + 0.18 * ramp(peak_stability, 0.05, 0.46)
        + 0.16 * ramp(noise_burst_ms, 20.0, 360.0)
        + 0.14 * inverse_ramp(low_total, 0.05, 0.45)
    )
    scrape_rasp_score = clamp01(
        0.32 * onset_panel["scrape_onset_score"]
        + 0.24 * ramp(loop_noisy, 0.20, 0.90)
        + 0.18 * ramp(max(zcr, body_zcr), 0.10, 0.36)
        + 0.14 * ramp(noise_burst_ms, 120.0, 650.0)
        + 0.12 * inverse_ramp(clean_tone, 0.12, 0.70)
    )
    drum_hit_score = clamp01(
        0.34 * onset_panel["percussive_onset_score"]
        + 0.20 * loop_panel["role_one_shot_score"]
        + 0.18 * ramp(loop_percussive, 0.12, 0.90)
        + 0.14 * ramp(max(attack_noise, attack_high), 0.08, 0.44)
        + 0.14 * inverse_ramp(clean_tone, 0.20, 0.78)
    )
    fx_motion_score = clamp01(
        0.28 * ramp(abs(centroid_slope), 0.06, 0.34)
        + 0.22 * onset_panel["swell_onset_score"]
        + 0.18 * ramp(abs(noise_tail_decay), 0.04, 0.40)
        + 0.14 * ramp(stereo_width, 0.14, 0.70)
        + 0.10 * ramp(tail, 0.20, 0.86)
        + 0.08 * inverse_ramp(loop_panel["pulse_clarity"], 0.10, 0.78)
    )
    texture_bed_score = clamp01(
        0.24 * ramp(tail, 0.34, 0.96)
        + 0.22 * inverse_ramp(event_count, 1.0, 9.0)
        + 0.18 * ramp(stereo_width, 0.16, 0.76)
        + 0.16 * ramp(max(entropy, flatness), 0.36, 0.82)
        + 0.12 * inverse_ramp(abs(centroid_slope), 0.02, 0.28)
        + 0.08 * ramp(loop_sustained, 0.45, 1.0)
    )
    voice_score = clamp01(
        0.26 * reed_panel["formant_envelope_score"]
        + 0.22 * ramp(formant_spacing, 0.22, 0.88)
        + 0.18 * ramp(max(body_noise, tail_noise), 0.18, 0.52)
        + 0.14 * ramp(f0_voiced, 0.42, 0.92)
        + 0.12 * ramp(high, 0.08, 0.34)
        + 0.08 * ramp(loop_event_noise, 0.16, 0.64)
    )
    clean_low_bass_phrase_decoy = bool(
        low_total >= 0.88
        and high <= 0.08
        and mid <= 0.18
        and loop_panel["role_phrase_score"] >= loop_panel["role_one_shot_score"] + 0.18
        and max(clean_tone, pitch_conf, loop_pitched) >= 0.64
        and max(loop_percussive, loop_drumlike) <= 0.08
    )
    low_sub_kick_exception = bool(
        (sub >= 0.45 or (low_total >= 0.82 and 0.0 < low_peak_hz <= 95.0))
        and 0.0 < low_peak_hz <= 135.0
        and f0_voiced <= 0.22
        and mid <= 0.24
        and high <= 0.14
    )
    low_sub_kick_one_shot_exception = bool(
        low_sub_kick_exception
        and event_count <= 5.0
        and fast_attack >= 0.42
        and (onset_span <= 0.36 or (event_count <= 4.0 and loop_panel["role_one_shot_score"] >= 0.38))
    )
    clean_tonal_music_hit_decoy = bool(
        max(loop_pitched, loop_sustained, loop_non_event_tonal) >= 0.92
        and max(loop_percussive, loop_drumlike) <= 0.08
        and clean_tone >= 0.52
        and metallic_noise_score <= 0.48
        and high <= 0.16
        and pitch_conf >= 0.48
        and f0_voiced >= 0.62
        and not low_sub_kick_exception
    )
    compact_pitched_music_hit_decoy = bool(
        max(loop_pitched, loop_sustained, loop_non_event_tonal) >= 0.92
        and max(loop_percussive, loop_drumlike) <= 0.08
        and clean_tone >= 0.50
        and pitch_conf >= 0.38
        and event_count <= 4.0
        and tail <= 0.22
        and metallic_noise_score <= 0.50
        and not low_sub_kick_exception
    )

    compact_struck_tonal_percussion_score = clamp01(
        0.22 * loop_panel["role_one_shot_score"]
        + 0.20 * fast_attack
        + 0.16 * early_centroid
        + 0.14 * ramp(max(pitch_conf, attack_pitch, body_pitch, peak_stability), 0.24, 0.78)
        + 0.12 * ramp(max(inharmonicity, spectral_peak_count / 20.0, metallic_noise_score), 0.10, 0.66)
        + 0.10 * inverse_ramp(max(loop_panel["role_loop_score"], repeated_events), 0.08, 0.52)
        + 0.10 * inverse_ramp(max(loop_sustained, loop_non_event_tonal), 0.10, 0.78)
    )
    hand_drum_membrane_score = clamp01(
        0.24 * compact_struck_tonal_percussion_score
        + 0.20 * drum_hit_score
        + 0.18 * ramp(bass + mid, 0.34, 0.88)
        + 0.14 * ramp(max(pitch_conf, body_pitch, harmonic, fundamental), 0.22, 0.76)
        + 0.12 * inverse_ramp(high, 0.04, 0.42)
        + 0.10 * inverse_ramp(sub, 0.08, 0.56)
        + 0.08 * inverse_ramp(tail, 0.06, 0.58)
        - 0.12 * ramp(metallic_noise_score, 0.52, 0.90)
        - 0.08 * ramp(air, 0.10, 0.34)
    )
    pitched_metal_percussion_score = clamp01(
        0.24 * compact_struck_tonal_percussion_score
        + 0.24 * metallic_noise_score
        + 0.18 * ramp(max(inharmonicity, spectral_peak_count / 18.0, peak_stability), 0.10, 0.62)
        + 0.14 * ramp(max(pitch_conf, attack_pitch, body_pitch, harmonic), 0.30, 0.84)
        + 0.10 * ramp(max(high, attack_high, tail_high), 0.08, 0.58)
        + 0.08 * inverse_ramp(max(loop_sustained, loop_non_event_tonal), 0.12, 0.88)
        - 0.08 * ramp(voice_score, 0.72, 0.96)
    )
    struck_wood_score = clamp01(
        0.28 * compact_struck_tonal_percussion_score
        + 0.22 * fast_attack
        + 0.18 * ramp(mid + presence, 0.30, 0.92)
        + 0.14 * inverse_ramp(tail, 0.02, 0.28)
        + 0.10 * inverse_ramp(max(flatness, entropy), 0.16, 0.62)
        + 0.08 * ramp(max(pitch_conf, peak_stability), 0.16, 0.56)
        + 0.08 * inverse_ramp(low_total, 0.04, 0.48)
        - 0.10 * ramp(metallic_noise_score, 0.46, 0.86)
    )
    pitched_mallet_instrument_score = clamp01(
        0.24 * compact_struck_tonal_percussion_score
        + 0.20 * keys_panel["struck_keys_score"]
        + 0.18 * ramp(max(pitch_conf, body_pitch, harmonic), 0.38, 0.90)
        + 0.14 * ramp(peak_stability, 0.08, 0.54)
        + 0.12 * ramp(max(inharmonicity, spectral_peak_count / 20.0), 0.08, 0.54)
        + 0.10 * inverse_ramp(max(loop_percussive, loop_drumlike), 0.08, 0.44)
        + 0.08 * ramp(max(loop_panel["role_phrase_score"], loop_panel["role_one_shot_score"]), 0.32, 0.86)
    )

    clear_transient_drum_hit = bool(
        loop_panel["role_one_shot_score"] >= 0.70
        and drum_hit_score >= 0.56
        and onset_panel["percussive_onset_score"] >= 0.58
        and pitch_conf <= 0.42
        and low_total <= 0.28
        and 0.10 <= high <= 0.58
        and tail <= 0.16
        and event_count <= 4.0
        and max(loop_panel["role_loop_score"], repeated_events) <= 0.34
    )
    if clear_transient_drum_hit:
        clean_tonal_music_hit_decoy = False
        compact_pitched_music_hit_decoy = False

    # Source-blind anti-drum guard for voiced/reed/synth stabs.
    # Vocal FX hits, dry sax notes, and synth stabs can have a sharp attack and
    # mid/high energy. That is not enough to call them snare/clap/rim when the
    # measured frames are strongly pitched/voiced and contain no drumlike or
    # percussive-event evidence. Arpeggios and onsets require actual repeated
    # note/event structure; a clean sustained voiced event should not inflate a
    # drum source panel just because it has a fast front edge.
    tonal_voiced_non_drum_hit_guard = bool(
        max(f0_voiced, loop_pitched, clean_tone) >= 0.72
        and max(loop_percussive, loop_drumlike) <= 0.12
        and max(
            voice_score,
            reed_panel["reed_wind_score"],
            reed_panel["formant_envelope_score"],
            loop_sustained,
            loop_non_event_tonal,
        )
        >= 0.55
        and not low_sub_kick_exception
        and not clear_transient_drum_hit
    )
    pitched_metal_material_evidence = bool(
        pitched_metal_percussion_score >= 0.54
        and (metallic_noise_score >= 0.48 or (max(attack_high, high, tail_high) >= 0.18 and low_total <= 0.82))
        and not clean_low_bass_phrase_decoy
        and not clean_tonal_music_hit_decoy
        and not compact_pitched_music_hit_decoy
        and not low_sub_kick_exception
        and not clear_transient_drum_hit
    )
    hand_drum_material_evidence = bool(
        hand_drum_membrane_score >= 0.54
        and high <= 0.42
        and not clean_tonal_music_hit_decoy
        and not compact_pitched_music_hit_decoy
        and (
            max(loop_sustained, loop_non_event_tonal) <= 0.54
            or (hand_drum_membrane_score >= 0.82 and compact_struck_tonal_percussion_score >= 0.70)
        )
    )
    struck_wood_material_evidence = bool(
        struck_wood_score >= 0.56
        and tail <= 0.34
        and max(loop_sustained, loop_non_event_tonal) <= 0.46
        and metallic_noise_score <= 0.58
    )
    struck_percussion_guard_exception = bool(
        (pitched_metal_material_evidence or hand_drum_material_evidence or struck_wood_material_evidence)
        and compact_struck_tonal_percussion_score >= 0.42
        and fast_attack >= 0.32
        and loop_panel["role_one_shot_score"] >= 0.46
        and max(loop_panel["role_loop_score"], repeated_events) <= 0.58
        and not clean_low_bass_phrase_decoy
        and not clean_tonal_music_hit_decoy
        and not compact_pitched_music_hit_decoy
    )
    if tonal_voiced_non_drum_hit_guard and not struck_percussion_guard_exception:
        drum_hit_score = min(drum_hit_score, 0.34)

    bowed_string_score = clamp01(
        0.30 * ramp(max(loop_sustained, loop_non_event_tonal), 0.62, 1.0)
        + 0.24 * slow_attack
        + 0.18 * clean_tone
        + 0.16 * ramp(max(body_noise, tail_noise), 0.08, 0.38)
        + 0.12 * inverse_ramp(voice_score, 0.08, 0.62)
    )
    low_end_source_score = clamp01(
        0.28 * ramp(sub + bass, 0.34, 0.92)
        + 0.20 * inverse_ramp(low_peak_hz, 35.0, 190.0)
        + 0.16 * ramp(sub_sustain, 0.20, 0.90)
        + 0.14 * ramp(sub_decay_ms, 120.0, 1200.0)
        + 0.12 * inverse_ramp(sub_click_offset, 0.0, 90.0)
        + 0.10 * ramp(abs(kick_drop), 40.0, 900.0)
    )
    low_kick_score = clamp01(
        0.32 * drum_hit_score
        + 0.24 * ramp(sub, 0.20, 0.78)
        + 0.18 * inverse_ramp(sub_attack_ms, 0.0, 90.0)
        + 0.14 * inverse_ramp(sub_decay_ms, 80.0, 900.0)
        + 0.12 * ramp(abs(kick_drop), 80.0, 900.0)
    )
    if low_sub_kick_one_shot_exception:
        low_kick_score = max(low_kick_score, 0.70)
    sustained_bass_score = clamp01(
        0.34 * low_end_source_score
        + 0.24 * ramp(sub_sustain, 0.35, 0.95)
        + 0.18 * loop_panel["role_phrase_score"]
        + 0.14 * clean_tone
        + 0.10 * inverse_ramp(low_kick_score, 0.25, 0.86)
    )
    tonal_alert_siren_score = clamp01(
        0.22 * clean_tone
        + 0.17 * max(loop_panel["role_phrase_score"], loop_panel["role_loop_score"])
        + 0.15 * ramp(event_count, 8.0, 40.0)
        + 0.14 * ramp(event_rate, 3.0, 10.0)
        + 0.12 * ramp(formant_spacing, 1.2, 3.0)
        + 0.10 * inverse_ramp(low_total, 0.02, 0.28)
        + 0.08 * inverse_ramp(high + air, 0.02, 0.18)
        + 0.07 * ramp(tail, 0.20, 0.90)
        - 0.10 * drum_hit_score
    )

    # Category-coverage panels.  These are source-name-blind measured witnesses
    # for the real public folders present in the current brain: drum subtypes,
    # FX motion/action/material families, textures/ambiences, and human/animal
    # FX.  They do not route by themselves; physics_layers consumes the scores
    # as voter evidence.
    snare_source_score = clamp01(
        0.22 * drum_hit_score
        + 0.20 * ramp(mid + presence, 0.32, 0.88)
        + 0.18 * ramp(max(attack_noise, body_noise, noise_burst_ms / 260.0), 0.16, 0.62)
        + 0.14 * fast_attack
        + 0.12 * inverse_ramp(tail, 0.05, 0.38)
        + 0.08 * inverse_ramp(sub, 0.06, 0.42)
        + 0.06 * inverse_ramp(clean_tone, 0.18, 0.70)
    )
    clap_source_score = clamp01(
        0.24 * drum_hit_score
        + 0.22 * ramp(max(attack_noise, flatness), 0.18, 0.62)
        + 0.18 * ramp(mid, 0.34, 0.88)
        + 0.14 * inverse_ramp(tail, 0.03, 0.30)
        + 0.12 * fast_attack
        + 0.10 * inverse_ramp(low_total, 0.06, 0.44)
    )
    if tonal_voiced_non_drum_hit_guard and not struck_percussion_guard_exception:
        snare_source_score = min(snare_source_score, 0.36)
        clap_source_score = min(clap_source_score, 0.34)
    if clear_transient_drum_hit:
        snare_source_score = max(snare_source_score, min(0.86, 0.42 + 0.44 * drum_hit_score))
        clap_source_score = max(clap_source_score, min(0.84, 0.40 + 0.42 * onset_panel["percussive_onset_score"]))
    closed_hat_source_score = clamp01(
        0.28 * ramp(high, 0.22, 0.70)
        + 0.22 * fast_attack
        + 0.18 * inverse_ramp(tail, 0.02, 0.22)
        + 0.14 * ramp(max(attack_zcr, zcr, attack_noise), 0.08, 0.34)
        + 0.10 * inverse_ramp(low_total, 0.02, 0.28)
        + 0.08 * drum_hit_score
    )
    cymbal_source_score = clamp01(
        0.25 * ramp(high, 0.18, 0.66)
        + 0.20 * ramp(tail, 0.12, 0.82)
        + 0.18 * metallic_noise_score
        + 0.16 * ramp(max(high_decay, centroid_decay, noise_burst_ms / 420.0), 0.12, 0.70)
        + 0.11 * inverse_ramp(low_total, 0.04, 0.42)
        + 0.10 * ramp(max(air, tail_high), 0.05, 0.34)
    )
    if tonal_voiced_non_drum_hit_guard:
        closed_hat_source_score = min(closed_hat_source_score, 0.34)
        if not pitched_metal_material_evidence:
            cymbal_source_score = min(cymbal_source_score, 0.34)
    tom_conga_source_score = clamp01(
        0.24 * drum_hit_score
        + 0.22 * ramp(bass + mid, 0.36, 0.88)
        + 0.16 * ramp(max(clean_tone, pitch_conf), 0.24, 0.72)
        + 0.14 * inverse_ramp(high, 0.02, 0.30)
        + 0.12 * inverse_ramp(sub, 0.10, 0.54)
        + 0.12 * inverse_ramp(tail, 0.08, 0.50)
    )
    rim_stick_source_score = clamp01(
        0.28 * fast_attack
        + 0.22 * inverse_ramp(tail, 0.02, 0.22)
        + 0.18 * ramp(mid + presence, 0.28, 0.86)
        + 0.12 * inverse_ramp(low_total, 0.04, 0.38)
        + 0.10 * ramp(max(pitch_conf, peak_stability), 0.18, 0.58)
        + 0.10 * drum_hit_score
    )
    if tonal_voiced_non_drum_hit_guard:
        if not struck_wood_material_evidence:
            rim_stick_source_score = min(rim_stick_source_score, 0.34)
        if not hand_drum_material_evidence:
            tom_conga_source_score = min(tom_conga_source_score, 0.36)
        if high <= 0.34:
            cymbal_source_score = min(cymbal_source_score, 0.34)
    shaker_tambourine_source_score = clamp01(
        0.24 * repeated_events
        + 0.22 * ramp(event_count, 3.0, 16.0)
        + 0.18 * ramp(high, 0.16, 0.62)
        + 0.16 * ramp(max(loop_noisy, flatness, zcr), 0.16, 0.58)
        + 0.12 * inverse_ramp(low_total, 0.04, 0.38)
        + 0.08 * loop_panel["pulse_clarity"]
    )
    guiro_scrape_source_score = scrape_rasp_score
    metallic_percussion_source_score = metallic_noise_score
    if low_sub_kick_exception or clear_transient_drum_hit:
        metallic_percussion_source_score = min(metallic_percussion_source_score, 0.34)
    if tonal_voiced_non_drum_hit_guard and not pitched_metal_material_evidence:
        metallic_percussion_source_score = min(metallic_percussion_source_score, 0.34)
    if not tonal_voiced_non_drum_hit_guard or hand_drum_material_evidence:
        tom_conga_source_score = max(tom_conga_source_score, clamp01(0.84 * hand_drum_membrane_score))
    if not tonal_voiced_non_drum_hit_guard or struck_wood_material_evidence:
        rim_stick_source_score = max(rim_stick_source_score, clamp01(0.86 * struck_wood_score))
    if not tonal_voiced_non_drum_hit_guard or pitched_metal_material_evidence:
        metallic_percussion_source_score = max(
            metallic_percussion_source_score,
            clamp01(0.90 * pitched_metal_percussion_score),
        )

    # Drum-loop evidence cannot depend only on frame labels such as
    # ``loop_percussive_event_ratio`` or ``loop_drumlike_frame_ratio``.  Real
    # sampled breaks and synthetic hat loops often contain pitched low-end or
    # narrow high-frequency bodies, so those frame classifiers under-read them
    # as bass phrases, reeds, voices, or FX.  The invariant below is still
    # source-name blind: it asks whether the audio is a repeated, fast-attacked,
    # rhythmically distributed event stream with drum-band emphasis.  That is a
    # drum-loop role signal even when the timbre is tonal or synthetic.
    rhythmic_break_loop_score = clamp01(
        0.22 * loop_panel["role_loop_score"]
        + 0.17 * repeated_events
        + 0.15 * ramp(event_count, 6.0, 32.0)
        + 0.14 * ramp(onset_span, 0.42, 0.90)
        + 0.12 * loop_panel["pulse_clarity"]
        + 0.10 * loop_panel["onset_periodicity"]
        + 0.08 * fast_attack
        + 0.07 * max(ramp(loop_event_low, 0.42, 0.92), ramp(loop_event_high, 0.16, 0.70))
        + 0.05 * ramp(event_rate, 1.0, 5.5)
        - 0.10 * slow_attack
        - 0.08 * ramp(loop_sustained, 0.88, 1.0)
    )
    rhythmic_transient_loop_score = clamp01(
        0.24 * fast_attack
        + 0.22 * ramp(event_count, 6.0, 18.0)
        + 0.18 * ramp(onset_span, 0.40, 0.86)
        + 0.14 * ramp(event_rate, 1.2, 4.2)
        + 0.14 * max(ramp(loop_event_low, 0.50, 0.90), ramp(loop_event_high, 0.14, 0.55))
        + 0.08 * inverse_ramp(tail, 0.20, 0.82)
        - 0.10 * slow_attack
    )
    rhythmic_break_loop_score = max(rhythmic_break_loop_score, rhythmic_transient_loop_score)
    low_fast_pulse_loop_score = clamp01(
        0.22 * ramp(low_total, 0.70, 0.92)
        + 0.18 * ramp(event_count, 5.0, 28.0)
        + 0.16 * ramp(onset_span, 0.34, 0.88)
        + 0.14 * loop_panel["pulse_clarity"]
        + 0.12 * fast_attack
        + 0.10 * inverse_ramp(f0_voiced, 0.18, 0.55)
        + 0.08 * inverse_ramp(high, 0.04, 0.16)
    )
    low_fast_pulse_loop_candidate = bool(
        low_total >= 0.70
        and high <= 0.18
        and event_count >= 6.0
        and onset_span >= 0.30
        and fast_attack >= 0.55
        and f0_voiced <= 0.42
        and (loop_panel["role_loop_score"] >= 0.30 or repeated_events >= 0.38)
    )
    if not low_fast_pulse_loop_candidate:
        low_fast_pulse_loop_score = min(low_fast_pulse_loop_score, 0.38)
    pitched_repetition_phrase_score = clamp01(
        0.22 * repeated_events
        + 0.18 * ramp(max(pitch_conf, body_pitch, loop_event_pitch), 0.42, 0.90)
        + 0.16 * ramp(max(loop_pitched, loop_tonal_balance), 0.48, 0.95)
        + 0.14 * ramp(max(loop_sustained, loop_non_event_tonal), 0.45, 0.90)
        + 0.12 * ramp(onset_panel["pitched_onset_score"] - onset_panel["percussive_onset_score"], 0.04, 0.30)
        + 0.10 * max(plucked_panel["plucked_string_score"], keys_panel["struck_keys_score"], clean_tone)
        + 0.08 * inverse_ramp(max(loop_percussive, loop_drumlike), 0.10, 0.36)
        - 0.06 * noisy_air
    )
    drum_material_loop_witness_count = sum(
        1
        for witness in (
            snare_source_score,
            clap_source_score,
            closed_hat_source_score,
            cymbal_source_score,
            tom_conga_source_score,
            rim_stick_source_score,
            shaker_tambourine_source_score,
            metallic_percussion_source_score,
        )
        if witness >= 0.48
    )
    low_rhythmic_break_loop_candidate = bool(
        low_total >= 0.78
        and high <= 0.16
        and 0.0 < low_peak_hz <= 115.0
        and f0_voiced <= 0.32
        and event_count >= 4.5
        and onset_span >= 0.65
        and fast_attack >= 0.70
        and tail >= 0.40
        and (loop_panel["pulse_clarity"] >= 0.18 or repeated_events >= 0.24)
        and not clean_low_bass_phrase_decoy
    )
    pitched_repetition_loop_decoy_candidate = bool(
        pitched_repetition_phrase_score >= 0.58
        and max(loop_percussive, loop_drumlike) <= 0.34
        and onset_panel["pitched_onset_score"] >= onset_panel["percussive_onset_score"] - 0.02
        and not (low_total >= 0.82 and 0.0 < low_peak_hz <= 135.0 and f0_voiced <= 0.32)
        and not low_rhythmic_break_loop_candidate
    )
    noisy_percussion_loop_candidate = bool(
        event_count >= 6.0
        and onset_span >= 0.36
        and max(loop_panel["role_loop_score"], repeated_events) >= 0.30
        and drum_material_loop_witness_count >= 2
        and (
            loop_drumlike >= 0.16
            or loop_percussive >= 0.46
            or onset_panel["percussive_onset_score"] >= onset_panel["pitched_onset_score"] - 0.02
        )
        and max(drum_hit_score, cymbal_source_score, shaker_tambourine_source_score, metallic_percussion_source_score)
        >= 0.50
        and not clean_low_bass_phrase_decoy
        and not clean_tonal_music_hit_decoy
        and not compact_pitched_music_hit_decoy
        and not pitched_repetition_loop_decoy_candidate
    )
    if noisy_percussion_loop_candidate:
        rhythmic_break_loop_score = max(
            rhythmic_break_loop_score,
            min(0.72, 0.38 + 0.24 * repeated_events + 0.18 * max(drum_hit_score, metallic_percussion_source_score)),
        )
    if low_rhythmic_break_loop_candidate:
        rhythmic_break_loop_score = max(rhythmic_break_loop_score, 0.68)
        low_fast_pulse_loop_score = max(low_fast_pulse_loop_score, 0.58)
    pitched_repetition_drum_decoy = pitched_repetition_loop_decoy_candidate
    if pitched_repetition_drum_decoy:
        rhythmic_break_loop_score *= 1.0 - 0.45 * ramp(pitched_repetition_phrase_score, 0.58, 0.86)
    clean_low_bass_phrase = bool(
        max(loop_event_low, low_total) >= 0.88
        and high <= 0.08
        and pitch_conf >= 0.70
        and loop_pitched >= 0.90
        and max(loop_sustained, loop_non_event_tonal) >= 0.86
        and max(loop_percussive, loop_drumlike) <= 0.08
        and flatness <= 0.08
        and event_count >= 3.0
        and onset_panel["pitched_onset_score"] >= onset_panel["percussive_onset_score"] + 0.10
    )
    if clean_low_bass_phrase:
        rhythmic_break_loop_score = min(rhythmic_break_loop_score, 0.42)
        low_fast_pulse_loop_score = min(low_fast_pulse_loop_score, 0.42)
    drum_loop_source_score = clamp01(
        0.22 * loop_panel["role_loop_score"]
        + 0.18 * ramp(loop_percussive, 0.12, 0.84)
        + 0.16 * ramp(loop_drumlike, 0.10, 0.82)
        + 0.12 * drum_hit_score
        + 0.08 * repeated_events
        + 0.06 * ramp(event_count, 5.0, 28.0)
        + 0.05 * max(ramp(loop_event_low, 0.45, 0.92), ramp(loop_event_high, 0.18, 0.70))
        + 0.25 * rhythmic_break_loop_score
        - 0.08 * plucked_panel["plucked_string_score"]
        - 0.06 * keys_panel["struck_keys_score"]
        - 0.06 * reed_panel["reed_wind_score"]
        - 0.08 * loop_panel["delay_tail_likelihood"]
        - 0.12 * pitched_repetition_phrase_score
    )
    if pitched_repetition_drum_decoy and low_fast_pulse_loop_score < 0.58:
        drum_loop_source_score = min(
            drum_loop_source_score,
            max(0.20, 0.44 - 0.22 * ramp(pitched_repetition_phrase_score, 0.58, 0.86)),
        )
    if low_fast_pulse_loop_score >= 0.50:
        drum_loop_source_score = max(drum_loop_source_score, min(0.72, 0.34 + 0.46 * low_fast_pulse_loop_score))
    if low_rhythmic_break_loop_candidate:
        drum_loop_source_score = max(drum_loop_source_score, 0.56)
    if noisy_percussion_loop_candidate:
        drum_loop_source_score = max(
            drum_loop_source_score, 0.54 + 0.12 * ramp(drum_material_loop_witness_count, 2.0, 4.0)
        )
    if clean_low_bass_phrase:
        drum_loop_source_score = min(drum_loop_source_score, 0.32)
    if low_sub_kick_one_shot_exception:
        drum_loop_source_score = min(drum_loop_source_score, 0.24)

    reverse_fx_score = clamp01(
        0.30 * onset_panel["swell_onset_score"]
        + 0.24 * ramp(attack, 0.16, 0.64)
        + 0.16 * ramp(temporal, 0.42, 0.84)
        + 0.12 * ramp(tail, 0.18, 0.78)
        + 0.10 * inverse_ramp(event_count, 1.0, 8.0)
        + 0.08 * fx_motion_score
    )
    riser_build_score = clamp01(
        0.34 * ramp(centroid_slope, 0.05, 0.28)
        + 0.20 * fx_motion_score
        + 0.16 * onset_panel["swell_onset_score"]
        + 0.12 * ramp(tail, 0.20, 0.82)
        + 0.10 * ramp(stereo_width, 0.14, 0.70)
        + 0.08 * inverse_ramp(drum_loop_source_score, 0.18, 0.76)
    )
    drop_downlifter_score = clamp01(
        0.34 * ramp(-centroid_slope, 0.05, 0.28)
        + 0.20 * fx_motion_score
        + 0.14 * ramp(max(tail_low, low_total, loop_event_low), 0.18, 0.72)
        + 0.12 * ramp(tail, 0.16, 0.78)
        + 0.10 * ramp(stereo_width, 0.14, 0.70)
        + 0.10 * inverse_ramp(clean_tone, 0.10, 0.76)
    )
    whoosh_sweep_score = clamp01(
        0.26 * fx_motion_score
        + 0.24 * noisy_air
        + 0.18 * ramp(high, 0.08, 0.42)
        + 0.12 * ramp(stereo_width, 0.16, 0.78)
        + 0.10 * inverse_ramp(clean_tone, 0.10, 0.70)
        + 0.10 * ramp(abs(centroid_slope), 0.04, 0.26)
    )
    impact_fx_score = clamp01(
        0.26 * drum_hit_score
        + 0.22 * ramp(tail, 0.20, 0.86)
        + 0.18 * ramp(max(low_total, high), 0.14, 0.70)
        + 0.14 * ramp(stereo_width, 0.16, 0.76)
        + 0.10 * fast_attack
        + 0.10 * inverse_ramp(loop_panel["role_loop_score"], 0.12, 0.74)
    )
    glitch_stutter_score = clamp01(
        0.28 * ramp(event_count, 4.0, 28.0)
        + 0.20 * ramp(max(flux_var, loop_diversity), 0.08, 0.58)
        + 0.18 * inverse_ramp(regularity, 0.04, 0.62)
        + 0.14 * ramp(max(zcr, flatness, loop_noisy), 0.12, 0.58)
        + 0.12 * ramp(event_rate, 2.0, 9.0)
        + 0.08 * inverse_ramp(clean_tone, 0.08, 0.70)
    )
    blip_beep_score = clamp01(
        0.32 * clean_tone
        + 0.20 * fast_attack
        + 0.16 * inverse_ramp(tail, 0.02, 0.34)
        + 0.12 * inverse_ramp(low_total, 0.02, 0.36)
        + 0.10 * ramp(max(pitch_conf, peak_stability), 0.34, 0.90)
        + 0.10 * inverse_ramp(event_count, 1.0, 5.0)
    )
    formant_fx_score = clamp01(
        0.30 * reed_panel["formant_envelope_score"]
        + 0.22 * voice_score
        + 0.16 * ramp(formant_spacing, 0.24, 1.0)
        + 0.12 * ramp(max(body_noise, tail_noise), 0.14, 0.52)
        + 0.10 * fx_motion_score
        + 0.10
        * inverse_ramp(
            plucked_panel["plucked_string_score"] if 'plucked_panel["plucked_string_score"]' in locals() else 0.0,
            0.20,
            0.70,
        )
    )
    radio_electrical_score = clamp01(
        0.30 * noisy_air
        + 0.22 * ramp(zcr, 0.10, 0.38)
        + 0.16 * ramp(max(mid, presence), 0.24, 0.78)
        + 0.12 * inverse_ramp(low_total, 0.02, 0.42)
        + 0.10 * glitch_stutter_score
        + 0.10 * inverse_ramp(clean_tone, 0.08, 0.62)
    )
    machine_mechanical_score = clamp01(
        0.24 * ramp(max(loop_noisy, flatness, zcr), 0.16, 0.62)
        + 0.18 * repeated_events
        + 0.16 * ramp(max(bass, mid), 0.22, 0.76)
        + 0.14 * ramp(event_rate, 0.6, 5.0)
        + 0.12 * fx_motion_score
        + 0.10 * inverse_ramp(clean_tone, 0.08, 0.62)
        + 0.06 * ramp(tail, 0.10, 0.76)
    )
    foley_material_score = clamp01(
        0.22 * max(drum_hit_score, scrape_rasp_score, metallic_noise_score)
        + 0.20 * noisy_air
        + 0.16 * ramp(max(mid, high), 0.22, 0.74)
        + 0.14 * inverse_ramp(clean_tone, 0.08, 0.58)
        + 0.12 * inverse_ramp(loop_panel["role_loop_score"], 0.14, 0.70)
        + 0.08 * ramp(event_count, 1.0, 12.0)
        + 0.08 * inverse_ramp(low_total, 0.06, 0.60)
    )
    small_object_score = clamp01(
        0.26 * metallic_noise_score
        + 0.22 * ramp(event_count, 2.0, 14.0)
        + 0.18 * inverse_ramp(tail, 0.03, 0.46)
        + 0.14 * ramp(max(high, mid), 0.14, 0.66)
        + 0.10 * inverse_ramp(low_total, 0.04, 0.40)
        + 0.10 * inverse_ramp(loop_drumlike, 0.06, 0.44)
    )

    water_ocean_score = clamp01(
        0.26 * texture_bed_score
        + 0.24 * ramp(noisy_air, 0.22, 0.70)
        + 0.18 * ramp(stereo_width, 0.16, 0.76)
        + 0.14 * inverse_ramp(abs(centroid_slope), 0.02, 0.26)
        + 0.10 * ramp(tail, 0.32, 0.94)
        + 0.08 * inverse_ramp(clean_tone, 0.06, 0.50)
    )
    rain_score = clamp01(
        0.28 * texture_bed_score
        + 0.24 * ramp(high, 0.12, 0.52)
        + 0.18 * ramp(max(flatness, zcr), 0.18, 0.62)
        + 0.14 * inverse_ramp(loop_panel["pulse_clarity"], 0.04, 0.40)
        + 0.10 * inverse_ramp(low_total, 0.04, 0.40)
        + 0.06 * ramp(event_count, 4.0, 44.0)
    )
    wind_score = clamp01(
        0.28 * texture_bed_score
        + 0.24 * ramp(max(noisy_air, tail_noise), 0.22, 0.72)
        + 0.18 * ramp(abs(centroid_slope), 0.02, 0.18)
        + 0.14 * inverse_ramp(event_count, 1.0, 12.0)
        + 0.10 * ramp(stereo_width, 0.18, 0.78)
        + 0.06 * inverse_ramp(clean_tone, 0.04, 0.46)
    )
    fire_score = clamp01(
        0.26 * texture_bed_score
        + 0.24 * ramp(max(flatness, zcr, loop_noisy), 0.18, 0.66)
        + 0.16 * ramp(flux_var, 0.04, 0.30)
        + 0.14 * ramp(mid + high, 0.24, 0.82)
        + 0.12 * inverse_ramp(clean_tone, 0.06, 0.54)
        + 0.08 * ramp(event_count, 3.0, 30.0)
    )
    thunder_score = clamp01(
        0.28 * impact_fx_score
        + 0.24 * ramp(low_total, 0.28, 0.86)
        + 0.18 * ramp(tail, 0.36, 0.96)
        + 0.12 * ramp(stereo_width, 0.16, 0.76)
        + 0.10 * inverse_ramp(event_count, 1.0, 8.0)
        + 0.08 * noisy_air
    )
    noise_static_score = clamp01(
        0.34 * noisy_air
        + 0.24 * ramp(max(flatness, zcr), 0.22, 0.72)
        + 0.16 * inverse_ramp(clean_tone, 0.04, 0.52)
        + 0.12 * inverse_ramp(low_total, 0.04, 0.44)
        + 0.08 * texture_bed_score
        + 0.06 * radio_electrical_score
    )
    room_crowd_ambience_score = clamp01(
        0.24 * texture_bed_score
        + 0.20 * voice_score
        + 0.16 * ramp(mid, 0.28, 0.78)
        + 0.14 * ramp(stereo_width, 0.18, 0.76)
        + 0.12 * ramp(max(body_noise, flatness), 0.14, 0.46)
        + 0.08 * inverse_ramp(clean_tone, 0.08, 0.62)
        + 0.06 * ramp(event_count, 4.0, 48.0)
    )

    breath_mouth_score = clamp01(
        0.30 * voice_score
        + 0.24 * noisy_air
        + 0.16 * ramp(max(attack_noise, high), 0.12, 0.46)
        + 0.12 * inverse_ramp(low_total, 0.04, 0.42)
        + 0.10 * inverse_ramp(clean_tone, 0.08, 0.62)
        + 0.08 * inverse_ramp(event_count, 1.0, 8.0)
    )
    spoken_voice_score = clamp01(
        0.30 * voice_score
        + 0.22 * ramp(f0_voiced, 0.36, 0.90)
        + 0.18 * ramp(event_count, 3.0, 28.0)
        + 0.12 * ramp(mid + presence, 0.28, 0.86)
        + 0.10 * ramp(max(body_noise, flatness), 0.12, 0.42)
        + 0.08 * inverse_ramp(loop_drumlike, 0.04, 0.34)
    )
    scream_score = clamp01(
        0.26 * voice_score
        + 0.22 * ramp(high, 0.12, 0.52)
        + 0.18 * ramp(max(flatness, zcr, body_noise), 0.18, 0.62)
        + 0.14 * ramp(tail, 0.10, 0.72)
        + 0.10 * ramp(f0_voiced, 0.30, 0.86)
        + 0.10 * inverse_ramp(clean_tone, 0.10, 0.68)
    )
    applause_crowd_score = clamp01(
        0.24 * clap_source_score
        + 0.22 * repeated_events
        + 0.18 * ramp(event_count, 8.0, 60.0)
        + 0.14 * ramp(stereo_width, 0.18, 0.78)
        + 0.12 * ramp(max(flatness, high), 0.16, 0.58)
        + 0.10 * inverse_ramp(clean_tone, 0.08, 0.62)
    )
    bird_score = clamp01(
        0.30 * clean_tone
        + 0.20 * ramp(high, 0.14, 0.58)
        + 0.16 * ramp(max(pitch_conf, f0_voiced), 0.38, 0.94)
        + 0.14 * ramp(abs(f0_slope), 20.0, 260.0)
        + 0.10 * inverse_ramp(low_total, 0.02, 0.28)
        + 0.10 * inverse_ramp(tail, 0.04, 0.48)
    )
    animal_voice_score = clamp01(
        0.28 * voice_score
        + 0.22 * ramp(max(formant_spacing, f0_slope / 200.0), 0.18, 0.78)
        + 0.16 * ramp(max(body_noise, tail_noise), 0.14, 0.52)
        + 0.14 * ramp(tail, 0.08, 0.74)
        + 0.10 * inverse_ramp(plucked_panel["plucked_string_score"], 0.10, 0.64)
        + 0.10 * inverse_ramp(keys_panel["struck_keys_score"], 0.10, 0.64)
    )
    cricket_insect_score = clamp01(
        0.30 * ramp(high, 0.20, 0.70)
        + 0.24 * repeated_events
        + 0.16 * ramp(event_rate, 4.0, 16.0)
        + 0.12 * inverse_ramp(low_total, 0.02, 0.26)
        + 0.10 * ramp(max(zcr, flatness), 0.14, 0.56)
        + 0.08 * inverse_ramp(tail, 0.04, 0.38)
    )

    # Leaf panels below are conservative measured witnesses.  Per-category
    # calibration status is tracked in physics_category_panels.py so every
    # active brain leaf has a named panel instead of hidden ad-hoc logic.
    cat_voice_score = clamp01(
        0.58 * animal_voice_score
        + 0.18 * ramp(f0_slope, 40.0, 320.0)
        + 0.14 * ramp(mid + presence, 0.24, 0.82)
        + 0.10 * inverse_ramp(low_total, 0.04, 0.42)
    )
    dog_voice_score = clamp01(
        0.52 * animal_voice_score
        + 0.20 * ramp(low_total + mid, 0.34, 0.92)
        + 0.14 * ramp(max(body_noise, tail_noise), 0.12, 0.48)
        + 0.14 * ramp(event_count, 1.0, 10.0)
    )
    clean_musical_phrase_guard = bool(
        clean_tone >= 0.52
        and loop_pitched >= 0.78
        and max(loop_sustained, loop_non_event_tonal) >= 0.72
        and max(loop_percussive, loop_drumlike) <= 0.12
        and pitch_conf >= 0.55
        and max(plucked_panel["plucked_string_score"], keys_panel["struck_keys_score"], reed_panel["reed_wind_score"])
        >= 0.36
    )
    if clean_musical_phrase_guard:
        cat_voice_score = min(cat_voice_score, 0.30)
        dog_voice_score = min(dog_voice_score, 0.30)
        bird_score = min(bird_score, 0.34)
        animal_voice_score = min(animal_voice_score, 0.34)
    door_foley_score = clamp01(
        0.34 * foley_material_score
        + 0.20 * ramp(low_total + mid, 0.30, 0.90)
        + 0.18 * ramp(tail, 0.10, 0.62)
        + 0.14 * inverse_ramp(clean_tone, 0.06, 0.50)
        + 0.14 * ramp(max(impact_fx_score, small_object_score), 0.20, 0.74)
    )
    engine_machine_score = clamp01(
        0.42 * machine_mechanical_score
        + 0.20 * ramp(low_total + mid, 0.32, 0.92)
        + 0.16 * ramp(max(loop_sustained, tail), 0.35, 0.95)
        + 0.12 * ramp(max(noisy_air, body_noise), 0.18, 0.62)
        + 0.10 * inverse_ramp(clean_tone, 0.08, 0.58)
    )
    motor_machine_score = clamp01(
        0.38 * machine_mechanical_score
        + 0.22 * repeated_events
        + 0.16 * ramp(event_rate, 1.0, 7.0)
        + 0.14 * ramp(mid + bass, 0.28, 0.86)
        + 0.10 * inverse_ramp(high, 0.04, 0.34)
    )
    coin_object_score = clamp01(
        0.42 * small_object_score
        + 0.24 * metallic_noise_score
        + 0.14 * ramp(high, 0.12, 0.50)
        + 0.10 * inverse_ramp(tail, 0.04, 0.40)
        + 0.10 * ramp(event_count, 1.0, 12.0)
    )
    key_object_score = clamp01(
        0.36 * small_object_score
        + 0.20 * metallic_noise_score
        + 0.16 * ramp(event_count, 2.0, 16.0)
        + 0.14 * ramp(mid + high, 0.18, 0.72)
        + 0.14 * inverse_ramp(low_total, 0.04, 0.42)
    )
    boom_fx_score = clamp01(
        0.42 * impact_fx_score
        + 0.26 * ramp(low_total, 0.30, 0.90)
        + 0.16 * ramp(tail, 0.24, 0.92)
        + 0.10 * inverse_ramp(high, 0.02, 0.26)
        + 0.06 * inverse_ramp(event_count, 1.0, 5.0)
    )
    slam_fx_score = clamp01(
        0.42 * impact_fx_score
        + 0.22 * fast_attack
        + 0.16 * ramp(low_total + mid, 0.34, 0.94)
        + 0.10 * inverse_ramp(tail, 0.06, 0.56)
        + 0.10 * ramp(noisy_air, 0.14, 0.56)
    )
    sub_hit_fx_score = clamp01(
        0.44 * impact_fx_score
        + 0.30 * ramp(sub, 0.30, 0.88)
        + 0.14 * inverse_ramp(high, 0.02, 0.20)
        + 0.12 * inverse_ramp(event_count, 1.0, 5.0)
    )
    alarm_fx_score = clamp01(
        0.42 * tonal_alert_siren_score
        + 0.22 * repeated_events
        + 0.16 * ramp(event_rate, 1.0, 8.0)
        + 0.12 * clean_tone
        + 0.08 * inverse_ramp(low_total, 0.04, 0.34)
    )
    siren_fx_score = clamp01(
        0.46 * tonal_alert_siren_score
        + 0.20 * ramp(abs(f0_slope), 20.0, 260.0)
        + 0.16 * fx_motion_score
        + 0.10 * ramp(tail, 0.20, 0.88)
        + 0.08 * clean_tone
    )

    bass_808_score = clamp01(
        0.48 * low_kick_score
        + 0.24 * ramp(sub, 0.40, 0.92)
        + 0.16 * inverse_ramp(high, 0.01, 0.16)
        + 0.12 * ramp(sub_decay_ms, 90.0, 900.0)
    )
    bass_sub_score = clamp01(
        0.50 * sustained_bass_score
        + 0.24 * ramp(sub, 0.34, 0.92)
        + 0.14 * inverse_ramp(high, 0.02, 0.18)
        + 0.12 * ramp(sub_sustain, 0.25, 0.95)
    )
    bass_synth_score = clamp01(
        0.42 * sustained_bass_score
        + 0.22 * clean_tone
        + 0.14 * inverse_ramp(flatness, 0.01, 0.22)
        + 0.12 * ramp(loop_sustained, 0.42, 1.0)
        + 0.10 * inverse_ramp(high, 0.02, 0.22)
    )
    bass_electric_score = clamp01(
        0.34 * sustained_bass_score
        + 0.22 * plucked_panel["plucked_string_score"]
        + 0.16 * ramp(mid, 0.08, 0.42)
        + 0.14 * ramp(body_noise, 0.04, 0.28)
        + 0.14 * inverse_ramp(high, 0.02, 0.24)
    )
    bass_upright_score = clamp01(
        0.34 * sustained_bass_score
        + 0.22 * bowed_string_score
        + 0.16 * slow_attack
        + 0.14 * ramp(body_noise, 0.06, 0.32)
        + 0.14 * inverse_ramp(high, 0.02, 0.20)
    )
    guitar_acoustic_score = clamp01(
        0.42 * plucked_panel["plucked_string_score"]
        + 0.18 * plucked_panel["guitar_body_band_score"]
        + 0.16 * inverse_ramp(high, 0.04, 0.32)
        + 0.14 * inverse_ramp(body_noise, 0.04, 0.34)
        + 0.10 * inverse_ramp(low_total, 0.06, 0.58)
    )
    guitar_electric_score = clamp01(
        0.40 * plucked_panel["plucked_string_score"]
        + 0.20 * ramp(max(mid, presence), 0.30, 0.92)
        + 0.16 * ramp(max(body_noise, flatness), 0.08, 0.42)
        + 0.14 * ramp(tail, 0.18, 0.82)
        + 0.10 * inverse_ramp(low_total, 0.04, 0.62)
    )
    guitar_nylon_score = clamp01(
        0.42 * plucked_panel["plucked_string_score"]
        + 0.20 * inverse_ramp(high, 0.02, 0.22)
        + 0.16 * ramp(mid, 0.22, 0.74)
        + 0.12 * inverse_ramp(body_noise, 0.04, 0.30)
        + 0.10 * inverse_ramp(tail, 0.10, 0.68)
    )
    synth_lead_score = clamp01(
        0.40 * clean_tone
        + 0.22 * clean_tone
        + 0.16 * repeated_events
        + 0.12 * ramp(mid, 0.30, 0.86)
        + 0.10 * inverse_ramp(low_total, 0.04, 0.48)
    )
    synth_pad_score = clamp01(
        0.38 * clean_tone
        + 0.24 * ramp(max(loop_sustained, loop_non_event_tonal), 0.62, 1.0)
        + 0.16 * slow_attack
        + 0.12 * ramp(stereo_width, 0.16, 0.74)
        + 0.10 * inverse_ramp(event_count, 1.0, 8.0)
    )
    synth_chord_score = clamp01(
        0.36 * clean_tone
        + 0.22 * keys_panel["chord_density_score"]
        + 0.18 * repeated_events
        + 0.14 * clean_tone
        + 0.10 * inverse_ramp(flatness, 0.02, 0.22)
    )
    brass_trumpet_score = clamp01(
        0.34 * reed_panel["reed_wind_score"]
        + 0.24 * ramp(max(presence, high), 0.12, 0.48)
        + 0.18 * clean_tone
        + 0.14 * ramp(mid, 0.28, 0.82)
        + 0.10 * inverse_ramp(flatness, 0.04, 0.34)
    )
    woodwind_flute_score = clamp01(
        0.34 * reed_panel["reed_wind_score"]
        + 0.24 * ramp(max(air, high), 0.08, 0.46)
        + 0.16 * inverse_ramp(low_total, 0.03, 0.38)
        + 0.14 * clean_tone
        + 0.12 * reed_panel["breath_attack_score"]
    )
    woodwind_sax_score = clamp01(
        0.40 * reed_panel["reed_wind_score"]
        + 0.20 * reed_panel["formant_envelope_score"]
        + 0.16 * ramp(mid + bass, 0.28, 0.86)
        + 0.14 * reed_panel["reed_noise_score"]
        + 0.10 * inverse_ramp(high, 0.02, 0.34)
    )
    string_violin_score = clamp01(
        0.42 * bowed_string_score
        + 0.20 * ramp(high + presence, 0.10, 0.42)
        + 0.16 * clean_tone
        + 0.12 * slow_attack
        + 0.10 * inverse_ramp(low_total, 0.04, 0.44)
    )
    string_cello_score = clamp01(
        0.42 * bowed_string_score
        + 0.22 * ramp(low_total + mid, 0.34, 0.90)
        + 0.16 * clean_tone
        + 0.12 * slow_attack
        + 0.08 * inverse_ramp(high, 0.02, 0.26)
    )
    voice_choir_score = clamp01(
        0.34 * voice_score
        + 0.24 * ramp(max(loop_sustained, loop_non_event_tonal), 0.52, 1.0)
        + 0.18 * ramp(stereo_width, 0.16, 0.74)
        + 0.14 * clean_tone
        + 0.10 * ramp(f0_voiced, 0.40, 0.92)
    )

    # Calibrated authority scores: the raw panels above intentionally stay broad.
    # These authority scores add separation so several "kind of musical" panels
    # do not all vote equally.  They remain source-name blind and only use the
    # measured low-level descriptors already in this shared-facts block.
    plucked_string_authority_score = clamp01(
        0.18 * plucked_panel["plucked_string_score"]
        + 0.18 * plucked_panel["pluck_attack_score"]
        + 0.18 * plucked_panel["harmonic_stack_score"]
        + 0.14 * plucked_panel["note_event_consistency"]
        + 0.12 * plucked_panel["guitar_body_band_score"]
        + 0.10 * plucked_panel["string_decay_score"]
        + 0.10 * plucked_panel["pick_noise_score"]
        - 0.10 * reed_panel["reed_wind_score"]
        - 0.08 * keys_panel["struck_keys_score"]
        - 0.10 * drum_hit_score
    )
    reed_wind_authority_score = clamp01(
        0.28 * reed_panel["reed_wind_score"]
        + 0.18 * reed_panel["formant_envelope_score"]
        + 0.16 * reed_panel["breath_attack_score"]
        + 0.14 * reed_panel["non_plucked_attack_score"]
        + 0.12 * reed_panel["vibrato_score"]
        + 0.12 * reed_panel["legato_score"]
        - 0.18 * plucked_panel["plucked_string_score"]
        - 0.08 * keys_panel["struck_keys_score"]
    )
    struck_keys_authority_score = clamp01(
        0.28 * keys_panel["struck_keys_score"]
        + 0.18 * keys_panel["hammer_attack_score"]
        + 0.16 * keys_panel["partial_inharmonicity_score"]
        + 0.14 * keys_panel["chord_density_score"]
        + 0.12 * keys_panel["tonal_decay_score"]
        + 0.12 * max(keys_panel["electric_piano_tine_score"], keys_panel["organ_steady_tone_score"])
        - 0.08 * plucked_panel["plucked_string_score"]
        - 0.08 * reed_panel["reed_wind_score"]
        - 0.08 * drum_hit_score
    )
    synth_tonal_source_score = clamp01(
        0.22 * clean_tone
        + 0.20 * ramp(max(loop_sustained, loop_non_event_tonal), 0.62, 1.0)
        + 0.14 * ramp(max(loop_pitched, loop_event_pitch), 0.55, 1.0)
        + 0.12 * inverse_ramp(max(flatness, body_noise), 0.02, 0.24)
        + 0.10 * ramp(abs(centroid_slope), 0.02, 0.22)
        + 0.10 * ramp(stereo_width, 0.12, 0.64)
        + 0.07 * inverse_ramp(plucked_panel["plucked_string_score"], 0.18, 0.70)
        + 0.05 * inverse_ramp(reed_panel["reed_wind_score"], 0.18, 0.70)
    )
    fx_transition_authority_score = clamp01(
        0.34 * fx_motion_score
        + 0.18 * onset_panel["swell_onset_score"]
        + 0.15 * ramp(abs(centroid_slope), 0.08, 0.42)
        + 0.12 * ramp(stereo_width, 0.18, 0.72)
        + 0.10 * ramp(tail, 0.22, 0.88)
        + 0.07 * inverse_ramp(loop_panel["pulse_clarity"], 0.14, 0.82)
        + 0.04 * inverse_ramp(drum_loop_source_score, 0.18, 0.74)
    )

    source_summary = {
        "plucked_string_score": plucked_panel["plucked_string_score"],
        "plucked_string_authority_score": plucked_string_authority_score,
        "reed_wind_score": reed_panel["reed_wind_score"],
        "reed_wind_authority_score": reed_wind_authority_score,
        "struck_keys_score": keys_panel["struck_keys_score"],
        "struck_keys_authority_score": struck_keys_authority_score,
        "synth_tonal_source_score": synth_tonal_source_score,
        "bowed_string_score": bowed_string_score,
        "voice_score": voice_score,
        "drum_hit_score": drum_hit_score,
        "drum_loop_source_score": drum_loop_source_score,
        "rhythmic_break_loop_score": rhythmic_break_loop_score,
        "low_fast_pulse_loop_score": low_fast_pulse_loop_score,
        "drum_material_loop_witness_count": drum_material_loop_witness_count,
        "noisy_percussion_loop_candidate": noisy_percussion_loop_candidate,
        "low_rhythmic_break_loop_candidate": low_rhythmic_break_loop_candidate,
        "pitched_repetition_phrase_score": pitched_repetition_phrase_score,
        "pitched_repetition_drum_decoy": pitched_repetition_drum_decoy,
        "tonal_voiced_non_drum_hit_guard": tonal_voiced_non_drum_hit_guard,
        "compact_pitched_music_hit_decoy": compact_pitched_music_hit_decoy,
        "struck_percussion_guard_exception": struck_percussion_guard_exception,
        "pitched_metal_material_evidence": pitched_metal_material_evidence,
        "hand_drum_material_evidence": hand_drum_material_evidence,
        "struck_wood_material_evidence": struck_wood_material_evidence,
        "compact_struck_tonal_percussion_score": compact_struck_tonal_percussion_score,
        "hand_drum_membrane_score": hand_drum_membrane_score,
        "pitched_metal_percussion_score": pitched_metal_percussion_score,
        "struck_wood_score": struck_wood_score,
        "pitched_mallet_instrument_score": pitched_mallet_instrument_score,
        "metallic_noise_score": metallic_noise_score,
        "scrape_rasp_score": scrape_rasp_score,
        "fx_motion_score": fx_motion_score,
        "fx_transition_authority_score": fx_transition_authority_score,
        "tonal_alert_siren_score": tonal_alert_siren_score,
        "texture_bed_score": texture_bed_score,
        "low_end_source_score": low_end_source_score,
        "low_kick_score": low_kick_score,
        "sustained_bass_score": sustained_bass_score,
        "drum_kick_source_score": low_kick_score,
        "drum_snare_source_score": snare_source_score,
        "drum_clap_source_score": clap_source_score,
        "drum_closed_hat_source_score": closed_hat_source_score,
        "drum_cymbal_source_score": cymbal_source_score,
        "drum_tom_conga_source_score": tom_conga_source_score,
        "drum_rim_stick_source_score": rim_stick_source_score,
        "drum_shaker_tambourine_source_score": shaker_tambourine_source_score,
        "drum_guiro_scrape_source_score": guiro_scrape_source_score,
        "drum_metallic_percussion_source_score": metallic_percussion_source_score,
        "fx_riser_build_score": riser_build_score,
        "fx_drop_downlifter_score": drop_downlifter_score,
        "fx_whoosh_sweep_score": whoosh_sweep_score,
        "fx_reverse_score": reverse_fx_score,
        "fx_impact_score": impact_fx_score,
        "fx_glitch_stutter_score": glitch_stutter_score,
        "fx_blip_beep_score": blip_beep_score,
        "fx_formant_score": formant_fx_score,
        "fx_radio_electrical_score": radio_electrical_score,
        "fx_machine_mechanical_score": machine_mechanical_score,
        "fx_foley_material_score": foley_material_score,
        "fx_small_object_score": small_object_score,
        "texture_water_ocean_score": water_ocean_score,
        "texture_rain_score": rain_score,
        "texture_wind_score": wind_score,
        "texture_fire_score": fire_score,
        "texture_thunder_score": thunder_score,
        "texture_noise_static_score": noise_static_score,
        "texture_room_crowd_ambience_score": room_crowd_ambience_score,
        "human_breath_mouth_score": breath_mouth_score,
        "human_spoken_voice_score": spoken_voice_score,
        "human_scream_score": scream_score,
        "human_applause_crowd_score": applause_crowd_score,
        "animal_bird_score": bird_score,
        "animal_voice_score": animal_voice_score,
        "animal_cricket_insect_score": cricket_insect_score,
        "animal_cat_score": cat_voice_score,
        "animal_dog_score": dog_voice_score,
        "fx_door_foley_score": door_foley_score,
        "fx_engine_machine_score": engine_machine_score,
        "fx_motor_machine_score": motor_machine_score,
        "fx_coin_object_score": coin_object_score,
        "fx_key_object_score": key_object_score,
        "fx_boom_score": boom_fx_score,
        "fx_slam_score": slam_fx_score,
        "fx_sub_hit_score": sub_hit_fx_score,
        "fx_alarm_score": alarm_fx_score,
        "fx_siren_score": siren_fx_score,
        "bass_808_score": bass_808_score,
        "bass_sub_score": bass_sub_score,
        "bass_synth_score": bass_synth_score,
        "bass_electric_score": bass_electric_score,
        "bass_upright_score": bass_upright_score,
        "guitar_acoustic_score": guitar_acoustic_score,
        "guitar_electric_score": guitar_electric_score,
        "guitar_nylon_score": guitar_nylon_score,
        "synth_lead_score": synth_lead_score,
        "synth_pad_score": synth_pad_score,
        "synth_chord_score": synth_chord_score,
        "brass_trumpet_score": brass_trumpet_score,
        "woodwind_flute_score": woodwind_flute_score,
        "woodwind_sax_score": woodwind_sax_score,
        "string_violin_score": string_violin_score,
        "string_cello_score": string_cello_score,
        "voice_choir_score": voice_choir_score,
    }
    category_panels = build_category_panel_scores(values, source_summary)
    category_flat = category_panels.get("flat", {}) if isinstance(category_panels, dict) else {}
    if not isinstance(category_flat, dict):
        category_flat = {}

    flattened = {
        **{f"onset_{key}": value for key, value in onset_panel.items()},
        **{f"plucked_{key}": value for key, value in plucked_panel.items()},
        **{f"reed_{key}": value for key, value in reed_panel.items()},
        **{f"keys_{key}": value for key, value in keys_panel.items()},
        **{f"loop_{key}": value for key, value in loop_panel.items()},
        **source_summary,
        **category_flat,
        "role_loop_score": loop_panel["role_loop_score"],
        "role_one_shot_score": loop_panel["role_one_shot_score"],
        "role_phrase_score": loop_panel["role_phrase_score"],
        "physics_subpanel_event_count": event_count,
        "physics_subpanel_clean_tone": clean_tone,
        "physics_subpanel_noisy_air": noisy_air,
    }
    return {
        "onset": round_score_map(onset_panel),
        "plucked_string": round_score_map(plucked_panel),
        "reed_wind": round_score_map(reed_panel),
        "struck_keys": round_score_map(keys_panel),
        "loop_role": round_score_map(loop_panel),
        "drum_family": round_score_map(
            {
                "kick": low_kick_score,
                "snare": snare_source_score,
                "clap": clap_source_score,
                "closed_hat": closed_hat_source_score,
                "cymbal": cymbal_source_score,
                "tom_conga": tom_conga_source_score,
                "rim_stick": rim_stick_source_score,
                "shaker_tambourine": shaker_tambourine_source_score,
                "guiro_scrape": guiro_scrape_source_score,
                "metallic_percussion": metallic_percussion_source_score,
            }
        ),
        "fx_family": round_score_map(
            {
                "riser_build": riser_build_score,
                "drop_downlifter": drop_downlifter_score,
                "whoosh_sweep": whoosh_sweep_score,
                "reverse": reverse_fx_score,
                "impact": impact_fx_score,
                "glitch_stutter": glitch_stutter_score,
                "blip_beep": blip_beep_score,
                "formant": formant_fx_score,
                "radio_electrical": radio_electrical_score,
                "machine_mechanical": machine_mechanical_score,
                "foley_material": foley_material_score,
                "small_object": small_object_score,
                "door": door_foley_score,
                "engine": engine_machine_score,
                "motor": motor_machine_score,
                "coin": coin_object_score,
                "key": key_object_score,
                "boom": boom_fx_score,
                "slam": slam_fx_score,
                "sub_hit": sub_hit_fx_score,
                "alarm": alarm_fx_score,
                "siren": siren_fx_score,
            }
        ),
        "texture_environment": round_score_map(
            {
                "water_ocean": water_ocean_score,
                "rain": rain_score,
                "wind": wind_score,
                "fire": fire_score,
                "thunder": thunder_score,
                "noise_static": noise_static_score,
                "room_crowd_ambience": room_crowd_ambience_score,
            }
        ),
        "human_animal": round_score_map(
            {
                "breath_mouth": breath_mouth_score,
                "spoken_voice": spoken_voice_score,
                "scream": scream_score,
                "applause_crowd": applause_crowd_score,
                "bird": bird_score,
                "animal_voice": animal_voice_score,
                "cricket_insect": cricket_insect_score,
                "cat": cat_voice_score,
                "dog": dog_voice_score,
            }
        ),
        "instrument_family": round_score_map(
            {
                "bass_808": bass_808_score,
                "bass_sub": bass_sub_score,
                "bass_synth": bass_synth_score,
                "bass_electric": bass_electric_score,
                "bass_upright": bass_upright_score,
                "guitar_acoustic": guitar_acoustic_score,
                "guitar_electric": guitar_electric_score,
                "guitar_nylon": guitar_nylon_score,
                "synth_lead": synth_lead_score,
                "synth_pad": synth_pad_score,
                "synth_chord": synth_chord_score,
                "brass_trumpet": brass_trumpet_score,
                "woodwind_flute": woodwind_flute_score,
                "woodwind_sax": woodwind_sax_score,
                "string_violin": string_violin_score,
                "string_cello": string_cello_score,
                "voice_choir": voice_choir_score,
            }
        ),
        "category_panels": category_panels,
        "panel_coverage": category_panels.get("coverage", {}) if isinstance(category_panels, dict) else {},
        "source_identity": round_score_map(source_summary),
        "flat": round_score_map(flattened),
    }
