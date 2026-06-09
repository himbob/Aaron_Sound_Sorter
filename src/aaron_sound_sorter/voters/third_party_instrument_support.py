# SOURCE-NAME BLINDNESS INVARIANT:
# This module reads only decoded-audio measurements already attached to
# SharedAudioFacts. It must never inspect filenames, folders, ZIP members, or
# sample-pack names as classification evidence.
"""Source-blind third-party instrument evidence.

Librosa exposes useful primitive audio measurements, but raw keys such as
``librosa_spectral_contrast_mean`` are too low-level for branch voters to use
safely.  This module turns those primitive measurements into conservative
instrument-support scores.  The scores are *witnesses*, not final labels: they
can lift a low-level physics branch, but they do not own routing.

Design rule: separate *role/gesture* from *source identity*.  A clean stable
beep, a synth pad, a sax sustain, and a bowed violin can all be pitched; the
branch evidence below therefore uses pitch stability only in combination with
attack/event structure, spectral contrast, breath/noise texture, and motion.
"""

from __future__ import annotations

from typing import Any

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.voters.physics_layer_utils import clamp01, inverse_ramp, ramp, safe_float


def _evidence(facts: SharedAudioFacts | None) -> dict[str, Any]:
    data = getattr(facts, "evidence", {}) if facts is not None else {}
    return data if isinstance(data, dict) else {}


def _number(data: dict[str, Any], name: str, default: float = 0.0) -> float:
    return safe_float(data.get(name, default), default)


def _mean_strength(*parts: float) -> float:
    usable = [clamp01(part) for part in parts]
    return float(sum(usable) / max(1, len(usable)))


def _geo_strength(*parts: float) -> float:
    """Soft AND for evidence that should require several witnesses.

    Arithmetic means let one very loud primitive hide missing counter-evidence.
    The square-rooted product keeps scores conservative without making them
    brittle when one primitive is absent from an older adapter output.
    """
    vals = [max(0.0, clamp01(part)) for part in parts]
    if not vals:
        return 0.0
    product = 1.0
    for value in vals:
        product *= max(0.001, value)
    return float(product ** (1.0 / len(vals)))


def _gesture_profile(data: dict[str, Any]) -> dict[str, float]:
    """Compute reusable role/gesture witnesses from librosa primitives."""
    tonal = _number(data, "librosa_tonal_confidence")
    harmonic = _number(data, "librosa_harmonic_energy_ratio")
    percussive = _number(data, "librosa_percussive_confidence")
    loop_conf = _number(data, "librosa_loop_confidence")
    onset_events = _number(data, "librosa_onset_event_count")
    onset_density = _number(data, "librosa_onset_density_hz")
    chroma_peak = _number(data, "librosa_chroma_peak_mean")
    chroma_entropy = _number(data, "librosa_chroma_entropy_norm")
    chroma_change = _number(data, "librosa_chroma_frame_change_mean")
    spectral_contrast = _number(data, "librosa_spectral_contrast_mean")
    contrast_std = _number(data, "librosa_spectral_contrast_std")
    centroid = _number(data, "librosa_spectral_centroid_mean")
    centroid_std = _number(data, "librosa_spectral_centroid_std")
    centroid_slope = abs(_number(data, "librosa_spectral_centroid_slope_norm"))
    flatness = _number(data, "librosa_spectral_flatness_mean")
    zcr = _number(data, "librosa_zero_crossing_rate_mean")
    zcr_std = _number(data, "librosa_zero_crossing_rate_std")
    bandwidth = _number(data, "librosa_spectral_bandwidth_mean")
    bandwidth_std = _number(data, "librosa_spectral_bandwidth_std")
    rms_std = _number(data, "librosa_rms_std")
    mfcc_motion = max(_number(data, "librosa_mfcc_delta_std"), _number(data, "librosa_mfcc_delta2_std"))
    f0_voiced = _number(data, "librosa_yin_voiced_ratio")
    f0_stability = _number(data, "librosa_yin_f0_stability")
    f0_motion = _number(data, "librosa_yin_f0_motion_cents")
    duration = _number(data, "duration_sec")

    clean_tone = _mean_strength(
        ramp(tonal, 0.42, 0.84),
        ramp(harmonic, 0.42, 0.86),
        inverse_ramp(flatness, 0.02, 0.25),
        inverse_ramp(zcr, 0.02, 0.16),
    )
    sustained_pitch = _mean_strength(
        ramp(f0_voiced, 0.42, 0.88),
        ramp(f0_stability, 0.20, 0.78),
        ramp(chroma_peak, 0.18, 0.42),
        inverse_ramp(chroma_entropy, 0.45, 0.92),
    )
    eventful_phrase = max(ramp(onset_events, 3.0, 14.0), ramp(loop_conf, 0.34, 0.82))
    short_event = _mean_strength(
        inverse_ramp(duration, 0.28, 1.20) if duration > 0.0 else 0.0,
        inverse_ramp(onset_events, 1.0, 4.0),
        inverse_ramp(loop_conf, 0.12, 0.42),
    )
    phrase_scale = max(ramp(duration, 0.70, 2.80) if duration > 0.0 else 0.45, eventful_phrase)
    breath_noise = _mean_strength(
        ramp(max(flatness, zcr), 0.07, 0.36),
        ramp(max(zcr_std, centroid_std / 6000.0), 0.02, 0.16),
        ramp(mfcc_motion, 4.0, 26.0),
        inverse_ramp(percussive, 0.12, 0.66),
    )
    smooth_bow_noise = _mean_strength(
        ramp(max(centroid_std, bandwidth_std), 180.0, 1900.0),
        ramp(max(f0_motion, mfcc_motion), 8.0, 120.0),
        inverse_ramp(onset_density, 0.18, 2.5),
        inverse_ramp(percussive, 0.08, 0.46),
        inverse_ramp(flatness, 0.10, 0.38),
    )
    stable_electronic = _mean_strength(
        clean_tone,
        ramp(chroma_peak, 0.18, 0.50),
        inverse_ramp(chroma_change, 0.015, 0.18),
        inverse_ramp(mfcc_motion, 2.0, 18.0),
        inverse_ramp(zcr, 0.01, 0.12),
        inverse_ramp(centroid_slope, 0.02, 0.22),
    )
    bright_tonal = _mean_strength(
        clean_tone,
        ramp(centroid, 1100.0, 4200.0),
        ramp(_number(data, "librosa_spectral_rolloff85_mean"), 2400.0, 7800.0),
        inverse_ramp(flatness, 0.02, 0.32),
    )
    harmonic_clarity = _mean_strength(
        clean_tone,
        ramp(spectral_contrast, 12.0, 36.0),
        ramp(chroma_peak, 0.14, 0.44),
        inverse_ramp(chroma_entropy, 0.50, 0.94),
    )
    attack_phrase = _mean_strength(
        ramp(onset_density, 0.45, 4.2),
        ramp(onset_events, 2.0, 14.0),
        ramp(mfcc_motion, 3.0, 22.0),
        inverse_ramp(rms_std, 0.20, 0.75),
    )

    return {
        "tonal": tonal,
        "harmonic": harmonic,
        "percussive": percussive,
        "loop_conf": loop_conf,
        "onset_events": onset_events,
        "onset_density": onset_density,
        "chroma_peak": chroma_peak,
        "chroma_entropy": chroma_entropy,
        "chroma_change": chroma_change,
        "spectral_contrast": spectral_contrast,
        "contrast_std": contrast_std,
        "centroid": centroid,
        "centroid_std": centroid_std,
        "flatness": flatness,
        "zcr": zcr,
        "bandwidth": bandwidth,
        "mfcc_motion": mfcc_motion,
        "f0_voiced": f0_voiced,
        "f0_stability": f0_stability,
        "f0_motion": f0_motion,
        "clean_tone": clean_tone,
        "sustained_pitch": sustained_pitch,
        "eventful_phrase": eventful_phrase,
        "short_event": short_event,
        "phrase_scale": phrase_scale,
        "breath_noise": breath_noise,
        "smooth_bow_noise": smooth_bow_noise,
        "stable_electronic": stable_electronic,
        "bright_tonal": bright_tonal,
        "harmonic_clarity": harmonic_clarity,
        "attack_phrase": attack_phrase,
    }


def third_party_instrument_support(facts: SharedAudioFacts | None) -> dict[str, float]:
    """Return conservative branch support scores from third-party measurements.

    The score names intentionally describe broad source families. They should be
    consumed only as supporting evidence beside the project's own physics
    features, shape vote, and brain candidates.
    """
    data = _evidence(facts)
    if not data:
        return _empty_scores()

    g = _gesture_profile(data)
    f0_median = _number(data, "librosa_yin_f0_median_hz")
    chroma_std = _number(data, "librosa_chroma_std")

    not_drumlike = inverse_ramp(g["percussive"], 0.12, 0.58)
    not_short_fx_event = inverse_ramp(g["short_event"], 0.42, 0.86)

    # Voice: moving F0/formant-ish timbre, breath/noise texture, and phrase scale.
    voice = _mean_strength(
        ramp(g["f0_voiced"], 0.42, 0.88),
        ramp(g["breath_noise"], 0.28, 0.80),
        ramp(g["f0_motion"], 22.0, 240.0),
        ramp(g["mfcc_motion"], 5.0, 32.0),
        ramp(g["centroid_std"], 260.0, 2800.0),
        not_drumlike,
        g["phrase_scale"],
    )
    voice *= inverse_ramp(g["stable_electronic"], 0.54, 0.92)

    # Reed/woodwind/sax: sustained pitch plus breath/noise and spectral contrast.
    reed_wind = _mean_strength(
        g["sustained_pitch"],
        ramp(g["spectral_contrast"], 14.0, 38.0),
        ramp(g["contrast_std"], 1.5, 9.0),
        ramp(g["breath_noise"], 0.24, 0.72),
        ramp(g["centroid"], 650.0, 3400.0),
        not_drumlike,
        not_short_fx_event,
    )
    reed_wind *= inverse_ramp(g["stable_electronic"], 0.60, 0.96)
    reed_wind *= inverse_ramp(g["smooth_bow_noise"], 0.72, 0.98)

    # Bowed strings: sustained pitch, smooth non-event motion, low attack density,
    # moderate harmonic noise; not breath/formant heavy and not clean-electronic.
    bowed_string = _mean_strength(
        g["sustained_pitch"],
        g["bright_tonal"],
        ramp(g["smooth_bow_noise"], 0.32, 0.84),
        ramp(g["f0_motion"], 6.0, 150.0),
        ramp(g["bandwidth"], 900.0, 4300.0),
        inverse_ramp(g["onset_density"], 0.20, 3.2),
        not_drumlike,
        not_short_fx_event,
    )
    bowed_string *= inverse_ramp(g["stable_electronic"], 0.68, 0.98)
    bowed_string *= inverse_ramp(g["breath_noise"], 0.70, 0.96)
    bowed_string *= inverse_ramp(g["attack_phrase"], 0.52, 0.92)
    bowed_string *= ramp(g["f0_stability"], 0.24, 0.74)

    # Guitar/plucked strings: tonal percussive event structure, fast changing
    # envelope/timbre, not bowed sustain and not drum/noise loops.
    plucked_string = _mean_strength(
        g["clean_tone"],
        ramp(g["percussive"], 0.10, 0.46),
        g["attack_phrase"],
        ramp(g["mfcc_motion"], 3.5, 22.0),
        inverse_ramp(g["f0_voiced"], 0.18, 0.72),
        inverse_ramp(g["flatness"], 0.04, 0.24),
        g["eventful_phrase"],
    )
    plucked_string *= inverse_ramp(g["flatness"], 0.04, 0.36)
    plucked_string *= inverse_ramp(g["percussive"], 0.18, 0.78)
    plucked_string *= inverse_ramp(bowed_string, 0.54, 0.92)

    synth = _mean_strength(
        g["stable_electronic"],
        ramp(g["clean_tone"], 0.48, 0.90),
        ramp(chroma_std, 0.04, 0.34),
        inverse_ramp(g["flatness"], 0.01, 0.20),
        inverse_ramp(g["breath_noise"], 0.40, 0.86),
    )
    if g["loop_conf"] >= 0.42 or g["onset_events"] >= 3.0:
        synth = max(synth, 0.88 * _mean_strength(g["stable_electronic"], g["eventful_phrase"], g["clean_tone"]))
    # Short clean beeps are FX blips, not synth instruments.  Let the FX layer
    # keep those; longer/eventful clean tones may still support synth.
    synth *= max(0.55, not_short_fx_event)
    synth *= inverse_ramp(max(g["breath_noise"], g["f0_motion"] / 220.0), 0.52, 1.0)

    keys = _mean_strength(
        g["clean_tone"],
        ramp(g["percussive"], 0.08, 0.34),
        ramp(g["spectral_contrast"], 10.0, 30.0),
        inverse_ramp(g["flatness"], 0.01, 0.18),
        ramp(g["chroma_peak"], 0.14, 0.40),
        g["eventful_phrase"],
    )
    keys *= inverse_ramp(g["breath_noise"], 0.52, 0.90)
    keys *= inverse_ramp(bowed_string, 0.62, 0.94)

    strongest_identity = max(voice, reed_wind, plucked_string, synth, keys, bowed_string)
    second_identity = sorted([voice, reed_wind, plucked_string, synth, keys, bowed_string], reverse=True)[1]
    mixed_loop = _mean_strength(
        ramp(g["loop_conf"], 0.38, 0.82),
        ramp(g["onset_events"], 4.0, 18.0),
        ramp(g["chroma_change"], 0.04, 0.28),
        ramp(g["mfcc_motion"], 4.0, 26.0),
        ramp(strongest_identity, 0.20, 0.70),
    )
    mixed_loop = max(
        mixed_loop,
        _mean_strength(
            ramp(second_identity, 0.28, 0.66),
            ramp(strongest_identity, 0.38, 0.78),
            ramp(g["loop_conf"], 0.30, 0.78),
        ),
    )

    # Confidence margin diagnostics.  These are not final labels, but they make
    # debugging much easier and help downstream claim producers know when the API
    # witness is broad rather than decisive.
    identity_scores = {
        "voice": voice,
        "reed_wind": reed_wind,
        "bowed_string": bowed_string,
        "plucked_string": plucked_string,
        "synth": synth,
        "keys": keys,
    }
    ordered = sorted(identity_scores.items(), key=lambda item: item[1], reverse=True)
    top_name, top_score = ordered[0]
    second_score = ordered[1][1]
    top_margin = clamp01(top_score - second_score)

    return {
        "api_clean_tonal_source": round(clamp01(g["clean_tone"]), 6),
        "api_sustained_pitch_source": round(clamp01(g["sustained_pitch"]), 6),
        "api_plucked_string_support": round(clamp01(plucked_string), 6),
        "api_voice_support": round(clamp01(voice), 6),
        "api_reed_wind_support": round(clamp01(reed_wind), 6),
        "api_bowed_string_support": round(clamp01(bowed_string), 6),
        "api_synth_support": round(clamp01(synth), 6),
        "api_keys_support": round(clamp01(keys), 6),
        "api_mixed_loop_support": round(clamp01(mixed_loop), 6),
        "api_low_register_pitch": round(clamp01(inverse_ramp(f0_median, 45.0, 220.0) if f0_median > 0.0 else 0.0), 6),
        "api_mid_register_pitch": round(clamp01(ramp(f0_median, 130.0, 620.0) if f0_median > 0.0 else 0.0), 6),
        "api_high_register_pitch": round(clamp01(ramp(f0_median, 500.0, 1800.0) if f0_median > 0.0 else 0.0), 6),
        "api_eventful_tonal_phrase": round(clamp01(_mean_strength(g["eventful_phrase"], g["clean_tone"], not_drumlike)), 6),
        "api_short_clean_tonal_event": round(clamp01(_geo_strength(g["short_event"], g["clean_tone"], g["stable_electronic"])), 6),
        "api_breath_noise_texture": round(clamp01(g["breath_noise"]), 6),
        "api_smooth_bow_texture": round(clamp01(g["smooth_bow_noise"]), 6),
        "api_instrument_identity_top_score": round(clamp01(top_score), 6),
        "api_instrument_identity_second_score": round(clamp01(second_score), 6),
        "api_instrument_identity_margin": round(clamp01(top_margin), 6),
        # Numeric codes keep the evidence dict manifest-friendly without
        # string labels becoming routing inputs.
        "api_instrument_identity_top_code": float(
            {"voice": 1, "reed_wind": 2, "bowed_string": 3, "plucked_string": 4, "synth": 5, "keys": 6}[top_name]
        ),
    }


def _empty_scores() -> dict[str, float]:
    return {
        "api_clean_tonal_source": 0.0,
        "api_sustained_pitch_source": 0.0,
        "api_plucked_string_support": 0.0,
        "api_voice_support": 0.0,
        "api_reed_wind_support": 0.0,
        "api_bowed_string_support": 0.0,
        "api_synth_support": 0.0,
        "api_keys_support": 0.0,
        "api_mixed_loop_support": 0.0,
        "api_low_register_pitch": 0.0,
        "api_mid_register_pitch": 0.0,
        "api_high_register_pitch": 0.0,
        "api_eventful_tonal_phrase": 0.0,
        "api_short_clean_tonal_event": 0.0,
        "api_breath_noise_texture": 0.0,
        "api_smooth_bow_texture": 0.0,
        "api_instrument_identity_top_score": 0.0,
        "api_instrument_identity_second_score": 0.0,
        "api_instrument_identity_margin": 0.0,
        "api_instrument_identity_top_code": 0.0,
    }
