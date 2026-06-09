# SOURCE-NAME BLINDNESS INVARIANT:
# This module may inspect only decoded audio arrays and sample rates. It must
# never inspect producer filenames, source folders, ZIP member names, sample-pack
# labels, or user corrections as classification evidence.
"""Third-party measured audio feature adapters.

The sorter keeps its own stable fingerprint and brain dimensions.  Third-party
libraries feed extra measured facts into voters and claim producers; they do not
choose final folders directly.

``librosa`` is a required runtime dependency for audio analysis, but this
module must still be import-safe.  Test collection, CLI help, packaging checks,
and lightweight metadata commands should not fail just because the active shell
points at a Python without optional wheels installed.  The adapter lazy-loads
librosa at feature-extraction time and reports a controlled missing-dependency
status instead of crashing package import.  Heavier model-based scouts such as
Essentia, YAMNet, or MediaPipe are represented as explicit adapter slots, but
they require model files and are not allowed to own final routing.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any
import warnings

import numpy as np

HOP_LENGTH = 512
N_FFT = 2048
MAX_THIRD_PARTY_SECONDS = 4.0


@lru_cache(maxsize=1)
def _load_librosa() -> Any:
    """Import librosa lazily so package/test import stays robust.

    Librosa is required for the enhanced measured-audio path, but importing the
    whole package at module import time makes every pytest collection fail when
    the shell accidentally uses a Python environment that lacks librosa.  Keep
    the failure local to actual third-party feature extraction.
    """
    try:
        import librosa as _librosa  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return None
    except Exception:
        return None
    return _librosa


def librosa_is_available() -> bool:
    """Return True when the active interpreter can import librosa."""
    return _load_librosa() is not None


def _finite_float(value: object, default: float = 0.0) -> float:
    """Coerce a scalar to a finite float."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except Exception:
        return default
    if not np.isfinite(number):
        return default
    return number


def _safe_mean(values: object) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else 0.0


def _safe_std(values: object) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if arr.size else 0.0


def _safe_max(values: object) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else 0.0


def _safe_median(values: object) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else 0.0


def _safe_percentile(values: object, percentile: float) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, percentile)) if arr.size else 0.0


def _safe_sum(values: object) -> float:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.sum(arr)) if arr.size else 0.0


def _bounded_mono(mono: np.ndarray, sr: int) -> np.ndarray:
    """Return a normalized finite mono vector bounded for analysis cost."""
    y = np.asarray(mono, dtype=np.float32).reshape(-1)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    max_len = max(N_FFT, int(MAX_THIRD_PARTY_SECONDS * max(1, int(sr))))
    if y.size > max_len:
        y = y[:max_len]
    if y.size < N_FFT:
        y = np.pad(y, (0, N_FFT - y.size), mode="constant")
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-9:
        y = y / peak
    return y.astype(np.float32, copy=False)


def _fast_onset_envelope(y: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Cheap source-blind onset summary for per-file sorter runs.

    Librosa's onset_strength is accurate, but its first call can JIT/initialize
    slowly in short-lived smoke processes.  This envelope uses frame RMS and
    spectral-flux-style positive deltas, so the sorter keeps fast rhythmic
    evidence without paying the heavy onset detector cost for every file.
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size < N_FFT:
        y = np.pad(y, (0, N_FFT - y.size), mode="constant")
    frame_count = 1 + max(0, (y.size - N_FFT) // HOP_LENGTH)
    if frame_count <= 0:
        return np.zeros(1, dtype=np.float32), 0
    window = np.hanning(N_FFT).astype(np.float32)
    env = np.zeros(frame_count, dtype=np.float32)
    for index in range(frame_count):
        start = index * HOP_LENGTH
        frame = y[start : start + N_FFT]
        if frame.size < N_FFT:
            frame = np.pad(frame, (0, N_FFT - frame.size), mode="constant")
        weighted = frame * window
        env[index] = float(np.sqrt(np.mean(weighted * weighted)))
    diff = np.maximum(0.0, np.diff(env, prepend=env[0]))
    if diff.size <= 2 or _safe_max(diff) <= 1e-9:
        return diff.astype(np.float32), 0
    threshold = max(_safe_mean(diff) + 1.75 * _safe_std(diff), _safe_max(diff) * 0.20)
    min_gap = max(1, int(round(0.075 * max(1, sr) / HOP_LENGTH)))
    count = 0
    last = -9999
    for index, value in enumerate(diff):
        if value >= threshold and index - last >= min_gap:
            count += 1
            last = index
    return diff.astype(np.float32), int(count)


def third_party_feature_profile_from_audio(mono: np.ndarray, sr: int) -> dict[str, Any]:
    """Return required librosa features plus declared third-party scout slots.

    The returned dictionary is diagnostic/evidence data.  It intentionally avoids
    folder names and classifier labels so it can be safely consumed by measured
    voters without reintroducing source-name routing.
    """
    y = _bounded_mono(mono, int(sr))
    duration = float(y.size / max(1, int(sr)))
    profile: dict[str, Any] = {
        "status": "ok",
        "duration_sec": duration,
        "dependency_policy": "librosa_required_runtime",
        "adapters": {},
        "flat": {},
    }
    flat: dict[str, float] = {}
    librosa_result = _librosa_feature_profile(y, int(sr), duration)
    profile["adapters"]["librosa"] = librosa_result
    if librosa_result.get("status") != "ok":
        profile["status"] = str(librosa_result.get("status", "librosa_error"))
    if isinstance(librosa_result.get("flat"), dict):
        flat.update(
            {str(name): _finite_float(value) for name, value in librosa_result["flat"].items() if isinstance(name, str)}
        )
    # Heavy model-based APIs are deliberately visible, not implicit.  They can be
    # wired later as scout voters without changing the stable sorter fingerprint.
    profile["adapters"].update(_declared_model_scout_slots())
    profile["flat"] = flat
    return profile



def _empty_yin_summary() -> dict[str, float]:
    return {
        "librosa_yin_f0_median_hz": 0.0,
        "librosa_yin_voiced_ratio": 0.0,
        "librosa_yin_f0_stability": 0.0,
        "librosa_yin_f0_motion_cents": 0.0,
    }


def _librosa_yin_summary(librosa: Any, y: np.ndarray, sr: int) -> dict[str, float]:
    """Return bounded YIN F0 facts without letting pitch own routing."""
    yin_fn = getattr(librosa, "yin", None)
    if yin_fn is None:
        return {
            "librosa_yin_f0_median_hz": 0.0,
            "librosa_yin_voiced_ratio": 0.0,
            "librosa_yin_f0_stability": 0.0,
            "librosa_yin_f0_motion_cents": 0.0,
        }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f0 = np.asarray(
                yin_fn(
                    y,
                    fmin=55.0,
                    fmax=1760.0,
                    sr=sr,
                    frame_length=N_FFT,
                    hop_length=HOP_LENGTH,
                    trough_threshold=0.12,
                ),
                dtype=np.float32,
            ).reshape(-1)
    except Exception:
        return {
            "librosa_yin_f0_median_hz": 0.0,
            "librosa_yin_voiced_ratio": 0.0,
            "librosa_yin_f0_stability": 0.0,
            "librosa_yin_f0_motion_cents": 0.0,
        }
    finite = f0[np.isfinite(f0) & (f0 > 0.0)]
    if finite.size < 3:
        return {
            "librosa_yin_f0_median_hz": 0.0,
            "librosa_yin_voiced_ratio": 0.0,
            "librosa_yin_f0_stability": 0.0,
            "librosa_yin_f0_motion_cents": 0.0,
        }
    median = _safe_median(finite)
    if median <= 0.0:
        return {
            "librosa_yin_f0_median_hz": 0.0,
            "librosa_yin_voiced_ratio": 0.0,
            "librosa_yin_f0_stability": 0.0,
            "librosa_yin_f0_motion_cents": 0.0,
        }
    cents = 1200.0 * np.log2(np.maximum(finite, 1e-6) / max(median, 1e-6))
    cents_iqr = max(0.0, _safe_percentile(cents, 75.0) - _safe_percentile(cents, 25.0))
    diffs = np.abs(np.diff(cents)) if cents.size >= 2 else np.zeros(1, dtype=np.float32)
    # YIN always returns a candidate, so treat only candidates close to the
    # median contour as voiced. This keeps noisy FX from looking fully pitched.
    close = np.abs(cents) <= 90.0
    voiced_ratio = float(np.sum(close) / max(1, f0.size))
    stability = float(np.clip(1.0 - cents_iqr / 360.0, 0.0, 1.0))
    motion = _safe_mean(diffs)
    return {
        "librosa_yin_f0_median_hz": median,
        "librosa_yin_voiced_ratio": voiced_ratio,
        "librosa_yin_f0_stability": stability,
        "librosa_yin_f0_motion_cents": motion,
    }


def _librosa_feature_profile(y: np.ndarray, sr: int, duration: float) -> dict[str, Any]:
    """Return librosa-derived rhythm, tonal, timbre, and HPSS support features."""
    librosa = _load_librosa()
    if librosa is None:
        return {
            "status": "missing_dependency:librosa",
            "flat": {},
            "message": "Install project dependencies before running audio analysis.",
        }
    try:
        onset_env, onset_event_count = _fast_onset_envelope(y, sr)

        tempo_bpm = 0.0
        tempo_fn = getattr(getattr(librosa, "feature", None), "tempo", None)
        if tempo_fn is None:
            tempo_fn = getattr(getattr(librosa, "beat", None), "tempo", None)
        if tempo_fn is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tempo_values = tempo_fn(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
            tempo_bpm = _safe_max(tempo_values)

        tempogram_peak = 0.0
        tempogram_mean = 0.0
        tempogram_ratio_peak = 0.0
        tempogram_fn = getattr(getattr(librosa, "feature", None), "tempogram", None)
        if tempogram_fn is not None:
            tempogram = tempogram_fn(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
            tempogram_peak = _safe_max(tempogram)
            tempogram_mean = _safe_mean(tempogram)
        # tempogram_ratio is useful but comparatively expensive on long smoke runs.
        # Keep the output field stable and let tempogram_peak carry rhythmic support.
        tempogram_ratio_peak = 0.0

        chroma_peak = 0.0
        chroma_entropy = 0.0
        chroma_std = 0.0
        chroma_frame_change_mean = 0.0
        chroma_peak_std = 0.0
        chroma_fn = getattr(getattr(librosa, "feature", None), "chroma_stft", None)
        if chroma_fn is not None:
            # Passing tuning=0.0 prevents librosa from trying to estimate tuning
            # on unpitched FX/noise files, which produced noisy smoke-test warnings.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                chroma = np.asarray(
                    chroma_fn(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, tuning=0.0),
                    dtype=np.float32,
                )
            if chroma.size:
                chroma_sum = np.sum(np.maximum(chroma, 0.0), axis=0) + 1e-9
                probs = np.maximum(chroma, 0.0) / chroma_sum[None, :]
                frame_entropy = -np.sum(probs * np.log2(probs + 1e-12), axis=0) / np.log2(12.0)
                frame_peaks = np.max(probs, axis=0)
                chroma_peak = _safe_mean(frame_peaks)
                chroma_peak_std = _safe_std(frame_peaks)
                chroma_entropy = _safe_mean(frame_entropy)
                chroma_std = _safe_std(chroma)
                if chroma.shape[-1] >= 2:
                    chroma_frame_change_mean = _safe_mean(np.mean(np.abs(np.diff(probs, axis=1)), axis=0))

        spectral_contrast_mean = 0.0
        spectral_contrast_std = 0.0
        contrast_fn = getattr(getattr(librosa, "feature", None), "spectral_contrast", None)
        if contrast_fn is not None:
            contrast = contrast_fn(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
            spectral_contrast_mean = _safe_mean(contrast)
            spectral_contrast_std = _safe_std(contrast)

        spectral_centroid_mean = 0.0
        spectral_centroid_std = 0.0
        spectral_centroid_slope_norm = 0.0
        centroid_fn = getattr(getattr(librosa, "feature", None), "spectral_centroid", None)
        if centroid_fn is not None:
            centroid = np.asarray(
                centroid_fn(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH),
                dtype=np.float32,
            ).reshape(-1)
            spectral_centroid_mean = _safe_mean(centroid)
            spectral_centroid_std = _safe_std(centroid)
            finite_centroid = centroid[np.isfinite(centroid)]
            if finite_centroid.size >= 5:
                window = max(1, finite_centroid.size // 5)
                start = _safe_mean(finite_centroid[:window])
                end = _safe_mean(finite_centroid[-window:])
                spectral_centroid_slope_norm = float(np.clip((end - start) / (max(1, sr) / 2.0), -1.0, 1.0))

        spectral_rolloff85_mean = 0.0
        rolloff_fn = getattr(getattr(librosa, "feature", None), "spectral_rolloff", None)
        if rolloff_fn is not None:
            rolloff = rolloff_fn(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, roll_percent=0.85)
            spectral_rolloff85_mean = _safe_mean(rolloff)

        spectral_flatness_mean = 0.0
        flatness_fn = getattr(getattr(librosa, "feature", None), "spectral_flatness", None)
        if flatness_fn is not None:
            flatness = flatness_fn(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)
            spectral_flatness_mean = _safe_mean(flatness)

        rms_mean = 0.0
        rms_std = 0.0
        rms_fn = getattr(getattr(librosa, "feature", None), "rms", None)
        if rms_fn is not None:
            rms = rms_fn(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
            rms_mean = _safe_mean(rms)
            rms_std = _safe_std(rms)

        zcr_mean = 0.0
        zcr_std = 0.0
        zcr_fn = getattr(getattr(librosa, "feature", None), "zero_crossing_rate", None)
        if zcr_fn is not None:
            zcr = zcr_fn(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
            zcr_mean = _safe_mean(zcr)
            zcr_std = _safe_std(zcr)

        spectral_bandwidth_mean = 0.0
        spectral_bandwidth_std = 0.0
        bandwidth_fn = getattr(getattr(librosa, "feature", None), "spectral_bandwidth", None)
        if bandwidth_fn is not None:
            bandwidth = bandwidth_fn(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
            spectral_bandwidth_mean = _safe_mean(bandwidth)
            spectral_bandwidth_std = _safe_std(bandwidth)

        mfcc_delta_std = 0.0
        mfcc_delta2_std = 0.0
        mfcc_fn = getattr(getattr(librosa, "feature", None), "mfcc", None)
        if mfcc_fn is not None:
            mfcc = np.asarray(mfcc_fn(y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP_LENGTH), dtype=np.float32)
            frame_count = int(mfcc.shape[-1]) if mfcc.ndim >= 2 else 0
            if mfcc.size and frame_count >= 3:
                # librosa.feature.delta defaults to width=9.  Short one-shots can
                # have fewer frames, so choose the largest valid odd width that
                # fits the actual frame count instead of letting the whole adapter
                # fail on short audio.
                delta_width = min(9, frame_count if frame_count % 2 == 1 else frame_count - 1)
                delta_width = max(3, delta_width)
                delta = librosa.feature.delta(mfcc, width=delta_width, order=1)
                delta2 = librosa.feature.delta(mfcc, width=delta_width, order=2)
                mfcc_delta_std = _safe_std(delta)
                mfcc_delta2_std = _safe_std(delta2)

        # YIN is useful for instrument identity, but it is one of the more
        # expensive API calls.  Gate it with cheap tonal evidence so unpitched
        # drums/FX do not pay for F0 tracking on every smoke case.
        yin_tonal_gate = bool(
            chroma_peak >= 0.14
            or spectral_contrast_mean >= 12.0
            or spectral_flatness_mean <= 0.18
        )
        yin_noise_skip = bool(
            onset_event_count >= 1
            and chroma_peak < 0.10
            and spectral_flatness_mean >= 0.26
            and spectral_contrast_mean < 16.0
        )
        yin_summary = (
            _librosa_yin_summary(librosa, y, sr)
            if yin_tonal_gate and not yin_noise_skip
            else _empty_yin_summary()
        )

        onset_strength_mean = _safe_mean(onset_env)
        onset_strength_std = _safe_std(onset_env)
        onset_strength_max = _safe_max(onset_env)
        onset_density_hz = float(onset_event_count / max(duration, 0.001))
        loop_confidence = _librosa_loop_confidence(
            duration=duration,
            onset_event_count=onset_event_count,
            onset_density_hz=onset_density_hz,
            onset_strength_std=onset_strength_std,
            tempogram_peak=tempogram_peak,
            tempogram_ratio_peak=tempogram_ratio_peak,
            tempo_bpm=tempo_bpm,
        )
        harmonic_energy_ratio = float(
            np.clip(
                0.58 * chroma_peak
                + 0.24 * min(1.0, spectral_contrast_mean / 35.0)
                + 0.18 * max(0.0, 1.0 - min(1.0, zcr_mean * 12.0)),
                0.0,
                1.0,
            )
        )
        percussive_energy_ratio = float(
            np.clip(
                0.50 * min(1.0, onset_density_hz / 6.0)
                + 0.35 * min(1.0, onset_strength_max / 1.5)
                + 0.15 * max(0.0, 1.0 - harmonic_energy_ratio),
                0.0,
                1.0,
            )
        )
        hpss_balance = harmonic_energy_ratio - percussive_energy_ratio
        tonal_confidence = float(
            np.clip(
                0.50 * chroma_peak
                + 0.22 * min(1.0, spectral_contrast_mean / 35.0)
                + 0.18 * harmonic_energy_ratio
                + 0.10 * max(0.0, 1.0 - min(1.0, zcr_mean * 12.0)),
                0.0,
                1.0,
            )
        )
        percussive_confidence = float(
            np.clip(
                0.45 * percussive_energy_ratio
                + 0.25 * min(1.0, onset_density_hz / 6.0)
                + 0.20 * min(1.0, onset_strength_max / 8.0)
                + 0.10 * min(1.0, spectral_bandwidth_mean / 4500.0),
                0.0,
                1.0,
            )
        )
        flat = {
            "duration_sec": duration,
            "librosa_onset_strength_mean": onset_strength_mean,
            "librosa_onset_strength_std": onset_strength_std,
            "librosa_onset_strength_max": onset_strength_max,
            "librosa_onset_event_count": float(onset_event_count),
            "librosa_onset_density_hz": onset_density_hz,
            "librosa_tempo_bpm": tempo_bpm,
            "librosa_tempogram_peak": tempogram_peak,
            "librosa_tempogram_mean": tempogram_mean,
            "librosa_tempogram_ratio_peak": tempogram_ratio_peak,
            "librosa_chroma_peak_mean": chroma_peak,
            "librosa_chroma_peak_std": chroma_peak_std,
            "librosa_chroma_entropy_norm": chroma_entropy,
            "librosa_chroma_std": chroma_std,
            "librosa_chroma_frame_change_mean": chroma_frame_change_mean,
            "librosa_spectral_contrast_mean": spectral_contrast_mean,
            "librosa_spectral_contrast_std": spectral_contrast_std,
            "librosa_spectral_centroid_mean": spectral_centroid_mean,
            "librosa_spectral_centroid_std": spectral_centroid_std,
            "librosa_spectral_centroid_slope_norm": spectral_centroid_slope_norm,
            "librosa_spectral_rolloff85_mean": spectral_rolloff85_mean,
            "librosa_spectral_flatness_mean": spectral_flatness_mean,
            "librosa_rms_mean": rms_mean,
            "librosa_rms_std": rms_std,
            "librosa_zero_crossing_rate_mean": zcr_mean,
            "librosa_zero_crossing_rate_std": zcr_std,
            "librosa_spectral_bandwidth_mean": spectral_bandwidth_mean,
            "librosa_spectral_bandwidth_std": spectral_bandwidth_std,
            "librosa_mfcc_delta_std": mfcc_delta_std,
            "librosa_mfcc_delta2_std": mfcc_delta2_std,
            "librosa_harmonic_energy_ratio": harmonic_energy_ratio,
            "librosa_percussive_energy_ratio": percussive_energy_ratio,
            "librosa_hpss_balance": hpss_balance,
            "librosa_loop_confidence": loop_confidence,
            "librosa_tonal_confidence": tonal_confidence,
            "librosa_percussive_confidence": percussive_confidence,
            **yin_summary,
        }
        return {"status": "ok", "flat": flat}
    except Exception as exc:
        return {"status": "error:" + str(exc)[:120], "flat": {}}


def _declared_model_scout_slots() -> dict[str, dict[str, Any]]:
    """Declare heavier model APIs without making them hidden routing inputs.

    These entries make the architecture explicit: future model outputs should be
    added as scout evidence beside measured facts, never as final folder owners.
    """
    return {
        "essentia": {
            "status": "not_configured",
            "role": "future high-level music/audio descriptor scout",
            "owns_final_folder": False,
        },
        "yamnet": {
            "status": "not_configured",
            "role": "future AudioSet event-label scout",
            "owns_final_folder": False,
        },
        "mediapipe_audio_classifier": {
            "status": "not_configured",
            "role": "future ranked audio-category scout",
            "owns_final_folder": False,
        },
    }


def _librosa_loop_confidence(
    *,
    duration: float,
    onset_event_count: int,
    onset_density_hz: float,
    onset_strength_std: float,
    tempogram_peak: float,
    tempogram_ratio_peak: float,
    tempo_bpm: float,
) -> float:
    """Summarize librosa rhythm features into a conservative loop-support score."""
    if duration < 1.25 or onset_event_count < 2:
        return 0.0
    event_score = min(1.0, max(0.0, (float(onset_event_count) - 2.0) / 6.0))
    density_score = min(1.0, max(0.0, onset_density_hz / 4.0))
    tempogram_score = min(1.0, max(0.0, tempogram_peak / 4.0))
    ratio_score = min(1.0, max(0.0, tempogram_ratio_peak / 4.0))
    tempo_score = 1.0 if 45.0 <= tempo_bpm <= 240.0 else 0.0
    modulation_score = min(1.0, max(0.0, onset_strength_std / 1.5))
    return float(
        np.clip(
            0.30 * event_score
            + 0.20 * density_score
            + 0.20 * tempogram_score
            + 0.10 * ratio_score
            + 0.10 * tempo_score
            + 0.10 * modulation_score,
            0.0,
            1.0,
        )
    )
