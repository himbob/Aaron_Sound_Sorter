# Auto-split from Aaron_Sound_Sorter.py.
# This is a component module, not a legacy wrapper.
from __future__ import annotations

import signal
import threading

from .core import *
from .io_utils import *
from .third_party_audio_features import third_party_feature_profile_from_audio
from .training_labels import *
from .training_labels import (
    _expanded_decay_and_low_source_descriptors,
    _frame_level_f0_stats,
    _harmonic_descriptors,
    _segment_descriptor,
    _spectral_peak_descriptors,
)

_UNCACHED_READ_AUDIO = read_audio
_READ_AUDIO_CACHE_LOCK = threading.Lock()
_READ_AUDIO_CACHE: dict[tuple[str, int, int], tuple[np.ndarray, int]] = {}
_READ_AUDIO_CACHE_ORDER: list[tuple[str, int, int]] = []
_READ_AUDIO_CACHE_MAX_ITEMS = 64


def read_audio(path: Path) -> Tuple[np.ndarray, int]:
    """Read audio with a tiny per-process cache for multi-view analysis.

    Sorting intentionally analyzes full, direct/body, wetness, and sometimes
    harmonic-core views of the same file.  Decoding the same WAV/AIFF/FLAC four
    times per sample is pure overhead, so cache only the decoded raw audio and
    hand callers a copy to preserve the existing mutation safety.
    """
    expanded = Path(path).expanduser()
    try:
        stat = expanded.stat()
        key = (str(expanded), int(stat.st_mtime_ns), int(stat.st_size))
    except Exception:
        key = (str(expanded), 0, 0)
    with _READ_AUDIO_CACHE_LOCK:
        cached = _READ_AUDIO_CACHE.get(key)
        if cached is not None:
            audio, sr = cached
            return np.array(audio, copy=True), int(sr)
    audio, sr = _UNCACHED_READ_AUDIO(expanded)
    audio = ensure_2d(np.asarray(audio, dtype=np.float32))
    with _READ_AUDIO_CACHE_LOCK:
        _READ_AUDIO_CACHE[key] = (np.array(audio, copy=True), int(sr))
        _READ_AUDIO_CACHE_ORDER.append(key)
        while len(_READ_AUDIO_CACHE_ORDER) > _READ_AUDIO_CACHE_MAX_ITEMS:
            old_key = _READ_AUDIO_CACHE_ORDER.pop(0)
            _READ_AUDIO_CACHE.pop(old_key, None)
    return np.array(audio, copy=True), int(sr)


def expanded_physics_descriptors(
    mono: np.ndarray,
    frames: np.ndarray,
    power: np.ndarray,
    freqs: np.ndarray,
    sr: int,
    centroid_frames: np.ndarray,
) -> Dict[str, float]:
    """Compute the appended v0.5 physics atlas descriptor block."""
    y = np.asarray(mono, dtype=np.float32).reshape(-1)
    n = int(y.size)
    f0_stats = _frame_level_f0_stats(y, sr)
    f0_hz = float(f0_stats.get("f0_median_hz", 0.0) or 0.0)
    harmonic = _harmonic_descriptors(power, freqs, f0_hz)
    peak_desc = _spectral_peak_descriptors(power, freqs, f0_hz)
    decay_low = _expanded_decay_and_low_source_descriptors(y, frames, power, freqs, sr, centroid_frames, f0_stats)
    a_end = max(1, int(n * 0.15))
    b_end = max(a_end + 1, int(n * 0.60))
    attack = _segment_descriptor(y[:a_end], sr)
    body = _segment_descriptor(y[a_end:b_end], sr)
    tail = _segment_descriptor(y[b_end:], sr)
    out: Dict[str, float] = {}
    out.update(f0_stats)
    out.update(harmonic)
    out.update(
        {
            "attack_pitch_confidence": attack["pitch"],
            "body_pitch_confidence": body["pitch"],
            "tail_pitch_confidence": tail["pitch"],
            "attack_flatness": attack["flatness"],
            "body_flatness": body["flatness"],
            "tail_flatness": tail["flatness"],
            "attack_entropy": attack["entropy"],
            "body_entropy": body["entropy"],
            "tail_entropy": tail["entropy"],
            "attack_zcr": attack["zcr"],
            "body_zcr": body["zcr"],
            "tail_zcr": tail["zcr"],
            "attack_high_ratio": attack["high_ratio"],
            "body_high_ratio": body["high_ratio"],
            "tail_high_ratio": tail["high_ratio"],
            "attack_low_ratio": attack["low_ratio"],
            "body_low_ratio": body["low_ratio"],
            "tail_low_ratio": tail["low_ratio"],
            "attack_noise_ratio": attack["noise_ratio"],
            "body_noise_ratio": body["noise_ratio"],
            "tail_noise_ratio": tail["noise_ratio"],
        }
    )
    out.update(decay_low)
    out.update(peak_desc)
    return {
        name: float(np.nan_to_num(out.get(name, 0.0), nan=0.0, posinf=0.0, neginf=0.0))
        for name in EXTRA_PHYSICS_FEATURE_NAMES
    }


def _safe_mean(values: list[float]) -> float:
    """Mean of finite values, or zero for empty lists."""
    finite = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else 0.0


def _ratio_true(values: list[bool]) -> float:
    """Ratio of true values, or zero for empty lists."""
    return float(sum(1 for value in values if value) / max(1, len(values))) if values else 0.0


def loop_long_segment_descriptors(mono: np.ndarray, frames: np.ndarray, sr: int) -> Dict[str, float]:
    """Measured segment-population features for loop and long material.

    Version: v20260512_LOOP_LONG_SEGMENT_FEATURES

    This does not choose a category.  It measures how much of the active audio
    behaves like pitched/tonal material, noisy/percussive events, or mixed
    content.  Training and sorting use the same descriptors, so the brain can
    learn the normal population shape of drum loops, instrument loops, Long FX,
    and mixed material without filename rules or after-the-fact penalties.
    """
    y = np.asarray(mono, dtype=np.float32).reshape(-1)
    empty = {name: 0.0 for name in LOOP_LONG_SEGMENT_FEATURE_NAMES}
    if y.size < max(128, int(0.05 * sr)) or float(np.max(np.abs(y))) <= 1e-9:
        return empty

    peaks = clean_onset_peaks(frames, sr)
    max_events = 32
    if len(peaks) > max_events:
        pick = np.linspace(0, len(peaks) - 1, max_events).round().astype(int)
        event_peaks = [peaks[int(i)] for i in pick]
    else:
        event_peaks = list(peaks)

    event_pitch: list[float] = []
    event_noise: list[float] = []
    event_low: list[float] = []
    event_high: list[float] = []
    event_categories: list[int] = []

    pre = int(0.030 * sr)
    post = int(0.220 * sr)
    for peak_frame in event_peaks:
        center = int(peak_frame * HOP)
        start = max(0, center - pre)
        end = min(y.size, center + post)
        if end - start < 96:
            continue
        desc = _segment_descriptor(y[start:end], sr)
        pitch = float(desc.get("pitch", 0.0))
        noise = float(desc.get("noise_ratio", 0.0))
        low = float(desc.get("low_ratio", 0.0))
        high = float(desc.get("high_ratio", 0.0))
        event_pitch.append(pitch)
        event_noise.append(noise)
        event_low.append(low)
        event_high.append(high)
        # Broad measured event bins used only to compute diversity.  This is not
        # a label decision: 0 tonal/pitched, 1 noisy/percussive, 2 low/body, 3 other.
        if pitch >= 0.30 and noise <= 0.70:
            event_categories.append(0)
        elif pitch < 0.25 and noise >= 0.55:
            event_categories.append(1)
        elif low >= 0.45 and pitch < 0.35:
            event_categories.append(2)
        else:
            event_categories.append(3)

    pitched_event_ratio = _ratio_true([v >= 0.30 for v in event_pitch])
    noisy_event_ratio = _ratio_true([v >= 0.55 for v in event_noise])
    # Percussive here means short event-like and aperiodic/noisy enough to be
    # drum/foley-like.  It remains just a feature, never a routing command.
    percussive_event_ratio = _ratio_true(
        [
            (p < 0.28 and n >= 0.48) or (p < 0.20 and (lo >= 0.42 or hi >= 0.42))
            for p, n, lo, hi in zip(event_pitch, event_noise, event_low, event_high)
        ]
    )

    diversity = 0.0
    if event_categories:
        counts = np.asarray([event_categories.count(i) for i in sorted(set(event_categories))], dtype=np.float32)
        probs = counts / max(1.0, float(np.sum(counts)))
        if probs.size > 1:
            diversity = float(-np.sum(probs * np.log2(probs + 1e-12)) / math.log2(max(2, probs.size)))

    # Frame population features over a bounded set of short windows.  This finds
    # sustained pitched content between transients, which global onset counts miss.
    frame_pitch: list[float] = []
    frame_noise: list[float] = []
    frame_low: list[float] = []
    frame_high: list[float] = []
    non_event_tonal_flags: list[bool] = []
    tonal_flags: list[bool] = []
    drumlike_flags: list[bool] = []
    win = max(512, int(0.185 * sr))
    hop = max(256, int(0.092 * sr))
    starts = list(range(0, max(1, y.size - win + 1), hop))
    max_windows = 48
    if len(starts) > max_windows:
        pick = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = [starts[int(i)] for i in pick]
    onset_frame_set = set(int(p) for p in peaks)
    for start in starts:
        seg = y[start : start + win]
        if seg.size < 96:
            continue
        desc = _segment_descriptor(seg, sr)
        pitch = float(desc.get("pitch", 0.0))
        noise = float(desc.get("noise_ratio", 0.0))
        low = float(desc.get("low_ratio", 0.0))
        high = float(desc.get("high_ratio", 0.0))
        frame_pitch.append(pitch)
        frame_noise.append(noise)
        frame_low.append(low)
        frame_high.append(high)
        tonal = pitch >= 0.28 and noise <= 0.72
        drumlike = pitch < 0.24 and noise >= 0.50
        tonal_flags.append(tonal)
        drumlike_flags.append(drumlike)
        frame_idx = int(start / max(1, HOP))
        near_onset = any(abs(frame_idx - p) <= 2 for p in onset_frame_set)
        if not near_onset:
            non_event_tonal_flags.append(tonal)

    sustained_tonal_frame_ratio = _ratio_true(tonal_flags)
    drumlike_frame_ratio = _ratio_true(drumlike_flags)
    non_event_tonal_ratio = _ratio_true(non_event_tonal_flags)
    tonal_to_percussive_balance = float(np.clip(sustained_tonal_frame_ratio - drumlike_frame_ratio, -1.0, 1.0))

    return {
        "loop_pitched_event_ratio": pitched_event_ratio,
        "loop_percussive_event_ratio": percussive_event_ratio,
        "loop_noisy_event_ratio": noisy_event_ratio,
        "loop_event_timbre_diversity": diversity,
        "loop_sustained_tonal_frame_ratio": sustained_tonal_frame_ratio,
        "loop_drumlike_frame_ratio": drumlike_frame_ratio,
        "loop_tonal_to_percussive_balance": tonal_to_percussive_balance,
        "loop_mean_event_pitch_confidence": _safe_mean(event_pitch),
        "loop_mean_event_noise_ratio": _safe_mean(event_noise),
        "loop_mean_event_low_ratio": _safe_mean(event_low),
        "loop_mean_event_high_ratio": _safe_mean(event_high),
        "loop_non_event_tonal_ratio": non_event_tonal_ratio,
    }


def make_fingerprint(path: Path) -> Tuple[np.ndarray, float, str]:
    try:
        y, sr0 = read_audio(path)
        y, sr = resample_linear(y, sr0, TARGET_SR)
        y, status = trim_and_normalize(y)
        duration = float(y.shape[0] / sr) if sr else 0.0
        if status != "ok":
            return np.zeros(FP_SIZE, dtype=np.float32), duration, status

        mono = np.mean(y, axis=1).astype(np.float32)
        if mono.size < N_FFT:
            mono = np.pad(mono, (0, N_FFT - mono.size), mode="constant")

        frames = frame_audio(mono)
        mag = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32) + 1e-12
        power = np.nan_to_num(mag * mag, nan=0.0, posinf=0.0, neginf=0.0)
        freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr).astype(np.float32)
        total = np.sum(power, axis=1) + 1e-12
        sub_bass_ratio = band_energy_ratio(power, freqs, 0.0, 150.0)
        bass_ratio = band_energy_ratio(power, freqs, 150.0, 500.0)
        mid_band_ratio = band_energy_ratio(power, freqs, 500.0, 2000.0)
        presence_ratio = band_energy_ratio(power, freqs, 2000.0, 8000.0)
        air_ratio = band_energy_ratio(power, freqs, 8000.0, None)

        fb = mel_filterbank(sr)
        mel = np.nan_to_num(np.dot(power, fb.T), nan=0.0, posinf=0.0, neginf=0.0)
        log_mel = np.nan_to_num(np.log(np.maximum(mel, 0.0) + 1e-10), nan=0.0, posinf=0.0, neginf=0.0)
        mfcc = dct2(log_mel, axis=1)[:, :N_MFCC]

        centroid_frames = np.sum(power * freqs[None, :], axis=1) / total
        cs = np.cumsum(power, axis=1)
        roll_idx = np.array(
            [min(np.searchsorted(cs[i], total[i] * 0.85), len(freqs) - 1) for i in range(power.shape[0])]
        )
        rolloff = float(np.mean(freqs[roll_idx]))
        flatness = float(np.mean(np.exp(np.mean(np.log(mag), axis=1)) / (np.mean(mag, axis=1) + 1e-12)))
        entropy = float(np.mean(spectral_entropy(power)))
        norm_spec = mag / np.maximum(np.sum(mag, axis=1, keepdims=True), 1e-12)
        flux_frames = np.sqrt(np.mean(np.diff(norm_spec, axis=0, prepend=norm_spec[:1]) ** 2, axis=1))
        flux = float(np.mean(flux_frames))
        flux_variance = float(np.std(flux_frames))
        crest = float(np.max(np.abs(mono)) / (np.sqrt(np.mean(mono**2)) + 1e-9))
        decay_start = mono[: max(64, min(mono.size, int(0.05 * sr)))]
        tail = mono[mono.size // 2 :] if mono.size else mono
        decay = (
            float(np.sqrt(np.mean(tail**2)) / (np.sqrt(np.mean(decay_start**2)) + 1e-9))
            if tail.size and decay_start.size
            else 0.0
        )
        width, ms_ratio = stereo_metrics(y)
        transients = onset_count(frames, sr)
        zcr_mean = zero_crossing_rate_mean(frames)
        temporal_centroid = temporal_centroid_ratio(mono)
        onset_regularity = onset_interval_regularity(frames, sr)
        attack_rise = attack_rise_time_norm(mono)
        onset_span = onset_span_ratio(frames, sr)
        event_rate = event_rate_hz(frames, sr)
        tail_ratio = tail_energy_ratio(mono)
        pitch_conf = pitch_confidence(mono, sr)

        if centroid_frames.size >= 5:
            third = max(1, centroid_frames.size // 5)
            slope = (float(np.mean(centroid_frames[-third:])) - float(np.mean(centroid_frames[:third]))) / (
                sr / 2.0 + 1e-9
            )
        else:
            slope = 0.0

        fp = np.zeros(FP_SIZE, dtype=np.float32)
        fp[0:13] = np.mean(mfcc, axis=0)
        fp[13:26] = np.std(mfcc, axis=0)
        fp[26] = math.log1p(max(0.0, rolloff))
        fp[27] = math.log1p(max(0.0, crest))
        fp[28] = flux
        fp[29] = flatness
        fp[30] = entropy
        fp[31] = math.log1p(max(0.0, decay))
        fp[32] = width
        fp[33] = ms_ratio
        fp[34] = slope
        fp[35] = math.log1p(max(0, transients))
        fp[36] = sub_bass_ratio
        fp[37] = bass_ratio
        fp[38] = mid_band_ratio
        fp[39] = presence_ratio
        fp[40] = air_ratio
        fp[41] = zcr_mean
        fp[42] = temporal_centroid
        fp[43] = onset_regularity
        fp[44] = attack_rise
        fp[45] = onset_span
        fp[46] = event_rate
        fp[47] = tail_ratio
        fp[48] = flux_variance
        fp[49] = pitch_conf

        # v0.5.5 committee physics agreement.  These descriptors are part of the
        # required v0.5 brain. Old smaller-feature brains are rejected.
        extra = expanded_physics_descriptors(mono, frames, power, freqs, sr, centroid_frames)
        for name in EXTRA_PHYSICS_FEATURE_NAMES:
            idx = FEATURE_NAMES.index(name)
            fp[idx] = float(extra.get(name, 0.0) or 0.0)

        # v20260512 loop/long segment population descriptors. These are computed
        # for every file, but become especially informative on loops and Long FX.
        segment_extra = loop_long_segment_descriptors(mono, frames, sr)
        for name in LOOP_LONG_SEGMENT_FEATURE_NAMES:
            idx = FEATURE_NAMES.index(name)
            fp[idx] = float(segment_extra.get(name, 0.0) or 0.0)

        fp = np.nan_to_num(fp, nan=0.0, posinf=0.0, neginf=0.0)
        return fp, duration, "ok"
    except Exception as exc:
        return np.zeros(FP_SIZE, dtype=np.float32), 0.0, "read_error:" + str(exc)[:120]


def _fingerprint_timeout_handler(signum, frame):
    raise FingerprintTimeout("fingerprint timed out")


def _signals_available_in_current_thread() -> bool:
    """Return True when SIGALRM can be installed safely."""
    return bool(hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread())


def make_fingerprint_safe(
    path: Path, timeout_sec: float = FINGERPRINT_TIMEOUT_SECONDS
) -> Tuple[np.ndarray, float, str]:
    """Bound fingerprint extraction so one bad decoder/file cannot hang a run."""
    # Timeout design note: decoder timeouts stay signal-based on the main thread.
    # Process-isolated worker decoding is intentionally deferred until the CLI
    # runner owns a full worker pool and can preserve deterministic manifests.
    if timeout_sec <= 0 or not _signals_available_in_current_thread():
        return make_fingerprint(path)
    old_handler = signal.signal(signal.SIGALRM, _fingerprint_timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
    try:
        return make_fingerprint(path)
    except FingerprintTimeout:
        return np.zeros(FP_SIZE, dtype=np.float32), 0.0, "fingerprint_timeout"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def third_party_feature_profile(path: Path) -> dict[str, object]:
    """Return automatic third-party measured features for one audio file.

    This is analysis evidence only.  Adapters may use installed third-party DSP
    libraries, but they cannot inspect source names or emit final folder labels.
    """
    try:
        y, sr0 = read_audio(path)
        y, sr = resample_linear(y, sr0, TARGET_SR)
        y, status = trim_and_normalize(y)
        if status != "ok" or y.size == 0:
            return {"status": status, "adapters": {}, "flat": {}}
        mono = np.mean(y, axis=1).astype(np.float32)
        return third_party_feature_profile_from_audio(mono, sr)
    except Exception as exc:
        return {"status": "third_party_feature_error:" + str(exc)[:120], "adapters": {}, "flat": {}}


# ---------------------------------------------------------------------------
# Wet/smeared-state and harmonic-core recall features
# ---------------------------------------------------------------------------


def _rolling_median_2d(arr: np.ndarray, axis: int, width: int) -> np.ndarray:
    """Small NumPy median filter used for HPSS-style analysis.

    This deliberately avoids scipy/librosa dependencies.  It is not a mastering
    dereverb tool.  It is only an analysis transform so wet/smeared musical
    samples can expose their harmonic body to the baby-brain recall lanes.
    """
    x = np.asarray(arr, dtype=np.float32)
    width = int(max(1, width))
    if width <= 1 or x.size == 0:
        return x.copy()
    if width % 2 == 0:
        width += 1
    pad = width // 2
    try:
        from numpy.lib.stride_tricks import sliding_window_view

        if axis == 0:
            padded = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
            windows = sliding_window_view(padded, window_shape=width, axis=0)
            return np.median(windows, axis=-1).astype(np.float32)
        padded = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
        windows = sliding_window_view(padded, window_shape=width, axis=1)
        return np.median(windows, axis=-1).astype(np.float32)
    except Exception:
        out = np.empty_like(x)
        if axis == 0:
            for i in range(x.shape[0]):
                lo = max(0, i - pad)
                hi = min(x.shape[0], i + pad + 1)
                out[i] = np.median(x[lo:hi], axis=0)
        else:
            for j in range(x.shape[1]):
                lo = max(0, j - pad)
                hi = min(x.shape[1], j + pad + 1)
                out[:, j] = np.median(x[:, lo:hi], axis=1)
        return out.astype(np.float32)


def _overlap_add(frames: np.ndarray, original_len: int) -> np.ndarray:
    """Reconstruct a mono signal from windowed frames."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[0] == 0:
        return np.zeros(max(1, int(original_len)), dtype=np.float32)
    n_frames, n_fft = frames.shape
    out_len = max(int(original_len), (n_frames - 1) * HOP + n_fft)
    out = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)
    win = np.hanning(n_fft).astype(np.float32)
    for i in range(n_frames):
        start = i * HOP
        end = start + n_fft
        out[start:end] += frames[i] * win
        norm[start:end] += win * win
    valid = norm > 1e-7
    out[valid] /= norm[valid]
    if original_len > 0:
        out = out[:original_len]
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1e-9:
        out = out / peak
    return np.nan_to_num(out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def harmonic_core_signal(mono: np.ndarray, *, harmonic_margin: float = 8.0) -> tuple[np.ndarray, dict[str, float]]:
    """Return a harmonic-core analysis signal using a small HPSS-style mask.

    This is not destructive dereverberation.  It is a recall view.  Raw audio
    remains the primary truth; the harmonic-core view only helps underrepresented
    instrument identities survive when room/tail/smear dominates raw features.
    """
    y = np.asarray(mono, dtype=np.float32).reshape(-1)
    original_len = int(y.size)
    if original_len < 128 or float(np.max(np.abs(y))) <= 1e-9:
        return np.zeros(max(1, original_len), dtype=np.float32), {
            "harmonic_energy_ratio": 0.0,
            "percussive_energy_ratio": 0.0,
        }
    y_work = np.pad(y, (0, N_FFT - y.size), mode="constant") if y.size < N_FFT else y
    frames = frame_audio(y_work)
    complex_spec = np.fft.rfft(frames, axis=1)
    mag = np.abs(complex_spec).astype(np.float32) + 1e-12
    # Time-median favors horizontal/harmonic continuity; frequency-median favors
    # vertical/percussive bursts.  This mirrors the HPSS idea without requiring librosa.
    harmonic_ref = _rolling_median_2d(mag, axis=0, width=17)
    percussive_ref = _rolling_median_2d(mag, axis=1, width=31)
    margin = float(max(1.0, harmonic_margin))
    h2 = harmonic_ref * harmonic_ref
    p2 = (percussive_ref * margin) * (percussive_ref * margin)
    mask = h2 / np.maximum(h2 + p2, 1e-12)
    mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    harmonic_frames = np.fft.irfft(complex_spec * mask, n=N_FFT, axis=1).astype(np.float32)
    harmonic = _overlap_add(harmonic_frames, original_len)
    total_energy = float(np.sum(mag * mag)) + 1e-12
    h_energy = float(np.sum((mag * mask) ** 2))
    p_energy = float(np.sum((mag * (1.0 - mask)) ** 2))
    return harmonic, {
        "harmonic_energy_ratio": float(h_energy / total_energy),
        "percussive_energy_ratio": float(p_energy / total_energy),
        "harmonic_mask_mean": float(np.mean(mask)),
    }


def wetness_profile(path: Path) -> dict[str, float | str]:
    """Measure wet/smeared-state evidence for diagnostics and scoring context.

    This deliberately says wet/smeared, not just reverb.  Pads, ensembles,
    crowds, long tails, and actual reverb can all smear identity features.
    """
    try:
        y, sr0 = read_audio(path)
        y, sr = resample_linear(y, sr0, TARGET_SR)
        y, status = trim_and_normalize(y)
        if status != "ok" or y.size == 0:
            return {"status": status, "wetness_score": 0.0}
        mono = np.mean(y, axis=1).astype(np.float32)
        n = int(mono.size)
        duration = float(n / max(1, sr))
        peak_i = int(np.argmax(np.abs(mono))) if n else 0
        early_end = min(n, max(1, peak_i + int(0.35 * sr)))
        mid_start = min(n, max(0, peak_i + int(0.35 * sr)))
        mid_end = min(n, max(mid_start + 1, peak_i + int(1.25 * sr)))
        tail_start = min(n, max(0, peak_i + int(1.25 * sr)))
        early = mono[max(0, peak_i) : early_end]
        mid = mono[mid_start:mid_end]
        tail = mono[tail_start:]

        def rms(a: np.ndarray) -> float:
            return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float32) ** 2))) if a.size else 0.0

        early_r = rms(early)
        mid_r = rms(mid)
        tail_r = rms(tail)
        tail_to_early = float(tail_r / (early_r + 1e-9))
        tail_to_mid = float(tail_r / (mid_r + 1e-9))
        abs_y = np.abs(mono)
        peak = float(np.max(abs_y)) + 1e-12
        above = np.where(abs_y[peak_i:] >= peak * 0.01)[0]  # about -40 dB
        post_peak_above_40 = float((int(above[-1]) / sr) if above.size else 0.0)
        body_start = min(n, max(0, peak_i + int(0.15 * sr)))
        body_end = min(n, max(body_start + 1, peak_i + int(1.25 * sr)))

        def stereo_width_segment(seg: np.ndarray) -> float:
            if seg.ndim != 2 or seg.shape[1] < 2 or seg.shape[0] < 32:
                return 0.0
            left = seg[:, 0].astype(np.float32)
            right = seg[:, 1].astype(np.float32)
            mid_sig = 0.5 * (left + right)
            side_sig = 0.5 * (left - right)
            mid_rms = rms(mid_sig)
            side_rms = rms(side_sig)
            return float(side_rms / (mid_rms + side_rms + 1e-9))

        body_width = stereo_width_segment(y[body_start:body_end])
        tail_width = stereo_width_segment(y[tail_start:])
        harmonic, h_metrics = harmonic_core_signal(mono)
        # Conservative scalar: long post-peak energy and tail/early ratio dominate;
        # width and harmonic smear add context but cannot alone call something wet.
        long_tail = min(1.0, post_peak_above_40 / 3.0)
        tail_ratio_score = min(1.0, max(tail_to_early, tail_to_mid) / 0.75)
        width_score = min(1.0, max(body_width, tail_width) / 0.55)
        harmonic_sustain = min(1.0, float(h_metrics.get("harmonic_energy_ratio", 0.0)) / 0.65)
        wetness = float(
            np.clip(0.45 * long_tail + 0.35 * tail_ratio_score + 0.10 * width_score + 0.10 * harmonic_sustain, 0.0, 1.0)
        )
        return {
            "status": "ok",
            "duration_sec": duration,
            "peak_time_sec": float(peak_i / max(1, sr)),
            "tail_to_early_rms": tail_to_early,
            "tail_to_mid_rms": tail_to_mid,
            "post_peak_above_minus40db_sec": post_peak_above_40,
            "stereo_width_body": body_width,
            "stereo_width_tail": tail_width,
            "harmonic_core_energy_ratio": float(h_metrics.get("harmonic_energy_ratio", 0.0)),
            "percussive_residual_energy_ratio": float(h_metrics.get("percussive_energy_ratio", 0.0)),
            "wetness_score": wetness,
        }
    except Exception as exc:
        return {"status": "wetness_error:" + str(exc)[:120], "wetness_score": 0.0}


def make_fingerprint_from_preprocessed_audio(
    y: np.ndarray, sr: int, status: str = "ok"
) -> tuple[np.ndarray, float, str]:
    """Create the normal fingerprint from already loaded/preprocessed audio."""
    try:
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.size == 0:
            return np.zeros(FP_SIZE, dtype=np.float32), 0.0, "empty"
        duration = float(y.shape[0] / max(1, sr))
        if status != "ok":
            return np.zeros(FP_SIZE, dtype=np.float32), duration, status
        mono = np.mean(y, axis=1).astype(np.float32)
        if mono.size < N_FFT:
            mono = np.pad(mono, (0, N_FFT - mono.size), mode="constant")
        frames = frame_audio(mono)
        mag = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32) + 1e-12
        power = np.nan_to_num(mag * mag, nan=0.0, posinf=0.0, neginf=0.0)
        freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr).astype(np.float32)
        total = np.sum(power, axis=1) + 1e-12
        sub_bass_ratio = band_energy_ratio(power, freqs, 0.0, 150.0)
        bass_ratio = band_energy_ratio(power, freqs, 150.0, 500.0)
        mid_band_ratio = band_energy_ratio(power, freqs, 500.0, 2000.0)
        presence_ratio = band_energy_ratio(power, freqs, 2000.0, 8000.0)
        air_ratio = band_energy_ratio(power, freqs, 8000.0, None)
        fb = mel_filterbank(sr)
        mel = np.nan_to_num(np.dot(power, fb.T), nan=0.0, posinf=0.0, neginf=0.0)
        log_mel = np.nan_to_num(np.log(np.maximum(mel, 0.0) + 1e-10), nan=0.0, posinf=0.0, neginf=0.0)
        mfcc = dct2(log_mel, axis=1)[:, :N_MFCC]
        centroid_frames = np.sum(power * freqs[None, :], axis=1) / total
        cs = np.cumsum(power, axis=1)
        roll_idx = np.array(
            [min(np.searchsorted(cs[i], total[i] * 0.85), len(freqs) - 1) for i in range(power.shape[0])]
        )
        rolloff = float(np.mean(freqs[roll_idx]))
        flatness = float(np.mean(np.exp(np.mean(np.log(mag), axis=1)) / (np.mean(mag, axis=1) + 1e-12)))
        entropy = float(np.mean(spectral_entropy(power)))
        norm_spec = mag / np.maximum(np.sum(mag, axis=1, keepdims=True), 1e-12)
        flux_frames = np.sqrt(np.mean(np.diff(norm_spec, axis=0, prepend=norm_spec[:1]) ** 2, axis=1))
        flux = float(np.mean(flux_frames))
        flux_variance = float(np.std(flux_frames))
        crest = float(np.max(np.abs(mono)) / (np.sqrt(np.mean(mono**2)) + 1e-9))
        decay_start = mono[: max(64, min(mono.size, int(0.05 * sr)))]
        tail = mono[mono.size // 2 :] if mono.size else mono
        decay = (
            float(np.sqrt(np.mean(tail**2)) / (np.sqrt(np.mean(decay_start**2)) + 1e-9))
            if tail.size and decay_start.size
            else 0.0
        )
        width, ms_ratio = stereo_metrics(y)
        transients = onset_count(frames, sr)
        zcr_mean = zero_crossing_rate_mean(frames)
        temporal_centroid = temporal_centroid_ratio(mono)
        onset_regularity = onset_interval_regularity(frames, sr)
        attack_rise = attack_rise_time_norm(mono)
        onset_span = onset_span_ratio(frames, sr)
        event_rate = event_rate_hz(frames, sr)
        tail_ratio = tail_energy_ratio(mono)
        pitch_conf = pitch_confidence(mono, sr)
        if centroid_frames.size >= 5:
            third = max(1, centroid_frames.size // 5)
            slope = (float(np.mean(centroid_frames[-third:])) - float(np.mean(centroid_frames[:third]))) / (
                sr / 2.0 + 1e-9
            )
        else:
            slope = 0.0
        fp = np.zeros(FP_SIZE, dtype=np.float32)
        fp[0:13] = np.mean(mfcc, axis=0)
        fp[13:26] = np.std(mfcc, axis=0)
        fp[26] = math.log1p(max(0.0, rolloff))
        fp[27] = math.log1p(max(0.0, crest))
        fp[28] = flux
        fp[29] = flatness
        fp[30] = entropy
        fp[31] = math.log1p(max(0.0, decay))
        fp[32] = width
        fp[33] = ms_ratio
        fp[34] = slope
        fp[35] = math.log1p(max(0, transients))
        fp[36] = sub_bass_ratio
        fp[37] = bass_ratio
        fp[38] = mid_band_ratio
        fp[39] = presence_ratio
        fp[40] = air_ratio
        fp[41] = zcr_mean
        fp[42] = temporal_centroid
        fp[43] = onset_regularity
        fp[44] = attack_rise
        fp[45] = onset_span
        fp[46] = event_rate
        fp[47] = tail_ratio
        fp[48] = flux_variance
        fp[49] = pitch_conf
        extra = expanded_physics_descriptors(mono, frames, power, freqs, sr, centroid_frames)
        for name in EXTRA_PHYSICS_FEATURE_NAMES:
            idx = FEATURE_NAMES.index(name)
            fp[idx] = float(extra.get(name, 0.0) or 0.0)
        segment_extra = loop_long_segment_descriptors(mono, frames, sr)
        for name in LOOP_LONG_SEGMENT_FEATURE_NAMES:
            idx = FEATURE_NAMES.index(name)
            fp[idx] = float(segment_extra.get(name, 0.0) or 0.0)
        return np.nan_to_num(fp, nan=0.0, posinf=0.0, neginf=0.0), duration, "ok"
    except Exception as exc:
        return np.zeros(FP_SIZE, dtype=np.float32), 0.0, "harmonic_core_error:" + str(exc)[:120]


def direct_body_audio_view(
    y: np.ndarray,
    sr: int,
    *,
    max_events: int = 32,
    pre_ms: float = 25.0,
    post_ms: float = 340.0,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Return a tail-reduced direct/body analysis view for any audio file.

    The full file remains the main fingerprint. This auxiliary view extracts
    short onset-centered windows, or a peak-centered body window when no onset
    list is available. It lets the voters inspect the direct source body before
    reverb, delay, or long decay dominates physics such as pitch, bass energy,
    and stereo width. It is intentionally deterministic and source-name blind.
    """
    samples = np.asarray(y, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    if samples.size == 0 or samples.shape[0] < 8 or sr <= 0:
        return np.zeros((1, 1), dtype=np.float32), {"status": "empty", "selected_event_count": 0}

    mono = np.mean(samples, axis=1).astype(np.float32)
    if float(np.max(np.abs(mono))) <= 1e-9:
        return np.zeros((1, samples.shape[1]), dtype=np.float32), {"status": "silent", "selected_event_count": 0}

    frames = frame_audio(mono)
    peaks = clean_onset_peaks(frames, sr)
    if not peaks:
        peak_sample = int(np.argmax(np.abs(mono)))
        peaks = [max(0, int(round(peak_sample / max(1, HOP))))]

    if len(peaks) > max_events:
        chosen = np.linspace(0, len(peaks) - 1, max_events).round().astype(int)
        peaks = [peaks[int(index)] for index in chosen]

    pre = max(0, int((pre_ms / 1000.0) * sr))
    post = max(32, int((post_ms / 1000.0) * sr))
    gap = np.zeros((max(1, int(0.025 * sr)), samples.shape[1]), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for peak_frame in peaks:
        center = int(peak_frame * HOP)
        start = max(0, center - pre)
        stop = min(samples.shape[0], center + post)
        if stop - start < 32:
            continue
        pieces.append(samples[start:stop, :].astype(np.float32, copy=False))
        pieces.append(gap)

    if not pieces:
        peak_sample = int(np.argmax(np.abs(mono)))
        start = max(0, peak_sample - pre)
        stop = min(samples.shape[0], peak_sample + post)
        pieces = [samples[start:stop, :].astype(np.float32, copy=False)]

    direct = np.vstack(pieces).astype(np.float32, copy=False)
    max_len = max(1, int(8.0 * sr))
    if direct.shape[0] > max_len:
        direct = direct[:max_len, :]
    peak = float(np.max(np.abs(direct))) if direct.size else 0.0
    if peak > 1e-9:
        direct = direct / peak
    duration = float(direct.shape[0] / max(1, sr))
    return np.nan_to_num(direct, nan=0.0, posinf=0.0, neginf=0.0), {
        "status": "ok",
        "selected_event_count": int(max(0, len(peaks))),
        "duration_sec": duration,
        "pre_ms": float(pre_ms),
        "post_ms": float(post_ms),
    }


def make_direct_body_fingerprint(path: Path) -> tuple[np.ndarray, float, str, dict[str, float | int | str]]:
    """Create a tail-reduced direct/body fingerprint for one file.

    This view is computed for every sorted file and carried as auxiliary evidence.
    It never overwrites the full-file fingerprint; voters can compare both views
    and prefer a label only when the direct/body evidence is compatible.
    """
    try:
        y, sr0 = read_audio(path)
        y, sr = resample_linear(y, sr0, TARGET_SR)
        y, status = trim_and_normalize(y)
        if status != "ok" or y.size == 0:
            return np.zeros(FP_SIZE, dtype=np.float32), 0.0, status, {"status": status, "selected_event_count": 0}
        direct, meta = direct_body_audio_view(y, sr)
        fp, duration, fp_status = make_fingerprint_from_preprocessed_audio(direct, sr, "ok")
        meta = dict(meta)
        meta["full_duration_sec"] = float(y.shape[0] / max(1, sr))
        return fp, duration, fp_status, meta
    except Exception as exc:
        return (
            np.zeros(FP_SIZE, dtype=np.float32),
            0.0,
            "direct_body_error:" + str(exc)[:120],
            {"status": "error", "selected_event_count": 0},
        )


def make_direct_body_fingerprint_safe(
    path: Path,
    timeout_sec: float = FINGERPRINT_TIMEOUT_SECONDS,
) -> tuple[np.ndarray, float, str, dict[str, float | int | str]]:
    """Bounded direct/body fingerprint extraction."""
    if timeout_sec <= 0 or not _signals_available_in_current_thread():
        return make_direct_body_fingerprint(path)
    old_handler = signal.signal(signal.SIGALRM, _fingerprint_timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
    try:
        return make_direct_body_fingerprint(path)
    except FingerprintTimeout:
        return (
            np.zeros(FP_SIZE, dtype=np.float32),
            0.0,
            "direct_body_timeout",
            {"status": "timeout", "selected_event_count": 0},
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def make_harmonic_core_fingerprint(
    path: Path,
    wet_profile: dict[str, float | str] | None = None,
) -> tuple[np.ndarray, float, str, dict[str, float | str]]:
    """Return a harmonic-core recall fingerprint plus wetness metrics.

    Args:
        path: Audio file to analyze.
        wet_profile: Optional wetness profile already measured for the same file
            in the current run. When omitted, this function measures wetness.

    Returns:
        Tuple of fingerprint, duration seconds, status, and wetness metrics.

    Side Effects:
        Reads audio from disk. Does not write files.

    Raises:
        No intentional exceptions; failures are converted into a status string.

    Important Constraints:
        The optional profile is measured audio evidence only. It must not contain
        routing decisions, labels, or user corrections.
    """
    wet = dict(wet_profile) if isinstance(wet_profile, dict) else wetness_profile(path)
    try:
        y, sr0 = read_audio(path)
        y, sr = resample_linear(y, sr0, TARGET_SR)
        y, status = trim_and_normalize(y)
        if status != "ok":
            return np.zeros(FP_SIZE, dtype=np.float32), 0.0, status, wet
        mono = np.mean(y, axis=1).astype(np.float32)
        harmonic, h_metrics = harmonic_core_signal(mono, harmonic_margin=8.0)
        # Keep a mono harmonic-core view.  This is intentional: the recall lane
        # should discount room width/tail dominance, not classify by it.
        fp, duration, fp_status = make_fingerprint_from_preprocessed_audio(harmonic.reshape(-1, 1), sr, "ok")
        wet = dict(wet)
        wet.update({f"harmonic_core_{k}": v for k, v in h_metrics.items()})
        return fp, duration, fp_status, wet
    except Exception as exc:
        return np.zeros(FP_SIZE, dtype=np.float32), 0.0, "harmonic_core_error:" + str(exc)[:120], wet


def make_harmonic_core_fingerprint_safe(
    path: Path,
    timeout_sec: float = FINGERPRINT_TIMEOUT_SECONDS,
    wet_profile: dict[str, float | str] | None = None,
) -> tuple[np.ndarray, float, str, dict[str, float | str]]:
    """Bounded harmonic-core fingerprint extraction.

    Args:
        path: Audio file to analyze.
        timeout_sec: Main-thread SIGALRM timeout in seconds.
        wet_profile: Optional wetness metrics already measured for this file.

    Returns:
        Tuple of fingerprint, duration seconds, status, and wetness metrics.

    Side Effects:
        Reads audio from disk and may install a temporary SIGALRM handler on the
        main thread.

    Raises:
        No intentional exceptions; timeout returns a zero fingerprint.

    Important Constraints:
        Reusing ``wet_profile`` only removes duplicate wetness measurement. It
        must not alter harmonic-core audio processing.
    """
    if timeout_sec <= 0 or not _signals_available_in_current_thread():
        return make_harmonic_core_fingerprint(path, wet_profile=wet_profile)
    old_handler = signal.signal(signal.SIGALRM, _fingerprint_timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
    try:
        return make_harmonic_core_fingerprint(path, wet_profile=wet_profile)
    except FingerprintTimeout:
        return (
            np.zeros(FP_SIZE, dtype=np.float32),
            0.0,
            "harmonic_core_timeout",
            {"status": "harmonic_core_timeout", "wetness_score": 0.0},
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def weighted_label_spread(vectors: np.ndarray) -> Dict[str, float]:
    """Measure one folder's physical spread without using names.

    Vectors are already scaled for their structure lane.  Feature weights make
    punch, decay, noise, motion, and other physical dimensions matter more than
    raw MFCC similarity alone.
    """
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] <= 1:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    weights = FEATURE_WEIGHTS.astype(np.float32)
    center = np.mean(arr, axis=0, keepdims=True)
    dists = np.linalg.norm((arr - center) * weights[None, :], axis=1)
    return {
        "mean": float(np.mean(dists)),
        "median": float(np.median(dists)),
        "p90": float(np.percentile(dists, 90)),
        "max": float(np.max(dists)),
    }


def choose_adaptive_label_model(label: str, vectors: np.ndarray, max_centroids: int) -> Dict[str, object]:
    """Choose the internal model for a trusted training folder.

    The output label remains the folder path.  Physics only decides whether that
    folder is best represented by one centroid, multiple centroids, or real
    exemplar anchors.
    """
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    count = int(arr.shape[0])
    cap = max(1, int(max_centroids or 1))
    spread = weighted_label_spread(arr)
    mean_spread = float(spread["mean"])
    p90_spread = float(spread["p90"])

    if count <= 3:
        mode = "exemplar_only"
        centroid_count = min(count, cap)
        exemplar_count = count
    elif mean_spread < 0.50 and p90_spread < 0.90:
        mode = "single_centroid"
        centroid_count = 1
        exemplar_count = min(3, count)
    elif mean_spread < 1.15 and p90_spread < 1.75:
        mode = "multi_centroid"
        centroid_count = min(3, cap, count)
        exemplar_count = min(6, count)
    else:
        mode = "multi_centroid_exemplar"
        centroid_count = min(5, cap, count)
        exemplar_count = min(10, count)

    return {
        "model_mode": mode,
        "training_count": count,
        "centroid_count": int(max(1, centroid_count)),
        "exemplar_count": int(max(1, exemplar_count)),
        "spread_mean": mean_spread,
        "spread_median": float(spread["median"]),
        "spread_p90": p90_spread,
        "spread_max": float(spread["max"]),
        "model_reason": f"count={count}; spread_mean={mean_spread:.3f}; spread_p90={p90_spread:.3f}; mode={mode}; folder_truth_preserved",
    }


def select_deterministic_exemplars(vectors: np.ndarray, max_exemplars: int) -> np.ndarray:
    """Pick real teacher anchors deterministically.

    This keeps tiny and messy folders from becoming ghost averages.  It does not
    create public subcategories and does not inspect filenames.
    """
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 0:
        return np.zeros((0, FP_SIZE), dtype=np.float32)
    count = int(arr.shape[0])
    k = min(max(1, int(max_exemplars or 1)), count)
    weights = FEATURE_WEIGHTS.astype(np.float32)
    weighted = arr * weights[None, :]
    mean = np.mean(weighted, axis=0, keepdims=True)
    first = int(np.argmin(np.linalg.norm(weighted - mean, axis=1)))
    chosen = [first]
    while len(chosen) < k:
        chosen_vecs = weighted[chosen]
        d = np.linalg.norm(weighted[:, None, :] - chosen_vecs[None, :, :], axis=2)
        min_d = np.min(d, axis=1)
        for idx in chosen:
            min_d[idx] = -1.0
        chosen.append(int(np.argmax(min_d)))
    return arr[chosen].astype(np.float32)


def adaptive_centroid_count(label: str, n: int, max_centroids: int, top: str = "") -> int:
    """Compatibility wrapper. New code uses choose_adaptive_label_model()."""
    n = int(n)
    if n <= 0:
        return 1
    return min(max(1, int(max_centroids or 1)), n)


def deterministic_centroids(vectors: np.ndarray, max_k: int = 3) -> np.ndarray:
    """Deterministic K-means in already scaled feature space.

    v0.4.5 uses spread/farthest-point initialization instead of low/mid/high L2
    norm seeding.  This is still deterministic, but it is less likely to waste a
    centroid on the same dense neighborhood when a category has real variety.
    """
    n = int(vectors.shape[0])
    if n == 0:
        return np.zeros((0, FP_SIZE), dtype=np.float32)
    if n == 1:
        return vectors[:1].copy()
    k = min(max(1, int(max_k)), n)
    if k == 1:
        return np.mean(vectors, axis=0, keepdims=True).astype(np.float32)

    center_indices: List[int] = []
    mean = np.mean(vectors, axis=0, keepdims=True)
    first = int(np.argmin(np.linalg.norm(vectors - mean, axis=1)))
    center_indices.append(first)
    while len(center_indices) < k:
        chosen = vectors[center_indices]
        d = np.linalg.norm(vectors[:, None, :] - chosen[None, :, :], axis=2)
        min_d = np.min(d, axis=1)
        for idx in center_indices:
            min_d[idx] = -1.0
        center_indices.append(int(np.argmax(min_d)))
    centers = vectors[center_indices].astype(np.float32)

    for _ in range(40):
        d = np.linalg.norm(vectors[:, None, :] - centers[None, :, :], axis=2)
        assign = np.argmin(d, axis=1)
        new_centers = []
        for j in range(k):
            members = vectors[assign == j]
            if members.size:
                new_centers.append(np.mean(members, axis=0))
            else:
                new_centers.append(centers[j])
        new_centers = np.vstack(new_centers).astype(np.float32)
        if np.max(np.abs(new_centers - centers)) < 1e-5:
            centers = new_centers
            break
        centers = new_centers
    return centers


def balanced_cap_eval_rows(
    eval_candidates: List[Dict[str, str]], max_eval_total: int, random_seed: int = 20260503
) -> List[Dict[str, str]]:
    """Cap eval rows without accidentally testing only early sorted labels.

    Tiny Phase 3 runs should still build a brain from every selected label.  Only
    the held-out eval rows should be capped.  The cap is balanced by learned top
    family and label so small tests include Drums, Instruments, FX, and Textures
    when those families are present in the source data.
    """
    if max_eval_total <= 0 or len(eval_candidates) <= max_eval_total:
        return list(eval_candidates)

    queues: Dict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in eval_candidates:
        top = str(row.get("_expected_top", "") or top_for_public_label(str(row.get("_expected_label", ""))))
        label = str(row.get("_expected_label", ""))
        queues[top][label].append(row)

    for top, by_label in queues.items():
        for label, rows in by_label.items():
            rows.sort(
                key=lambda r: (
                    int(str(r.get("_eval_rank", "0") or "0")),
                    stable_random_key(top, label, str(r.get("source_path", "")), seed=random_seed),
                )
            )

    selected: List[Dict[str, str]] = []
    top_names = sorted(queues)
    label_names_by_top = {top: sorted(by_label) for top, by_label in queues.items()}
    label_cursor_by_top = {top: 0 for top in top_names}

    def pop_next_for_top(top: str) -> Optional[Dict[str, str]]:
        labels = label_names_by_top.get(top, [])
        if not labels:
            return None
        start = label_cursor_by_top.get(top, 0) % len(labels)
        for offset in range(len(labels)):
            idx = (start + offset) % len(labels)
            label = labels[idx]
            queue = queues[top].get(label, [])
            if queue:
                label_cursor_by_top[top] = (idx + 1) % len(labels)
                return queue.pop(0)
        return None

    while len(selected) < max_eval_total:
        progressed = False
        for top in top_names:
            if len(selected) >= max_eval_total:
                break
            row = pop_next_for_top(top)
            if row is None:
                continue
            rr = dict(row)
            rr["_eval_cap_method"] = f"balanced_round_robin_top_then_label_limit_{max_eval_total}"
            rr["_eval_global_rank"] = str(len(selected) + 1)
            selected.append(rr)
            progressed = True
        if not progressed:
            break
    return selected


def select_rows(
    clean_rows: List[Dict[str, str]],
    allowed_top: set[str],
    max_train_per_group: int,
    max_eval_per_label: int,
    max_eval_total: int,
    random_seed: int = 20260503,
    min_train_after_holdout: int = MIN_ACTIVE_TRAIN_PER_LABEL,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Select randomized training and eval rows by real hierarchy label.

    v0.4.5 trains producer paths with One Shots / Loops at the leaf.

    It does not create _ONE_SHOTS:: or _LOOPS:: labels.  It also does not
    pass the expected structure into prediction during eval.  Folder hierarchy
    supplies training labels; audio structure evidence is logged as warning only.
    """
    train: List[Dict[str, str]] = []
    eval_candidates: List[Dict[str, str]] = []
    skipped: List[Dict[str, str]] = []

    def prepare_row(row: Dict[str, str]) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, str]]]:
        group_key = normalize_locked_training_group_key(str(row.get("group_key", "")).replace("\\", "/").strip())
        source = str(row.get("source_path", "")).strip()
        status = row.get("outlier_status", "CLEAN_CANDIDATE") or "CLEAN_CANDIDATE"
        top = group_key.split("/", 1)[0] if group_key else ""
        dur = row_duration_sec(row)
        group_low = group_key.lower()
        raw_structure_from_folder = ""
        if "_long_fx" in group_low or "long fx" in group_low or "long_fx" in group_low:
            raw_structure_from_folder = "long_fx"
        elif "_long_running" in group_low or "long running" in group_low or "long_running" in group_low:
            raw_structure_from_folder = "long_running"
        base_structure = structure_lane_for_row(row, group_key)
        structure = repair_structure_lane_for_row(row, group_key, base_structure)

        if raw_structure_from_folder and structure != raw_structure_from_folder:
            remap_reason = (
                f"STRUCTURE_MARKER_CHANGED_UNEXPECTEDLY_{raw_structure_from_folder}_TO_{structure}_REPORT_ONLY"
            )
        else:
            remap_reason = ""
        conflict_warning = structure_conflict_warning(row, group_key, structure)

        if status != "CLEAN_CANDIDATE":
            return None, {"source_path": source, "group_key": group_key, "reason": f"not clean: {status}"}
        if top not in allowed_top:
            return None, {"source_path": source, "group_key": group_key, "reason": f"top not allowed: {top}"}
        if not group_key:
            return None, {"source_path": source, "group_key": group_key, "reason": "missing group_key"}
        if is_near_silent_or_too_tiny_training_row(row):
            return None, {"source_path": source, "group_key": group_key, "reason": "near_silent_or_too_tiny_removed"}
        if is_dirty_non_fx_training_row(row):
            return None, {"source_path": source, "group_key": group_key, "reason": "dirty_or_fx_like_non_fx_removed"}
        if is_reviewed_bad_training_row(row):
            return None, {"source_path": source, "group_key": group_key, "reason": "reviewed_bad_removed"}

        public = training_public_label_from_group(group_key, row)
        internal = scoped_label(structure, public)
        simplified = simplified_hint_for_group(group_key, dur)

        r = dict(row)
        r["_public_label"] = public
        r["_expected_label"] = internal
        r["_expected_top"] = top_for_public_label(public)
        r["_structure_before_remap"] = raw_structure_from_folder or base_structure
        r["_structure"] = structure
        r["_structure_remap_reason"] = remap_reason
        r["_structure_conflict_warning"] = conflict_warning
        r["_simplified_hint_label"] = simplified
        return r, None

    prepared_by_label: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for row in clean_rows:
        prepared, skip = prepare_row(row)
        if skip:
            skipped.append(skip)
            continue
        assert prepared is not None
        label = str(prepared.get("_expected_label", ""))
        prepared_by_label[label].append(prepared)

    for label, rows in sorted(prepared_by_label.items()):
        force_train = [r for r in rows if str(r.get("source", "")).strip() == "supplemental_training"]
        normal_rows = [r for r in rows if str(r.get("source", "")).strip() != "supplemental_training"]
        shuffled = sorted(
            normal_rows, key=lambda r: stable_random_key(label, str(r.get("source_path", "")), seed=random_seed)
        )
        len(shuffled)
        if max_train_per_group <= 0:
            # v0.4.57: Aaron's locked folder tree is ground truth.
            # Train-all means every readable row becomes a teacher.  Evaluation
            # rows in this mode are diagnostic training-recall checks, not
            # independent holdout proof.
            holdout = []
            train_selected = list(shuffled)
        else:
            train_selected = shuffled[:max_train_per_group]
            train_sources_for_limited = {str(r.get("source_path", "")).strip() for r in train_selected}
            holdout = [r for r in shuffled if str(r.get("source_path", "")).strip() not in train_sources_for_limited]
        list(train_selected)
        if force_train:
            forced = sorted(
                force_train, key=lambda r: stable_random_key(label, str(r.get("source_path", "")), seed=random_seed)
            )
            forced_sources = {str(r.get("source_path", "")).strip() for r in forced}
            train_selected = forced + [
                r for r in train_selected if str(r.get("source_path", "")).strip() not in forced_sources
            ]
        {str(r.get("source_path", "")).strip() for r in train_selected}

        for rank, r in enumerate(train_selected, 1):
            rr = dict(r)
            rr["_sampling_rank"] = str(rank)
            rr["_sampling_pool_size"] = str(len(rows))
            rr["_sampling_method"] = (
                f"supplemental_force_train_{len(force_train)}_stable_random_seed_{random_seed}_train_all_minus_holdout_{len(holdout)}"
                if max_train_per_group <= 0
                else f"supplemental_force_train_{len(force_train)}_stable_random_seed_{random_seed}_train_up_to_{max_train_per_group}"
            )
            train.append(rr)

        if max_train_per_group <= 0:
            # Diagnostic recall only: these rows are also teachers.  This catches
            # catastrophic prototype/anchor failures without pretending it is
            # holdout accuracy.
            eval_pool = list(train_selected)
            eval_reused = "1"
            eval_method = "diagnostic_training_recall_reused_source_train_all_no_holdout"
        else:
            # Limited-training mode keeps independent holdout rows when available.
            eval_pool = holdout
            eval_reused = "0"
            eval_method = "strict_holdout_only_no_training_reuse"
        for rank, r in enumerate(eval_pool[:max_eval_per_label], 1):
            rr = dict(r)
            rr["_eval_rank"] = str(rank)
            rr["_eval_pool_size"] = str(len(eval_pool))
            rr["_eval_reused_training_source"] = eval_reused
            rr["_eval_selection_method"] = eval_method
            eval_candidates.append(rr)

    eval_rows = balanced_cap_eval_rows(eval_candidates, max_eval_total, random_seed=random_seed)
    return train, eval_rows, skipped


def scan_training_tree_rows(root: Path, source_name: str = "training_tree_scan") -> List[Dict[str, str]]:
    """Read the trusted Sorted samples tree directly instead of a stale manifest.

    The manifest can lag behind folders Aaron added or corrected by hand.  Phase 3
    should train from the actual trusted folder tree, preserving labels such as
    Sub Kick and Short Kick instead of only whatever the old manifest knew.
    """
    rows: List[Dict[str, str]] = []
    root = root.expanduser().resolve()
    if not root.exists():
        return rows
    skip_dir_names = {
        "_manifests",
        "_training_data",
        "_locked_audio_sources",
        "training data",
        "training_preview",
        "training_curated_keep_preview",
        "training_curated_reject_preview",
        "sorted_preview",
        "real_sorted_preview",
        "listen_review_pack",
        "__macosx",
        "_unsure_structure",
        "_to_review",
        "to_review",
        "review",
        "_quarantined_by_physics_review",
    }
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        kept_dirnames = []
        for name in dirnames:
            low = name.lower().replace("-", "_").replace(" ", "_")
            if low in skip_dir_names or name.startswith("."):
                continue
            kept_dirnames.append(name)
        dirnames[:] = kept_dirnames

        rel_dir = here.relative_to(root)
        if any(part.startswith(".") for part in rel_dir.parts):
            continue
        group_key = "" if rel_dir.parts == (".",) else normalize_locked_training_group_key("/".join(rel_dir.parts))
        if not group_key:
            continue
        top = group_key.split("/", 1)[0]
        for filename in filenames:
            p = here / filename
            if p.suffix.lower() not in AUDIO_EXTS:
                continue
            if is_apple_or_hidden_junk_path(p):
                continue
            rows.append(
                {
                    "source_path": str(p),
                    "group_key": group_key,
                    "outlier_status": "CLEAN_CANDIDATE",
                    "source": source_name,
                    "top": top,
                    "balance_group": source_balance_key_for_path(str(p)),
                }
            )
    return rows


def dry_core_augmentation_kind_for_training_label(
    *,
    group_key: str,
    label: str,
    public_label_text: str,
    top: str,
    policy: str,
) -> str:
    """Return the voice/sax dry-core augmentation kind for trusted training rows.

    This is training-label scoped, not filename scoped.  Folder paths remain the
    trusted label, and source file names are never inspected to decide whether a
    dry-core teacher is added.  The augmentation is intentionally opt-in so the
    normal brain family rebuild is unchanged unless Aaron requests the experiment.
    """
    normalized_policy = str(policy or "").strip().lower().replace("-", "_")
    if normalized_policy not in {
        "voice_sax",
        "voice_and_sax",
        "voice,sax",
        "voice sax",
        "1",
        "true",
        "yes",
    }:
        return ""
    text = (
        " ".join(
            [
                str(group_key or ""),
                str(label or ""),
                str(public_label_text or ""),
            ]
        )
        .lower()
        .replace("\\", "/")
    )
    top_text = str(top or "").strip().lower()
    if top_text and top_text != "instruments" and "instruments" not in text:
        return ""
    tokens = text.replace("/", " ").replace("_", " ").replace("-", " ").split()
    token_set = set(tokens)
    if "sax" in token_set or "saxophone" in token_set or "saxophones" in token_set:
        return "sax"
    if token_set.intersection({"voice", "voices", "vocal", "vocals", "vox"}):
        return "voice"
    return ""


def make_dry_core_training_feature_row(
    *,
    source: Path,
    base_row: FeatureRow,
    augmentation_kind: str,
    timeout_sec: float,
) -> tuple[FeatureRow | None, dict[str, str]]:
    """Create one dry-core training teacher for a voice/sax trusted row.

    The added row keeps the same folder-truth label as the original audio.  Only
    the fingerprint view changes.  This does not write transformed audio and does
    not modify Aaron's training files.
    """
    dry_fp, dry_duration, dry_status, dry_meta = make_harmonic_core_fingerprint_safe(
        source,
        timeout_sec=timeout_sec,
    )
    manifest_extra = {
        "training_view": f"dry_core_{augmentation_kind}",
        "dry_core_status": str(dry_status),
        "dry_core_wetness_score": f"{float(dry_meta.get('wetness_score', 0.0) or 0.0):.6f}"
        if isinstance(dry_meta, dict)
        else "0.000000",
        "dry_core_harmonic_energy_ratio": f"{float(dry_meta.get('harmonic_core_harmonic_energy_ratio', 0.0) or 0.0):.6f}"
        if isinstance(dry_meta, dict)
        else "0.000000",
        "dry_core_percussive_energy_ratio": f"{float(dry_meta.get('harmonic_core_percussive_energy_ratio', 0.0) or 0.0):.6f}"
        if isinstance(dry_meta, dict)
        else "0.000000",
    }
    if dry_status != "ok" or not np.any(np.asarray(dry_fp, dtype=np.float32)):
        return None, manifest_extra
    reason = (
        (base_row.structure_remap_reason + " | ") if str(base_row.structure_remap_reason or "").strip() else ""
    ) + f"dry_core_training_augmentation:{augmentation_kind}:same_folder_truth_label"
    augmented = FeatureRow(
        path=base_row.path,
        group_key=base_row.group_key,
        label=base_row.label,
        top=base_row.top,
        structure=base_row.structure,
        duration_sec=float(dry_duration),
        fingerprint=np.asarray(dry_fp, dtype=np.float32).astype(float).tolist(),
        read_status="ok",
        eval_reused_training_source=base_row.eval_reused_training_source,
        eval_selection_method=((base_row.eval_selection_method + "|") if base_row.eval_selection_method else "")
        + f"dry_core_training_augmentation_{augmentation_kind}",
        sampling_pool_size=base_row.sampling_pool_size,
        source_pack=base_row.source_pack,
        structure_remap_reason=reason,
        structure_conflict_warning=base_row.structure_conflict_warning,
        training_reject_reason=base_row.training_reject_reason,
        training_active_status=base_row.training_active_status,
        label_source_group_count=base_row.label_source_group_count,
        label_clean_available=base_row.label_clean_available,
        balance_group=base_row.balance_group,
    )
    return augmented, manifest_extra


def extract_feature_rows(
    rows: List[Dict[str, str]],
    reports_dir: Path,
    name: str,
    *,
    dry_core_augment_policy: str = "",
    dry_core_augment_timeout_sec: float = FINGERPRINT_TIMEOUT_SECONDS,
) -> List[FeatureRow]:
    out = []
    manifest_rows = []
    for idx, row in enumerate(rows, 1):
        source = Path(row.get("source_path", "")).expanduser()
        group_key = str(row.get("group_key", "")).replace("\\", "/").strip()
        label = row.get("_expected_label", "")
        public = row.get("_public_label", "") or public_label(label)
        top = row.get("_expected_top", "") or top_for_public_label(public)
        base_structure = row.get("_structure", "") or structure_lane_for_row(row, group_key)
        structure = repair_structure_lane_for_row(row, group_key, base_structure)
        simplified_hint = row.get("_simplified_hint_label", "") or simplified_hint_for_group(
            group_key, row_duration_sec(row)
        )
        fp, duration, status = make_fingerprint_safe(source)
        source_pack = source_pack_key_for_path(str(source))
        balance_group = (
            str(row.get("balance_group", "") or row.get("_balance_group", "")).strip()
            or source_balance_key_for_path(str(source))
            or source_pack
        )
        fingerprint_reason = ""
        if status == "ok":
            _repaired_structure, fingerprint_reason = repair_structure_lane_with_fingerprint(
                structure, fp, duration, group_key
            )
            # Locked training policy: never change public, label, top, or structure
            # after audio read. Physics disagreement remains report-only.
        remap_reason_parts = [str(row.get("_structure_remap_reason", "")).strip()]
        if fingerprint_reason:
            remap_reason_parts.append(fingerprint_reason)
        remap_reason = " | ".join(part for part in remap_reason_parts if part)
        conflict_warning = str(row.get("_structure_conflict_warning", ""))
        if fingerprint_reason:
            conflict_warning = (conflict_warning + " | " if conflict_warning else "") + fingerprint_reason
        eval_reused_training_source = str(row.get("_eval_reused_training_source", "0")) == "1"
        eval_selection_method = str(row.get("_eval_selection_method", ""))
        try:
            sampling_pool_size = int(str(row.get("_sampling_pool_size", row.get("_eval_pool_size", "0")) or "0"))
        except Exception:
            sampling_pool_size = 0
        feature_row = FeatureRow(
            str(source),
            group_key,
            label,
            top,
            structure,
            duration,
            fp.astype(float).tolist(),
            status,
            eval_reused_training_source,
            eval_selection_method,
            sampling_pool_size,
            source_pack,
            remap_reason,
            conflict_warning,
            "",
        )
        feature_row.balance_group = balance_group
        out.append(feature_row)
        manifest_rows.append(
            {
                "index": str(idx),
                "source_path": str(source),
                "training_view": "full_audio",
                "dry_core_status": "",
                "dry_core_wetness_score": "",
                "dry_core_harmonic_energy_ratio": "",
                "dry_core_percussive_energy_ratio": "",
                "group_key": group_key,
                "structure_before_remap": str(row.get("_structure_before_remap", "")),
                "expected_structure": structure,
                "structure_remap_reason": remap_reason,
                "structure_conflict_warning": conflict_warning,
                "expected_internal_label": label,
                "expected_label": public,
                "simplified_hint_label": simplified_hint,
                "expected_top": top,
                "duration_sec": f"{duration:.6f}",
                "physics_tags": fingerprint_physics_tags(fp, duration),
                "physics_summary": fingerprint_physics_summary(fp, duration),
                **fingerprint_physics_dict(fp, duration),
                "read_status": status,
                "source_pack": source_pack,
                "balance_group": balance_group,
                "sampling_rank": str(row.get("_sampling_rank", "")),
                "sampling_pool_size": str(row.get("_sampling_pool_size", "")),
                "sampling_method": str(row.get("_sampling_method", "")),
                "eval_rank": str(row.get("_eval_rank", "")),
                "eval_pool_size": str(row.get("_eval_pool_size", "")),
                "eval_reused_training_source": "1" if eval_reused_training_source else "0",
                "eval_selection_method": eval_selection_method,
            }
        )
        augmentation_kind = dry_core_augmentation_kind_for_training_label(
            group_key=group_key,
            label=label,
            public_label_text=public,
            top=top,
            policy=dry_core_augment_policy if name == "train" else "",
        )
        if status == "ok" and augmentation_kind:
            dry_row, dry_meta = make_dry_core_training_feature_row(
                source=source,
                base_row=feature_row,
                augmentation_kind=augmentation_kind,
                timeout_sec=float(dry_core_augment_timeout_sec),
            )
            if dry_row is not None:
                out.append(dry_row)
                manifest_rows.append(
                    {
                        "index": f"{idx}.dry_core",
                        "source_path": str(source),
                        "training_view": dry_meta.get("training_view", f"dry_core_{augmentation_kind}"),
                        "dry_core_status": dry_meta.get("dry_core_status", ""),
                        "dry_core_wetness_score": dry_meta.get("dry_core_wetness_score", ""),
                        "dry_core_harmonic_energy_ratio": dry_meta.get("dry_core_harmonic_energy_ratio", ""),
                        "dry_core_percussive_energy_ratio": dry_meta.get("dry_core_percussive_energy_ratio", ""),
                        "group_key": group_key,
                        "structure_before_remap": str(row.get("_structure_before_remap", "")),
                        "expected_structure": structure,
                        "structure_remap_reason": dry_row.structure_remap_reason,
                        "structure_conflict_warning": dry_row.structure_conflict_warning,
                        "expected_internal_label": label,
                        "expected_label": public,
                        "simplified_hint_label": simplified_hint,
                        "expected_top": top,
                        "duration_sec": f"{dry_row.duration_sec:.6f}",
                        "physics_tags": fingerprint_physics_tags(dry_row.fingerprint, dry_row.duration_sec),
                        "physics_summary": fingerprint_physics_summary(dry_row.fingerprint, dry_row.duration_sec),
                        **fingerprint_physics_dict(dry_row.fingerprint, dry_row.duration_sec),
                        "read_status": dry_row.read_status,
                        "source_pack": source_pack,
                        "balance_group": balance_group,
                        "sampling_rank": str(row.get("_sampling_rank", "")),
                        "sampling_pool_size": str(row.get("_sampling_pool_size", "")),
                        "sampling_method": str(row.get("_sampling_method", "")),
                        "eval_rank": str(row.get("_eval_rank", "")),
                        "eval_pool_size": str(row.get("_eval_pool_size", "")),
                        "eval_reused_training_source": "1" if eval_reused_training_source else "0",
                        "eval_selection_method": dry_row.eval_selection_method,
                    }
                )
        if idx % 25 == 0:
            print(f"{name}: analyzed {idx}/{len(rows)}", flush=True)

    write_csv(
        reports_dir / f"{name}_feature_manifest.csv",
        manifest_rows,
        [
            "index",
            "source_path",
            "training_view",
            "dry_core_status",
            "dry_core_wetness_score",
            "dry_core_harmonic_energy_ratio",
            "dry_core_percussive_energy_ratio",
            "group_key",
            "structure_before_remap",
            "expected_structure",
            "structure_remap_reason",
            "structure_conflict_warning",
            "expected_internal_label",
            "expected_label",
            "simplified_hint_label",
            "expected_top",
            "duration_sec",
            *PHYSICS_REPORT_FIELDS,
            "read_status",
            "source_pack",
            "balance_group",
            "sampling_rank",
            "sampling_pool_size",
            "sampling_method",
            "eval_rank",
            "eval_pool_size",
            "eval_reused_training_source",
            "eval_selection_method",
        ],
    )
    return out


def ensure_eval_labels_have_training_support(
    train_features: List[FeatureRow],
    eval_features: List[FeatureRow],
    reports_dir: Path,
) -> List[FeatureRow]:
    """Report eval labels that lack teachers, but never add eval rows to training.

    Earlier Stage 4 code reused readable eval rows as provisional teachers when a
    label had no train row. That makes accuracy meaningless because the model is
    allowed to learn the answer being tested. Locked training now stays honest:
    eval rows are never injected into training.
    """
    readable_train_labels = {str(r.label) for r in train_features if r.read_status == "ok"}
    rows = []
    for row in eval_features:
        if row.read_status != "ok":
            continue
        if str(row.label) in readable_train_labels:
            continue
        rows.append(
            {
                "action": "REPORT_ONLY_EVAL_LABEL_HAS_NO_TRAINING_TEACHER",
                "internal_label": row.label,
                "public_label": public_label(row.label),
                "top": row.top,
                "structure": row.structure,
                "source_path": row.path,
                "duration_sec": f"{float(row.duration_sec):.6f}",
                "reason": "eval row was not added to training; accuracy must remain holdout-only",
                "physics_tags": fingerprint_physics_tags(row.fingerprint, row.duration_sec),
                "physics_summary": fingerprint_physics_summary(row.fingerprint, row.duration_sec),
                "read_status": row.read_status,
                "source_pack": row.source_pack,
                "balance_group": row.balance_group,
            }
        )
    write_csv(
        reports_dir / "locked_corpus_eval_label_training_fallback.csv",
        rows,
        [
            "action",
            "internal_label",
            "public_label",
            "top",
            "structure",
            "source_path",
            "duration_sec",
            "reason",
            "physics_tags",
            "physics_summary",
            "read_status",
            "source_pack",
            "balance_group",
        ],
    )
    if rows:
        print(
            f"Eval honesty: {len(rows)} readable eval row(s) had no active training teacher; none were added to training"
        )
    return train_features


def _resolved_training_identity(path_text: str) -> str:
    try:
        p = Path(str(path_text)).expanduser()
        if p.exists() or p.is_symlink():
            return str(p.resolve())
        return str(p)
    except Exception:
        return str(path_text or "")


def remove_eval_training_overlaps(
    train_features: List[FeatureRow],
    eval_features: List[FeatureRow],
    reports_dir: Path,
) -> List[FeatureRow]:
    """Remove eval rows that resolve to the same audio target as training rows.

    This catches symlink-target overlap. It intentionally does not read filenames
    for semantic routing. The filename may appear in the report path only so Aaron
    can identify which file was excluded.
    """
    train_ids = {_resolved_training_identity(r.path) for r in train_features if r.read_status == "ok"}
    kept: List[FeatureRow] = []
    rows: List[Dict[str, str]] = []
    for row in eval_features:
        ident = _resolved_training_identity(row.path)
        overlap = ident in train_ids
        if overlap:
            row.eval_reused_training_source = True
            row.eval_selection_method = (
                row.eval_selection_method + "|" if row.eval_selection_method else ""
            ) + "excluded_realpath_training_overlap"
            rows.append(
                {
                    "action": "EXCLUDE_EVAL_TRAINING_OVERLAP",
                    "internal_label": row.label,
                    "public_label": public_label(row.label),
                    "top": row.top,
                    "structure": row.structure,
                    "source_path": row.path,
                    "resolved_identity": ident,
                    "reason": "same resolved audio target already used as training teacher",
                }
            )
            continue
        kept.append(row)
    write_csv(
        reports_dir / "eval_training_overlap_exclusions.csv",
        rows,
        ["action", "internal_label", "public_label", "top", "structure", "source_path", "resolved_identity", "reason"],
    )
    if rows:
        print(f"Eval honesty: excluded {len(rows)} eval row(s) that overlapped training by resolved path")
    return kept


def write_eval_activation_audit(brain: dict, eval_features: List[FeatureRow], reports_dir: Path) -> Path:
    """Write a hard audit of eval labels that are not active in the brain."""
    labels = {str(x) for x in brain.get("labels", [])}
    readable_by_label: Dict[str, List[FeatureRow]] = defaultdict(list)
    for row in eval_features:
        if row.read_status == "ok":
            readable_by_label[str(row.label)].append(row)
    rows = []
    for label, label_rows in sorted(readable_by_label.items()):
        if label in labels:
            continue
        sample = label_rows[0]
        rows.append(
            {
                "internal_label": label,
                "public_label": public_label(label),
                "top": sample.top,
                "structure": sample.structure,
                "readable_eval_count": str(len(label_rows)),
                "example_source_path": sample.path,
                "reason": "readable_eval_label_not_active_in_built_brain",
            }
        )
    out = reports_dir / "eval_label_activation_audit.csv"
    write_csv(
        out,
        rows,
        ["internal_label", "public_label", "top", "structure", "readable_eval_count", "example_source_path", "reason"],
    )
    return out


def training_quarantine_reason(row: FeatureRow) -> str:
    """Return reason this row cannot be read.

    Locked training policy: folder labels are Aaron-owned.  The code no longer
    rejects training rows for filename text, loop/one-shot disagreement, riser
    shape, or other physics judgments.  Those issues are reported for later
    human cleanup, but readable rows remain eligible teachers.
    """
    if row.read_status != "ok":
        return "read_status_not_ok"
    return ""


def write_loop_component_spy_report(kept_rows: List[FeatureRow], reports_dir: Path) -> None:
    """Report whether loop teachers resemble available one-shot teachers.

    This is Aaron's "spy on the loops" idea. It is not crazy, but it is only a
    first-pass macro check here: compare every kept loop fingerprint to nearby
    one-shot teacher fingerprints in the same broad top/family. A future version
    can segment loop events and compare each hit against one-shot prototypes.
    """
    one_shots = [r for r in kept_rows if r.structure == "one_shot" and r.read_status == "ok"]
    loops = [r for r in kept_rows if r.structure == "loop" and r.read_status == "ok"]
    rows: List[Dict[str, str]] = []
    if not one_shots or not loops:
        write_csv(
            reports_dir / "loop_component_spy.csv",
            rows,
            [
                "loop_source_path",
                "loop_label",
                "loop_top",
                "nearest_one_shot_label",
                "nearest_one_shot_top",
                "distance",
                "same_top",
                "note",
            ],
        )
        return
    X1 = np.asarray([r.fingerprint for r in one_shots], dtype=np.float32)
    center = np.median(X1, axis=0)
    mad = np.median(np.abs(X1 - center[None, :]), axis=0)
    scale = np.where(mad < 1e-6, 1.0, mad * 1.4826)
    W = FEATURE_WEIGHTS.astype(np.float32)
    one_z = ((X1 - center[None, :]) / scale[None, :]) * W[None, :]
    for loop in loops:
        x = ((np.asarray(loop.fingerprint, dtype=np.float32) - center) / scale) * W
        dists = np.linalg.norm(one_z - x[None, :], axis=1)
        order = np.argsort(dists)[:5]
        nearest = one_shots[int(order[0])]
        same_top_count = sum(1 for i in order if one_shots[int(i)].top == loop.top)
        rows.append(
            {
                "loop_source_path": loop.path,
                "loop_label": loop.label,
                "loop_top": loop.top,
                "loop_structure": loop.structure,
                "nearest_one_shot_label": nearest.label,
                "nearest_one_shot_top": nearest.top,
                "distance": f"{float(dists[int(order[0])]):.6f}",
                "same_top": "1" if nearest.top == loop.top else "0",
                "nearest5_same_top_count": str(same_top_count),
                "note": "macro_fingerprint_spy_not_event_segmentation",
            }
        )
    write_csv(
        reports_dir / "loop_component_spy.csv",
        rows,
        [
            "loop_source_path",
            "loop_label",
            "loop_top",
            "loop_structure",
            "nearest_one_shot_label",
            "nearest_one_shot_top",
            "distance",
            "same_top",
            "nearest5_same_top_count",
            "note",
        ],
    )


def write_training_contradiction_report(kept: List[FeatureRow], reports_dir: Path) -> None:
    """Report training rows where folder structure and audio evidence disagree.

    This is a report-only diagnostic. It never removes, moves, or relabels any
    training example. Aaron decides what to do with flagged rows.

    Three contradiction types:

    FOLDER_LOOP_AUDIO_ONE_SHOT
      Folder or label says loop but the audio fingerprint lacks repeated-event
      evidence. Could be a long single sustain, a one-shot misfiled in a loop
      folder, or an FX sound with no clear pulse. High severity.

    FOLDER_ONE_SHOT_AUDIO_LOOP
      Folder says one-shot but the fingerprint shows strong repeated-event
      evidence. Could be a loop misfiled in a one-shot folder. High severity.

    FILENAME_SUGGESTS_SINGLE_HIT_IN_LOOP_FOLDER
      Folder says loop but the filename matches common single-hit naming
      conventions used by FreeSound datasets and public audio packs (e.g.
      bass_drum_train_*, snare_drum_val_*, hi_hat_val_*). This is a listening
      audit hint only. Filenames are never classification truth. Medium severity.

    The report is sorted: high severity first, then by label, then by path.
    """
    SINGLE_HIT_FILENAME_RE = re.compile(
        r"(?:^|[_\-\s])(?:"
        r"bass_drum|snare_drum|hi_hat|hihat|crash_cymbal|kick_drum|"
        r"cowbell_val|tambourine_val|tambourine_eval|"
        r"_val_\d|_train_\d|_eval_\d"
        r")(?:[_\-\s]|$)",
        re.IGNORECASE,
    )

    severity_order = {"high": 0, "medium": 1, "low": 2}
    rows_out: List[Dict[str, str]] = []

    for row in kept:
        if row.read_status != "ok":
            continue
        folder_structure = row.structure or label_default_structure(row.label)
        has_loop_audio, loop_reason = fingerprint_loop_evidence_detail(row.fingerprint, row.duration_sec)
        filename = Path(row.path).name

        if folder_structure == "loop" and not has_loop_audio:
            rows_out.append(
                {
                    "contradiction_type": "FOLDER_LOOP_AUDIO_ONE_SHOT",
                    "severity": "high",
                    "internal_label": row.label,
                    "public_label": public_label(row.label),
                    "top": row.top,
                    "folder_structure": folder_structure,
                    "audio_loop_evidence": "False",
                    "detail": loop_reason,
                    "source_path": row.path,
                    "group_key": row.group_key,
                    "duration_sec": f"{row.duration_sec:.3f}",
                    "physics_tags": fingerprint_physics_tags(row.fingerprint, row.duration_sec),
                    "action": "REVIEW_NEEDED_DO_NOT_AUTO_REMOVE",
                }
            )

        elif folder_structure == "one_shot" and has_loop_audio:
            rows_out.append(
                {
                    "contradiction_type": "FOLDER_ONE_SHOT_AUDIO_LOOP",
                    "severity": "high",
                    "internal_label": row.label,
                    "public_label": public_label(row.label),
                    "top": row.top,
                    "folder_structure": folder_structure,
                    "audio_loop_evidence": "True",
                    "detail": loop_reason,
                    "source_path": row.path,
                    "group_key": row.group_key,
                    "duration_sec": f"{row.duration_sec:.3f}",
                    "physics_tags": fingerprint_physics_tags(row.fingerprint, row.duration_sec),
                    "action": "REVIEW_NEEDED_DO_NOT_AUTO_REMOVE",
                }
            )

        if folder_structure == "loop" and SINGLE_HIT_FILENAME_RE.search(filename):
            rows_out.append(
                {
                    "contradiction_type": "FILENAME_SUGGESTS_SINGLE_HIT_IN_LOOP_FOLDER",
                    "severity": "medium",
                    "internal_label": row.label,
                    "public_label": public_label(row.label),
                    "top": row.top,
                    "folder_structure": folder_structure,
                    "audio_loop_evidence": str(has_loop_audio),
                    "detail": f"filename={filename}; filenames_are_not_truth_listen_before_acting",
                    "source_path": row.path,
                    "group_key": row.group_key,
                    "duration_sec": f"{row.duration_sec:.3f}",
                    "physics_tags": fingerprint_physics_tags(row.fingerprint, row.duration_sec),
                    "action": "REVIEW_NEEDED_DO_NOT_AUTO_REMOVE",
                }
            )

    rows_out.sort(
        key=lambda r: (
            severity_order.get(r["severity"], 9),
            r["internal_label"],
            r["source_path"],
        )
    )

    fields = [
        "contradiction_type",
        "severity",
        "internal_label",
        "public_label",
        "top",
        "folder_structure",
        "audio_loop_evidence",
        "detail",
        "source_path",
        "group_key",
        "duration_sec",
        "physics_tags",
        "action",
    ]
    write_csv(reports_dir / "training_contradiction_report.csv", rows_out, fields)

    high = sum(1 for r in rows_out if r["severity"] == "high")
    med = sum(1 for r in rows_out if r["severity"] == "medium")
    print(
        f"Training contradiction report: {len(rows_out)} total "
        f"({high} high severity, {med} medium severity) "
        f"-> {reports_dir / 'training_contradiction_report.csv'}"
    )


def curate_training_features(
    train_features: List[FeatureRow],
    reports_dir: Path,
    run_dir: Path,
    min_group_size: int = 1,
    max_keep_per_label: int = 100,
    fx_max_keep_per_label: int = 100,
    fx_central_pool_fraction: float = 1.00,
    max_reject_fraction: float = 1.00,
    preview_keep_per_label: int = 5,
    preview_reject_per_label: int = 5,
    central_pool_fraction: float = 1.00,
    random_seed: int = 20260503,
    source_cap_per_label: int = DEFAULT_SOURCE_CAP_PER_LABEL,
    min_active_train_per_label: int = MIN_ACTIVE_TRAIN_PER_LABEL,
    min_source_groups_per_label: int = DEFAULT_MIN_SOURCE_GROUPS_PER_LABEL,
    copy_gold_workspace: bool = True,
) -> List[FeatureRow]:
    """Build a folder-balanced gold-candidate teacher set.

    v0.4.34 policy:
      - only One Shots and Loops are valid structures
      - old _LONG_RUNNING is ignored as truth
      - structure conflicts are report-only
      - clean rare labels become active/provisional teachers instead of disappearing
      - every readable locked row is kept; no source caps or central-pool trimming
      - unreadable rows are reported, not hidden
    """
    good = [r for r in train_features if r.read_status == "ok"]
    by_label: Dict[str, List[FeatureRow]] = defaultdict(list)
    for row in good:
        by_label[row.label].append(row)

    kept: List[FeatureRow] = []
    manifest_rows: List[Dict[str, str]] = []
    label_summary: List[Dict[str, str]] = []
    activation_rows: List[Dict[str, str]] = []

    keep_preview_root = run_dir / "training_curated_keep_preview"
    reject_preview_root = run_dir / "training_curated_reject_preview"
    gold_root = run_dir / "gold_training_candidate_workspace" / "02_GOLD_CANDIDATES_TO_REVIEW"
    quarantine_root = run_dir / "gold_training_candidate_workspace" / "01_QUARANTINED_NOT_TRAINING"

    for label, rows in sorted(by_label.items()):
        pub = public_label(label)
        top = top_for_public_label(label)
        structure = rows[0].structure if rows else label_default_structure(label)
        disabled_label_reason = public_label_is_disabled_for_training(pub)

        clean_rows: List[FeatureRow] = []
        row_reject_reasons: Dict[int, str] = {}
        for idx, row in enumerate(rows):
            reason = training_quarantine_reason(row)
            if reason:
                row.training_reject_reason = reason
                row_reject_reasons[idx] = reason
            else:
                clean_rows.append(row)

        source_groups = sorted(
            set((r.balance_group or r.source_pack or source_balance_key_for_path(r.path)) for r in clean_rows)
        )
        available_clean = len(clean_rows)
        active_status = "ACTIVE_TRUSTED"
        inactive_reason = ""
        active_reason = "clean_source_diverse_label"
        if disabled_label_reason:
            active_status = "DISABLED_BAD_TRAINING_DATA"
            inactive_reason = disabled_label_reason
            active_reason = disabled_label_reason
        elif available_clean < max(min_group_size, MIN_PROVISIONAL_TRAIN_PER_LABEL):
            active_status = "DISCOVERY_ONLY_NO_CLEAN_EXAMPLES"
            inactive_reason = (
                f"clean_examples_{available_clean}_below_min_{max(min_group_size, MIN_PROVISIONAL_TRAIN_PER_LABEL)}"
            )
            active_reason = inactive_reason
        elif available_clean < min_active_train_per_label:
            active_status = "ACTIVE_PROVISIONAL_LOW_COUNT"
            active_reason = f"clean_examples_{available_clean}_below_trusted_min_{min_active_train_per_label}"
        elif len(source_groups) < min_source_groups_per_label:
            active_status = "ACTIVE_PROVISIONAL_SINGLE_SOURCE"
            active_reason = f"source_groups_{len(source_groups)}_below_trusted_min_{min_source_groups_per_label}"

        chosen_clean_indices: Set[int] = set()
        dist_by_clean_index: Dict[int, float] = {}
        rank_by_clean_index: Dict[int, int] = {}
        central_pool_size = 0
        effective_keep_target = 0
        reason_policy = ""
        if clean_rows and is_active_training_status(active_status):
            X = np.asarray([r.fingerprint for r in clean_rows], dtype=np.float32)
            center = np.median(X, axis=0)
            mad = np.median(np.abs(X - center[None, :]), axis=0)
            scale = np.where(mad < 1e-6, 1.0, mad * 1.4826)
            Z = (X - center[None, :]) / scale[None, :]
            weighted = Z * FEATURE_WEIGHTS[None, :]
            dist = np.linalg.norm(weighted, axis=1)
            order = [int(i) for i in np.argsort(dist)]
            for rank, idx in enumerate(order, 1):
                dist_by_clean_index[idx] = float(dist[idx])
                rank_by_clean_index[idx] = rank

            # Locked-training policy: readable rows are Aaron-owned teachers.
            # Do not functionally delete them with source caps, central-pool caps,
            # outlier trimming, or loop/one-shot repair.  Selection limits belong
            # in the earlier explicit --max-train-per-group sampling step.
            central_pool_size = len(clean_rows)
            effective_keep_target = len(clean_rows)
            chosen_clean_indices = set(range(len(clean_rows)))
            reason_policy = (
                f"{active_status} {active_reason}; KEEP_ALL_READABLE_LOCKED_TRAINING_ROWS "
                "no_source_cap_no_central_pool_no_physics_relabel"
            )

        elif not is_active_training_status(active_status):
            reason_policy = "label_not_active_no_training_keep"

        clean_idx_by_identity = {id(r): idx for idx, r in enumerate(clean_rows)}
        label_kept = 0
        label_rejected = 0
        for row in rows:
            clean_idx = clean_idx_by_identity.get(id(row), -1)
            quarantine_reason = row.training_reject_reason or ("" if clean_idx >= 0 else "quarantined_before_curation")
            row.training_active_status = active_status
            row.label_source_group_count = len(source_groups)
            row.label_clean_available = available_clean
            if not is_active_training_status(active_status) and clean_idx >= 0:
                quarantine_reason = inactive_reason
            action = (
                "KEEP"
                if clean_idx >= 0 and clean_idx in chosen_clean_indices and is_active_training_status(active_status)
                else "REJECT_FOR_TRAINING"
            )
            if action == "KEEP":
                kept.append(row)
                label_kept += 1
            else:
                label_rejected += 1
            dist = dist_by_clean_index.get(clean_idx, 0.0) if clean_idx >= 0 else 0.0
            rank = rank_by_clean_index.get(clean_idx, 999999) if clean_idx >= 0 else 999999
            manifest_rows.append(
                {
                    "action": action,
                    "internal_label": label,
                    "public_label": pub,
                    "top": top,
                    "structure": structure,
                    "source_pack": row.source_pack or source_pack_key_for_path(row.path),
                    "balance_group": row.balance_group or row.source_pack or source_balance_key_for_path(row.path),
                    "rank_within_label": str(rank),
                    "distance_to_label_median": f"{float(dist):.6f}",
                    "source_path": row.path,
                    "group_key": row.group_key,
                    "duration_sec": f"{row.duration_sec:.6f}",
                    "physics_tags": fingerprint_physics_tags(row.fingerprint, row.duration_sec),
                    "physics_summary": fingerprint_physics_summary(row.fingerprint, row.duration_sec),
                    **fingerprint_physics_dict(row.fingerprint, row.duration_sec),
                    "read_status": row.read_status,
                    "policy": reason_policy,
                    "central_pool_size": str(central_pool_size),
                    "active_status": active_status,
                    "active_reason": active_reason,
                    "training_reject_reason": quarantine_reason,
                    "structure_remap_reason": row.structure_remap_reason,
                    "structure_conflict_warning": row.structure_conflict_warning,
                }
            )

        def copy_selected_preview(
            root: Path,
            selected_rows: List[FeatureRow],
            prefix: str,
            limit: int,
            top_name: str = top,
            public_name: str = pub,
        ) -> None:
            copied = 0
            for row in selected_rows:
                if copied >= limit:
                    break
                src = Path(row.path)
                if not src.exists() and not src.is_symlink():
                    continue
                parts = folder_parts_for_public_label(row.label)
                dest_dir = (
                    root.joinpath(*parts)
                    if parts
                    else root / safe_folder_name(top_name) / safe_folder_name(public_name)
                )
                dest_dir.mkdir(parents=True, exist_ok=True)
                suffix = src.suffix.lower() or ".wav"
                dest = dest_dir / f"{prefix}_{copied + 1:03d}__{safe_folder_name(src.stem)[:90]}{suffix}"
                try:
                    made = materialize_preview_audio(src, dest)
                    if not made:
                        continue
                    copied += 1
                except Exception:
                    pass

        kept_for_label = [r for r in rows if r in kept]
        rejected_for_label = [r for r in rows if r not in kept]
        copy_selected_preview(keep_preview_root, kept_for_label, "keep", preview_keep_per_label)
        copy_selected_preview(reject_preview_root, rejected_for_label, "reject", preview_reject_per_label)
        if copy_gold_workspace:
            copy_selected_preview(gold_root, kept_for_label, "gold", DEFAULT_GOLD_WORKSPACE_KEEP_LIMIT)
            # Quarantine copy is capped to avoid huge upload folders.
            copy_selected_preview(quarantine_root, rejected_for_label, "quarantine", min(preview_reject_per_label, 5))

        label_summary.append(
            {
                "internal_label": label,
                "public_label": pub,
                "top": top,
                "structure": structure,
                "available": str(len(rows)),
                "clean_available": str(available_clean),
                "source_group_count": str(len(source_groups)),
                "kept": str(label_kept),
                "target_keep": str(effective_keep_target),
                "shortage": str(max(0, effective_keep_target - label_kept)),
                "rejected": str(label_rejected),
                "policy": reason_policy,
                "central_pool_size": str(central_pool_size),
                "active_status": active_status,
                "active_reason": active_reason,
                "inactive_reason": inactive_reason,
            }
        )
        activation_rows.append(
            {
                "internal_label": label,
                "public_label": pub,
                "top": top,
                "structure": structure,
                "active_status": active_status,
                "active_reason": active_reason,
                "clean_available": str(available_clean),
                "source_group_count": str(len(source_groups)),
                "kept": str(label_kept),
                "min_active_train_per_label": str(min_active_train_per_label),
                "min_source_groups_per_label": str(min_source_groups_per_label),
                "inactive_reason": inactive_reason,
            }
        )

    write_csv(
        reports_dir / "training_curation_manifest.csv",
        manifest_rows,
        [
            "action",
            "internal_label",
            "public_label",
            "top",
            "structure",
            "source_pack",
            "balance_group",
            "rank_within_label",
            "distance_to_label_median",
            "source_path",
            "group_key",
            "duration_sec",
            *PHYSICS_REPORT_FIELDS,
            "read_status",
            "policy",
            "central_pool_size",
            "active_status",
            "active_reason",
            "training_reject_reason",
            "structure_remap_reason",
            "structure_conflict_warning",
        ],
    )
    write_csv(
        reports_dir / "training_curation_summary_by_label.csv",
        label_summary,
        [
            "internal_label",
            "public_label",
            "top",
            "structure",
            "available",
            "clean_available",
            "source_group_count",
            "kept",
            "target_keep",
            "shortage",
            "rejected",
            "policy",
            "central_pool_size",
            "active_status",
            "active_reason",
            "inactive_reason",
        ],
    )
    write_csv(
        reports_dir / "label_activation_status.csv",
        activation_rows,
        [
            "internal_label",
            "public_label",
            "top",
            "structure",
            "active_status",
            "active_reason",
            "clean_available",
            "source_group_count",
            "kept",
            "min_active_train_per_label",
            "min_source_groups_per_label",
            "inactive_reason",
        ],
    )
    shortage_rows = [row for row in label_summary if int(row.get("shortage", "0") or 0) > 0]
    write_csv(
        reports_dir / "training_curation_shortages.csv",
        shortage_rows,
        [
            "internal_label",
            "public_label",
            "top",
            "structure",
            "available",
            "clean_available",
            "source_group_count",
            "kept",
            "target_keep",
            "shortage",
            "rejected",
            "policy",
            "central_pool_size",
            "active_status",
            "active_reason",
            "inactive_reason",
        ],
    )
    quarantine_rows = [row for row in manifest_rows if row.get("action") == "REJECT_FOR_TRAINING"]
    write_csv(
        reports_dir / "training_quarantine_manifest.csv",
        quarantine_rows,
        [
            "action",
            "internal_label",
            "public_label",
            "top",
            "structure",
            "source_pack",
            "balance_group",
            "training_reject_reason",
            "structure_remap_reason",
            "structure_conflict_warning",
            "source_path",
            "group_key",
            "duration_sec",
            *PHYSICS_REPORT_FIELDS,
            "active_status",
        ],
    )
    write_loop_component_spy_report(kept, reports_dir)
    write_training_physics_summary(kept, reports_dir)
    write_training_contradiction_report(kept, reports_dir)

    def write_tree(root: Path, out_name: str) -> None:
        lines = []
        if root.exists():
            for p in sorted(root.rglob("*")):
                rel = p.relative_to(root)
                depth = len(rel.parts) - 1
                indent = "  " * depth
                suffix = "/" if p.is_dir() else ""
                lines.append(f"{indent}{p.name}{suffix}")
        (reports_dir / out_name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_tree(keep_preview_root, "training_curated_keep_preview_tree.txt")
    write_tree(reject_preview_root, "training_curated_reject_preview_tree.txt")
    write_tree(gold_root, "gold_training_candidate_workspace_tree.txt")
    write_tree(quarantine_root, "training_quarantine_preview_tree.txt")

    print(f"Training curation kept {len(kept)} of {len(good)} readable rows")
    return kept
