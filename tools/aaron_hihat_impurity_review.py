#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT = Path("/Volumes/T9/testbed/Aaron_Sound_Sorter")
TRAINING_ROOT = PROJECT / "training/locked_curated_v1"
HIHAT_ROOT = TRAINING_ROOT / "Drums/Hi Hats"
OUT_ROOT = PROJECT / "reports/hihat_impurity_review" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".ogg"}

try:
    import soundfile as sf
except Exception:
    sf = None


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTS and not path.name.startswith("._")


def read_audio(path: Path):
    if sf is not None:
        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        y = np.mean(data, axis=1).astype(np.float32)
        return y, int(sr)

    if path.suffix.lower() != ".wav":
        raise RuntimeError("soundfile missing and not a WAV")

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nchan = wf.getnchannels()
        sampw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sampw == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampw == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError("unsupported WAV sample width")

    if nchan > 1:
        arr = arr.reshape(-1, nchan)
        y = np.mean(arr, axis=1)
    else:
        y = arr
    return y.astype(np.float32), int(sr)


def frame_audio(y, n_fft=2048, hop=512):
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    starts = range(0, max(1, len(y) - n_fft + 1), hop)
    win = np.hanning(n_fft).astype(np.float32)
    frames = []
    for s in starts:
        f = y[s : s + n_fft]
        if len(f) < n_fft:
            f = np.pad(f, (0, n_fft - len(f)))
        frames.append(f * win)
    return np.vstack(frames)


def pitch_confidence(y, sr):
    if len(y) < int(0.06 * sr):
        return 0.0
    max_len = min(len(y), int(sr * 1.0))
    x = y[:max_len].astype(np.float32)
    x = x - np.mean(x)
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    if rms < 1e-6:
        return 0.0

    fmin, fmax = 80.0, 2500.0
    min_lag = max(1, int(sr / fmax))
    max_lag = min(int(sr / fmin), len(x) - 1)
    if min_lag >= max_lag:
        return 0.0

    n = int(2 ** math.ceil(math.log2(max(1, len(x) * 2 - 1))))
    sp = np.fft.rfft(x, n=n)
    corr = np.fft.irfft(sp * np.conj(sp), n=n)[: len(x)]
    corr0 = float(corr[0]) + 1e-9
    corr[:min_lag] = 0
    lag = int(np.argmax(corr[min_lag:max_lag])) + min_lag
    return max(0.0, min(1.0, float(corr[lag] / corr0)))


def metrics(y, sr):
    y = y.astype(np.float32)
    y = y - np.mean(y)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-9:
        y = y / peak

    duration = float(len(y) / max(sr, 1))
    frames = frame_audio(y)
    mag = np.abs(np.fft.rfft(frames, axis=1)) + 1e-12
    power = mag**2
    freqs = np.fft.rfftfreq(2048, 1.0 / sr)

    spec = np.sum(power, axis=0) + 1e-12
    total = float(np.sum(spec)) + 1e-12

    def ratio(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(np.sum(spec[m]) / total) if np.any(m) else 0.0

    sub = ratio(20, 80)
    low = ratio(80, 250)
    low_mid = ratio(250, 1000)
    mid = ratio(1000, 2500)
    presence = ratio(2500, 8000)
    air = ratio(8000, sr / 2)
    high = presence + air
    low_total = sub + low + low_mid

    flatness_frames = np.exp(np.mean(np.log(mag), axis=1)) / (np.mean(mag, axis=1) + 1e-12)
    flatness = float(np.mean(flatness_frames))

    p = power / np.maximum(np.sum(power, axis=1, keepdims=True), 1e-12)
    entropy_frames = -np.sum(p * np.log2(p + 1e-12), axis=1) / math.log2(power.shape[1])
    entropy = float(np.mean(entropy_frames))

    env = np.sqrt(np.mean(frames**2, axis=1))
    diff = np.maximum(0, np.diff(env, prepend=env[0]))
    if diff.size:
        med = float(np.median(diff))
        mad = float(np.median(np.abs(diff - med))) + 1e-12
        thr = max(med + 3.0 * mad, float(np.max(diff)) * 0.18)
        peaks = np.where(diff >= thr)[0]
        events = int(len(peaks))
    else:
        events = 0

    temporal = float(np.argmax(env) / max(len(env), 1)) if env.size else 0.0

    tail_start = max(0, int(len(y) * 0.55))
    start_end = max(1, int(len(y) * 0.20))
    start_rms = float(np.sqrt(np.mean(y[:start_end] ** 2))) if start_end else 0.0
    tail_rms = float(np.sqrt(np.mean(y[tail_start:] ** 2))) if tail_start < len(y) else 0.0
    tail_ratio = tail_rms / (start_rms + 1e-9)

    pitch = pitch_confidence(y, sr)

    return {
        "duration": duration,
        "events": events,
        "sub": sub,
        "low": low,
        "low_mid": low_mid,
        "mid": mid,
        "presence": presence,
        "air": air,
        "high": high,
        "low_total": low_total,
        "flatness": flatness,
        "entropy": entropy,
        "temporal": temporal,
        "tail": tail_ratio,
        "pitch": pitch,
    }


def symlink_to_bucket(src: Path, bucket: str, reason: str):
    bucket_dir = OUT_ROOT / bucket
    bucket_dir.mkdir(parents=True, exist_ok=True)
    dest = bucket_dir / src.name
    i = 1
    while dest.exists() or dest.is_symlink():
        dest = bucket_dir / f"{src.stem}__{i}{src.suffix}"
        i += 1
    os.symlink(src, dest)
    with (bucket_dir / "_REASONS.txt").open("a", encoding="utf-8") as f:
        f.write(f"{dest.name}\n  source={src}\n  {reason}\n\n")


def classify(m):
    d = m["duration"]
    events = m["events"]
    high = m["high"]
    air = m["air"]
    presence = m["presence"]
    low_total = m["low_total"]
    flatness = m["flatness"]
    entropy = m["entropy"]
    pitch = m["pitch"]
    temporal = m["temporal"]
    tail = m["tail"]

    reason = (
        f"duration={d:.3f} events={events} low_total={low_total:.3f} "
        f"presence={presence:.3f} air={air:.3f} high={high:.3f} "
        f"flatness={flatness:.3f} entropy={entropy:.3f} pitch={pitch:.3f} "
        f"temporal={temporal:.3f} tail={tail:.3f}"
    )

    # Likely clean hat: short, bright/noisy, low low-end, not too pitched.
    if (
        d <= 0.85
        and high >= 0.45
        and low_total <= 0.22
        and flatness >= 0.22
        and entropy >= 0.45
        and pitch <= 0.45
        and tail <= 0.45
    ):
        return "_KEEP_LIKELY_HIHATS", "likely clean hi-hat: " + reason

    # Low/body contamination. These are probably not hats.
    if low_total >= 0.40 and high <= 0.45 and d >= 0.12:
        return "_MOVE_CANDIDATES/Low_Body_Contaminated_Not_Hat", "low/body contaminated hat suspect: " + reason

    # Pitched ring hiding in hats.
    if pitch >= 0.55 and entropy <= 0.58 and flatness <= 0.42 and d >= 0.18:
        return "_MOVE_CANDIDATES/Pitched_Ring_or_Bell_Not_Hat", "pitched ring/bell suspect in hats: " + reason

    # Crash/open cymbal tail hiding in hats.
    if d >= 1.40 and high >= 0.45 and tail >= 0.35 and entropy >= 0.48:
        return "_MOVE_CANDIDATES/Open_Cymbal_or_Crash_Not_Hat", "open cymbal/crash-like tail suspect: " + reason

    # Multi-hit or drum-mix contamination.
    if d >= 0.70 and events >= 5 and (low_total >= 0.25 or tail >= 0.35):
        return "_MOVE_CANDIDATES/Multi_Hit_or_Drum_Mix_Not_Hat", "multi-hit/drum-mix suspect in hats: " + reason

    # Long weird hats, not safe for clean one-shot training.
    if d >= 2.00 or events >= 10:
        return "_MOVE_CANDIDATES/Long_or_Distributed_Not_Clean_Hat", "long/distributed hat suspect: " + reason

    return "_IGNORE_WEAK_OR_AMBIGUOUS", "weak or ambiguous, do not move now: " + reason


def main():
    if not HIHAT_ROOT.exists():
        raise SystemExit(f"Missing hi-hat root: {HIHAT_ROOT}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    total = 0
    errors = 0
    for path in sorted(HIHAT_ROOT.rglob("*")):
        if "_QUARANTINED_BY_PHYSICS_REVIEW" in str(path):
            continue
        if not path.is_file() or not is_audio(path):
            continue
        total += 1
        try:
            y, sr = read_audio(path)
            m = metrics(y, sr)
            bucket, reason = classify(m)
            symlink_to_bucket(path, bucket, reason)
        except Exception as exc:
            errors += 1
            symlink_to_bucket(path, "_ERRORS", f"error={exc}")

    print("DONE")
    print(f"Scanned: {total}")
    print(f"Errors: {errors}")
    print(f"Review folder: {OUT_ROOT}")
    os.system(f'open "{OUT_ROOT}"')


if __name__ == "__main__":
    main()
