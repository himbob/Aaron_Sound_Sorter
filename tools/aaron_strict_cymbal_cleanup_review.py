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
CYMBAL_ROOT = TRAINING_ROOT / "Drums/Cymbals"
OUT_ROOT = (
    PROJECT / "reports/metallic_percussion_review" / f"strict_cymbal_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

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
    return np.vstack(
        [
            (
                y[s : s + n_fft]
                if len(y[s : s + n_fft]) == n_fft
                else np.pad(y[s : s + n_fft], (0, n_fft - len(y[s : s + n_fft])))
            )
            * win
            for s in starts
        ]
    )


def spectral_features(y, sr):
    y = y.astype(np.float32)
    y = y - np.mean(y)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-9:
        y = y / peak

    frames = frame_audio(y)
    mag = np.abs(np.fft.rfft(frames, axis=1)) + 1e-12
    power = mag**2
    freqs = np.fft.rfftfreq(2048, 1.0 / sr)

    spec = np.sum(power, axis=0) + 1e-12
    total = float(np.sum(spec)) + 1e-12

    def ratio(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return float(np.sum(spec[m]) / total) if np.any(m) else 0.0

    presence = ratio(2000, 8000)
    air = ratio(8000, sr / 2)
    high = presence + air
    low = ratio(20, 250)

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

    duration = float(len(y) / max(sr, 1))
    temporal = float(np.argmax(env) / max(len(env), 1)) if env.size else 0.0

    pitch_conf = pitch_confidence(y, sr)

    return {
        "duration": duration,
        "presence": presence,
        "air": air,
        "high": high,
        "low": low,
        "flatness": flatness,
        "entropy": entropy,
        "events": events,
        "temporal": temporal,
        "pitch": pitch_conf,
    }


def pitch_confidence(y, sr):
    if len(y) < int(0.08 * sr):
        return 0.0
    max_len = min(len(y), int(sr * 1.2))
    x = y[:max_len].astype(np.float32)
    x = x - np.mean(x)
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    if rms < 1e-6:
        return 0.0
    fmin, fmax = 80.0, 2000.0
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


def symlink_to_bucket(src: Path, bucket: str, reason: str):
    bucket_dir = OUT_ROOT / bucket
    bucket_dir.mkdir(parents=True, exist_ok=True)
    name = src.name
    dest = bucket_dir / name
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
    flatness = m["flatness"]
    entropy = m["entropy"]
    pitch = m["pitch"]
    temporal = m["temporal"]

    reason = (
        f"duration={d:.3f} events={events} high={high:.3f} presence={presence:.3f} "
        f"air={air:.3f} flatness={flatness:.3f} entropy={entropy:.3f} "
        f"pitch={pitch:.3f} temporal={temporal:.3f}"
    )

    # Low/body pitched resonant metal wrongly living in cymbals.
    # Example: long low ringing metal/gong-like files with strong pitch and little high-air wash.
    if (
        d >= 1.2
        and pitch >= 0.55
        and m.get("low", 0.0) >= 0.45
        and high <= 0.22
        and entropy <= 0.62
        and flatness <= 0.34
    ):
        return (
            "_MOVE_CANDIDATES/Low_Rings_Gongs_and_Resonant_Metal_STRONG",
            "strong low/body resonant metal suspect: " + reason,
        )

    # Real cymbal protection. Long, noisy, diffuse, shimmering cymbal tails.
    if d >= 1.8 and high >= 0.45 and entropy >= 0.50 and flatness >= 0.22 and pitch <= 0.42:
        return "_KEEP_LIKELY_CYMBALS", "likely real cymbal: " + reason

    # Strong high ring/chime suspect. Shorter, clearer pitch, less diffuse than crash cymbal.
    if d <= 1.4 and high >= 0.55 and pitch >= 0.50 and entropy <= 0.48 and flatness <= 0.36:
        return "_MOVE_CANDIDATES/High_Rings_and_Chimes_STRONG", "strong high ring/chime suspect: " + reason

    # Medium suspect. Still needs your ear before moving.
    if d <= 2.2 and high >= 0.50 and pitch >= 0.42 and entropy <= 0.55 and flatness <= 0.40:
        return "_MOVE_CANDIDATES/High_Rings_and_Chimes_MEDIUM", "medium high ring/chime suspect: " + reason

    # Cowbell/block-like metallic percussion tends to be shorter and not diffuse cymbal wash.
    if d <= 1.2 and pitch >= 0.35 and entropy <= 0.58 and flatness <= 0.42 and high < 0.65:
        return "_MOVE_CANDIDATES/Cowbell_Block_or_Other_Metallic", "cowbell/block/metallic suspect: " + reason

    return "_IGNORE_WEAK_OR_AMBIGUOUS", "weak or ambiguous, do not move now: " + reason


def main():
    if not CYMBAL_ROOT.exists():
        raise SystemExit(f"Missing cymbal root: {CYMBAL_ROOT}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    total = 0
    errors = 0
    for path in sorted(CYMBAL_ROOT.rglob("*")):
        if not path.is_file() or not is_audio(path):
            continue
        total += 1
        try:
            y, sr = read_audio(path)
            metrics = spectral_features(y, sr)
            bucket, reason = classify(metrics)
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
