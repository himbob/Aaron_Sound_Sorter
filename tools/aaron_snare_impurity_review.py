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
SNARE_ROOT = TRAINING_ROOT / "Drums/Snares"
OUT_ROOT = PROJECT / "reports/snare_impurity_review" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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

    fmin, fmax = 60.0, 1800.0
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
    kick = ratio(80, 160)
    low_body = ratio(160, 350)
    snare_body = ratio(350, 1000)
    crack = ratio(1000, 3500)
    presence = ratio(3500, 8000)
    air = ratio(8000, sr / 2)
    low_total = sub + kick + low_body
    snare_mid = snare_body + crack
    high_total = presence + air

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

    crest = float(np.max(np.abs(y)) / (np.sqrt(np.mean(y * y)) + 1e-9)) if y.size else 0.0
    pitch = pitch_confidence(y, sr)

    return {
        "duration": duration,
        "events": events,
        "sub": sub,
        "kick": kick,
        "low_body": low_body,
        "snare_body": snare_body,
        "crack": crack,
        "presence": presence,
        "air": air,
        "low_total": low_total,
        "snare_mid": snare_mid,
        "high_total": high_total,
        "flatness": flatness,
        "entropy": entropy,
        "temporal": temporal,
        "tail": tail_ratio,
        "crest": crest,
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
    sub = m["sub"]
    kick = m["kick"]
    low_body = m["low_body"]
    low_total = m["low_total"]
    snare_mid = m["snare_mid"]
    high = m["high_total"]
    air = m["air"]
    flatness = m["flatness"]
    entropy = m["entropy"]
    temporal = m["temporal"]
    tail = m["tail"]
    pitch = m["pitch"]
    crest = m["crest"]

    reason = (
        f"duration={d:.3f} events={events} sub={sub:.3f} kick={kick:.3f} "
        f"low_body={low_body:.3f} low_total={low_total:.3f} snare_mid={snare_mid:.3f} "
        f"high={high:.3f} air={air:.3f} flatness={flatness:.3f} entropy={entropy:.3f} "
        f"pitch={pitch:.3f} crest={crest:.3f} temporal={temporal:.3f} tail={tail:.3f}"
    )

    # Likely clean snare: one short event, sharp/noisy, mid/high crack, not too much kick/sub.
    if (
        d <= 1.25
        and events <= 4
        and low_total <= 0.42
        and snare_mid >= 0.22
        and high >= 0.18
        and flatness >= 0.18
        and entropy >= 0.40
        and temporal <= 0.30
        and tail <= 0.55
    ):
        return "_KEEP_LIKELY_SNARES", "likely clean snare: " + reason

    # Kick mixed into snare or mislabeled kick/body hit.
    if (sub + kick) >= 0.32 and low_total >= 0.48 and high <= 0.32 and d <= 1.60:
        return (
            "_MOVE_CANDIDATES/Kick_or_Low_Body_Contaminated_Not_Clean_Snare",
            "kick/low-body contamination suspect: " + reason,
        )

    # Tom/body percussion mixed in or mislabeled as snare.
    if low_body >= 0.35 and snare_mid <= 0.34 and high <= 0.28 and pitch >= 0.35 and d <= 1.80:
        return "_MOVE_CANDIDATES/Tom_or_Body_Pitched_Not_Clean_Snare", "tom/body pitched percussion suspect: " + reason

    # Hat/cymbal tail or noisy wash hiding in snare.
    if d >= 1.10 and high >= 0.45 and air >= 0.06 and tail >= 0.30 and low_total <= 0.38:
        return "_MOVE_CANDIDATES/Hat_Cymbal_or_Airy_Tail_Not_Clean_Snare", "hat/cymbal/airy-tail suspect: " + reason

    # Clap/rim/click instead of snare: very short, thin, high crack, little body.
    if d <= 0.65 and low_total <= 0.22 and snare_mid <= 0.30 and high >= 0.35 and tail <= 0.25:
        return (
            "_MOVE_CANDIDATES/Clap_Rim_Click_or_Thin_Hit_Not_Clean_Snare",
            "clap/rim/click/thin hit suspect: " + reason,
        )

    # Multi-hit or stacked drum mix.
    if (events >= 5 and d >= 0.75) or d >= 2.00:
        return (
            "_MOVE_CANDIDATES/Multi_Hit_Fill_Loop_or_Stacked_Drum_Mix",
            "multi-hit/fill/loop/stacked drum mix suspect: " + reason,
        )

    # Pitched ring or metallic hit hiding in snare.
    if pitch >= 0.55 and entropy <= 0.58 and flatness <= 0.42 and high >= 0.20 and d >= 0.20:
        return (
            "_MOVE_CANDIDATES/Pitched_Ring_Metallic_or_Bell_Not_Clean_Snare",
            "pitched ring/metallic suspect: " + reason,
        )

    return "_IGNORE_WEAK_OR_AMBIGUOUS", "weak or ambiguous, do not move now: " + reason


def main():
    if not SNARE_ROOT.exists():
        raise SystemExit(f"Missing snare root: {SNARE_ROOT}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    total = 0
    errors = 0
    for path in sorted(SNARE_ROOT.rglob("*")):
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
