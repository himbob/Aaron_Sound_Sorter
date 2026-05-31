"""Synthetic test fixtures for audio regression tests.

This module generates lightweight WAV files and support fixtures when real
external audio assets are absent. It is intentionally conservative and only
creates the named files needed by skipped regression tests.
"""

from __future__ import annotations

import json
import wave
import zipfile
from pathlib import Path

import numpy as np

from aaron_sound_sorter import api as mod

SAMPLE_RATE = 22050


def _normalize(samples: np.ndarray) -> np.ndarray:
    y = np.asarray(samples, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(y))) if y.size else 1.0
    if peak < 1e-9:
        peak = 1.0
    return (y / peak) * 0.95


def write_wav(path: Path, samples: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = _normalize(samples)
    pcm = np.clip(y, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm16.tobytes())
    return path


def _fade_in_out(samples: np.ndarray, sr: int, fade_seconds: float = 0.02) -> np.ndarray:
    length = len(samples)
    fade_length = min(int(sr * fade_seconds), length // 10)
    if fade_length <= 0:
        return samples
    env = np.ones(length, dtype=np.float32)
    env[:fade_length] = np.linspace(0.0, 1.0, fade_length, dtype=np.float32)
    env[-fade_length:] = np.linspace(1.0, 0.0, fade_length, dtype=np.float32)
    return samples * env


def _sine(freq: float, dur: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(sr * dur), dtype=np.float32) / sr
    return np.sin(2.0 * np.pi * freq * t)


def synth_tone(freq: float = 440.0, dur: float = 1.8, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(sr * dur), dtype=np.float32) / sr
    env = np.sin(np.pi * t / dur) ** 2
    tone = np.sin(2.0 * np.pi * freq * t)
    tone *= np.exp(-1.3 * t)
    return _normalize(tone * env)


def synth_harmonic_phrase(freq: float = 440.0, dur: float = 1.8) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * dur), dtype=np.float32) / SAMPLE_RATE
    vibrato = 5.0 * np.sin(2.0 * np.pi * 5.0 * t)
    phase = 2.0 * np.pi * np.cumsum((freq + vibrato) / SAMPLE_RATE)
    harmonics = np.sin(phase)
    harmonics += 0.55 * np.sin(2.0 * phase)
    harmonics += 0.25 * np.sin(3.0 * phase)
    env = np.sin(np.pi * t / dur) ** 2 * np.exp(-0.8 * t)
    return _normalize(harmonics * env)


def synth_voice_like(freq: float = 220.0, dur: float = 1.5) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * dur), dtype=np.float32) / SAMPLE_RATE
    vibrato = 6.0 * np.sin(2.0 * np.pi * 4.0 * t)
    phase = 2.0 * np.pi * np.cumsum((freq + vibrato) / SAMPLE_RATE)
    base = np.sin(phase)
    harmonic = 0.5 * np.sin(2.0 * phase)
    formants = 0.25 * np.sin(2.0 * np.pi * (freq * 3.2 + 80.0) * t) * np.exp(-(((t - 0.2) * 5.0) ** 2))
    noise = np.random.default_rng(0).normal(0, 0.07, size=t.shape).astype(np.float32)
    env = np.sin(np.pi * t / dur) ** 2 * np.minimum(1.0, 4.0 * t)
    return _normalize((base + harmonic * 0.7 + formants + noise * 0.4) * env)


def synth_guitar_like(freq: float = 110.0, dur: float = 2.4) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * dur), dtype=np.float32) / SAMPLE_RATE
    pluck = np.sin(2.0 * np.pi * freq * t) * np.exp(-5.0 * t)
    harmonic = 0.5 * np.sin(2.0 * np.pi * 2.0 * freq * t) * np.exp(-6.0 * t)
    noise = np.random.default_rng(1).normal(0, 0.03, size=t.shape).astype(np.float32)
    env = np.sin(np.pi * t / dur) ** 2
    return _normalize((pluck + harmonic + noise * 0.25) * env)


def synth_piano_like(freq: float = 392.0, dur: float = 2.5) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * dur), dtype=np.float32) / SAMPLE_RATE
    chord = np.sin(2.0 * np.pi * freq * t)
    chord += 0.75 * np.sin(2.0 * np.pi * (freq * 1.25) * t)
    chord += 0.5 * np.sin(2.0 * np.pi * (freq * 1.5) * t)
    env = np.minimum(1.0, 4.0 * t) * np.exp(-1.8 * t)
    noise = np.random.default_rng(2).normal(0, 0.02, size=t.shape).astype(np.float32)
    return _normalize((chord + noise) * env)


def synth_kick(dur: float = 0.45) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * dur), dtype=np.float32) / SAMPLE_RATE
    f0 = 110.0 * np.exp(-6.0 * t) + 40.0
    phase = 2.0 * np.pi * np.cumsum(f0 / SAMPLE_RATE)
    body = np.sin(phase) * np.exp(-12.0 * t)
    click = np.random.default_rng(3).normal(0, 0.08, size=t.shape).astype(np.float32) * np.exp(-120.0 * t)
    return _normalize((body + click) * np.exp(-1.6 * t))


def synth_hat(dur: float = 0.16) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * dur), dtype=np.float32) / SAMPLE_RATE
    noise = np.random.default_rng(4).normal(0, 0.7, size=t.shape).astype(np.float32)
    return _normalize(noise * np.exp(-75.0 * t))


def synth_snare(dur: float = 0.24) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * dur), dtype=np.float32) / SAMPLE_RATE
    noise = np.random.default_rng(5).normal(0, 0.5, size=t.shape).astype(np.float32)
    tone = np.sin(2.0 * np.pi * 180.0 * t) * np.exp(-6.0 * t)
    return _normalize((noise + tone * 0.45) * np.exp(-10.0 * t))


def synth_fx_hit(dur: float = 2.2) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * dur), dtype=np.float32) / SAMPLE_RATE
    sweep = np.sin(2.0 * np.pi * (200.0 + 1200.0 * t) * t)
    noise = np.random.default_rng(6).normal(0, 0.25, size=t.shape).astype(np.float32)
    env = np.sin(np.pi * t / dur) ** 2 * np.exp(-1.4 * t)
    return _normalize((sweep * 0.6 + noise * 0.4) * env)


def synth_percussion_hit(dur: float = 0.32) -> np.ndarray:
    """Build a clearly percussive broadband one-shot fixture.

    The earlier fallback mixed a tonal kick body with a hat and could look like
    a voiced/pitched one-shot to the low-level panels.  Regression stand-ins for
    numbered percussion files need to be unambiguously drum/percussion-like so
    the tests validate routing instead of a bad synthetic fixture.
    """
    t = np.arange(int(SAMPLE_RATE * dur), dtype=np.float32) / SAMPLE_RATE
    rng = np.random.default_rng(8)
    noise = rng.normal(0, 0.85, size=t.shape).astype(np.float32)
    snap_env = np.exp(-42.0 * t)
    body = np.sin(2.0 * np.pi * 185.0 * t) * np.exp(-22.0 * t)
    click = rng.normal(0, 1.0, size=t.shape).astype(np.float32) * np.exp(-180.0 * t)
    return _normalize(noise * snap_env + 0.25 * body + 0.35 * click)


def synth_drum_loop(name: str, dur: float = 2.4) -> np.ndarray:
    sample = np.zeros(int(SAMPLE_RATE * dur), dtype=np.float32)
    beat_length = int(SAMPLE_RATE * 0.6)
    for i in range(0, len(sample), beat_length):
        pos = min(i, len(sample) - 1)
        pad = synth_kick(0.28)
        sample[pos : pos + len(pad)] += pad[: max(0, len(sample) - pos)]
        sn = synth_snare(0.18)
        pos2 = pos + beat_length // 2
        if pos2 < len(sample):
            sample[pos2 : pos2 + len(sn)] += sn[: max(0, len(sample) - pos2)]
        hh = synth_hat(0.12)
        pos3 = pos + beat_length // 4
        if pos3 < len(sample):
            sample[pos3 : pos3 + len(hh)] += hh[: max(0, len(sample) - pos3)]
    return _normalize(_fade_in_out(sample, SAMPLE_RATE, 0.05))


def _classify_name(name: str) -> str:
    key = name.lower()
    if any(token in key for token in ["piano", "key", "grand piano", "keyboard"]):
        return "piano"
    if any(token in key for token in ["guitar"]):
        return "guitar"
    if any(token in key for token in ["sax", "brass", "reed", "woodwind"]):
        return "sax"
    if any(token in key for token in ["voice", "vocal", "stabs", "shout", "rap"]):
        return "voice"
    if any(token in key for token in ["drum loop", "drum_loop", "drums loop", "drum loops", "beat", "loop"]):
        return "drum_loop"
    if any(
        token in key for token in ["kick", "snare", "hat", "cymbal", "clap", "tambourine", "bongo", "percussion", "hit"]
    ):
        return "drum_hit"
    if any(token in key for token in ["fx", "riser", "noise", "transition", "wet", "ambience", "police", "outlaw"]):
        return "fx"
    return "tone"


def ensure_fixture_wav(path: Path) -> Path:
    if path.exists():
        return path
    category = _classify_name(path.name)
    if category == "piano":
        samples = synth_piano_like(392.0, dur=2.4)
    elif category == "guitar":
        samples = synth_guitar_like(110.0, dur=2.8)
    elif category == "sax":
        samples = synth_harmonic_phrase(440.0, dur=2.4)
    elif category == "voice":
        samples = synth_voice_like(220.0, dur=1.6)
    elif category == "drum_loop":
        samples = synth_drum_loop(path.name)
    elif category == "drum_hit":
        samples = synth_percussion_hit(dur=0.28)
    elif category == "fx":
        samples = synth_fx_hit(dur=2.1)
    else:
        samples = synth_tone(440.0, dur=1.8)
    return write_wav(path, samples)


def ensure_wav_dir(directory: Path, names: list[str]) -> None:
    if directory.exists() and any(directory.rglob("*.wav")):
        return
    for name in names:
        ensure_fixture_wav(directory / name)


def ensure_optional_kick_fixtures() -> None:
    root = Path.cwd() / "tests" / "fixtures"
    ensure_wav_dir(root, ["COY Kick 7.wav", "Ed Kick 5.wav", "Ed Kick 7.wav"])


def ensure_scratch_brain_zip(fixture_dir: Path) -> Path:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    zip_path = fixture_dir / "phase3_pure_scratch_brain.json.zip"
    if zip_path.exists():
        return zip_path
    audio_names = ["Piano_G.wav", "Acoustic Guitar Rhythm_25_KeyDm_100bpm.wav"]
    for name in audio_names:
        ensure_fixture_wav(fixture_dir / name)

    rows = []
    for name, label, structure in [
        ("Piano_G.wav", "Instruments/Piano/One Shots", "one_shot"),
        ("Acoustic Guitar Rhythm_25_KeyDm_100bpm.wav", "Instruments/Guitar/Acoustic Guitar/Loops", "loop"),
    ]:
        path = fixture_dir / name
        fp, duration, status = mod.make_fingerprint_safe(path)
        assert status == "ok"
        rows.append(
            mod.FeatureRow(
                path=str(path),
                group_key=label,
                label=label,
                top="Instruments",
                structure=structure,
                duration_sec=duration,
                fingerprint=fp.tolist(),
                read_status="ok",
            )
        )
    brain = mod.build_brain(rows, max_centroids=3)
    brain_path = fixture_dir / "phase3_pure_scratch_brain.json"
    brain_path.write_text(json.dumps(brain), encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(brain_path, brain_path.name)
    brain_path.unlink()
    return zip_path


def ensure_synthetic_ledger_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    payload = {
        "classification_categories": {
            "test_category": {
                "primary_folder_path": "Drums/Percussion/Generic Percussion",
                "label_family": "Drums",
                "display_name": "Generic Percussion",
                "source_name_template": "Generic Percussion",
                "classifier_label_hints": ["percussion", "drum", "hit"],
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
