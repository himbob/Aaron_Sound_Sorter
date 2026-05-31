"""Regression coverage for dependency-free IEEE-float WAV decoding."""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np

from aaron_sound_sorter.features import make_fingerprint_safe
from aaron_sound_sorter.training_labels import read_audio


def write_ieee_float_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write a minimal RIFF/WAVE_FORMAT_IEEE_FLOAT file."""
    data = np.asarray(samples, dtype="<f4")
    channels = int(data.shape[1])
    payload = data.tobytes()
    fmt_chunk = struct.pack(
        "<HHIIHH",
        3,
        channels,
        sample_rate,
        sample_rate * channels * 4,
        channels * 4,
        32,
    )
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", 4 + (8 + len(fmt_chunk)) + (8 + len(payload)))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt_chunk))
        + fmt_chunk
        + b"data"
        + struct.pack("<I", len(payload))
        + payload
    )


def test_ieee_float_wav_reads_without_soundfile_dependency(tmp_path: Path) -> None:
    sr = 48_000
    t = np.linspace(0.0, 0.35, int(sr * 0.35), endpoint=False, dtype=np.float32)
    tone = 0.45 * np.sin(2.0 * math.pi * 220.0 * t)
    stereo = np.column_stack([tone, tone]).astype(np.float32)
    wav_path = tmp_path / "float32_fixture.wav"
    write_ieee_float_wav(wav_path, stereo, sr)

    audio, read_sr = read_audio(wav_path)
    fingerprint, duration, status = make_fingerprint_safe(wav_path)

    assert read_sr == sr
    assert audio.shape == stereo.shape
    assert status == "ok"
    assert duration > 0.30
    assert fingerprint.shape[0] > 50
