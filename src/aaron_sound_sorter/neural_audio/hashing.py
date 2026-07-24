"""Content hashing helpers used by caches and leakage guards."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash file bytes without using its name or source folder as evidence."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def decoded_audio_sha256(path: Path) -> str:
    """Hash decoded PCM content to catch lossless container-level duplicates.

    The fingerprint includes sample rate, channel count, frame count, and
    canonical little-endian float32 PCM. It can therefore identify the same
    lossless audio stored in different containers while remaining independent
    of file and folder names. Lossy or transformed near-duplicates still
    require a later perceptual-fingerprint audit.

    Args:
        path: Audio file to decode.

    Returns:
        SHA-256 digest of canonical decoded audio content.

    Raises:
        OSError: If the audio cannot be opened.
        RuntimeError: If libsndfile cannot decode the audio.
    """
    audio, sample_rate = sf.read(Path(path), always_2d=True, dtype="float32")
    canonical = np.nan_to_num(
        np.asarray(audio, dtype="<f4"),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )
    digest = hashlib.sha256()
    digest.update(b"aaron-decoded-audio-v1\0")
    digest.update(int(sample_rate).to_bytes(8, byteorder="little", signed=False))
    digest.update(int(canonical.shape[0]).to_bytes(8, byteorder="little", signed=False))
    digest.update(int(canonical.shape[1]).to_bytes(4, byteorder="little", signed=False))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def normalized_audio_sha256(path: Path) -> str:
    """Hash gain-normalized, silence-trimmed mono PCM for duplicate grouping.

    This secondary digest catches practical derived copies that differ only by
    a lossless container, channel layout, uniform gain, or leading/trailing
    silence. It is deliberately conservative: sample rate remains part of the
    identity, and lossy or time-stretched copies require later perceptual
    review.

    Args:
        path: Audio file to decode.

    Returns:
        SHA-256 digest of normalized quantized PCM.

    Raises:
        OSError: If the audio cannot be opened.
        RuntimeError: If libsndfile cannot decode the audio.
        ValueError: If the decoded audio has no frames.
    """
    audio, sample_rate = sf.read(Path(path), always_2d=True, dtype="float32")
    canonical = np.nan_to_num(
        np.asarray(audio, dtype=np.float32),
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    )
    if canonical.shape[0] == 0:
        raise ValueError("decoded audio has no frames")
    mono = np.mean(canonical, axis=1)
    peak = float(np.max(np.abs(mono)))
    if peak > 1e-8:
        mono = mono / peak
        active = np.flatnonzero(np.abs(mono) >= 1e-4)
        if active.size:
            mono = mono[int(active[0]) : int(active[-1]) + 1]
    quantized = np.rint(np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2")
    digest = hashlib.sha256()
    digest.update(b"aaron-normalized-audio-v1\0")
    digest.update(int(sample_rate).to_bytes(8, byteorder="little", signed=False))
    digest.update(int(quantized.shape[0]).to_bytes(8, byteorder="little", signed=False))
    digest.update(quantized.tobytes(order="C"))
    return digest.hexdigest()
