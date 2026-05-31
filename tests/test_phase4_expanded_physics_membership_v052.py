#!/usr/bin/env python3
"""
Phase 4 v0.5.2 expanded-physics and membership-gate regression tests.

These tests are intentionally small. They prove the new physics atlas is real,
gets stored in a rebuilt baby brain, and is used by membership/fact machinery
instead of existing only as unused column names.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest


def _module_path() -> Path:
    env = os.environ.get("SORTER_UNDER_TEST", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    # Prefer the active project-local sorter first.  Parent copies are often stale
    # artifacts from previous bundles.
    for c in (
        here.parent / "Aaron_Sound_Sorter.py",
        Path.cwd() / "Aaron_Sound_Sorter.py",
        here.parent.parent / "Aaron_Sound_Sorter.py",
    ):
        if c.exists():
            return c.resolve()
    raise FileNotFoundError("Set SORTER_UNDER_TEST=/path/to/Aaron_Sound_Sorter.py")


def load_module():
    from aaron_sound_sorter import api

    return api


def write_wav(path: Path, y: np.ndarray, sr: int = 22050) -> None:
    y = np.asarray(y, dtype=np.float32)
    y = y / max(1e-9, float(np.max(np.abs(y))))
    data = (np.clip(y, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


def sine(sr=22050, dur=0.5, hz=440.0):
    t = np.arange(int(sr * dur), dtype=np.float32) / float(sr)
    env = np.exp(-t * 1.2).astype(np.float32)
    return 0.8 * np.sin(2 * np.pi * hz * t).astype(np.float32) * env


def noise_hit(sr=22050, dur=0.35, seed=1):
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float32) / float(sr)
    env = np.exp(-t * 22.0).astype(np.float32)
    return rng.normal(0, 1, n).astype(np.float32) * env


def kick_like(sr=22050, dur=0.45, seed=1):
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float32) / float(sr)
    f0 = 95.0 * np.exp(-t * 18.0) + 42.0
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sub = np.sin(phase).astype(np.float32) * np.exp(-t * 7.0).astype(np.float32)
    click = rng.normal(0, 1, n).astype(np.float32) * np.exp(-t * 90.0).astype(np.float32) * 0.18
    return sub + click


def idx(m, name: str) -> int:
    return m.FEATURE_NAMES.index(name)


def row(m, path: str, label: str, structure: str, features: dict[str, float], duration: float = 0.5):
    public = label.strip("/")
    internal = f"{public}/{'Loops' if structure == 'loop' else 'One Shots'}"
    fp = np.zeros(m.FP_SIZE, dtype=np.float32)
    for name, value in features.items():
        assert name in m.FEATURE_NAMES, name
        fp[idx(m, name)] = float(value)
    return m.FeatureRow(
        path=path,
        group_key=public,
        label=internal,
        top=public.split("/", 1)[0],
        structure=structure,
        duration_sec=duration,
        fingerprint=fp.tolist(),
        read_status="ok",
    )


def test_expanded_physics_columns_are_filled_from_audio(tmp_path):
    m = load_module()
    assert m.FP_SIZE >= 100
    assert len(m.FEATURE_NAMES) == m.FP_SIZE
    required = [
        "f0_median_hz",
        "f0_voiced_ratio",
        "harmonic_energy_ratio",
        "body_pitch_confidence",
        "body_flatness",
        "noise_burst_duration_ms",
        "low_peak_frequency_hz",
        "sub_decay_time_ms",
        "spectral_peak_count",
        "top1_peak_frequency_hz",
    ]
    for name in required:
        assert name in m.FEATURE_NAMES

    tone_path = tmp_path / "tone.wav"
    hit_path = tmp_path / "noise_hit.wav"
    write_wav(tone_path, sine(hz=440.0))
    write_wav(hit_path, noise_hit(seed=2))

    tone_fp, tone_dur, tone_status = m.make_fingerprint_safe(tone_path)
    hit_fp, hit_dur, hit_status = m.make_fingerprint_safe(hit_path)
    assert tone_status == "ok"
    assert hit_status == "ok"
    assert tone_fp.shape[0] == m.FP_SIZE
    assert hit_fp.shape[0] == m.FP_SIZE

    assert 380.0 <= float(tone_fp[idx(m, "f0_median_hz")]) <= 500.0
    assert float(tone_fp[idx(m, "f0_voiced_ratio")]) >= 0.75
    assert float(tone_fp[idx(m, "harmonic_energy_ratio")]) >= 0.70
    assert float(tone_fp[idx(m, "body_pitch_confidence")]) >= 0.70
    assert float(tone_fp[idx(m, "body_flatness")]) < float(hit_fp[idx(m, "body_flatness")])
    assert float(tone_fp[idx(m, "top1_peak_frequency_hz")]) > 300.0
    assert float(hit_fp[idx(m, "noise_burst_duration_ms")]) > 0.0


def test_baby_brain_stores_expanded_physics_fact_profiles_and_uses_membership():
    m = load_module()
    rows: list[object] = []
    # Learned noisy percussion folder: high flatness/noise/ZCR, low harmonicity.
    for i in range(24):
        rows.append(
            row(
                m,
                f"/train/noise_hit_{i}.wav",
                "Drums/Percussion/Noisy Hit",
                "one_shot",
                {
                    "spectral_flatness_mean": 0.62 + (i % 3) * 0.01,
                    "spectral_entropy_mean": 0.72,
                    "zcr_mean": 0.42,
                    "air_ratio_gt_8000hz": 0.30,
                    "presence_ratio_2000_8000hz": 0.42,
                    "pitch_confidence": 0.05,
                    "f0_voiced_ratio": 0.0,
                    "harmonic_energy_ratio": 0.04,
                    "body_pitch_confidence": 0.05,
                    "body_flatness": 0.66,
                    "body_entropy": 0.74,
                    "body_zcr": 0.42,
                    "body_high_ratio": 0.72,
                    "body_noise_ratio": 0.88,
                    "harmonic_to_noise_ratio": 0.02,
                    "spectral_peak_count": 18,
                },
            )
        )
    # Learned tonal instrument folder: stable pitch/harmonics, low noise.
    for i in range(24):
        rows.append(
            row(
                m,
                f"/train/tonal_{i}.wav",
                "Instruments/Wind/Tonal Short",
                "one_shot",
                {
                    "spectral_flatness_mean": 0.03,
                    "spectral_entropy_mean": 0.25,
                    "zcr_mean": 0.03,
                    "air_ratio_gt_8000hz": 0.002,
                    "presence_ratio_2000_8000hz": 0.03,
                    "pitch_confidence": 0.96,
                    "f0_median_hz": 440.0 + (i % 3),
                    "f0_voiced_ratio": 1.0,
                    "f0_stability_cents": 8.0,
                    "harmonic_energy_ratio": 0.91,
                    "body_pitch_confidence": 0.97,
                    "tail_pitch_confidence": 0.90,
                    "body_flatness": 0.035,
                    "body_entropy": 0.24,
                    "body_zcr": 0.03,
                    "body_high_ratio": 0.03,
                    "body_noise_ratio": 0.08,
                    "harmonic_to_noise_ratio": 4.2,
                    "spectral_peak_count": 3,
                    "top1_peak_frequency_hz": 440.0,
                },
            )
        )

    brain = m.build_brain(rows, max_centroids=3)
    assert brain["feature_size"] == m.FP_SIZE
    assert "body_pitch_confidence" in brain["feature_names"]
    assert "Drums/Percussion/Noisy Hit/One Shots" in brain["category_fact_profiles"]
    stats = brain["category_fact_profiles"]["Drums/Percussion/Noisy Hit/One Shots"]["feature_stats"]
    assert "body_pitch_confidence" in stats
    assert "harmonic_energy_ratio" in stats

    mystery = row(
        m,
        "/mystery/flute_like.wav",
        "Unknown",
        "one_shot",
        {
            "spectral_flatness_mean": 0.03,
            "spectral_entropy_mean": 0.24,
            "zcr_mean": 0.03,
            "air_ratio_gt_8000hz": 0.002,
            "presence_ratio_2000_8000hz": 0.02,
            "pitch_confidence": 0.98,
            "f0_median_hz": 445.0,
            "f0_voiced_ratio": 1.0,
            "f0_stability_cents": 6.0,
            "harmonic_energy_ratio": 0.94,
            "body_pitch_confidence": 0.98,
            "tail_pitch_confidence": 0.92,
            "body_flatness": 0.03,
            "body_entropy": 0.22,
            "body_zcr": 0.025,
            "body_high_ratio": 0.025,
            "body_noise_ratio": 0.06,
            "harmonic_to_noise_ratio": 4.5,
            "spectral_peak_count": 2,
            "top1_peak_frequency_hz": 445.0,
        },
    ).fingerprint

    bad_label = "Drums/Percussion/Noisy Hit/One Shots"
    good_label = "Instruments/Wind/Tonal Short/One Shots"
    bad_ev = m.folder_membership_evidence(brain, mystery, bad_label)
    good_ev = m.folder_membership_evidence(brain, mystery, good_label)
    assert bad_ev["enabled"]
    assert bad_ev["blocked"], bad_ev
    assert int(bad_ev["severe_count"]) >= 3, bad_ev
    assert good_ev["enabled"]
    assert not good_ev["blocked"], good_ev
    assert float(good_ev["membership_score"]) > float(bad_ev["membership_score"])

    # The actual product gate should not let the bad nearest label auto-place.
    label, top, action, ev = m.learned_membership_gate_decision(
        brain, mystery, bad_label, "Drums", [(bad_label, 0.1), (good_label, 0.2)], "Drums"
    )
    assert action in {"switch", "review"}
    assert label != bad_label


def test_real_uploaded_kick_files_collect_low_end_physics_and_make_a_baby_brain(tmp_path):
    m = load_module()
    from tests.synthetic_audio_fixtures import ensure_optional_kick_fixtures

    ensure_optional_kick_fixtures()
    kick_paths = [
        Path.cwd() / "tests" / "fixtures" / "COY Kick 7.wav",
        Path.cwd() / "tests" / "fixtures" / "Ed Kick 5.wav",
        Path.cwd() / "tests" / "fixtures" / "Ed Kick 7.wav",
    ]
    present = [p for p in kick_paths if p.exists()]
    if len(present) < 2:
        pytest.skip("real uploaded kick fixtures are optional and are not present in this bundle")

    rows = []
    for _i, p in enumerate(present):
        fp_arr, dur, status = m.make_fingerprint_safe(p)
        assert status == "ok"
        assert fp_arr.shape[0] == m.FP_SIZE
        assert float(fp_arr[idx(m, "low_peak_frequency_hz")]) > 0.0
        assert float(fp_arr[idx(m, "sub_decay_time_ms")]) >= 0.0
        rows.append(
            m.FeatureRow(
                path=str(p),
                group_key="Drums/Kick Drums/Test Kick",
                label="Drums/Kick Drums/Test Kick/One Shots",
                top="Drums",
                structure="one_shot",
                duration_sec=dur,
                fingerprint=fp_arr.tolist(),
                read_status="ok",
            )
        )

    # Add synthetic non-kick noisy hits so the baby brain has a rival folder.
    for i in range(8):
        rows.append(
            row(
                m,
                f"/synthetic/noisy_{i}.wav",
                "Drums/Percussion/Noisy Hit",
                "one_shot",
                {
                    "spectral_flatness_mean": 0.70,
                    "spectral_entropy_mean": 0.74,
                    "zcr_mean": 0.45,
                    "body_noise_ratio": 0.90,
                    "body_high_ratio": 0.65,
                    "low_peak_frequency_hz": 900.0,
                    "sub_sustain_ratio": 0.02,
                    "pitch_confidence": 0.05,
                    "body_pitch_confidence": 0.04,
                },
            )
        )

    brain = m.build_brain(rows, max_centroids=3)
    assert brain["feature_size"] == m.FP_SIZE
    kick_profile = brain["category_fact_profiles"]["Drums/Kick Drums/Test Kick/One Shots"]
    assert "low_peak_frequency_hz" in kick_profile["feature_stats"]
    assert "sub_decay_time_ms" in kick_profile["feature_stats"]

    # Re-test one real kick against the rebuilt baby brain. It should not be blocked
    # by its own learned kick membership profile.
    test_fp = rows[0].fingerprint
    ev = m.folder_membership_evidence(brain, test_fp, "Drums/Kick Drums/Test Kick/One Shots")
    assert ev.get("enabled")
    assert not ev.get("blocked"), ev


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
