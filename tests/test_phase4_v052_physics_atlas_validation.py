#!/usr/bin/env python3
"""
Phase 4 / v0.5.2 physics-atlas validation tests.

These are product-level micro-tests, not broad smoke tests. They validate that:
  1. The clean v0.5.2 brain format is really 100 features and rejects old brains.
  2. Expanded physics features are populated from real audio, not left as zero-only placeholders.
  3. Folder fact-profile statistics are tabulated correctly with robust percentiles.
  4. Membership helpers use expanded physics features when approving/rejecting candidate folders.
  5. The final helper logic can choose candidate #3 when #1/#2 fail physics but #3 passes.

No filename/category rescue logic is used here. Labels are arbitrary learned folder paths.
"""

from __future__ import annotations

import math
import os
import wave
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def _candidate_sorter_paths():
    env = os.environ.get("AARON_SORTER_PATH")
    if env:
        yield Path(env)
    yield HERE.parent / "Aaron_Sound_Sorter.py"
    yield Path.cwd() / "Aaron_Sound_Sorter.py"
    yield HERE / "Aaron_Sound_Sorter.py"


def load_module():
    from aaron_sound_sorter import api

    return api


def idx(m, name: str) -> int:
    return m.FEATURE_NAMES.index(name)


def fp(m, **values):
    x = np.zeros(m.FP_SIZE, dtype=np.float32)
    for name, value in values.items():
        x[idx(m, name)] = float(value)
    return x.tolist()


def make_row(m, label: str, values: dict, i: int = 0, structure: str = "one_shot"):
    top = label.split("/", 1)[0]
    return m.FeatureRow(
        path=f"/synthetic/{label.replace('/', '_')}_{i}.wav",
        group_key=label,
        label=label,
        top=top,
        structure=structure,
        duration_sec=0.35,
        fingerprint=fp(m, **values),
        read_status="ok",
    )


def write_wav(path: Path, samples: np.ndarray, sr: int = 22050) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(samples, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(y))) if y.size else 1.0
    if peak > 1.0:
        y = y / peak
    pcm = np.clip(y, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())
    return path


def synth_tone(sr=22050, dur=0.55, freq=560.0):
    t = np.arange(int(sr * dur), dtype=np.float32) / sr
    env = np.exp(-2.2 * t)
    return 0.85 * np.sin(2 * np.pi * freq * t) * env


def synth_kick(sr=22050, dur=0.45):
    t = np.arange(int(sr * dur), dtype=np.float32) / sr
    f0 = 130.0 * np.exp(-8.0 * t) + 42.0
    phase = 2 * np.pi * np.cumsum(f0) / sr
    body = np.sin(phase) * np.exp(-10.0 * t)
    click = np.random.default_rng(1).normal(0, 0.18, size=t.shape).astype(np.float32) * np.exp(-130.0 * t)
    return 0.9 * body + click


def synth_noise_hit(sr=22050, dur=0.25):
    rng = np.random.default_rng(2)
    t = np.arange(int(sr * dur), dtype=np.float32) / sr
    return rng.normal(0, 0.7, size=t.shape).astype(np.float32) * np.exp(-25.0 * t)


BASE_PERC = {
    "spectral_flatness_mean": 0.82,
    "spectral_entropy_mean": 0.72,
    "air_ratio_gt_8000hz": 0.45,
    "zcr_mean": 0.42,
    "pitch_confidence": 0.08,
    "f0_voiced_ratio": 0.05,
    "harmonic_energy_ratio": 0.04,
    "harmonic_to_noise_ratio": 0.10,
    "body_pitch_confidence": 0.04,
    "tail_pitch_confidence": 0.03,
    "attack_noise_ratio": 0.88,
    "body_noise_ratio": 0.78,
    "tail_noise_ratio": 0.62,
    "spectral_peak_stability": 0.05,
}

BASE_TONAL = {
    "spectral_flatness_mean": 0.05,
    "spectral_entropy_mean": 0.18,
    "air_ratio_gt_8000hz": 0.01,
    "zcr_mean": 0.025,
    "pitch_confidence": 0.96,
    "f0_voiced_ratio": 0.94,
    "harmonic_energy_ratio": 0.86,
    "harmonic_to_noise_ratio": 0.90,
    "body_pitch_confidence": 0.95,
    "tail_pitch_confidence": 0.90,
    "attack_noise_ratio": 0.05,
    "body_noise_ratio": 0.04,
    "tail_noise_ratio": 0.04,
    "spectral_peak_stability": 0.88,
}

BASE_FX_NOISE = {
    "spectral_flatness_mean": 0.78,
    "spectral_entropy_mean": 0.80,
    "air_ratio_gt_8000hz": 0.18,
    "zcr_mean": 0.30,
    "pitch_confidence": 0.10,
    "f0_voiced_ratio": 0.06,
    "harmonic_energy_ratio": 0.08,
    "harmonic_to_noise_ratio": 0.12,
    "body_pitch_confidence": 0.06,
    "tail_pitch_confidence": 0.06,
    "noise_tail_decay_slope": -0.6,
    "spectral_peak_stability": 0.12,
}


def build_three_label_brain(m):
    rows = []
    labels = [
        ("Drums/Percussion/Arbitrary Perc/One Shots", BASE_PERC),
        ("FX/Noise/Arbitrary Noise/One Shots", BASE_FX_NOISE),
        ("Instruments/Wind/Arbitrary Tonal/One Shots", BASE_TONAL),
    ]
    for label, base in labels:
        for i in range(8):
            # Keep the profile tight so the membership helper can prove it uses
            # each learned physics knob as a hard envelope, not a fuzzy label hint.
            rows.append(make_row(m, label, dict(base), i=i))
    brain = m.build_brain(rows, max_centroids=2)
    # Make the product gate decisive for micro-tests. These are brain knobs, not category rules.
    brain["membership_gate_enabled"] = True
    brain["membership_min_profile_count"] = 3
    brain["membership_min_reliability"] = 0.45
    brain["membership_block_severe_count"] = 1
    brain["membership_block_weighted_severity"] = 0.25
    brain["membership_alt_max_score_gap"] = 2.0
    brain["membership_alt_min_severity_improvement"] = 0.25
    return brain


def test_v052_brain_format_is_clean_100_feature_only():
    m = load_module()
    assert m.BASE_FP_SIZE == 50
    assert m.FP_SIZE >= 100
    assert len(m.FEATURE_NAMES) == m.FP_SIZE
    assert len(set(m.FEATURE_NAMES)) == m.FP_SIZE
    assert len(m.FEATURE_WEIGHTS) == m.FP_SIZE
    assert len(m.STRUCTURE_FEATURE_WEIGHTS) == m.FP_SIZE
    for required in [
        "f0_voiced_ratio",
        "harmonic_energy_ratio",
        "body_pitch_confidence",
        "noise_tail_decay_slope",
        "low_peak_frequency_hz",
        "spectral_peak_stability",
    ]:
        assert required in m.FEATURE_NAMES

    with pytest.raises(SystemExit):
        m.validate_v052_brain_or_die(
            {
                "feature_size": 50,
                "feature_names": m.FEATURE_NAMES[:50],
                "category_fact_profiles": {},
                "feature_weights": [1.0] * 50,
                "scaler_mean": [0.0] * 50,
                "scaler_std": [1.0] * 50,
                "labels": [],
            }
        )


def test_expanded_audio_physics_are_finite_populated_and_group_covered(tmp_path):
    m = load_module()
    paths = [
        write_wav(tmp_path / "tone.wav", synth_tone()),
        write_wav(tmp_path / "kick.wav", synth_kick()),
        write_wav(tmp_path / "noise_hit.wav", synth_noise_hit()),
    ]
    fps = []
    for p in paths:
        f, duration, status = m.make_fingerprint(p)
        assert status == "ok"
        assert duration > 0.05
        assert f.shape == (m.FP_SIZE,)
        assert np.all(np.isfinite(f))
        fps.append(f)
    arr = np.vstack(fps)
    groups = {
        "harmonic_pitch_identity": m.EXTRA_PHYSICS_FEATURE_NAMES[0:10],
        "attack_body_tail": m.EXTRA_PHYSICS_FEATURE_NAMES[10:28],
        "noise_decay_shape": m.EXTRA_PHYSICS_FEATURE_NAMES[28:35],
        "low_end_source_identity": m.EXTRA_PHYSICS_FEATURE_NAMES[35:42],
        "spectral_peak_formant_shape": m.EXTRA_PHYSICS_FEATURE_NAMES[42:50],
    }
    for group, names in groups.items():
        cols = [idx(m, n) for n in names]
        vals = np.abs(arr[:, cols])
        assert float(np.max(vals)) > 1e-6, f"group stayed zero: {group}"
    # At least several appended features should vary across simple sources.
    varied = 0
    for name in m.EXTRA_PHYSICS_FEATURE_NAMES:
        col = arr[:, idx(m, name)]
        if float(np.max(col) - np.min(col)) > 1e-6:
            varied += 1
    assert varied >= 18


def test_fact_profile_tabulation_matches_numpy_percentiles_for_all_100_features():
    m = load_module()
    label = "Instruments/Test/Stats/One Shots"
    values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    rows = []
    for i, value in enumerate(values):
        rows.append(
            make_row(
                m,
                label,
                {
                    "f0_voiced_ratio": value,
                    "harmonic_energy_ratio": 1.0 - value / 2.0,
                    "body_pitch_confidence": value + 0.03,
                    "spectral_peak_stability": value + 0.01,
                },
                i=i,
            )
        )
    brain = m.build_brain(rows, max_centroids=2)
    assert brain["feature_size"] == m.FP_SIZE
    assert len(brain["feature_names"]) == m.FP_SIZE
    assert len(brain["scaler_mean"]) == m.FP_SIZE
    assert len(brain["scaler_std"]) == m.FP_SIZE
    assert len(brain["feature_weights"]) == m.FP_SIZE

    profile = brain["category_fact_profiles"][label]
    stats = profile["feature_stats"]["f0_voiced_ratio"]
    arr = np.asarray(values, dtype=float)
    expected = {
        "median": np.median(arr),
        "p10": np.percentile(arr, 10),
        "p25": np.percentile(arr, 25),
        "p75": np.percentile(arr, 75),
        "p90": np.percentile(arr, 90),
        "iqr": np.percentile(arr, 75) - np.percentile(arr, 25),
        "mad": np.median(np.abs(arr - np.median(arr))),
    }
    for key, val in expected.items():
        assert stats[key] == pytest.approx(round(float(val), 6), abs=1e-6)
    assert stats["valid_count"] == len(values)
    assert math.isfinite(float(stats["reliability"]))

    # Every feature in the new 100-wide layout has a stats slot.
    assert set(profile["feature_stats"].keys()) == set(m.FEATURE_NAMES)


def test_membership_helper_uses_representative_expanded_features_as_knobs():
    m = load_module()
    label = "Drums/Test/Zero Profile/One Shots"
    rows = [make_row(m, label, {}, i=i) for i in range(8)]
    brain = m.build_brain(rows, max_centroids=2)
    brain["membership_block_severe_count"] = 1
    brain["membership_block_weighted_severity"] = 0.10
    brain["membership_min_reliability"] = 0.45

    representative_knobs = {
        # harmonic / pitch identity
        "f0_voiced_ratio": 0.95,
        "harmonic_energy_ratio": 0.90,
        # attack/body/tail split
        "body_pitch_confidence": 0.92,
        # noise-burst / decay shape
        "high_band_decay_slope": 1.50,
        # low-end source identity
        "low_peak_frequency_hz": 180.0,
        # spectral peak / formant-like shape
        "spectral_peak_stability": 0.90,
    }
    for feature_name, value in representative_knobs.items():
        ev = m.folder_membership_evidence(brain, fp(m, **{feature_name: value}), label)
        assert ev["enabled"] is True
        assert ev["blocked"] is True, f"membership ignored {feature_name}: {ev}"
        joined = " | ".join(ev.get("severe_features", []) + ev.get("violation_features", []))
        assert feature_name in joined


def test_helper_reviews_instead_of_cross_family_switch_when_one_and_two_fail_physics():
    m = load_module()
    brain = build_three_label_brain(m)
    drums = "Drums/Percussion/Arbitrary Perc/One Shots"
    fx = "FX/Noise/Arbitrary Noise/One Shots"
    tonal = "Instruments/Wind/Arbitrary Tonal/One Shots"
    tonal_sample = fp(m, **BASE_TONAL)

    drum_ev = m.folder_membership_evidence(brain, tonal_sample, drums)
    fx_ev = m.folder_membership_evidence(brain, tonal_sample, fx)
    tonal_ev = m.folder_membership_evidence(brain, tonal_sample, tonal)
    assert drum_ev["blocked"] is True
    assert fx_ev["blocked"] is True
    assert tonal_ev["blocked"] is False

    final_label, final_top, action, meta = m.learned_membership_gate_decision(
        brain=brain,
        fingerprint=tonal_sample,
        final_label=drums,
        final_top="Drums",
        top5=[(drums, 0.00), (fx, 0.25), (tonal, 0.80)],
        predicted_top="Drums",
    )
    # v0.6.4: membership fallback may switch inside the same broad family,
    # but a Drums -> Instruments repair is a cross-family reclassification.
    # That must go to review unless the brain explicitly opts into that unsafe mode.
    assert action == "review"
    assert final_top == "_TO_REVIEW"
    assert "review_reason" in meta


def test_helper_can_choose_candidate_three_inside_same_family_when_one_and_two_fail_physics():
    m = load_module()
    brain = build_three_label_brain(m)
    bad_one = "Instruments/Bad Perc-Like/One Shots"
    bad_two = "Instruments/Bad Noise-Like/One Shots"
    tonal = "Instruments/Wind/Arbitrary Tonal/One Shots"
    # Re-map tops so this is a same-family repair, not a broad-family jump.
    brain["top_by_label"][bad_one] = "Instruments"
    brain["top_by_label"][bad_two] = "Instruments"
    brain["top_by_label"][tonal] = "Instruments"
    # Reuse existing profiles by copying the learned ranges from the synthetic labels.
    brain["category_fact_profiles"][bad_one] = brain["category_fact_profiles"][
        "Drums/Percussion/Arbitrary Perc/One Shots"
    ]
    brain["category_fact_profiles"][bad_two] = brain["category_fact_profiles"]["FX/Noise/Arbitrary Noise/One Shots"]
    brain["category_fact_profiles"][tonal] = brain["category_fact_profiles"][
        "Instruments/Wind/Arbitrary Tonal/One Shots"
    ]
    tonal_sample = fp(m, **BASE_TONAL)

    final_label, final_top, action, meta = m.learned_membership_gate_decision(
        brain=brain,
        fingerprint=tonal_sample,
        final_label=bad_one,
        final_top="Instruments",
        top5=[(bad_one, 0.00), (bad_two, 0.25), (tonal, 0.80)],
        predicted_top="Instruments",
    )
    assert action == "switch"
    assert final_label == tonal
    assert final_top == "Instruments"
    assert "alternative_reason" in meta


def test_old_exact_leaf_gets_review_when_no_candidate_passes_membership():
    m = load_module()
    brain = build_three_label_brain(m)
    drums = "Drums/Percussion/Arbitrary Perc/One Shots"
    impossible = fp(
        m,
        pitch_confidence=0.98,
        f0_voiced_ratio=0.98,
        harmonic_energy_ratio=0.99,
        spectral_flatness_mean=0.99,
        air_ratio_gt_8000hz=0.99,
        zcr_mean=0.99,
        body_pitch_confidence=0.99,
        spectral_peak_stability=0.99,
        low_peak_frequency_hz=800.0,
    )
    final_label, final_top, action, meta = m.learned_membership_gate_decision(
        brain=brain,
        fingerprint=impossible,
        final_label=drums,
        final_top="Drums",
        top5=[(drums, 0.00)],
        predicted_top="Drums",
    )
    assert action == "review"
    assert final_top == "_TO_REVIEW"
    assert "Learned Folder Membership Failed" in final_label
    assert int(meta.get("severe_count", 0)) >= 1


def _find_optional_kick_fixtures():
    names = ["COY Kick 7.wav", "Ed Kick 5.wav", "Ed Kick 7.wav"]
    roots = [
        Path(os.environ.get("AARON_KICK_FIXTURE_DIR", "")) if os.environ.get("AARON_KICK_FIXTURE_DIR") else None,
        Path.cwd(),
        Path.cwd() / "tests" / "fixtures",
        Path.home() / "Downloads",
    ]
    found = []
    for root in [r for r in roots if r is not None]:
        for name in names:
            p = root / name
            if p.exists():
                found.append(p)
    # De-dupe while preserving order.
    out = []
    seen = set()
    for p in found:
        if str(p) not in seen:
            out.append(p)
            seen.add(str(p))
    return out


def test_optional_real_kick_fixtures_have_low_end_physics_when_available():
    from tests.synthetic_audio_fixtures import ensure_optional_kick_fixtures

    ensure_optional_kick_fixtures()
    m = load_module()
    paths = _find_optional_kick_fixtures()
    if len(paths) < 2:
        pytest.skip(
            "real kick fixtures not found locally; set AARON_KICK_FIXTURE_DIR or place files in Downloads/tests/fixtures"
        )
    for path in paths[:3]:
        f, duration, status = m.make_fingerprint(path)
        assert status == "ok"
        assert duration > 0.03
        assert f.shape == (m.FP_SIZE,)
        assert np.all(np.isfinite(f))
        # Real kicks should at least populate the low-end source identity section.
        low_group = [
            "low_peak_frequency_hz",
            "low_peak_bandwidth_hz",
            "sub_attack_time_ms",
            "sub_decay_time_ms",
            "sub_sustain_ratio",
        ]
        assert max(abs(float(f[idx(m, name)])) for name in low_group) > 1e-6


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", __file__]))
