from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

from aaron_sound_sorter.domain.roles import measured_roles_from_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SORTER = PROJECT_ROOT / "Aaron_Sound_Sorter.py"
AUDIO_DIR = PROJECT_ROOT / "tests" / "regression_audio"
OUT_ROOT = PROJECT_ROOT / "_pytest_regression_outputs" / "v31138_clean_pitched_stab"


def test_clean_low_mid_pitched_stab_is_not_measured_as_drum_one_shot() -> None:
    """A clean pitched chord stab may be front-loaded but is not a drum role."""
    roles = measured_roles_from_features(
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        feature_values={
            "sub_bass_ratio_lt_150hz": 0.000002,
            "bass_ratio_150_500hz": 0.822113,
            "mid_ratio_500_2000hz": 0.171645,
            "presence_ratio_2000_8000hz": 0.00624,
            "air_ratio_gt_8000hz": 0.0,
            "pitch_confidence": 0.44301,
            "f0_voiced_ratio": 0.133333,
            "formant_like_peak_spacing": 0.0,
            "spectral_flatness_mean": 0.071468,
            "log_crest": 2.011515,
            "attack_rise_time_norm": 0.097636,
            "temporal_centroid_ratio": 0.17062,
            "tail_energy_ratio": 0.05497,
            "log_transient_count": 1.098612,
            "loop_pitched_event_ratio": 1.0,
            "loop_percussive_event_ratio": 0.0,
            "loop_drumlike_frame_ratio": 0.0,
            "loop_tonal_to_percussive_balance": 1.0,
            "loop_sustained_tonal_frame_ratio": 1.0,
            "loop_non_event_tonal_ratio": 1.0,
            "low_peak_frequency_hz": 290.698242,
        },
    )

    assert roles.pitched_music_phrase >= 0.70
    assert roles.percussive_one_shot <= 0.15
    assert roles.evidence is not None
    assert roles.evidence["clean_pitched_hit_raw"] >= 0.80


def test_clean_mid_pitched_stab_is_not_measured_as_voiced_one_shot() -> None:
    """A clean synthetic/metallic tonal stab must not become a vocal role."""
    roles = measured_roles_from_features(
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        feature_values={
            "sub_bass_ratio_lt_150hz": 0.0,
            "bass_ratio_150_500hz": 0.021136,
            "mid_ratio_500_2000hz": 0.978407,
            "presence_ratio_2000_8000hz": 0.000016,
            "air_ratio_gt_8000hz": 0.0,
            "pitch_confidence": 0.957453,
            "f0_voiced_ratio": 1.0,
            "formant_like_peak_spacing": 0.0,
            "spectral_flatness_mean": 0.167499,
            "log_crest": 1.863259,
            "attack_rise_time_norm": 0.035729,
            "temporal_centroid_ratio": 0.090358,
            "tail_energy_ratio": 0.00961,
            "log_transient_count": 1.098612,
            "loop_pitched_event_ratio": 1.0,
            "loop_percussive_event_ratio": 0.0,
            "loop_drumlike_frame_ratio": 0.0,
            "loop_tonal_to_percussive_balance": 1.0,
            "loop_sustained_tonal_frame_ratio": 1.0,
            "loop_non_event_tonal_ratio": 1.0,
        },
    )

    assert roles.voiced_one_shot < 0.55
    assert roles.pitched_music_phrase >= 0.60


def test_short_repeated_tonal_phrase_gets_music_role_without_long_loop_gate() -> None:
    """A short multi-event tonal phrase is musical evidence, not FX/percussion."""
    roles = measured_roles_from_features(
        is_loop_like=False,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=False,
        feature_values={
            "sub_bass_ratio_lt_150hz": 0.78,
            "bass_ratio_150_500hz": 0.07,
            "mid_ratio_500_2000hz": 0.09,
            "presence_ratio_2000_8000hz": 0.04,
            "air_ratio_gt_8000hz": 0.01,
            "pitch_confidence": 0.81,
            "f0_voiced_ratio": 0.35,
            "formant_like_peak_spacing": 0.0,
            "spectral_flatness_mean": 0.27,
            "log_crest": 1.90,
            "attack_rise_time_norm": 0.13,
            "temporal_centroid_ratio": 0.51,
            "tail_energy_ratio": 0.64,
            "log_transient_count": 2.08,
            "loop_pitched_event_ratio": 1.0,
            "loop_percussive_event_ratio": 0.0,
            "loop_drumlike_frame_ratio": 0.0,
            "loop_tonal_to_percussive_balance": 1.0,
            "loop_sustained_tonal_frame_ratio": 1.0,
            "loop_non_event_tonal_ratio": 1.0,
            "low_peak_frequency_hz": 65.0,
        },
    )

    assert roles.pitched_music_phrase >= 0.58
    assert roles.pitched_music_loop == 0.0
    assert roles.percussive_one_shot == 0.0
    assert roles.evidence is not None
    assert roles.evidence["short_repeated_tonal_phrase_raw"] >= 0.58


def test_short_repeated_low_sub_kick_does_not_get_music_phrase_role() -> None:
    """A low sub kick may be pitched, but it is not a musical phrase."""
    roles = measured_roles_from_features(
        is_loop_like=False,
        is_single_event_like=False,
        is_short_hit_like=True,
        is_long=False,
        feature_values={
            "sub_bass_ratio_lt_150hz": 0.99,
            "bass_ratio_150_500hz": 0.0,
            "mid_ratio_500_2000hz": 0.0,
            "presence_ratio_2000_8000hz": 0.0,
            "air_ratio_gt_8000hz": 0.0,
            "pitch_confidence": 0.96,
            "f0_voiced_ratio": 0.0,
            "formant_like_peak_spacing": 0.0,
            "spectral_flatness_mean": 0.20,
            "log_crest": 1.71,
            "attack_rise_time_norm": 0.008,
            "temporal_centroid_ratio": 0.10,
            "tail_energy_ratio": 0.012,
            "log_transient_count": 2.20,
            "loop_pitched_event_ratio": 1.0,
            "loop_percussive_event_ratio": 0.0,
            "loop_drumlike_frame_ratio": 0.0,
            "loop_tonal_to_percussive_balance": 1.0,
            "loop_sustained_tonal_frame_ratio": 1.0,
            "loop_non_event_tonal_ratio": 1.0,
            "low_peak_frequency_hz": 32.3,
        },
    )

    assert roles.pitched_music_phrase < 0.20
    assert roles.percussive_one_shot >= 0.70
    assert roles.evidence is not None
    assert roles.evidence["short_repeated_tonal_phrase_raw"] == 0.0


def test_uploaded_chord_stab_routes_to_instruments_not_drums() -> None:
    sample = AUDIO_DIR / "Chord 1_G.wav"
    assert sample.exists(), sample
    out_dir = OUT_ROOT / "chord_1_g"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            str(SORTER),
            "sort",
            str(sample),
            str(out_dir),
            "--no-zip",
            "--workers",
            "1",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout
    manifest = out_dir / "Aaron_Sorted_Sounds_manifest.csv"
    row = list(csv.DictReader(manifest.open(encoding="utf-8")))[0]
    label = (row.get("final_label") or row.get("new_relative_path") or "").replace("\\", "/")
    assert label.startswith("Instruments/"), label
    assert label.startswith(("Instruments/Keys/", "Instruments/Synths/")), label
    assert "Drums/" not in label, label
