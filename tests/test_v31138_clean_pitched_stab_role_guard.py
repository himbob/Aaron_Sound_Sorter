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
