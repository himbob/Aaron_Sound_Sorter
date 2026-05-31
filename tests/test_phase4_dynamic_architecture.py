#!/usr/bin/env python3
"""
Fast architecture regression tests for Aaron Sound Sorter Stage 4 v0.4.82+.

Purpose:
  These tests are intentionally small. Most tests use synthetic fingerprints so
  future code edits can be checked quickly without full-library runs. The final
  optional real-sample smoke test auto-finds Aaron's local sample ZIPs on his Mac
  or the uploaded project attachments in the AI sandbox.

How to run from the project root:
  python3 -m pytest -q tests/test_phase4_dynamic_architecture.py

Direct-run fallback also works now:
  python3 tests/test_phase4_dynamic_architecture.py

Useful environment variables:
  SORTER_UNDER_TEST=/path/to/Aaron_Sound_Sorter.py
  AARON_DRUM_ZIP=/path/to/one_shot_percussive_sounds.zip
  AARON_META_ZIP=/path/to/sound_info_analysis.json.zip
  AARON_SAMPLE_ROOT=/path/to/sample/root

If the sample ZIPs are not found, the optional real-sample smoke test falls back to generated artifacts.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import subprocess
import sys
import wave
import zipfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

csv.field_size_limit(sys.maxsize)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _project_root_guess() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "tests":
        return here.parent.parent
    return here.parent


def _module_path() -> Path:
    env = os.environ.get("SORTER_UNDER_TEST", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    candidates = [
        # Prefer the explicit project-local file or an environment override.
        here.parent.parent / "Aaron_Sound_Sorter.py",
        here.parent / "Aaron_Sound_Sorter.py",
        Path.cwd() / "Aaron_Sound_Sorter.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find Aaron_Sound_Sorter.py. Set SORTER_UNDER_TEST=/path/to/Aaron_Sound_Sorter.py"
    )


def _first_existing_file(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        try:
            p = candidate.expanduser()
            if p.is_file():
                return p.resolve()
        except Exception:
            continue
    return None


def _sample_zip_candidates(filename: str, env_var: str) -> list[Path]:
    project_root = _project_root_guess()
    cwd = Path.cwd()
    sample_root_env = os.environ.get("AARON_SAMPLE_ROOT", "").strip()
    candidates: list[Path] = []
    if os.environ.get(env_var, "").strip():
        candidates.append(Path(os.environ[env_var]))
    if sample_root_env:
        candidates.append(Path(sample_root_env) / filename)
    candidates.extend(
        [
            project_root / filename,
            project_root.parent / filename,
            cwd / filename,
            cwd.parent / filename,
        ]
    )
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[Path] = []
    for p in candidates:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def resolve_project_sample_zips() -> tuple[Path | None, Path | None, str]:
    """Find drum and metadata ZIPs in Aaron's Mac paths or AI sandbox attachments."""
    drum = _first_existing_file(_sample_zip_candidates("one_shot_percussive_sounds.zip", "AARON_DRUM_ZIP"))
    meta = _first_existing_file(_sample_zip_candidates("sound_info_analysis.json.zip", "AARON_META_ZIP"))
    checked = [
        "AARON_DRUM_ZIP / AARON_META_ZIP",
        "AARON_SAMPLE_ROOT",
        "project root and parent",
        "current directory and parent",
    ]
    return drum, meta, "; ".join(checked)


def _write_wav_bytes(freq: float = 440.0, duration: float = 0.75, sr: int = 22050) -> bytes:
    t = np.arange(int(sr * duration), dtype=np.float32) / float(sr)
    y = np.sin(2.0 * np.pi * freq * t) * np.exp(-t * 3.5)
    pcm = (np.clip(y, -1.0, 1.0) * 32767.0).astype("<i2")
    buff = io.BytesIO()
    with wave.open(buff, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())
    return buff.getvalue()


def _create_synthetic_drum_and_metadata_zips(tmp_path: Path) -> tuple[Path, Path]:
    drum_zip = tmp_path / "synthetic_drum.zip"
    meta_zip = tmp_path / "synthetic_meta.zip"
    drums = [
        ("kick1", "Kick Drum", ["kick", "bass drum"], 0.85, 80.0),
        ("kick2", "Kick Drum", ["kick", "bass drum"], 0.78, 85.0),
        ("kick3", "Kick Drum", ["kick", "bass drum"], 0.72, 90.0),
        ("kick4", "Kick Drum", ["kick", "bass drum"], 0.70, 95.0),
        ("kick5", "Kick Drum", ["kick", "bass drum"], 0.75, 100.0),
        ("kick6", "Kick Drum", ["kick", "bass drum"], 0.80, 88.0),
        ("kick7", "Kick Drum", ["kick", "bass drum"], 0.82, 92.0),
        ("snare1", "Snare", ["snare"], 0.65, 220.0),
        ("snare2", "Snare", ["snare"], 0.55, 240.0),
        ("snare3", "Snare", ["snare"], 0.60, 230.0),
        ("snare4", "Snare", ["snare"], 0.58, 250.0),
        ("snare5", "Snare", ["snare"], 0.62, 210.0),
        ("snare6", "Snare", ["snare"], 0.68, 235.0),
        ("snare7", "Snare", ["snare"], 0.59, 245.0),
        ("tom1", "Tom", ["tom"], 0.75, 120.0),
        ("tom2", "Tom", ["tom"], 0.68, 100.0),
        ("tom3", "Tom", ["tom"], 0.70, 130.0),
        ("tom4", "Tom", ["tom"], 0.72, 110.0),
        ("tom5", "Tom", ["tom"], 0.66, 115.0),
        ("tom6", "Tom", ["tom"], 0.74, 125.0),
        ("tom7", "Tom", ["tom"], 0.71, 105.0),
        ("clap1", "Clap", ["clap"], 0.35, 180.0),
        ("clap2", "Clap", ["clap"], 0.30, 190.0),
        ("clap3", "Clap", ["clap"], 0.28, 200.0),
        ("clap4", "Clap", ["clap"], 0.32, 210.0),
        ("clap5", "Clap", ["clap"], 0.33, 220.0),
        ("clap6", "Clap", ["clap"], 0.34, 230.0),
        ("clap7", "Clap", ["clap"], 0.31, 205.0),
    ]
    with zipfile.ZipFile(drum_zip, "w") as zf:
        for id_name, _name, _tags, duration, freq in drums:
            zf.writestr(f"{id_name}.wav", _write_wav_bytes(freq=freq, duration=duration))

    rows = []
    for id_name, name, tags, duration, _freq in drums:
        rows.append({"id": id_name, "name": name, "tags": tags, "duration": str(duration)})

    with zipfile.ZipFile(meta_zip, "w") as zf:
        zf.writestr("sound_info_analysis.json", json.dumps(rows))

    return drum_zip, meta_zip


# ---------------------------------------------------------------------------
# Module fixture and synthetic factories
# ---------------------------------------------------------------------------


def load_sorter_module():
    from aaron_sound_sorter import api

    return api


@pytest.fixture(scope="session")
def m():
    return load_sorter_module()


class SyntheticFactory:
    """Create FeatureRow objects from named feature dictionaries."""

    def __init__(self, module):
        self.m = module
        self.name_to_idx = {name: idx for idx, name in enumerate(module.FEATURE_NAMES)}

    def fingerprint(self, features: dict[str, float]) -> list[float]:
        fp = np.zeros(self.m.FP_SIZE, dtype=np.float32)
        for name, value in features.items():
            assert name in self.name_to_idx, f"Unknown feature name in test: {name}"
            fp[self.name_to_idx[name]] = float(value)
        return fp.tolist()

    def row(
        self,
        *,
        path: str,
        label: str,
        features: dict[str, float],
        duration: float = 1.0,
        structure: str | None = None,
    ):
        top = label.split("/", 1)[0]
        if structure is None:
            structure = "loop" if label.lower().replace("\\", "/").endswith("/loops") else "one_shot"
        return self.m.FeatureRow(
            path=path,
            group_key=label,
            label=label,
            top=top,
            structure=structure,
            duration_sec=float(duration),
            fingerprint=self.fingerprint(features),
            read_status="ok",
        )


def repeated_rows(factory: SyntheticFactory, label: str, base: dict[str, float], n: int, path_word: str):
    """Create deterministic variation without using filenames as evidence."""
    rows = []
    for i in range(n):
        varied = dict(base)
        jitter = ((i % 7) - 3) * 0.01
        for key, value in list(varied.items()):
            varied[key] = float(value) + jitter
        rows.append(factory.row(path=f"/synthetic/{path_word}_{i:03d}.wav", label=label, features=varied))
    return rows


def predict_label(module, brain: dict, features: list[float], structure: str = "") -> str:
    label, _top, _sim, _margin, _top5 = module.predict(brain, features, structure)
    return label


def assert_has_dynamic_rival_support(module):
    assert hasattr(module, "rival_contrast_second_pass_choose_label")
    assert hasattr(module, "score_sample_against_rival_contrast_facts")
    assert hasattr(module, "compute_category_fact_profiles")
    assert hasattr(module, "compute_category_rival_contrast_facts")


# ---------------------------------------------------------------------------
# Synthetic architecture tests
# ---------------------------------------------------------------------------


def test_brain_builds_fact_profiles_and_rival_facts_for_arbitrary_labels(m):
    """Fact/rival learning must work for arbitrary folder labels, not named drum hacks."""
    assert_has_dynamic_rival_support(m)
    f = SyntheticFactory(m)
    label_a = "AlphaTop/Transient Low Body/One Shots"
    label_b = "AlphaTop/Transient Mid Body/One Shots"

    rows = []
    rows += repeated_rows(
        f,
        label_a,
        {
            "sub_bass_ratio_lt_150hz": 0.80,
            "bass_ratio_150_500hz": 0.20,
            "presence_ratio_2000_8000hz": 0.18,
            "pitch_confidence": 0.12,
        },
        24,
        "misleading_mid_body_name",
    )
    rows += repeated_rows(
        f,
        label_b,
        {
            "sub_bass_ratio_lt_150hz": 0.20,
            "bass_ratio_150_500hz": 0.75,
            "presence_ratio_2000_8000hz": 0.20,
            "pitch_confidence": 0.18,
        },
        24,
        "misleading_low_body_name",
    )

    brain = m.build_brain(rows, max_centroids=3)
    assert "category_fact_profiles" in brain
    assert "category_rival_contrast_facts" in brain
    assert label_a in brain["category_fact_profiles"]
    assert label_b in brain["category_fact_profiles"]
    assert brain["category_fact_profiles"][label_a]["fact_profile_strength"] in {"good", "strong"}
    assert brain["category_rival_contrast_facts"].get(label_a, {})

    sample = f.fingerprint(
        {
            "sub_bass_ratio_lt_150hz": 0.83,
            "bass_ratio_150_500hz": 0.22,
            "presence_ratio_2000_8000hz": 0.18,
            "pitch_confidence": 0.12,
        }
    )
    pred = predict_label(m, brain, sample)
    assert pred == label_a
    meta = brain.get("_last_fact_meta", {})
    assert "fact_score" in meta
    assert "top_rival_label" in meta


def test_filename_words_do_not_override_fingerprint_evidence(m):
    """Training paths are intentionally deceptive. Fingerprints must still win."""
    f = SyntheticFactory(m)
    label_a = "NeutralTop/Bright Noise/One Shots"
    label_b = "NeutralTop/Low Tone/One Shots"

    rows = []
    rows += repeated_rows(
        f,
        label_a,
        {
            "air_ratio_gt_8000hz": 0.82,
            "zcr_mean": 0.75,
            "sub_bass_ratio_lt_150hz": 0.05,
        },
        24,
        "low_tone_filename_trap",
    )
    rows += repeated_rows(
        f,
        label_b,
        {
            "air_ratio_gt_8000hz": 0.05,
            "zcr_mean": 0.08,
            "sub_bass_ratio_lt_150hz": 0.76,
        },
        24,
        "bright_noise_filename_trap",
    )

    brain = m.build_brain(rows, max_centroids=2)
    bright_sample = f.fingerprint({"air_ratio_gt_8000hz": 0.86, "zcr_mean": 0.77, "sub_bass_ratio_lt_150hz": 0.04})
    low_sample = f.fingerprint({"air_ratio_gt_8000hz": 0.04, "zcr_mean": 0.07, "sub_bass_ratio_lt_150hz": 0.78})

    assert predict_label(m, brain, bright_sample) == label_a
    assert predict_label(m, brain, low_sample) == label_b


def test_structure_lane_separates_loop_like_and_one_shot_like_material(m):
    """Structure separation should come from timing features and learned labels."""
    f = SyntheticFactory(m)
    loop_label = "StructureTop/Pulsed Pattern/Loops"
    shot_label = "StructureTop/Single Event/One Shots"

    rows = []
    rows += repeated_rows(
        f,
        loop_label,
        {
            "log_transient_count": 2.1,
            "onset_interval_regularity": 0.15,
            "onset_span_ratio": 0.88,
            "event_rate_hz": 4.2,
            "temporal_centroid_ratio": 0.50,
        },
        24,
        "loop_training",
    )
    rows += repeated_rows(
        f,
        shot_label,
        {
            "log_transient_count": 0.4,
            "onset_interval_regularity": 0.90,
            "onset_span_ratio": 0.08,
            "event_rate_hz": 0.35,
            "temporal_centroid_ratio": 0.12,
        },
        24,
        "shot_training",
    )

    brain = m.build_brain(rows, max_centroids=2)
    loop_sample = f.fingerprint(
        {
            "log_transient_count": 2.0,
            "onset_interval_regularity": 0.18,
            "onset_span_ratio": 0.86,
            "event_rate_hz": 4.0,
            "temporal_centroid_ratio": 0.48,
        }
    )
    shot_sample = f.fingerprint(
        {
            "log_transient_count": 0.35,
            "onset_interval_regularity": 0.92,
            "onset_span_ratio": 0.06,
            "event_rate_hz": 0.30,
            "temporal_centroid_ratio": 0.10,
        }
    )

    assert predict_label(m, brain, loop_sample) == loop_label
    assert predict_label(m, brain, shot_sample) == shot_label


def test_direct_second_pass_swaps_only_with_reliable_dynamic_contrast(m):
    """Second pass should be active, dynamic, and guarded by usable fact profiles."""
    assert_has_dynamic_rival_support(m)
    f = SyntheticFactory(m)
    current = "SameTop/Prototype Winner/One Shots"
    challenger = "SameTop/Contrast Winner/One Shots"
    feature = "sub_bass_ratio_lt_150hz"
    raw_x = np.asarray(f.fingerprint({feature: 0.86}), dtype=np.float32)

    brain = {
        "labels": [current, challenger],
        "top_by_label": {current: "SameTop", challenger: "SameTop"},
        "structure_by_label": {current: "one_shot", challenger: "one_shot"},
        "rival_second_pass_enabled": True,
        "rival_second_pass_max_score_gap": 0.55,
        "rival_second_pass_min_challenger_score": 0.34,
        "rival_second_pass_min_score_edge": 0.20,
    }
    rows = [
        {"label": current, "current_score": 0.10, "fact_score": 0.45},
        {"label": challenger, "current_score": 0.18, "fact_score": 0.48},
    ]
    profiles = {
        current: {"effective_count": 24, "fact_profile_strength": "good"},
        challenger: {"effective_count": 24, "fact_profile_strength": "good"},
    }
    facts = {
        challenger: {
            current: {
                "best_discriminators": [
                    {
                        "feature": feature,
                        "direction": "label_higher",
                        "robust_effect_size": 1.0,
                        "label_median": 0.80,
                        "rival_median": 0.20,
                        "reliability": 0.90,
                    }
                ]
            }
        },
        current: {
            challenger: {
                "best_discriminators": [
                    {
                        "feature": feature,
                        "direction": "rival_higher",
                        "robust_effect_size": 1.0,
                        "label_median": 0.20,
                        "rival_median": 0.80,
                        "reliability": 0.90,
                    }
                ]
            }
        },
    }

    chosen, meta = m.rival_contrast_second_pass_choose_label(
        brain=brain,
        rows=rows,
        current_label=current,
        raw_x=raw_x,
        feature_names_list=list(m.FEATURE_NAMES),
        fact_profiles=profiles,
        rival_contrast_facts=facts,
    )
    assert chosen == challenger
    assert meta["swapped"] is True
    assert meta["from_label"] == current
    assert meta["to_label"] == challenger
    assert feature in meta["matched_key_facts"]


def test_second_pass_refuses_cross_top_cross_structure_and_weak_profiles(m):
    """Guardrails prevent the second pass from becoming a hidden rescue rule."""
    f = SyntheticFactory(m)
    current = "TopA/Winner/One Shots"
    challenger = "TopB/Other/Loops"
    feature = "air_ratio_gt_8000hz"
    raw_x = np.asarray(f.fingerprint({feature: 0.90}), dtype=np.float32)
    rows = [
        {"label": current, "current_score": 0.10, "fact_score": 0.40},
        {"label": challenger, "current_score": 0.11, "fact_score": 0.80},
    ]
    profiles = {
        current: {"effective_count": 24, "fact_profile_strength": "good"},
        challenger: {"effective_count": 24, "fact_profile_strength": "good"},
    }
    facts = {
        challenger: {
            current: {
                "best_discriminators": [
                    {
                        "feature": feature,
                        "direction": "label_higher",
                        "robust_effect_size": 1.0,
                        "label_median": 0.8,
                        "rival_median": 0.2,
                        "reliability": 0.9,
                    }
                ]
            }
        }
    }
    brain = {
        "top_by_label": {current: "TopA", challenger: "TopB"},
        "structure_by_label": {current: "one_shot", challenger: "loop"},
        "rival_second_pass_enabled": True,
    }

    chosen, meta = m.rival_contrast_second_pass_choose_label(
        brain=brain,
        rows=rows,
        current_label=current,
        raw_x=raw_x,
        feature_names_list=list(m.FEATURE_NAMES),
        fact_profiles=profiles,
        rival_contrast_facts=facts,
    )
    assert chosen == current
    assert meta["swapped"] is False

    # Same top/structure but weak profile must still refuse.
    brain["top_by_label"][challenger] = "TopA"
    brain["structure_by_label"][challenger] = "one_shot"
    profiles[challenger] = {"effective_count": 4, "fact_profile_strength": "tentative"}
    chosen, meta = m.rival_contrast_second_pass_choose_label(
        brain=brain,
        rows=rows,
        current_label=current,
        raw_x=raw_x,
        feature_names_list=list(m.FEATURE_NAMES),
        fact_profiles=profiles,
        rival_contrast_facts=facts,
    )
    assert chosen == current
    assert meta["swapped"] is False
    assert "reliable" in str(meta["reason"]) or "challenger" in str(meta["reason"])


def test_predict_records_second_pass_manifest_meta_keys_even_without_swap(m):
    """Every prediction should leave manifest-ready fact and second-pass metadata."""
    f = SyntheticFactory(m)
    label_a = "MetaTop/A/One Shots"
    label_b = "MetaTop/B/One Shots"
    rows = []
    rows += repeated_rows(f, label_a, {"presence_ratio_2000_8000hz": 0.80, "tail_energy_ratio": 0.10}, 24, "a")
    rows += repeated_rows(f, label_b, {"presence_ratio_2000_8000hz": 0.20, "tail_energy_ratio": 0.70}, 24, "b")
    brain = m.build_brain(rows, max_centroids=2)
    sample = f.fingerprint({"presence_ratio_2000_8000hz": 0.78, "tail_energy_ratio": 0.12})
    _ = predict_label(m, brain, sample)
    meta = brain.get("_last_fact_meta", {})
    for key in [
        "fact_score",
        "fact_penalty",
        "rival_contrast_score",
        "top_rival_label",
        "matched_key_facts",
        "failed_key_facts",
        "rival_second_pass_swapped",
        "rival_second_pass_from_label",
        "rival_second_pass_to_label",
        "rival_second_pass_reason",
        "rival_second_pass_score_gap",
    ]:
        assert key in meta, f"missing manifest/audit key: {key}"


def test_no_obvious_active_filename_rescue_markers_in_second_pass_runtime_constants(m):
    """The second-pass function must not have active filename/category rescue constants."""
    code = m.rival_contrast_second_pass_choose_label.__code__
    constants = "\n".join(str(c).lower() for c in code.co_consts[1:] if isinstance(c, str))
    forbidden_patterns = [
        r"\bfilename\b",
        r"\bfile_name\b",
        r"\bpath_hint\b",
        r"\bsource_hint\b",
        r"\bkick\b",
        r"\bsnare\b",
        r"\btom\b",
        r"\bhat\b",
        r"\bcymbal\b",
        r"\bshaker\b",
        r"\bclap\b",
        r"\bbongo\b",
        r"\bconga\b",
    ]
    offenders = [pat for pat in forbidden_patterns if re.search(pat, constants)]
    assert not offenders, f"second-pass chooser has active forbidden shortcut constants: {offenders}"


# ---------------------------------------------------------------------------
# Optional real-sample smoke test
# ---------------------------------------------------------------------------


def _load_sound_metadata(meta_zip: Path) -> list[dict]:
    with zipfile.ZipFile(meta_zip) as z:
        json_names = [n for n in z.namelist() if n.lower().endswith(".json")]
        assert json_names, f"No JSON metadata file found in {meta_zip}"
        preferred = "sound_info_analysis.json"
        name = preferred if preferred in json_names else json_names[0]
        return json.loads(z.read(name))


def _audio_members_by_id(drum_zip: Path) -> dict[str, str]:
    with zipfile.ZipFile(drum_zip) as z:
        return {
            Path(info.filename).stem: info.filename
            for info in z.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".wav")
        }


def _metadata_text(row: dict) -> str:
    pieces = [str(row.get("name", ""))]
    tags = row.get("tags", []) or []
    if isinstance(tags, list):
        pieces.extend(str(t) for t in tags)
    return re.sub(r"\s+", " ", " ".join(pieces).lower().replace("_", " ").replace("-", " ")).strip()


def _select_real_drum_examples(meta_rows: list[dict], id_to_member: dict[str, str]) -> dict[str, list[tuple[str, str]]]:
    """Pick a tiny but useful real set from the drum metadata.

    Filename/metadata use here is only to build a temporary test fixture from the
    known ZIP database. It is not sorter logic and does not get passed as evidence
    to prediction except as normal source paths in reports.
    """
    patterns = {
        "kick": ("Drums/Kick Drums/Generic Kick/_ONE_SHOTS", [r"\bkick\b", r"\bbass drum\b"]),
        "snare": ("Drums/Snares/Generic Snare/_ONE_SHOTS", [r"\bsnare\b"]),
        "tom": ("Drums/Toms/Generic Tom/_ONE_SHOTS", [r"\btom\b", r"\btoms\b"]),
        "clap": ("Drums/Claps Snaps Slaps/Hand Clap/_ONE_SHOTS", [r"\bclap\b", r"\bclaps\b"]),
    }
    selected: dict[str, list[tuple[str, str]]] = {key: [] for key in patterns}
    for row in meta_rows:
        sid = str(row.get("id", ""))
        if sid not in id_to_member:
            continue
        text = _metadata_text(row)
        if re.search(r"\b(loop|beat|groove|pattern)\b", text):
            continue
        try:
            duration = float(row.get("duration") or 0.0)
        except Exception:
            duration = 0.0
        if duration <= 0.05 or duration > 3.0:
            continue
        for key, (_label_path, pats) in patterns.items():
            if len(selected[key]) >= 7:
                continue
            if any(re.search(pat, text) for pat in pats):
                selected[key].append((sid, id_to_member[sid]))
                break
    return selected


def test_optional_real_project_drum_zip_tiny_end_to_end(tmp_path):
    """Train a tiny real baby brain and sort holdouts when project sample ZIPs exist.

    This auto-resolves both Aaron's Mac path and the AI sandbox attachment path.
    It is deliberately small, so it stays useful as a fast smoke test instead of
    becoming a full validation run.
    """
    sorter = _module_path()
    drum_zip, meta_zip, checked = resolve_project_sample_zips()
    using_synthetic = False
    if not drum_zip or not meta_zip:
        drum_zip, meta_zip = _create_synthetic_drum_and_metadata_zips(tmp_path)
        using_synthetic = True

    meta_rows = _load_sound_metadata(meta_zip)
    id_to_member = _audio_members_by_id(drum_zip)
    selected = _select_real_drum_examples(meta_rows, id_to_member)
    missing = {key: len(vals) for key, vals in selected.items() if len(vals) < 5}
    if missing and not using_synthetic:
        pytest.skip(f"Not enough real drum examples for tiny smoke fixture: {missing}")
    assert not missing, f"Tiny smoke fixture missing required categories: {missing}"

    label_paths = {
        "kick": "Drums/Kick Drums/Generic Kick/_ONE_SHOTS",
        "snare": "Drums/Snares/Generic Snare/_ONE_SHOTS",
        "tom": "Drums/Toms/Generic Tom/_ONE_SHOTS",
        "clap": "Drums/Claps Snaps Slaps/Hand Clap/_ONE_SHOTS",
    }
    train_root = tmp_path / "train_tree"
    holdout_root = tmp_path / "holdout"
    sort_out = tmp_path / "sort_out"
    run_dir = tmp_path / "run"
    brain_path = tmp_path / "tiny_real_drum_brain.json"

    with zipfile.ZipFile(drum_zip) as z:
        for key, items in selected.items():
            for i, (_sid, member) in enumerate(items[:4]):
                dest = train_root / label_paths[key] / f"{key}_train_{i:02d}.wav"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(member))
            for i, (_sid, member) in enumerate(items[4:6]):
                dest = holdout_root / key / f"{key}_holdout_{i:02d}.wav"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(member))

    train_cmd = [
        sys.executable,
        str(sorter),
        "train-brain",
        str(train_root),
        "--save-brain",
        str(brain_path),
        "--project-dir",
        str(tmp_path),
        "--run-dir",
        str(run_dir / "train"),
        "--max-centroids",
        "3",
        "--max-eval-per-label",
        "0",
        "--max-eval-total",
        "0",
        "--training-preview-per-label",
        "0",
    ]
    trained = subprocess.run(train_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
    assert trained.returncode == 0, trained.stdout
    assert brain_path.exists()

    sort_cmd = [
        sys.executable,
        str(sorter),
        "sort",
        str(holdout_root),
        str(sort_out),
        "--brain",
        str(brain_path),
        "--no-zip",
    ]
    sorted_run = subprocess.run(sort_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
    assert sorted_run.returncode == 0, sorted_run.stdout

    manifest = sort_out / "Aaron_Sorted_Sounds_manifest.csv"
    assert manifest.exists(), f"manifest not found: {manifest}\n{sorted_run.stdout}"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", errors="replace")))
    assert len(rows) == 8

    # Current OO manifest exposes the same debug contract through shared facts,
    # role audit, and shared candidate JSON columns instead of the legacy
    # fact_score/rival_* flat columns.
    required_cols = {
        "brain_vote_1",
        "physics_vote_1",
        "consensus_status",
        "decision_reason",
        "parent_role_audit_json",
        "shared_facts_json",
        "shared_candidates_json",
    }
    assert required_cols.issubset(set(rows[0].keys()))

    if using_synthetic:
        return

    final_tops = [str(row.get("final_top") or row.get("top") or row.get("official_top") or "") for row in rows]
    # v0.6.0 safety policy: tiny four-example baby brains must not confidently
    # leak drum holdouts into FX/Instruments/Textures. Review is acceptable when
    # the exact drum leaf is ambiguous.
    assert all(top in {"Drums", "_TO_REVIEW"} for top in final_tops), rows
    assert any(str(row.get("top_rival_label", "")).strip() for row in rows) or any(
        str(row.get("shared_candidates_json", "")).strip() for row in rows
    )


# ---------------------------------------------------------------------------
# Direct-run convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Lets Aaron run this file directly with /usr/bin/python3 path/to/test.py
    # instead of remembering the pytest module command.
    raise SystemExit(pytest.main(["-q", str(Path(__file__).resolve())]))
