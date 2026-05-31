#!/usr/bin/env python3
"""
Fast synthetic accuracy smoke tests for Aaron Sound Sorter Stage 4.

Purpose:
  These tests are deliberately tiny and deterministic. They create synthetic
  training folder rows using the real FeatureRow -> build_brain() -> predict()
  path, then classify synthetic holdout rows that were not in training.

What this protects:
  - obvious weak-folder examples must not be swallowed by stronger folders
  - imbalanced training counts must not dominate clear audio facts
  - close same-family labels must use measured features, not filenames
  - second-pass rival logic must not use category-name rescues

How to run from the project root:
  python3 -m pytest -q tests/test_phase4_synthetic_accuracy_smoke.py

Direct-run also works:
  python3 tests/test_phase4_synthetic_accuracy_smoke.py

Optional environment variable:
  SORTER_UNDER_TEST=/path/to/Aaron_Sound_Sorter.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest


def _module_path() -> Path:
    env = os.environ.get("SORTER_UNDER_TEST", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    candidates = [
        # In ChatGPT/container tests prefer the explicitly patched file.
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


def load_sorter_module():
    from aaron_sound_sorter import api

    return api


@pytest.fixture(scope="session")
def m():
    return load_sorter_module()


class FingerprintFactory:
    def __init__(self, module):
        self.m = module
        self.idx = {name: i for i, name in enumerate(module.FEATURE_NAMES)}

    def fingerprint(self, features: dict[str, float]) -> list[float]:
        fp = np.zeros(self.m.FP_SIZE, dtype=np.float32)
        for name, value in features.items():
            assert name in self.idx, f"unknown feature: {name}"
            fp[self.idx[name]] = float(value)
        return fp.tolist()

    def row(
        self,
        *,
        label: str,
        path: str,
        features: dict[str, float],
        duration: float = 0.70,
        structure: str = "one_shot",
    ):
        return self.m.FeatureRow(
            path=path,
            group_key=label,
            label=label,
            top=label.split("/", 1)[0],
            structure=structure,
            duration_sec=float(duration),
            fingerprint=self.fingerprint(features),
            read_status="ok",
        )


def jittered(base: dict[str, float], i: int, amount: float = 0.015) -> dict[str, float]:
    """Small deterministic variation so tests use a folder distribution, not clones."""
    out = dict(base)
    for offset, key in enumerate(sorted(out)):
        # alternating small jitter, bounded to avoid changing the identity.
        j = (((i + offset) % 5) - 2) * amount
        out[key] = float(out[key]) + j
    return out


def make_rows(
    f: FingerprintFactory,
    *,
    label: str,
    base: dict[str, float],
    count: int,
    deceptive_word: str,
    duration: float = 0.70,
) -> list[object]:
    rows = []
    for i in range(count):
        rows.append(
            f.row(
                label=label,
                path=f"/synthetic/deceptive_{deceptive_word}_{i:03d}.wav",
                features=jittered(base, i),
                duration=duration,
            )
        )
    return rows


def predict_one(
    module, brain: dict, f: FingerprintFactory, features: dict[str, float], structure: str = "one_shot"
) -> tuple[str, dict]:
    label, top, sim, margin, top5 = module.predict(brain, f.fingerprint(features), structure)
    meta = dict(brain.get("_last_fact_meta", {}))
    meta.update({"top": top, "similarity": sim, "margin": margin, "top5": top5})
    return label, meta


def score_holdouts(
    module, brain: dict, f: FingerprintFactory, expected_to_features: dict[str, list[dict[str, float]]]
) -> list[dict[str, object]]:
    rows = []
    for expected, feature_list in expected_to_features.items():
        for idx, features in enumerate(feature_list):
            predicted, meta = predict_one(module, brain, f, features)
            rows.append(
                {
                    "expected": expected,
                    "predicted": predicted,
                    "ok": predicted == expected,
                    "idx": idx,
                    "meta": meta,
                }
            )
    return rows


def assert_accuracy(rows: list[dict[str, object]], expected_label: str, min_correct: int) -> None:
    subset = [r for r in rows if r["expected"] == expected_label]
    correct = sum(1 for r in subset if r["ok"])
    details = "\n".join(f"expected={r['expected']} predicted={r['predicted']} meta={r['meta']}" for r in subset)
    assert correct >= min_correct, f"{expected_label} correct {correct}/{len(subset)} below {min_correct}\n{details}"


def test_imbalanced_strong_kick_weak_tom_still_classifies_unseen_holdouts(m):
    """User-requested shape: 6 strong Kick teachers, 2 weak Tom teachers, unseen holdouts.

    This is intentionally imbalanced. The weak Tom folder has only two examples,
    but the holdout Toms are physically obvious. A healthy learned brain should
    not let the larger Kick folder swallow them.
    """
    f = FingerprintFactory(m)
    kick_label = "Drums/Kick Drums/Generic Kick/One Shots"
    tom_label = "Drums/Toms/Low Tom/One Shots"

    kick_base = {
        "sub_bass_ratio_lt_150hz": 0.86,
        "bass_ratio_150_500hz": 0.18,
        "presence_ratio_2000_8000hz": 0.14,
        "air_ratio_gt_8000hz": 0.03,
        "log_crest": 1.25,
        "log_decay_ratio": 0.20,
        "temporal_centroid_ratio": 0.10,
        "attack_rise_time_norm": 0.04,
        "pitch_confidence": 0.10,
    }
    tom_base = {
        "sub_bass_ratio_lt_150hz": 0.22,
        "bass_ratio_150_500hz": 0.72,
        "presence_ratio_2000_8000hz": 0.26,
        "air_ratio_gt_8000hz": 0.05,
        "log_crest": 0.92,
        "log_decay_ratio": 0.56,
        "temporal_centroid_ratio": 0.18,
        "attack_rise_time_norm": 0.06,
        "pitch_confidence": 0.55,
    }

    rows = []
    # Deceptive paths are deliberate: kick rows say tom; tom rows say kick.
    rows += make_rows(f, label=kick_label, base=kick_base, count=6, deceptive_word="tom", duration=0.55)
    rows += make_rows(f, label=tom_label, base=tom_base, count=2, deceptive_word="kick", duration=0.85)
    brain = m.build_brain(rows, max_centroids=2)

    unseen_kicks = [jittered(kick_base, i + 100, amount=0.02) for i in range(5)]
    unseen_toms = [jittered(tom_base, i + 200, amount=0.02) for i in range(5)]
    results = score_holdouts(m, brain, f, {kick_label: unseen_kicks, tom_label: unseen_toms})

    assert_accuracy(results, kick_label, min_correct=4)
    assert_accuracy(results, tom_label, min_correct=4)
    # The weak folder should be present in the brain and not disabled by count.
    assert tom_label in brain.get("labels", [])
    assert int(brain.get("counts", {}).get(tom_label, 0)) == 2


def test_same_physics_accuracy_with_arbitrary_folder_names_proves_no_category_rescue(m):
    """Repeat the kick/tom physics test with arbitrary folder names.

    If this passes, the behavior is not dependent on literal words like Kick or Tom.
    """
    f = FingerprintFactory(m)
    low_sub_label = "Drums/Folder Alpha/One Shots"
    modal_body_label = "Drums/Folder Beta/One Shots"

    low_sub = {
        "sub_bass_ratio_lt_150hz": 0.84,
        "bass_ratio_150_500hz": 0.16,
        "presence_ratio_2000_8000hz": 0.12,
        "log_crest": 1.15,
        "log_decay_ratio": 0.22,
        "pitch_confidence": 0.12,
    }
    modal_body = {
        "sub_bass_ratio_lt_150hz": 0.24,
        "bass_ratio_150_500hz": 0.70,
        "presence_ratio_2000_8000hz": 0.24,
        "log_crest": 0.90,
        "log_decay_ratio": 0.58,
        "pitch_confidence": 0.58,
    }

    rows = []
    rows += make_rows(f, label=low_sub_label, base=low_sub, count=6, deceptive_word="folder_beta", duration=0.55)
    rows += make_rows(f, label=modal_body_label, base=modal_body, count=2, deceptive_word="folder_alpha", duration=0.85)
    brain = m.build_brain(rows, max_centroids=2)

    results = score_holdouts(
        m,
        brain,
        f,
        {
            low_sub_label: [jittered(low_sub, i + 10, amount=0.02) for i in range(5)],
            modal_body_label: [jittered(modal_body, i + 20, amount=0.02) for i in range(5)],
        },
    )
    assert_accuracy(results, low_sub_label, min_correct=4)
    assert_accuracy(results, modal_body_label, min_correct=4)


def test_accuracy_smoke_spans_multiple_close_drum_rivals_without_filename_help(m):
    """A broader close-drum test: kick, tom, snare, clap, hat, cymbal, shaker.

    This is not trying to be a final real-world score. It is a fast regression
    guard that basic synthetic holdouts remain separable when future edits touch
    weights, fact scoring, or second-pass logic.
    """
    f = FingerprintFactory(m)
    bases = {
        "Drums/Kick Drums/Generic Kick/One Shots": {
            "sub_bass_ratio_lt_150hz": 0.86,
            "bass_ratio_150_500hz": 0.18,
            "presence_ratio_2000_8000hz": 0.12,
            "air_ratio_gt_8000hz": 0.03,
            "log_crest": 1.25,
            "log_decay_ratio": 0.20,
            "pitch_confidence": 0.10,
        },
        "Drums/Toms/Low Tom/One Shots": {
            "sub_bass_ratio_lt_150hz": 0.24,
            "bass_ratio_150_500hz": 0.72,
            "presence_ratio_2000_8000hz": 0.24,
            "air_ratio_gt_8000hz": 0.05,
            "log_crest": 0.92,
            "log_decay_ratio": 0.58,
            "pitch_confidence": 0.56,
        },
        "Drums/Snares/Generic Snare/One Shots": {
            "sub_bass_ratio_lt_150hz": 0.10,
            "bass_ratio_150_500hz": 0.32,
            "presence_ratio_2000_8000hz": 0.78,
            "air_ratio_gt_8000hz": 0.18,
            "spectral_flatness_mean": 0.58,
            "log_crest": 1.18,
            "log_decay_ratio": 0.32,
        },
        "Drums/Claps Snaps Slaps/Hand Clap/One Shots": {
            "sub_bass_ratio_lt_150hz": 0.04,
            "bass_ratio_150_500hz": 0.18,
            "presence_ratio_2000_8000hz": 0.70,
            "air_ratio_gt_8000hz": 0.30,
            "spectral_flatness_mean": 0.72,
            "zcr_mean": 0.48,
            "log_decay_ratio": 0.18,
        },
        "Drums/Hi Hats/Closed Hat/One Shots": {
            "sub_bass_ratio_lt_150hz": 0.02,
            "bass_ratio_150_500hz": 0.05,
            "presence_ratio_2000_8000hz": 0.38,
            "air_ratio_gt_8000hz": 0.84,
            "spectral_flatness_mean": 0.82,
            "zcr_mean": 0.78,
            "tail_energy_ratio": 0.08,
        },
        "Drums/Cymbals/Crash Cymbal/One Shots": {
            "sub_bass_ratio_lt_150hz": 0.02,
            "bass_ratio_150_500hz": 0.07,
            "presence_ratio_2000_8000hz": 0.30,
            "air_ratio_gt_8000hz": 0.88,
            "spectral_flatness_mean": 0.78,
            "zcr_mean": 0.70,
            "tail_energy_ratio": 0.62,
            "log_decay_ratio": 0.82,
        },
        "Drums/Percussion/Shakers/One Shots": {
            "sub_bass_ratio_lt_150hz": 0.02,
            "bass_ratio_150_500hz": 0.10,
            "presence_ratio_2000_8000hz": 0.52,
            "air_ratio_gt_8000hz": 0.48,
            "spectral_flux_variance": 0.72,
            "zcr_mean": 0.55,
            "event_rate_hz": 2.4,
            "onset_span_ratio": 0.42,
        },
    }

    rows = []
    for n, (label, base) in enumerate(bases.items()):
        # Imbalanced by design: first two classes stronger, others smaller.
        count = 8 if n < 2 else 4
        rows += make_rows(f, label=label, base=base, count=count, deceptive_word=f"wrong_name_{n}")
    brain = m.build_brain(rows, max_centroids=2)

    expected_to_features = {
        label: [jittered(base, i + 500, amount=0.018) for i in range(3)] for label, base in bases.items()
    }
    results = score_holdouts(m, brain, f, expected_to_features)
    correct = sum(1 for r in results if r["ok"])
    total = len(results)
    details = "\n".join(f"expected={r['expected']} predicted={r['predicted']}" for r in results if not r["ok"])
    assert correct >= total - 2, f"synthetic close-drum accuracy too low: {correct}/{total}\n{details}"


def test_second_pass_runtime_constants_stay_free_of_filename_or_named_drum_rescues(m):
    """The dynamic second-pass chooser must not contain active shortcut constants."""
    assert hasattr(m, "rival_contrast_second_pass_choose_label")
    constants = "\n".join(
        str(c).lower() for c in m.rival_contrast_second_pass_choose_label.__code__.co_consts[1:] if isinstance(c, str)
    )
    forbidden = [
        r"\bfilename\b",
        r"\bfile_name\b",
        r"\bpath_hint\b",
        r"\bsource_hint\b",
        r"\bkick\b",
        r"\btom\b",
        r"\bsnare\b",
        r"\bclap\b",
        r"\bhat\b",
        r"\bcymbal\b",
        r"\bshaker\b",
        r"\bbongo\b",
        r"\bconga\b",
    ]
    offenders = [pat for pat in forbidden if re.search(pat, constants)]
    assert not offenders, f"active second-pass shortcut constants found: {offenders}"


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", __file__]))
