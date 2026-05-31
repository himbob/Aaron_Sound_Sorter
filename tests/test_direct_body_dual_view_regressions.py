"""Regression tests for full-file plus direct/body audio evidence.

The direct/body view is source-name blind. It lets the sorter compare the full
file against onset/peak-centered audio windows before long tails, reverb, or
delay dominate measured identity.
"""

from __future__ import annotations

import numpy as np

from aaron_sound_sorter.core import FEATURE_NAMES
from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.features import direct_body_audio_view, make_fingerprint_from_preprocessed_audio
from aaron_sound_sorter.voters.physics_voter import PhysicsVoter


def decaying_sine(sr: int = 22050) -> np.ndarray:
    """Build a synthetic low kick-like hit with a long artificial tail."""
    t = np.arange(int(1.5 * sr), dtype=np.float32) / sr
    attack = np.exp(-t * 26.0) * np.sin(2.0 * np.pi * 55.0 * t)
    tail = 0.22 * np.exp(-np.maximum(t - 0.08, 0.0) * 2.1) * np.sin(2.0 * np.pi * 60.0 * t)
    y = attack + tail
    y /= max(1e-9, float(np.max(np.abs(y))))
    return y.reshape(-1, 1).astype(np.float32)


def test_direct_body_view_reduces_tail_without_destroying_event_body() -> None:
    """The auxiliary view should keep the hit body and reduce tail dominance."""
    sr = 22050
    full_audio = decaying_sine(sr)
    direct_audio, meta = direct_body_audio_view(full_audio, sr, max_events=1, post_ms=260.0)

    full_fp, _full_dur, full_status = make_fingerprint_from_preprocessed_audio(full_audio, sr)
    direct_fp, direct_dur, direct_status = make_fingerprint_from_preprocessed_audio(direct_audio, sr)

    sub_index = FEATURE_NAMES.index("sub_bass_ratio_lt_150hz")
    assert full_status == "ok"
    assert direct_status == "ok"
    assert meta["status"] == "ok"
    assert direct_dur < 0.40
    assert direct_dur < _full_dur * 0.35
    assert float(direct_fp[sub_index]) > 0.70


def test_physics_voter_can_blend_compatible_direct_body_score(monkeypatch) -> None:
    """PhysicsVoter should inspect a compatible direct/body score, not only full tail."""
    voter = PhysicsVoter()
    values = {name: 0.0 for name in FEATURE_NAMES}
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "direct_body_view": {
                "available": True,
                "feature_values_by_name": values,
                "measured_roles": {"percussive_one_shot": 1.0},
            }
        },
    )
    brain = {
        "feature_names": list(FEATURE_NAMES),
        "labels": ["Drums/Kick Drums/Generic Kick/One Shots"],
        "top_by_label": {"Drums/Kick Drums/Generic Kick/One Shots": "Drums"},
        "structure_by_label": {"Drums/Kick Drums/Generic Kick/One Shots": "one_shot"},
    }

    def fake_profile_residual_score(fingerprint, profile, policy):
        arr = np.asarray(fingerprint, dtype=np.float32)
        # Full fingerprint below uses first value 1.0. Direct-body vector is all zero.
        score = 9.0 if float(arr[0]) > 0.5 else 3.0
        return score, {"used_feature_count": len(FEATURE_NAMES)}

    monkeypatch.setattr("aaron_sound_sorter.voters.physics_voter.profile_residual_score", fake_profile_residual_score)
    blended, evidence = voter.apply_direct_body_profile_check(
        label="Drums/Kick Drums/Generic Kick/One Shots",
        brain=brain,
        facts=facts,
        profile={"feature_stats": {"dummy": {}}},
        full_profile_score=9.0,
    )

    assert evidence["direct_body_profile_check"] == "compatible_score_blend"
    assert blended < 9.0
