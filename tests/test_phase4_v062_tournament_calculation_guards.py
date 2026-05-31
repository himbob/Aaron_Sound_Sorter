from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
from aaron_sound_sorter import api as mod


def _brain() -> dict:
    return {
        "model_ensemble_enabled": True,
        "model_ensemble_min_switch_advantage": 0.10,
        "model_ensemble_max_score_penalty_ratio": 10.0,
        "model_ensemble_allow_cross_family_switch": False,
        "model_ensemble_current_rank_weight": 0.40,
        "model_ensemble_neutral_rank_weight": 0.18,
        "model_ensemble_spread_rank_weight": 0.18,
        "model_ensemble_anchor_rank_weight": 0.08,
        "model_ensemble_linear_rank_weight": 0.16,
        "model_ensemble_frontend_router_rank_weight": 0.18,
        "top_by_label": {
            "inst_piano": "Instruments",
            "inst_synth_chord": "Instruments",
            "inst_guitar_loop": "Instruments",
            "inst_bass_loop": "Instruments",
            "drum_clap": "Drums",
            "drum_snare_loop": "Drums",
            "drum_cymbal": "Drums",
            "drum_kick": "Drums",
            "fx_riser": "FX",
            "fx_drop": "FX",
            "fx_impact": "FX",
            "texture_drone": "Textures",
            "texture_noise": "Textures",
        },
    }


def _row(
    label: str,
    *,
    current: float,
    neutral: float | None = None,
    spread: float | None = None,
    anchor: float | None = None,
    linear: float | None = None,
    router: float | None = None,
    blocked: bool = False,
) -> dict:
    return {
        "label": label,
        "current_score": current,
        "support_neutral_score": current if neutral is None else neutral,
        "spread_norm_score": current if spread is None else spread,
        "anchor_score": current if anchor is None else anchor,
        "linear_rank_score": 0.0 if linear is None else linear,
        "frontend_router_rank_score": 1.0 if router is None else router,
        "membership_blocked": blocked,
    }


def test_low_score_rank_map_is_tie_aware_not_alphabetical() -> None:
    ranks = mod._rank_map_from_scores(
        [
            ("Drums/a", 1.0),
            ("FX/b", 1.0),
            ("Instruments/c", 1.0),
            ("Textures/d", 2.0),
        ]
    )
    assert ranks["Drums/a"] == 1
    assert ranks["FX/b"] == 1
    assert ranks["Instruments/c"] == 1
    assert ranks["Textures/d"] == 4


def test_high_score_rank_map_is_tie_aware_not_alphabetical() -> None:
    ranks = mod._rank_from_high_score(
        {
            "Drums/a": 0.5,
            "FX/b": 0.5,
            "Instruments/c": 0.5,
            "Textures/d": 0.1,
        }
    )
    assert ranks["Drums/a"] == 1
    assert ranks["FX/b"] == 1
    assert ranks["Instruments/c"] == 1
    assert ranks["Textures/d"] == 4


def test_disabled_router_constant_scores_are_ignored_not_ranked_alphabetically() -> None:
    brain = _brain()
    rows = [
        _row("inst_synth_chord", current=1.00, neutral=1.00, spread=2.00, anchor=1.50, linear=0.00, router=1.0),
        _row("drum_clap", current=1.30, neutral=1.30, spread=1.80, anchor=0.10, linear=0.00, router=1.0),
        _row("fx_riser", current=1.40, neutral=1.40, spread=1.90, anchor=0.20, linear=0.00, router=1.0),
    ]
    final, meta = mod.ensemble_tournament_choose_label(brain, rows, "inst_synth_chord")
    assert final == "inst_synth_chord"
    assert "frontend_router_rank_score" in meta["ignored_heads"]
    assert meta["switched"] is False


def test_membership_blocked_candidate_cannot_win_tournament_even_with_best_anchor() -> None:
    brain = _brain()
    rows = [
        _row("inst_synth_chord", current=1.00, neutral=1.00, spread=2.00, anchor=2.00, linear=1.00, router=1.0),
        _row("drum_clap", current=1.05, neutral=1.05, spread=1.00, anchor=0.01, linear=-10.0, router=1.0, blocked=True),
    ]
    final, meta = mod.ensemble_tournament_choose_label(brain, rows, "inst_synth_chord")
    assert final == "inst_synth_chord"
    assert "drum_clap:membership_blocked" in meta["rejected_proposals"]


def test_cross_family_switch_is_blocked_when_current_label_is_not_blocked() -> None:
    brain = _brain()
    rows = [
        _row("inst_synth_chord", current=1.00, neutral=1.00, spread=8.00, anchor=8.00, linear=5.00, router=1.0),
        _row("drum_clap", current=1.04, neutral=1.04, spread=0.50, anchor=0.01, linear=-10.0, router=1.0),
    ]
    final, meta = mod.ensemble_tournament_choose_label(brain, rows, "inst_synth_chord")
    assert final == "inst_synth_chord"
    assert "drum_clap:cross_family_blocked" in meta["rejected_proposals"]


def test_same_family_switch_still_allowed_when_heads_agree() -> None:
    brain = _brain()
    rows = [
        _row("inst_synth_chord", current=1.00, neutral=1.20, spread=9.00, anchor=9.00, linear=5.0, router=1.0),
        _row("inst_piano", current=1.04, neutral=0.80, spread=0.60, anchor=0.30, linear=-10.0, router=1.0),
    ]
    final, meta = mod.ensemble_tournament_choose_label(brain, rows, "inst_synth_chord")
    assert final == "inst_piano"
    assert meta["switched"] is True


SYNTHETIC_SOUNDS = [
    ("piano hit", "inst_piano", "drum_clap"),
    ("organ chord", "inst_synth_chord", "drum_cymbal"),
    ("synth stab", "inst_synth_chord", "drum_snare_loop"),
    ("pluck arp", "inst_synth_chord", "drum_cymbal"),
    ("acoustic guitar rhythm", "inst_guitar_loop", "drum_snare_loop"),
    ("bass loop", "inst_bass_loop", "drum_kick"),
    ("drone texture", "texture_drone", "fx_riser"),
    ("noise texture", "texture_noise", "drum_cymbal"),
    ("short impact fx", "fx_impact", "drum_kick"),
    ("riser fx", "fx_riser", "drum_cymbal"),
    ("downlifter fx", "fx_drop", "drum_kick"),
    ("kick", "drum_kick", "fx_drop"),
    ("snare loop", "drum_snare_loop", "fx_riser"),
    ("clap", "drum_clap", "fx_impact"),
    ("cymbal shimmer", "drum_cymbal", "fx_riser"),
    ("tom-like kick", "drum_kick", "fx_impact"),
    ("instrument with shimmer", "inst_piano", "drum_cymbal"),
    ("musical phrase", "inst_synth_chord", "fx_riser"),
    ("wide pad texture", "texture_drone", "fx_riser"),
    ("foley impact", "fx_impact", "drum_snare_loop"),
]


@pytest.mark.parametrize("sound_name,current_label,bad_label", SYNTHETIC_SOUNDS)
def test_twenty_sound_type_tournament_cannot_jump_to_wrong_family_from_reasonable_current(
    sound_name: str, current_label: str, bad_label: str
) -> None:
    brain = _brain()
    rows = [
        # The current/raw brain winner is reasonable: best current score.
        _row(current_label, current=1.00, neutral=1.00, spread=9.0, anchor=9.0, linear=5.0, router=1.0),
        # The bad rival simulates the broken v0.6.1 failure mode: misleading
        # anchor/linear/router-style heads prefer it despite worse raw score.
        _row(bad_label, current=1.08, neutral=1.08, spread=0.4, anchor=0.01, linear=-10.0, router=1.0),
    ]
    final, meta = mod.ensemble_tournament_choose_label(brain, rows, current_label)
    assert final == current_label, f"{sound_name} wrongly switched to {bad_label}: {meta}"
    if brain["top_by_label"][current_label] != brain["top_by_label"][bad_label]:
        assert f"{bad_label}:cross_family_blocked" in meta["rejected_proposals"]
