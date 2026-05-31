from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
from aaron_sound_sorter import api as mod


def brain() -> dict:
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
            "inst_synth_arp": "Instruments",
            "inst_guitar_loop": "Instruments",
            "inst_bass_loop": "Instruments",
            "inst_voice_loop": "Instruments",
            "drum_clap": "Drums",
            "drum_snare_loop": "Drums",
            "drum_cymbal": "Drums",
            "drum_kick": "Drums",
            "drum_roll_fill": "Drums",
            "fx_riser": "FX",
            "fx_drop": "FX",
            "fx_impact": "FX",
            "fx_whoosh": "FX",
            "texture_drone": "Textures",
            "texture_noise": "Textures",
        },
        "structure_by_label": {
            "inst_piano": "one_shot",
            "inst_synth_chord": "one_shot",
            "inst_synth_arp": "loop",
            "inst_guitar_loop": "loop",
            "inst_bass_loop": "loop",
            "inst_voice_loop": "loop",
            "drum_clap": "one_shot",
            "drum_snare_loop": "loop",
            "drum_cymbal": "one_shot",
            "drum_kick": "one_shot",
            "drum_roll_fill": "loop",
            "fx_riser": "transition",
            "fx_drop": "transition",
            "fx_impact": "one_shot",
            "fx_whoosh": "transition",
            "texture_drone": "texture",
            "texture_noise": "texture",
        },
    }


def row(
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
    """Synthetic tournament row.

    current_score is the raw/prototype winner path.  The other fields simulate
    the helper heads that broke Piano_G, OrganChord, guitar rhythm, and drum fills.
    Lower scores are better except membership_score, which is only a diagnostic
    here.  router=1.0 represents the disabled-router constant value that used to
    become fake alphabetical evidence.
    """
    return {
        "label": label,
        "current_score": float(current),
        "support_neutral_score": float(current if neutral is None else neutral),
        "spread_norm_score": float(current if spread is None else spread),
        "anchor_score": float(current if anchor is None else anchor),
        "linear_rank_score": float(0.0 if linear is None else linear),
        "frontend_router_rank_score": float(1.0 if router is None else router),
        "membership_blocked": bool(blocked),
        "membership_score": 0.0 if blocked else 1.0,
    }


# At least 20 real musical roles.  Each case says: if the raw brain/current
# winner is reasonable, a helper-head temptation from another top family cannot
# steal the final label.  These are not filename tests.  They simulate the
# numeric shape that created the observed bad placements.
SOUND_CASES: list[tuple[str, str, str, str]] = [
    ("piano single hit", "inst_piano", "drum_clap", "short tonal attack must stay Instrument"),
    ("organ chord", "inst_synth_chord", "drum_cymbal", "pitched chord cannot become cymbal"),
    ("synth stab", "inst_synth_chord", "drum_snare_loop", "short synth cannot become snare loop"),
    ("plinky synth arp", "inst_synth_arp", "drum_cymbal", "pitched repetition cannot become cymbal"),
    ("acoustic guitar rhythm", "inst_guitar_loop", "drum_snare_loop", "guitar loop cannot become snare loop"),
    ("bass loop", "inst_bass_loop", "drum_kick", "bass loop cannot become kick"),
    ("vocal loop", "inst_voice_loop", "fx_whoosh", "vocal loop cannot become transition FX"),
    ("drone texture", "texture_drone", "fx_riser", "steady texture cannot become riser"),
    ("noise texture", "texture_noise", "drum_cymbal", "texture cannot become cymbal"),
    ("short impact FX", "fx_impact", "drum_kick", "impact FX cannot become kick by anchor only"),
    ("riser FX", "fx_riser", "drum_cymbal", "transition FX cannot become cymbal"),
    ("downlifter FX", "fx_drop", "drum_kick", "downlifter cannot become kick"),
    ("kick", "drum_kick", "fx_drop", "kick cannot become drop"),
    ("snare loop", "drum_snare_loop", "fx_riser", "snare loop cannot become riser"),
    ("clap", "drum_clap", "fx_impact", "clap cannot become generic FX impact"),
    ("cymbal shimmer", "drum_cymbal", "fx_riser", "cymbal cannot become riser"),
    ("drum fill", "drum_roll_fill", "fx_riser", "drum fill cannot become riser"),
    ("instrument with shimmer", "inst_piano", "drum_cymbal", "shimmer does not make an instrument a cymbal"),
    ("musical phrase", "inst_synth_arp", "fx_riser", "musical phrase cannot become riser"),
    ("wide pad texture", "texture_drone", "fx_riser", "wide sustained pad cannot become riser"),
    ("foley impact", "fx_impact", "drum_snare_loop", "foley impact cannot become snare loop"),
]


def test_factor_low_score_ranking_is_tie_aware() -> None:
    ranks = mod._rank_map_from_scores(
        [
            ("drum_clap", 1.0),
            ("fx_riser", 1.0),
            ("inst_piano", 1.0),
            ("texture_drone", 2.0),
        ]
    )
    assert ranks["drum_clap"] == ranks["fx_riser"] == ranks["inst_piano"] == 1
    assert ranks["texture_drone"] == 4


def test_factor_high_score_ranking_is_tie_aware() -> None:
    ranks = mod._rank_from_high_score(
        {
            "Drums": 0.5,
            "FX": 0.5,
            "Instruments": 0.5,
            "Textures": 0.1,
        }
    )
    assert ranks["Drums"] == ranks["FX"] == ranks["Instruments"] == 1
    assert ranks["Textures"] == 4


def test_factor_constant_head_has_no_information() -> None:
    assert mod._score_head_has_information([("a", 1.0), ("b", 1.0), ("c", 1.0)]) is False
    assert mod._score_head_has_information([("a", 1.0), ("b", 1.1), ("c", 1.0)]) is True


def test_factor_frontend_router_disabled_gives_no_fake_rank_evidence() -> None:
    penalty, rank_score, note = mod._frontend_route_penalty_for_label(
        brain(), "inst_piano", {"top": {}, "structure": {}, "parent": {}}
    )
    assert penalty == 0.0
    assert rank_score == 1.0
    assert note == "router_disabled"


def test_factor_same_family_identity_check() -> None:
    b = brain()
    assert mod._model_ensemble_same_top_family(b, "inst_piano", "inst_synth_chord") is True
    assert mod._model_ensemble_same_top_family(b, "inst_piano", "drum_clap") is False


def test_factor_membership_blocked_candidate_is_recorded_and_cannot_win() -> None:
    b = brain()
    rows = [
        row("inst_piano", current=1.00, spread=4.0, anchor=4.0, linear=5.0),
        row("drum_clap", current=1.05, spread=0.2, anchor=0.01, linear=-10.0, blocked=True),
    ]
    final, meta = mod.ensemble_tournament_choose_label(b, rows, "inst_piano")
    assert final == "inst_piano"
    assert "drum_clap:membership_blocked" in meta["rejected_proposals"]


@pytest.mark.parametrize("sound_name,current_label,bad_label,why", SOUND_CASES)
def test_factor_raw_current_winner_is_the_best_distance_candidate(
    sound_name: str, current_label: str, bad_label: str, why: str
) -> None:
    rows = [
        row(current_label, current=1.00, spread=8.0, anchor=8.0, linear=5.0),
        row(bad_label, current=1.08, spread=0.2, anchor=0.01, linear=-10.0),
    ]
    winner = min(rows, key=lambda r: (r["current_score"], r["label"]))["label"]
    assert winner == current_label, f"bad test setup for {sound_name}: {why}"


@pytest.mark.parametrize("sound_name,current_label,bad_label,why", SOUND_CASES)
def test_factor_bad_helper_heads_cannot_create_cross_family_switch(
    sound_name: str, current_label: str, bad_label: str, why: str
) -> None:
    b = brain()
    rows = [
        row(current_label, current=1.00, neutral=1.00, spread=9.0, anchor=9.0, linear=5.0, router=1.0),
        row(bad_label, current=1.08, neutral=1.08, spread=0.4, anchor=0.01, linear=-10.0, router=1.0),
    ]
    final, meta = mod.ensemble_tournament_choose_label(b, rows, current_label)
    assert final == current_label, f"{sound_name}: {why}; meta={meta}"
    assert meta["switched"] is False
    if b["top_by_label"][current_label] != b["top_by_label"][bad_label]:
        assert f"{bad_label}:cross_family_blocked" in meta["rejected_proposals"]


@pytest.mark.parametrize("sound_name,current_label,bad_label,why", SOUND_CASES)
def test_full_tournament_result_for_twenty_plus_synthetic_sound_types(
    sound_name: str, current_label: str, bad_label: str, why: str
) -> None:
    """Full model-tournament contract test for the observed failure class.

    This is the test that should have existed before the broken code shipped:
    if the raw brain has a reasonable source-family winner, a non-source helper
    head cannot turn it into Drums/FX/Textures/Instruments from a different top
    family just because attack, anchor, spread, or disabled-router ranks look
    attractive.
    """
    b = brain()
    rows = [
        row(current_label, current=1.00, neutral=1.00, spread=7.0, anchor=7.0, linear=4.0, router=1.0),
        row(bad_label, current=1.04, neutral=1.04, spread=0.1, anchor=0.01, linear=-12.0, router=1.0),
        # Add a same-family fallback so the tournament has a legal alternative.
        row(current_label, current=1.02, neutral=1.02, spread=1.5, anchor=1.5, linear=-1.0, router=1.0),
    ]
    # De-duplicate if the fallback is identical to current; keep the setup simple.
    unique: dict[str, dict] = {}
    for r in rows:
        unique[r["label"]] = min([r, unique.get(r["label"], r)], key=lambda x: x["current_score"])
    final, meta = mod.ensemble_tournament_choose_label(b, list(unique.values()), current_label)
    assert b["top_by_label"][final] == b["top_by_label"][current_label], f"{sound_name}: {why}; meta={meta}"
    assert final != bad_label, f"{sound_name}: bad cross-family label won; meta={meta}"


def test_full_same_family_refinement_is_still_allowed() -> None:
    b = brain()
    rows = [
        row("inst_synth_chord", current=1.00, neutral=1.20, spread=9.0, anchor=9.0, linear=5.0),
        row("inst_piano", current=1.04, neutral=0.80, spread=0.5, anchor=0.30, linear=-10.0),
        row("drum_clap", current=1.06, neutral=0.70, spread=0.2, anchor=0.01, linear=-12.0),
    ]
    final, meta = mod.ensemble_tournament_choose_label(b, rows, "inst_synth_chord")
    assert final == "inst_piano"
    assert meta["switched"] is True
    assert "drum_clap:cross_family_blocked" in meta["rejected_proposals"]


def test_old_brain_cross_family_flag_cannot_reenable_broken_tournament_behavior() -> None:
    b = brain()
    # Simulate the scratch brain setting that existed during the bad run.
    b["model_ensemble_allow_cross_family_switch"] = True
    rows = [
        row("inst_piano", current=1.00, neutral=1.00, spread=9.0, anchor=9.0, linear=5.0, router=1.0),
        row("fx_riser", current=1.03, neutral=1.03, spread=0.2, anchor=0.01, linear=-12.0, router=1.0),
    ]
    final, meta = mod.ensemble_tournament_choose_label(b, rows, "inst_piano")
    assert final == "inst_piano"
    assert meta["brain_requested_cross_family_switch"] is True
    assert meta["explicit_cross_family_opt_in"] is False
    assert meta["same_family_switch_only"] is True
    assert "fx_riser:cross_family_blocked" in meta["rejected_proposals"]


def test_current_membership_blocked_does_not_allow_arbitrary_cross_family_jump() -> None:
    b = brain()
    rows = [
        row("inst_piano", current=1.00, neutral=1.00, spread=9.0, anchor=9.0, linear=5.0, blocked=True),
        row("fx_riser", current=1.03, neutral=1.03, spread=0.2, anchor=0.01, linear=-12.0),
    ]
    final, meta = mod.ensemble_tournament_choose_label(b, rows, "inst_piano")
    assert final == "inst_piano"
    assert "fx_riser:cross_family_blocked" in meta["rejected_proposals"]


def test_factor_tournament_anchor_uses_raw_brain_winner_not_penalized_current_score() -> None:
    rows = [
        {"label": "inst_piano", "raw_distance": 1.00, "raw_score": 1.00, "current_score": 9.00},
        {"label": "fx_riser", "raw_distance": 1.80, "raw_score": 1.80, "current_score": 0.50},
        {"label": "drum_clap", "raw_distance": 2.00, "raw_score": 2.00, "current_score": 0.40},
    ]
    assert mod._raw_brain_winner_label(rows) == "inst_piano"


def test_committee_cannot_cross_family_switch_even_when_cross_family_score_is_higher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b = brain()
    # Simulate an old/bad brain requesting committee cross-family switching.
    # The code must ignore that unless the new explicit opt-in exists.
    b["committee_allow_cross_family_switch"] = True
    rows = [
        row("inst_piano", current=1.0),
        row("fx_riser", current=1.1),
        row("drum_clap", current=1.2),
    ]

    fake_scores = {
        "inst_piano": {
            "label": "inst_piano",
            "score": 0.40,
            "current_score": 1.0,
            "blocked": False,
            "votes": {},
            "membership": {},
        },
        "fx_riser": {
            "label": "fx_riser",
            "score": 0.99,
            "current_score": 1.1,
            "blocked": False,
            "votes": {},
            "membership": {},
        },
        "drum_clap": {
            "label": "drum_clap",
            "score": 0.95,
            "current_score": 1.2,
            "blocked": False,
            "votes": {},
            "membership": {},
        },
    }

    def fake_score(
        brain, fingerprint, r, candidate_rows, feature_names_list, top_scores, structure_scores, rival_contrast_facts
    ):
        return fake_scores[r["label"]]

    monkeypatch.setattr(mod, "committee_agreement_score_candidate", fake_score)
    final, meta = mod.committee_agreement_choose_label(b, [0.0] * 36, rows, "inst_piano", [], [], [], {})
    assert final == "inst_piano"
    assert meta["switched"] is False
    assert meta["same_family_switch_only"] is True
    assert meta["committee_requested_cross_family_switch"] is True
    assert meta["committee_explicit_cross_family_opt_in"] is False
    assert "fx_riser:cross_family_blocked" in meta["rejected_proposals"]
    assert "drum_clap:cross_family_blocked" in meta["rejected_proposals"]


def test_committee_same_family_refinement_still_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    b = brain()
    rows = [row("inst_synth_chord", current=1.0), row("inst_piano", current=1.1), row("drum_clap", current=1.2)]
    fake_scores = {
        "inst_synth_chord": {
            "label": "inst_synth_chord",
            "score": 0.40,
            "current_score": 1.0,
            "blocked": False,
            "votes": {},
            "membership": {},
        },
        "inst_piano": {
            "label": "inst_piano",
            "score": 0.99,
            "current_score": 1.1,
            "blocked": False,
            "votes": {},
            "membership": {},
        },
        "drum_clap": {
            "label": "drum_clap",
            "score": 1.00,
            "current_score": 1.2,
            "blocked": False,
            "votes": {},
            "membership": {},
        },
    }

    def fake_score(
        brain, fingerprint, r, candidate_rows, feature_names_list, top_scores, structure_scores, rival_contrast_facts
    ):
        return fake_scores[r["label"]]

    monkeypatch.setattr(mod, "committee_agreement_score_candidate", fake_score)
    final, meta = mod.committee_agreement_choose_label(b, [0.0] * 36, rows, "inst_synth_chord", [], [], [], {})
    assert final == "inst_piano"
    assert meta["switched"] is True
    assert "drum_clap:cross_family_blocked" in meta["rejected_proposals"]


def test_membership_safe_alternative_cannot_cross_family(monkeypatch: pytest.MonkeyPatch) -> None:
    b = brain()
    current_ev = {"weighted_severity": 10.0, "blocked": True, "enabled": True}
    top5 = [("inst_piano", 1.0), ("fx_riser", 1.01), ("drum_clap", 1.02)]

    def fake_membership(brain, fingerprint, label):
        return {"enabled": True, "blocked": False, "weighted_severity": 0.0, "membership_score": 1.0}

    monkeypatch.setattr(mod, "folder_membership_evidence", fake_membership)
    alt, reason = mod.choose_membership_safe_alternative(b, [0.0] * 36, "inst_piano", top5, current_ev)
    assert alt == ""
    assert reason == ""


def test_membership_safe_alternative_can_switch_same_family(monkeypatch: pytest.MonkeyPatch) -> None:
    b = brain()
    current_ev = {"weighted_severity": 10.0, "blocked": True, "enabled": True}
    top5 = [("inst_synth_chord", 1.0), ("inst_piano", 1.01), ("fx_riser", 1.02)]

    def fake_membership(brain, fingerprint, label):
        return {"enabled": True, "blocked": False, "weighted_severity": 0.0, "membership_score": 1.0}

    monkeypatch.setattr(mod, "folder_membership_evidence", fake_membership)
    alt, reason = mod.choose_membership_safe_alternative(b, [0.0] * 36, "inst_synth_chord", top5, current_ev)
    assert alt == "inst_piano"
    assert "membership_gate_switched_to_safer_learned_candidate" in reason
