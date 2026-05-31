"""v31.97 architecture locks: roles diagnose, voters rank raw evidence."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.consensus_winner import ConsensusWinnerPreselector
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path
from aaron_sound_sorter.voters.scoring_tools import role_compatibility_adjustment


def facts_with_percussive_role() -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "measured_roles": {"percussive_one_shot": 1.0},
            "shape_vote": {"primary_shape": "hit_with_tail", "confidence": 0.9},
        },
    )


def test_role_compatibility_never_changes_voter_score() -> None:
    facts = facts_with_percussive_role()
    delta, evidence = role_compatibility_adjustment(
        "Instruments/Bass/808 Bass/One Shots",
        {"labels": ["Instruments/Bass/808 Bass/One Shots"]},
        facts,
        voter_name="brain",
    )

    assert delta == 0.0
    assert evidence["role_adjustment_applied"] == 0.0
    assert evidence["role_adjustment_mode"] == "diagnostic_only"
    assert evidence["role_recommended_adjustment"] != 0.0


def test_winner_preselector_returns_raw_best_shared_candidate() -> None:
    shared = [
        {
            "label": "Instruments/Bass/808 Bass/One Shots",
            "folder_path": "Instruments/Bass/808 Bass/One Shots",
            "top_family": "Instruments",
            "combined_rank_score": 5.0,
            "brain_rank": 4,
            "physics_rank": 1,
        },
        {
            "label": "Drums/Toms/Generic Tom/One Shots",
            "folder_path": "Drums/Toms/Generic Tom/One Shots",
            "top_family": "Drums",
            "combined_rank_score": 13.0,
            "brain_rank": 5,
            "physics_rank": 8,
        },
    ]

    winner = ConsensusWinnerPreselector().choose(shared, facts_with_percussive_role())

    assert winner["folder_path"] == "Instruments/Bass/808 Bass/One Shots"


def test_profile_kick_shortcut_cannot_cross_family_from_specific_bass_raw() -> None:
    raw = claim_from_folder_path(
        folder_path="Instruments/Bass/808 Bass/One Shots",
        source="strong_consensus",
        reason="raw brain physics winner",
        shared=[],
        shared_winner="Instruments/Bass/808 Bass/One Shots",
        raw_candidate_score=5.0,
        brain_rank=4,
        physics_rank=1,
        can_override=False,
        strength=0.80,
        is_real_candidate=True,
    )
    shortcut = claim_from_folder_path(
        folder_path="Drums/Kick Drums/Generic Kick/One Shots",
        source="profile_candidate_kick_claim",
        reason="old shortcut",
        shared=[],
        shared_winner="Drums/Kick Drums/Generic Kick/One Shots",
        raw_candidate_score=3.0,
        brain_rank=1,
        physics_rank=2,
        can_override=True,
        strength=0.98,
        is_real_candidate=True,
    )

    final = FamilyClaimArbiter().pick_winner(raw_claim=raw, claims=[shortcut])

    assert final.folder_path == "Instruments/Bass/808 Bass/One Shots"
