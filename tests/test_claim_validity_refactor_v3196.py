"""Claim-validity locks for the post-v31.77 architecture refactor.

These tests protect the architectural rule that measured role/shape shortcuts
are evidence, not final routing.  Baby/full brain lane evidence remains usable,
but a shortcut claim may not beat a concrete raw candidate unless its real
candidate evidence actually wins the rank contest.
"""

from __future__ import annotations

from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_folder_path


def claim(
    *,
    path: str,
    source: str,
    strength: float,
    score: float | None,
    can_override: bool,
    is_real_candidate: bool,
) -> ConsensusClaim:
    """Build a compact test claim from a folder path."""
    return claim_from_folder_path(
        folder_path=path,
        source=source,
        reason="test claim",
        shared=[],
        raw_candidate_score=score,
        brain_rank=None,
        physics_rank=None,
        shared_winner=path,
        can_override=can_override,
        strength=strength,
        is_real_candidate=is_real_candidate,
    )


def test_profile_kick_shortcut_loses_to_strong_specific_fx_sub_hit() -> None:
    """A role-compatible kick shortcut must not erase strong FX/Sub Hit evidence."""
    raw = claim(
        path="FX/Impacts and Hits/Sub Hit/One Shots",
        source="strong_consensus",
        strength=0.82,
        score=3.0,
        can_override=False,
        is_real_candidate=True,
    )
    kick = claim(
        path="Drums/Kick Drums/Generic Kick/One Shots",
        source="profile_candidate_kick_claim",
        strength=0.98,
        score=3.5,
        can_override=True,
        is_real_candidate=True,
    )

    winner = FamilyClaimArbiter().pick_winner(raw_claim=raw, claims=[kick])

    assert winner.folder_path == "FX/Impacts and Hits/Sub Hit/One Shots"
    assert winner.source == "strong_consensus"


def test_profile_kick_candidate_still_cannot_cross_family_from_weak_specific_raw() -> None:
    """Profile kick shortcuts are old behavior, not cross-family routing authority."""
    raw = claim(
        path="FX/Impacts and Hits/Sub Hit/One Shots",
        source="strong_consensus",
        strength=0.40,
        score=8.0,
        can_override=False,
        is_real_candidate=True,
    )
    kick = claim(
        path="Drums/Kick Drums/Generic Kick/One Shots",
        source="profile_candidate_kick_claim",
        strength=0.98,
        score=4.0,
        can_override=True,
        is_real_candidate=True,
    )

    winner = FamilyClaimArbiter().pick_winner(raw_claim=raw, claims=[kick])

    assert winner.folder_path == "FX/Impacts and Hits/Sub Hit/One Shots"


def test_inferred_role_shape_broad_claim_loses_to_strong_specific_raw() -> None:
    """Measured role/shape evidence alone cannot cross families."""
    raw = claim(
        path="FX/Impacts and Hits/Sub Hit/One Shots",
        source="strong_consensus",
        strength=0.80,
        score=4.0,
        can_override=False,
        is_real_candidate=True,
    )
    broad = claim(
        path="Drums/Drum Loops/Loops",
        source="parent_eligibility_broad_bucket",
        strength=1.0,
        score=None,
        can_override=True,
        is_real_candidate=False,
    )

    winner = FamilyClaimArbiter().pick_winner(raw_claim=raw, claims=[broad])

    assert winner.folder_path == "FX/Impacts and Hits/Sub Hit/One Shots"
    assert winner.source == "strong_consensus"


def test_baby_brain_candidate_evidence_still_beats_generic_raw_bucket() -> None:
    """The refactor must protect useful baby-lane evidence, not hide it."""
    raw = claim(
        path="Instruments/Instrument Loops/Loops",
        source="strong_consensus",
        strength=0.40,
        score=10.0,
        can_override=False,
        is_real_candidate=True,
    )
    baby_reed = claim(
        path="Instruments/Brass and Woodwinds/Loops",
        source="baby_recall_brass_woodwind_claim",
        strength=0.92,
        score=5.0,
        can_override=True,
        is_real_candidate=True,
    )

    winner = FamilyClaimArbiter().pick_winner(raw_claim=raw, claims=[baby_reed])

    assert winner.folder_path == "Instruments/Brass and Woodwinds/Loops"
