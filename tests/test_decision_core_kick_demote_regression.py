"""Regression for supported kicks being demoted into other drum one-shot leaves."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import ConsensusDecision, SharedAudioFacts
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import EligibilityDecision


def candidate(path: str, score: float) -> dict:
    """Build a synthetic shared candidate row."""
    return {"folder_path": path, "label": path, "top_family": path.split("/", 1)[0], "combined_rank_score": score}


def test_short_hit_guard_must_not_demote_supported_kick_to_tom() -> None:
    """A guard can block unsafe families, but it must not change Kick into Tom."""
    raw = ConsensusDecision(
        final_label="Drums/Kick Drums/Generic Kick/One Shots",
        final_top="Drums",
        folder_path="Drums/Kick Drums/Generic Kick/One Shots",
        consensus_status="strong_consensus",
        reason="synthetic raw kick consensus",
        combined_rank_score=8.0,
        shared_candidates=[
            candidate("Drums/Toms/Generic Tom/One Shots", 7.0),
            candidate("Instruments/Bass/Synth Bass/One Shots", 6.0),
        ],
    )
    eligibility = EligibilityDecision(
        role_name="low_kick_like_hit",
        confidence=0.82,
        allowed_top_families=("Drums", "_TO_REVIEW"),
        blocked_path_fragments=("FX", "Instruments", "Drum Loops"),
        broad_folder_path="Drums/Kick Drums/Generic Kick/One Shots",
        reason="synthetic low kick-like hit",
    )
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={"shape_vote": {"primary_shape": "hit_with_tail", "confidence": 0.90}},
    )

    final = DecisionCoreV2().apply_eligibility(raw, eligibility, facts)

    assert final.folder_path == "Drums/Kick Drums/Generic Kick/One Shots"
