"""Regression guard: stable FX smoke winners must not become review.

These source-name-blind tests reproduce the post-true-bucket failure where a
measured conflict role moved stable vocal, sax/instrument, and drum-loop winners
into _TO_REVIEW/Measured Role Conflict.  The fix must preserve real high-risk
conflict protections such as guitar-like hits being forced to toms.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import ConsensusDecision
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import EligibilityDecision


def candidate(path: str, score: float, *, top: str | None = None) -> dict:
    return {
        "folder_path": path,
        "label": path,
        "top_family": top or path.split("/", 1)[0],
        "combined_rank_score": score,
        "candidate_role_signature": {},
    }


def raw(path: str, top: str, *, score: float = 8.0, candidates: list[dict] | None = None) -> ConsensusDecision:
    return ConsensusDecision(
        final_label=path,
        final_top=top,
        folder_path=path,
        consensus_status="strong_consensus",
        reason="synthetic raw decision",
        combined_rank_score=score,
        shared_candidates=candidates or [],
    )


def eligibility(role: str, allowed: tuple[str, ...], broad: str, *, confidence: float = 0.84) -> EligibilityDecision:
    return EligibilityDecision(
        role_name=role,
        confidence=confidence,
        allowed_top_families=allowed,
        blocked_path_fragments=(),
        broad_folder_path=broad,
        reason="synthetic eligibility",
    )


def conflict_eligibility(allowed: tuple[str, ...]) -> EligibilityDecision:
    return eligibility("measured_role_conflict", allowed, "_TO_REVIEW/Measured Role Conflict", confidence=0.78)


def test_stable_vocal_or_voice_winner_is_not_dumped_to_review_by_generic_conflict_role() -> None:
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "FX/Human and Voice FX/Vocal One Shots",
            "FX",
            score=8,
            candidates=[
                candidate("FX/Human and Voice FX/Vocal One Shots", 8),
                candidate("Instruments/Instrument Loops/Loops", 13),
            ],
        ),
        conflict_eligibility(("FX", "Instruments", "_TO_REVIEW")),
    )
    assert final.final_top == "FX"
    assert final.folder_path == "FX/Human and Voice FX/Vocal One Shots"


def test_stable_sax_or_reed_instrument_winner_is_not_dumped_to_review_by_generic_conflict_role() -> None:
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Brass and Woodwinds/Loops",
            "Instruments",
            score=7,
            candidates=[
                candidate("Instruments/Brass and Woodwinds/Loops", 7),
                candidate("Instruments/Instrument Loops/Loops", 10),
                candidate("FX/Human and Voice FX/Crowd/Long FX", 18),
            ],
        ),
        conflict_eligibility(("Instruments", "_TO_REVIEW")),
    )
    assert final.final_top == "Instruments"
    assert final.folder_path == "Instruments/Brass and Woodwinds/Loops"


def test_stable_drum_loop_winner_is_not_dumped_to_review_by_generic_conflict_role() -> None:
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Drums/Drum Loops/Loops",
            "Drums",
            score=8,
            candidates=[
                candidate("Drums/Drum Loops/Loops", 8),
                candidate("Instruments/Instrument Loops/Loops", 13),
            ],
        ),
        conflict_eligibility(("Drums", "_TO_REVIEW")),
    )
    assert final.final_top == "Drums"
    assert final.folder_path == "Drums/Drum Loops/Loops"


def test_allowed_generic_conflict_keeps_raw_instead_of_dumping_to_review() -> None:
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Instrument Loops/Loops",
            "Instruments",
            score=10,
            candidates=[
                candidate("Instruments/Instrument Loops/Loops", 10),
                candidate("FX/Structural and Transitional FX/Risers and Builds/Build", 10.5),
            ],
        ),
        conflict_eligibility(("Instruments", "FX", "_TO_REVIEW")),
    )
    assert final.final_top == "Instruments"
    assert final.folder_path == "Instruments/Instrument Loops/Loops"


def test_bad_guitar_like_tom_conflict_still_reviews() -> None:
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Drums/Toms/Generic Tom/One Shots",
            "Drums",
            score=13,
            candidates=[
                candidate("Instruments/Guitar/Electric Guitar/One Shots", 10),
                candidate("Drums/Toms/Generic Tom/One Shots", 13),
            ],
        ),
        eligibility(
            "protected_percussive_one_shot", ("Drums", "_TO_REVIEW"), "Drums/Percussion/Generic Percussion/One Shots"
        ),
    )
    assert final.final_top == "_TO_REVIEW"
    assert "pitched instrument one-shot conflict" in final.reason
