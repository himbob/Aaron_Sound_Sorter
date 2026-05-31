"""Regression guard: curated FX smoke examples must not be dumped to review.

These tests are source-name blind.  They encode the resolver failures observed in
FX_Aaron2 smoke manifests where broad curated examples landed in
_TO_REVIEW/Measured Role Conflict.  The fix must still allow genuine dangerous
conflicts to review, but it must not use review as the normal answer for stable
raw/broad winners.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import ConsensusDecision, SharedAudioFacts
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


def drum_loop_facts() -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "measured_roles": {
                "drum_loop": 0.86,
                "detected_parent_role": "drum_loop",
            },
            "shape_vote": {"primary_shape": "drum_loop", "confidence": 0.82},
        },
    )


def test_pitched_conflict_loop_keeps_allowed_generic_instrument_loop_instead_of_review() -> None:
    """Sax/reed-like curated loops may be broad Instruments, not review."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Instrument Loops/Loops",
            "Instruments",
            score=10,
            candidates=[
                candidate("Instruments/Instrument Loops/Loops", 10),
                candidate("FX/Designed Noise FX/Glitch/Long FX", 12),
            ],
        ),
        eligibility(
            "pitched_percussion_conflict_loop",
            ("Drums", "Instruments", "_TO_REVIEW"),
            "_TO_REVIEW/Measured Role Conflict",
            confidence=0.74,
        ),
    )
    assert final.final_top == "Instruments"
    assert final.folder_path == "Instruments/Instrument Loops/Loops"


def test_stable_drum_loop_keeps_raw_drum_loop_even_with_close_voice_candidate() -> None:
    """Drum-top loops from curated smoke packs should not become review."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Drums/Drum Loops/Loops",
            "Drums",
            score=8,
            candidates=[
                candidate("Drums/Drum Loops/Loops", 8),
                candidate("FX/Human and Voice FX/Breath/Long FX", 11),
            ],
        ),
        eligibility("drum_loop", ("Drums", "_TO_REVIEW"), "Drums/Drum Loops/Loops", confidence=0.90),
    )
    assert final.final_top == "Drums"
    assert final.folder_path == "Drums/Drum Loops/Loops"


def test_incompatible_fx_drop_with_drum_loop_role_broadens_to_drum_loop_not_review() -> None:
    """A measured drum loop whose raw leaf is FX/drop should become Drum Loops."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX",
            "FX",
            score=8,
            candidates=[
                candidate(
                    "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX", 8
                ),
                candidate("FX/Human and Voice FX/Breath/Long FX", 12),
            ],
        ),
        eligibility("drum_loop", ("Drums", "_TO_REVIEW"), "Drums/Drum Loops/Loops", confidence=0.78),
        facts=drum_loop_facts(),
    )
    assert final.final_top == "Drums"
    assert final.folder_path == "Drums/Drum Loops/Loops"


def test_vocal_phrase_without_facts_or_candidate_does_not_force_voice_bucket() -> None:
    """Vocal eligibility alone is not enough to move concrete machine FX."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "FX/Everyday Foley/Machines/Motor/One Shots",
            "FX",
            score=8,
            candidates=[
                candidate("FX/Everyday Foley/Machines/Motor/One Shots", 8),
                candidate("FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX", 10),
            ],
        ),
        EligibilityDecision(
            role_name="vocal_phrase",
            confidence=0.86,
            allowed_top_families=("FX", "Instruments", "_TO_REVIEW"),
            blocked_path_fragments=("Machines", "Motor", "Drums", "Percussion"),
            broad_folder_path="FX/Human and Voice FX",
            reason="synthetic vocal phrase",
        ),
    )
    assert final.final_top == "FX"
    assert final.folder_path == "FX/Everyday Foley/Machines/Motor/One Shots"


def test_generic_pitched_loop_with_close_fx_candidate_stays_nonreview_when_raw_is_allowed() -> None:
    """Do not dump pads/keys/sax-like loops to review from close FX evidence alone."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Instrument Loops/Loops",
            "Instruments",
            score=9,
            candidates=[
                candidate("Instruments/Instrument Loops/Loops", 9),
                candidate("FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX", 10),
            ],
        ),
        eligibility(
            "pitched_music_loop", ("Instruments", "_TO_REVIEW"), "Instruments/Instrument Loops/Loops", confidence=1.0
        ),
    )
    assert final.final_top != "_TO_REVIEW"
    assert final.folder_path in {
        "Instruments/Instrument Loops/Loops",
        "FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX",
    }
