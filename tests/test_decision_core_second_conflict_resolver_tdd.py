"""Second-wave TDD regressions for confirmed post-conflict-resolver bugs.

Rule: every sorter bug fix must first add a failing synthetic regression, prove
it fails, then patch.  These tests reproduce the remaining classes found in the
random-folder/failure-review-pack logs without storing real audio files.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import ConsensusDecision
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import EligibilityDecision


def raw(path: str, top: str, *, score: float = 12.0, candidates: list[dict] | None = None) -> ConsensusDecision:
    return ConsensusDecision(
        final_label=path,
        final_top=top,
        folder_path=path,
        consensus_status="strong_consensus",
        reason="synthetic raw decision",
        combined_rank_score=score,
        shared_candidates=candidates or [],
    )


def eligibility(role: str, top: tuple[str, ...], broad: str, *, confidence: float = 0.80) -> EligibilityDecision:
    return EligibilityDecision(
        role_name=role,
        confidence=confidence,
        allowed_top_families=top,
        blocked_path_fragments=(),
        broad_folder_path=broad,
        reason="synthetic eligibility",
    )


def unknown_eligibility() -> EligibilityDecision:
    return EligibilityDecision(role_name="unknown", confidence=0.0)


def candidate(path: str, score: float, *, top: str | None = None) -> dict:
    return {
        "folder_path": path,
        "label": path,
        "top_family": top or path.split("/", 1)[0],
        "combined_rank_score": score,
        "candidate_role_signature": {},
    }


def test_generic_instrument_loop_is_not_promoted_to_brass_woodwinds_by_weak_reed_role() -> None:
    """Mixed/vocal/melodic loops should stay broad, not be narrowed to woodwinds."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Instrument Loops/Loops",
            "Instruments",
            candidates=[
                candidate("Instruments/Instrument Loops/Loops", 10),
                candidate("Instruments/Brass and Woodwinds/Loops", 18),
            ],
        ),
        eligibility(
            "pitched_reed_or_instrument_loop",
            ("Instruments", "_TO_REVIEW"),
            "Instruments/Brass and Woodwinds/Loops",
        ),
    )
    assert final.folder_path == "Instruments/Instrument Loops/Loops"


def test_fx_or_vocal_loop_is_not_broadened_to_brass_woodwinds_by_weak_reed_role() -> None:
    """A weak reed-like role may review, but must not override into woodwinds."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX",
            "FX",
            candidates=[
                candidate("FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX", 9),
                candidate("FX/Human and Voice FX/Spoken Voice/Long FX", 12),
                candidate("Instruments/Brass and Woodwinds/Loops", 30),
            ],
        ),
        eligibility(
            "pitched_reed_or_instrument_loop",
            ("Instruments", "_TO_REVIEW"),
            "Instruments/Brass and Woodwinds/Loops",
        ),
    )
    assert final.final_top == "_TO_REVIEW"
    assert "reed/woodwind over-narrowing conflict" in final.reason


def test_exact_brass_woodwind_raw_winner_still_survives_weak_reed_role() -> None:
    """Do not break actual sax/woodwind decisions that voters already selected."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Brass and Woodwinds/Loops",
            "Instruments",
            candidates=[
                candidate("Instruments/Brass and Woodwinds/Loops", 7),
                candidate("Instruments/Instrument Loops/Loops", 13),
            ],
        ),
        eligibility(
            "pitched_reed_or_instrument_loop",
            ("Instruments", "_TO_REVIEW"),
            "Instruments/Brass and Woodwinds/Loops",
        ),
    )
    assert final.folder_path == "Instruments/Brass and Woodwinds/Loops"


def test_raw_fx_slam_with_close_drum_candidate_reviews_even_when_parent_role_unknown() -> None:
    """Snare/ride/shuffle material should not become FX/Slam when drum evidence is close."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "FX/Impacts and Hits/Slam/Long FX",
            "FX",
            score=11,
            candidates=[
                candidate("Drums/Snares/Generic Snare/One Shots", 9),
                candidate("FX/Impacts and Hits/Slam/Long FX", 11),
                candidate("Drums/Drum Loops/Loops", 15),
            ],
        ),
        unknown_eligibility(),
    )
    assert final.final_top == "_TO_REVIEW"
    assert "drum/FX conflict" in final.reason


def test_clear_fx_slam_without_close_drum_candidate_stays_fx() -> None:
    """Do not damage obvious FX impact/slam cases."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "FX/Impacts and Hits/Slam/Long FX",
            "FX",
            score=6,
            candidates=[
                candidate("FX/Impacts and Hits/Slam/Long FX", 6),
                candidate("FX/Impacts and Hits/Impact/Long FX", 8),
                candidate("Drums/Snares/Generic Snare/One Shots", 30),
            ],
        ),
        unknown_eligibility(),
    )
    assert final.folder_path == "FX/Impacts and Hits/Slam/Long FX"


def test_bass_loop_role_overrides_raw_drum_loop_to_bass_not_drums() -> None:
    """A measured bass-loop role must not be allowed to land in Drum Loops."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Drums/Drum Loops/Loops",
            "Drums",
            candidates=[
                candidate("Instruments/Bass/Bass Loops", 8),
                candidate("Drums/Drum Loops/Loops", 12),
            ],
        ),
        eligibility(
            "bass_loop",
            ("Instruments", "_TO_REVIEW"),
            "Instruments/Bass/Bass Loops",
        ),
    )
    assert final.folder_path == "Instruments/Bass/Bass Loops"


def test_vocal_or_voice_candidate_blocks_drum_loop_broadening_from_non_drum_raw() -> None:
    """If raw/candidates say instrument or voice, drum-loop broadening should review."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Instrument Loops/Loops",
            "Instruments",
            score=10,
            candidates=[
                candidate("Instruments/Instrument Loops/Loops", 9),
                candidate("FX/Human and Voice FX/Crowd/Long FX", 10),
                candidate("Drums/Drum Loops/Loops", 22),
            ],
        ),
        eligibility(
            "drum_loop",
            ("Drums", "_TO_REVIEW"),
            "Drums/Drum Loops/Loops",
        ),
    )
    assert final.final_top == "_TO_REVIEW"
    assert "vocal loop conflict" in final.reason
