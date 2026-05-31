from __future__ import annotations

from aaron_sound_sorter.domain.models import CategoryGuess, VoterResult
from aaron_sound_sorter.voters.brain_lane_competence import (
    brain_lane_candidate_diagnostics,
    calibrated_lane_confidence,
)
from aaron_sound_sorter.voters.brain_recall import combine_full_and_balanced_brain_votes


def guess(path: str, rank: int, score: float) -> CategoryGuess:
    return CategoryGuess(
        label=path,
        folder_path=path,
        top_family=path.split("/", 1)[0],
        score=score,
        confidence=1.0 / (1.0 + score),
        rank=rank,
        reason="test",
    )


def test_calibrated_lane_confidence_reflects_rank_gap() -> None:
    near_tie = calibrated_lane_confidence(rank=1, raw_confidence=0.80, rank_gap_to_next=0.05, lane_weight=1.0)
    clear_gap = calibrated_lane_confidence(rank=1, raw_confidence=0.80, rank_gap_to_next=3.0, lane_weight=1.0)

    assert clear_gap > near_tie


def test_brain_lane_diagnostics_tracks_exact_parent_and_top_agreement() -> None:
    sax = "Instruments/Woodwinds/Saxophone/Loops"
    flute = "Instruments/Woodwinds/Flute/Loops"
    lanes = {
        "core_baby": {
            sax: guess(sax, 1, 0.7),
            "FX/Human and Voice FX/Crowd/Long FX": guess("FX/Human and Voice FX/Crowd/Long FX", 2, 2.1),
        },
        "spread_baby": {sax: guess(sax, 1, 0.9)},
        "full": {flute: guess(flute, 1, 0.8)},
    }

    diagnostics = brain_lane_candidate_diagnostics(
        label=sax,
        present_lanes=["core_baby", "spread_baby"],
        lane_guesses=lanes,
        weight_profile="generic_pitched_full_guarded",
    ).to_dict()

    assert diagnostics["lane_exact_agreement_count"] == 2
    assert diagnostics["lane_parent_agreement_count"] == 2
    assert diagnostics["lane_top_family_agreement_count"] == 2
    assert diagnostics["outlier_only_candidate"] is False
    assert diagnostics["lane_calibrated_confidence"] > 0.0


def test_brain_lane_diagnostics_does_not_overcount_sibling_lanes_as_label_authority() -> None:
    sax = "Instruments/Woodwinds/Saxophone/Loops"
    flute = "Instruments/Woodwinds/Flute/Loops"
    vocal = "Instruments/Voice/Vocal Phrase/Loops"
    lanes = {
        "outlier_baby": {sax: guess(sax, 1, 0.6)},
        "core_baby": {flute: guess(flute, 1, 0.7)},
        "spread_baby": {vocal: guess(vocal, 1, 0.8)},
    }

    diagnostics = brain_lane_candidate_diagnostics(
        label=sax,
        present_lanes=["outlier_baby"],
        lane_guesses=lanes,
        weight_profile="generic_pitched_full_guarded",
    ).to_dict()

    assert diagnostics["lane_exact_agreement_count"] == 1
    assert diagnostics["lane_parent_agreement_count"] == 1
    assert diagnostics["lane_top_family_agreement_count"] == 1
    assert diagnostics["outlier_only_candidate"] is True
    assert "outlier-only" in diagnostics["lane_authority_reason"]


def test_combined_brain_vote_exposes_competence_diagnostics() -> None:
    sax = "Instruments/Woodwinds/Saxophone/Loops"
    full = VoterResult("brain_full", [guess("Instruments/Instrument Loops/Loops", 1, 0.8)])
    core = VoterResult("brain_core_baby", [guess(sax, 1, 0.7)])
    spread = VoterResult("brain_spread_baby", [guess(sax, 1, 0.9)])
    outlier = VoterResult("brain_outlier_baby", [guess("FX/Animals and Creatures/Bird/Long FX", 1, 0.6)])

    combined = combine_full_and_balanced_brain_votes(
        full_result=full,
        baby_results={"core_baby": core, "spread_baby": spread, "outlier_baby": outlier},
        max_guesses=10,
    )
    sax_guess = next(item for item in combined.guesses if item.label == sax)
    bird_guess = next(item for item in combined.guesses if item.label.startswith("FX/Animals"))

    assert sax_guess.evidence["brain_lane_competence_profile"]
    assert sax_guess.evidence["lane_exact_agreement_count"] == 2
    assert sax_guess.evidence["lane_parent_agreement_count"] >= 2
    assert bird_guess.evidence["outlier_only_candidate"] is True
    assert "outlier-only" in bird_guess.evidence["lane_authority_reason"]
