"""v31.87 architecture and lane-competence regression checks."""

from __future__ import annotations

from pathlib import Path

from aaron_sound_sorter.voters.brain_recall import brain_lane_vote_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_architecture_review_is_packaged_for_future_ai() -> None:
    required_doc = PROJECT_ROOT / "docs" / "REQUIRED_Aaron_SoundSorter_v3177_Architecture_Review.md"
    must_read_doc = PROJECT_ROOT / "docs" / "AI_MUST_READ_CURRENT_ARCHITECTURE.md"

    assert required_doc.exists()
    assert must_read_doc.exists()
    assert "Family Claim Decision Ownership" in required_doc.read_text(encoding="utf-8")
    assert "Final placement belongs" in must_read_doc.read_text(encoding="utf-8")


def test_lane_competence_weights_do_not_treat_all_baby_brains_as_global_truth() -> None:
    full_drum = brain_lane_vote_weight("full", weight_profile="drum_loop_full_priority")
    core_drum = brain_lane_vote_weight("core_baby", weight_profile="drum_loop_full_priority")
    spread_drum = brain_lane_vote_weight("spread_baby", weight_profile="drum_loop_full_priority")

    assert full_drum > core_drum
    assert full_drum > spread_drum

    outlier_reed = brain_lane_vote_weight(
        "outlier_baby",
        weight_profile="generic_pitched_full_guarded",
        label="Instruments/Woodwinds/Saxophone/One Shots",
    )
    outlier_generic = brain_lane_vote_weight(
        "outlier_baby",
        weight_profile="generic_pitched_full_guarded",
        label="Instruments/Synths/Synth Lead/One Shots",
    )

    assert outlier_reed > outlier_generic
    assert outlier_reed < full_drum
