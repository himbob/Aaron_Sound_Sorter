"""Regression checks for PEP/readability cleanup helper functions."""

from aaron_sound_sorter.domain.roles import average_strength, inverse_ramp, ramp
from aaron_sound_sorter.voters.brain_lane_competence import parent_path, top_family
from aaron_sound_sorter.voters.physics_layer_utils import (
    candidate_is_broad_instrument,
    candidate_is_sax,
    instrument_candidate_matches_branch,
)
from aaron_sound_sorter.voters.scoring_tools import role_strength


def test_role_strength_helpers_keep_existing_numeric_behavior() -> None:
    roles = {"bass_loop": 0.75, "diagnostics": {"nested": True}, "bad": "not-a-number"}

    assert role_strength(roles, "bass_loop") == 0.75
    assert role_strength(roles, "diagnostics") == 0.0
    assert role_strength(roles, "bad") == 0.0
    assert ramp(0.5, 0.0, 1.0) == 0.5
    assert inverse_ramp(0.25, 0.0, 1.0) == 0.75
    assert average_strength(1.0, 0.0, 0.5) == 0.5


def test_candidate_label_path_helpers_remain_source_name_blind() -> None:
    sax_label = "Instruments/Brass and Woodwinds/Sax/Loops"
    broad_instrument_label = "Instruments/Instrument Loops/Loops"

    assert candidate_is_sax(sax_label)
    assert not candidate_is_sax("Instruments/Voice/Vocal Loops")
    assert candidate_is_broad_instrument(broad_instrument_label)
    assert instrument_candidate_matches_branch(sax_label, "Woodwinds")
    assert not instrument_candidate_matches_branch("Drums/Drum Loops/Loops", "Woodwinds")


def test_brain_lane_path_helpers_keep_parent_and_top_rollups() -> None:
    label_path = "Instruments/Brass and Woodwinds/Sax/Loops"

    assert top_family(label_path) == "Instruments"
    assert parent_path(label_path) == "Instruments/Brass and Woodwinds"
