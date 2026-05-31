"""Regression tests for brain-lane expected-quality diagnostics."""

from __future__ import annotations

from pathlib import Path

from aaron_sound_sorter.infrastructure.brain_lane_validation import (
    DIRTY_EXPECTED,
    STRONG_EXPECTED,
    brain_lane_group_winner_rows,
    competence_summary_rows,
    expected_profile_from_source_path,
    groups_are_strict_match,
    label_group,
)


def test_expected_profile_reads_explicit_strong_expected_folder() -> None:
    profile = expected_profile_from_source_path(Path("run/input/strong_expected_sax_reed/sample.wav"))

    assert profile["expected_group"] == "sax_reed"
    assert profile["expected_source_quality"] == STRONG_EXPECTED


def test_fx_pack_melody_loop_is_dirty_subgroup_not_plain_fx() -> None:
    profile = expected_profile_from_source_path(
        Path("FX_Aaron2/Loops/WS_KIT_3_West_Coast_Melody_Loop_Gnarly_Dm_100BPM.wav")
    )

    assert profile["expected_group"] == "melody_loop_inside_fx_pack"
    assert profile["expected_source_quality"] == DIRTY_EXPECTED


def test_true_transition_fx_label_group_is_strict_match() -> None:
    predicted = label_group("FX/Risers and Builds/Riser/Long FX")

    assert predicted == "true_transition_fx"
    assert groups_are_strict_match(predicted, "true_transition_fx")


def test_dead_harmonic_lane_surfaces_in_group_winners() -> None:
    summary = competence_summary_rows(
        [
            {
                "source_path": "input/strong_expected_sax_reed/example.wav",
                "expected_group": "sax_reed",
                "expected_source_quality": STRONG_EXPECTED,
                "brain_ensemble_top1": "Instruments/Woodwinds/Saxophone/One Shots",
                "brain_ensemble_top1_group": "sax_reed",
                "brain_ensemble_top1_top_family": "Instruments",
                "brain_ensemble_top5_json": '["Instruments/Woodwinds/Saxophone/One Shots"]',
                "brain_ensemble_strict_correct": "True",
                "brain_ensemble_broad_safe": "True",
                "harmonic_core_baby_top1": "",
                "harmonic_core_baby_top5_json": "[]",
                "harmonic_core_baby_strict_correct": "False",
                "harmonic_core_baby_broad_safe": "False",
            }
        ]
    )
    winners = brain_lane_group_winner_rows(summary)

    assert "harmonic_core_baby" in winners[0]["dead_or_missing_brain_lanes"]
