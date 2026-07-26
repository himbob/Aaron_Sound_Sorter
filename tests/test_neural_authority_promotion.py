from __future__ import annotations

import json

from aaron_sound_sorter.neural_audio.authority_promotion import (
    assess_group_promotion,
    category_authority_group,
    enabled_authority_groups,
)


def test_authority_groups_are_audio_taxonomy_contracts() -> None:
    assert category_authority_group("Instruments/Voice/Vocal One Shots/One Shots") == "voice"
    assert category_authority_group("Drums/Snares/Acoustic Snare/One Shots") == "drums"
    assert category_authority_group("Instruments/Woodwinds/Saxophone/Alto/One Shots") == "isolated_instruments"
    assert category_authority_group("Instruments/Synths/Synth Pad/Loops") == ""


def test_promotion_fails_closed_on_current_style_insufficient_evidence() -> None:
    assessment = assess_group_promotion(
        group="voice",
        calibration_status={
            "status": "insufficient_data",
            "reviewed_unique_hash_count": 19,
            "accepted_count": 2,
            "rejected_count": 17,
        },
        heldout_metrics={
            "example_count": 20,
            "top1_accuracy": 0.9,
            "parent_family_accuracy": 0.95,
            "incorrect_auto_placement_rate": 0.0,
        },
    )

    assert assessment["passed"] is False
    assert "calibration_built" in assessment["failed_checks"]


def test_enabled_groups_require_built_calibration_and_explicit_promotion(tmp_path) -> None:
    status_path = tmp_path / "neural_artifacts/calibration/current/calibration_status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps(
            {
                "status": "built",
                "production_authority_enabled": True,
                "enabled_authority_groups": ["voice", "made_up"],
            }
        ),
        encoding="utf-8",
    )

    assert enabled_authority_groups(tmp_path) == frozenset({"voice"})
