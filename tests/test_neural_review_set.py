from __future__ import annotations

from tools.build_neural_review_set import (
    _discovery_score,
    _target_review_counts,
)


def test_review_target_counts_cover_sparse_labels_and_known_boundaries() -> None:
    counts = {
        "Drums/Kick Drums/Generic Kick/One Shots": 1,
        "Instruments/Voice/Vocal Loops/Loops": 3,
        "Instruments/Woodwinds/Saxophone/Loops": 6,
        "Instruments/Keys/Piano/Loops": 5,
    }

    targets = _target_review_counts(
        counts,
        candidates_per_gap=4,
        boundary_candidates=3,
    )

    assert targets["Drums/Kick Drums/Generic Kick/One Shots"] == 4
    assert targets["Instruments/Voice/Vocal Loops/Loops"] == 3
    assert targets["Instruments/Woodwinds/Saxophone/Loops"] == 3
    assert "Instruments/Keys/Piano/Loops" not in targets


def test_name_discovery_is_review_metadata_not_an_approved_target() -> None:
    sax_score = _discovery_score(
        "Instruments/Woodwinds/Saxophone/Loops",
        "pack loop brass saxophone wet audio wav",
    )
    vocal_score = _discovery_score(
        "Instruments/Voice/Vocal Loops/Loops",
        "pack loop vocal female phrase audio wav",
    )
    unrelated_score = _discovery_score(
        "Instruments/Woodwinds/Saxophone/Loops",
        "pack loop acoustic piano audio wav",
    )

    assert sax_score > 0.0
    assert vocal_score > 0.0
    assert unrelated_score == 0.0
