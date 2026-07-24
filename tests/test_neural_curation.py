from __future__ import annotations

from aaron_sound_sorter.neural_audio.curation import (
    CALIBRATION_USE,
    EXCLUDED_CONFLICT_USE,
    EXCLUDED_DUPLICATE_USE,
    EXCLUDED_PENDING_REVIEW_USE,
    FINAL_HELDOUT_USE,
    TRAINING_USE,
    VALIDATION_USE,
    ProvenanceCandidate,
    assign_leakage_safe_uses,
    readiness_tier,
)


def candidate(
    candidate_id: str,
    *,
    label: str = "Instruments/Voice/Vocal Loops/Loops",
    source_kind: str = "locked_human_seed",
    approved: bool = True,
    group: str | None = None,
) -> ProvenanceCandidate:
    digest = group or f"{int(candidate_id):064x}"
    return ProvenanceCandidate(
        candidate_id=candidate_id,
        audio_path=f"/display/{candidate_id}.wav",
        intended_label=label,
        label_source="test",
        source_kind=source_kind,
        source_group="panel",
        human_approved=approved,
        file_sha256=digest,
        decoded_audio_sha256=digest,
        normalized_audio_sha256=digest,
    )


def test_four_way_split_is_duplicate_group_safe_and_keeps_gui_feedback_in_training() -> None:
    rows = [candidate(str(index)) for index in range(1, 9)]
    rows.append(candidate("9", source_kind="recent_gui_correction"))
    duplicate = candidate("10", group=rows[0].normalized_audio_sha256)
    rows.append(duplicate)

    assigned = assign_leakage_safe_uses(rows)
    uses = {row.allowed_use for row in assigned}
    by_id = {row.candidate_id: row for row in assigned}

    assert {TRAINING_USE, VALIDATION_USE, CALIBRATION_USE, FINAL_HELDOUT_USE}.issubset(uses)
    assert by_id["9"].allowed_use == TRAINING_USE
    assert by_id["10"].allowed_use == EXCLUDED_DUPLICATE_USE


def test_untrusted_and_conflicting_supervision_are_excluded() -> None:
    shared = "a" * 64
    rows = [
        candidate("1", label="Drums/Kick Drums/Generic Kick/One Shots", group=shared),
        candidate("2", label="FX/Impacts and Hits/Generic Impact/One Shots", group=shared),
        candidate("3", approved=False),
    ]

    assigned = {row.candidate_id: row.allowed_use for row in assign_leakage_safe_uses(rows)}

    assert assigned["1"] == EXCLUDED_CONFLICT_USE
    assert assigned["2"] == EXCLUDED_CONFLICT_USE
    assert assigned["3"] == EXCLUDED_PENDING_REVIEW_USE


def test_readiness_tiers_do_not_overstate_sparse_categories() -> None:
    assert readiness_tier(trusted_duplicate_groups=8, trusted_source_groups=2, final_heldout_examples=1)[0] == "A"
    assert readiness_tier(trusted_duplicate_groups=5, trusted_source_groups=1, final_heldout_examples=0)[0] == "B"
    assert readiness_tier(trusted_duplicate_groups=2, trusted_source_groups=1, final_heldout_examples=0)[0] == "C"
    assert readiness_tier(trusted_duplicate_groups=1, trusted_source_groups=1, final_heldout_examples=0)[0] == "D"
