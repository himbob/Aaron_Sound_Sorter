from __future__ import annotations

from aaron_sound_sorter.taxonomy_contracts import (
    canonicalize_taxonomy_label,
    is_trainable_taxonomy_label,
    taxonomy_label_contract,
)
from aaron_sound_sorter.voters.scoring_tools import canonical_candidate_labels


def test_loop_named_category_does_not_keep_one_shot_terminal() -> None:
    label = "Instruments/Guitar/Guitar Loops/One Shots"

    contract = taxonomy_label_contract(label)

    assert contract.canonical_label == "Instruments/Guitar/Guitar Loops/Loops"
    assert "canonicalized_loop_category_terminal" in contract.reasons
    assert contract.valid


def test_one_shot_named_category_does_not_keep_loop_terminal() -> None:
    label = "Instruments/Voice/Vocal One Shots/Loops"

    assert canonicalize_taxonomy_label(label) == "Instruments/Voice/Vocal One Shots/One Shots"


def test_explicit_loop_category_without_terminal_gains_loop_terminal() -> None:
    contract = taxonomy_label_contract("Instruments/Synths/Synth Loops")

    assert contract.canonical_label == "Instruments/Synths/Synth Loops/Loops"
    assert "inferred_explicit_loop_category_terminal" in contract.reasons
    assert is_trainable_taxonomy_label(contract.canonical_label)


def test_fx_loop_terminal_is_not_trainable() -> None:
    label = "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Loops"

    contract = taxonomy_label_contract(label)

    assert not contract.valid
    assert not is_trainable_taxonomy_label(label)
    assert "fx_loop_terminal_not_trainable" in contract.reasons


def test_voter_candidate_filter_removes_noncanonical_duplicates() -> None:
    labels = [
        "Instruments/Guitar/Guitar Loops/One Shots",
        "Instruments/Guitar/Guitar Loops/Loops",
        "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Loops",
    ]

    assert canonical_candidate_labels(labels) == ["Instruments/Guitar/Guitar Loops/Loops"]
