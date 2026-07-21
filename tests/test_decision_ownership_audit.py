from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from decision_ownership_audit import (  # noqa: E402
    classify_decision_ownership,
    row_final_path,
    summary_lines,
)


def row(**values: str) -> dict[str, str]:
    base = {
        "folder_path": "Instruments/Voice/Vocal Loops/Loops",
        "final_label": "Instruments/Voice/Vocal Loops/Loops",
        "final_top": "Instruments",
        "consensus_status": "strong_consensus",
        "final_claim_source": "",
        "final_agrees_with_brain_ensemble": "false",
        "brain_ensemble_vote_1": "",
        "learned_voter_memory_matched": "false",
        "learned_voter_memory_label": "",
        "learned_physics_memory_matched": "false",
        "learned_physics_memory_label": "",
    }
    base.update(values)
    return base


def test_final_path_prefers_folder_path() -> None:
    assert row_final_path(row(folder_path="Instruments/Bass/Electric Bass/Loops")) == (
        "Instruments/Bass/Electric Bass/Loops"
    )


def test_learned_owner_source_classifies_as_learned_memory() -> None:
    ownership = classify_decision_ownership(
        row(
            final_claim_source="learned_owner_body_claim",
            learned_physics_memory_matched="true",
            learned_physics_memory_label="Instruments/Voice/Vocal Loops/Loops",
        )
    )

    assert ownership.owner_type == "learned_memory"
    assert ownership.learned_memory_matched is True
    assert ownership.learned_memory_agrees is True


def test_exact_memory_label_agreement_classifies_as_learned_memory() -> None:
    ownership = classify_decision_ownership(
        row(
            consensus_status="strong_consensus",
            learned_voter_memory_matched="true",
            learned_voter_memory_label="Instruments/Voice/Vocal Loops/Loops",
        )
    )

    assert ownership.owner_type == "learned_memory"


def test_strong_consensus_with_brain_agreement_classifies_as_trained_brain() -> None:
    ownership = classify_decision_ownership(
        row(
            final_agrees_with_brain_ensemble="true",
            brain_ensemble_vote_1="Instruments/Voice/Vocal Loops/Loops",
        )
    )

    assert ownership.owner_type == "trained_brain_consensus"
    assert ownership.brain_agrees is True


def test_static_contract_over_brain_is_flagged() -> None:
    ownership = classify_decision_ownership(
        row(
            final_claim_source="final_measured_voice_invariant",
            final_agrees_with_brain_ensemble="true",
            brain_ensemble_vote_1="Instruments/Voice/Vocal Loops/Loops",
        )
    )

    assert ownership.owner_type == "static_measured_contract"
    assert ownership.static_over_brain is True


def test_learned_memory_blocked_is_flagged() -> None:
    ownership = classify_decision_ownership(
        row(
            folder_path="FX/Designed Noise FX/Alarm/Long FX",
            final_label="FX/Designed Noise FX/Alarm/Long FX",
            final_top="FX",
            final_claim_source="final_measured_transition_fx_invariant",
            learned_physics_memory_matched="true",
            learned_physics_memory_label="Instruments/Bass/Electric Bass/Loops",
        )
    )

    assert ownership.owner_type == "static_measured_contract"
    assert ownership.learned_memory_matched is True
    assert ownership.learned_memory_agrees is False
    assert ownership.learned_memory_blocked is True


def test_review_classifies_as_review() -> None:
    ownership = classify_decision_ownership(
        row(
            folder_path="_TO_REVIEW/Measured Role Conflict",
            final_label="_TO_REVIEW/Measured Role Conflict",
            final_top="_TO_REVIEW",
            final_claim_source="raw_winner_contract_review",
        )
    )

    assert ownership.owner_type == "review"
    assert ownership.broad_fallback is True


def test_summary_counts_memory_and_static_rows() -> None:
    rows = [
        {
            "decision_owner_type": "learned_memory",
            "broad_fallback": "False",
            "learned_memory_matched": "True",
            "learned_memory_agrees": "True",
            "learned_memory_blocked": "False",
            "static_over_brain": "False",
        },
        {
            "decision_owner_type": "static_measured_contract",
            "broad_fallback": "True",
            "learned_memory_matched": "True",
            "learned_memory_agrees": "False",
            "learned_memory_blocked": "True",
            "static_over_brain": "True",
        },
    ]

    text = "\n".join(summary_lines(rows))

    assert "learned_memory: 1" in text
    assert "static_measured_contract: 1" in text
    assert "Learned memory matched but did not own final label: 1" in text
