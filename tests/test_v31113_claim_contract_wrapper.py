"""v31113 claim-contract adapter tests.

These tests lock the second architecture step: existing arbiter claims now have
explicit contract metadata, but routing behavior is not rewritten yet.
"""

from __future__ import annotations

from pathlib import Path

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.claim_contracts import contract_from_claim, contracts_from_claims
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


def make_claim(
    *,
    family: str = "Instruments",
    sub_family: str = "Instrument Loops",
    source: str = "strong_consensus",
    strength: float = 0.82,
    folder_path: str = "Instruments/Instrument Loops/Loops",
    can_override: bool = False,
    raw_candidate_score: float | None = 5.0,
    is_real_candidate: bool = True,
) -> ConsensusClaim:
    """Create a small claim without relying on audio filenames."""
    return ConsensusClaim(
        family=family,
        sub_family=sub_family,
        strength=strength,
        can_override=can_override,
        reason="test evidence only",
        source=source,
        raw_candidate_score=raw_candidate_score,
        label=folder_path,
        folder_path=folder_path,
        shared_winner=folder_path,
        shared_candidates=[],
        is_real_candidate=is_real_candidate,
    )


def test_shape_claim_contract_has_structure_scope_not_source_identity() -> None:
    shape_claim = make_claim(
        family="Instruments",
        sub_family="Instrument Loops",
        source="shape_sanity_consensus",
        folder_path="Instruments/Instrument Loops/Loops",
        can_override=True,
        raw_candidate_score=None,
        is_real_candidate=False,
    )

    contract = contract_from_claim(shape_claim, index=1)

    assert contract.claim_type == "structure"
    assert contract.scope == "structure_only"
    assert contract.authority == "may_block_or_broaden_structure"
    assert contract.role == "pitched_music_loop"
    assert contract.structure == "loop"
    assert "shape_vote" in contract.required_evidence
    assert "brain_topk" not in contract.source_features_used


def test_real_source_identity_contract_needs_no_shape_veto() -> None:
    sax_claim = make_claim(
        family="Instruments",
        sub_family="Brass Woodwinds",
        source="strong_consensus",
        folder_path="Instruments/Brass and Woodwinds/Sax/Loops",
        can_override=False,
        raw_candidate_score=4.0,
        is_real_candidate=True,
    )

    contract = contract_from_claim(sax_claim, index=2)

    assert contract.claim_type == "source_identity"
    assert contract.scope == "source_identity_when_not_vetoed"
    assert contract.authority == "may_choose_leaf_if_physics_and_shape_allow"
    assert "no_shape_veto" in contract.required_evidence
    assert "shape_structure_conflict" in contract.veto_evidence
    assert contract.allowed_transitions == ("same_family:Instruments",)


def test_broad_role_contract_is_not_source_identity() -> None:
    role_claim = make_claim(
        family="Drums",
        sub_family="Drum Loops",
        source="role_sanity_broad_bucket",
        folder_path="Drums/Drum Loops",
        can_override=True,
        raw_candidate_score=None,
        is_real_candidate=False,
    )

    contract = contract_from_claim(role_claim, index=3)

    assert contract.claim_type == "role"
    assert contract.scope == "role_and_broad_bucket"
    assert contract.role == "drum_loop"
    assert contract.allowed_transitions == ("same_family:Drums", "broad_bucket:Drums")


def test_contracts_from_claims_preserves_stable_order() -> None:
    raw = make_claim(source="strong_consensus", folder_path="Instruments/Keys/Electric Piano/Loops")
    review = make_claim(
        family="_TO_REVIEW",
        sub_family="Measured Role Conflict",
        source="measured_role_conflict_review",
        folder_path="_TO_REVIEW/Measured Role Conflict",
        is_real_candidate=False,
        raw_candidate_score=None,
    )

    contracts = contracts_from_claims([raw, review])

    assert [contract.claim_id.split(":", 1)[0] for contract in contracts] == ["001", "002"]
    assert contracts[0].claim_type == "role"
    assert contracts[1].claim_type == "safety_review"
    assert contracts[1].scope == "safety_review_only"


def test_arbiter_debug_trace_writes_contract_lines(tmp_path: Path, monkeypatch) -> None:
    raw_claim = make_claim(
        family="Instruments",
        sub_family="Brass Woodwinds",
        source="strong_consensus",
        folder_path="Instruments/Brass and Woodwinds/Sax/Loops",
        raw_candidate_score=5.0,
        is_real_candidate=True,
    )
    competing_claim = make_claim(
        family="Instruments",
        sub_family="Brass Woodwinds",
        source="shape_sanity_consensus",
        folder_path="Instruments/Brass and Woodwinds/Loops",
        can_override=True,
        raw_candidate_score=6.0,
        is_real_candidate=True,
    )
    debug_path = tmp_path / "claim_debug.txt"
    monkeypatch.setenv("AARON_DEBUG_CLAIMS_FILE", str(debug_path))

    FamilyClaimArbiter().adjudicate(
        raw_claim=raw_claim,
        consensus_claims=[competing_claim],
        eligibility_claims=[],
        facts=SharedAudioFacts(False, True, False, False, False),
    )

    text = debug_path.read_text(encoding="utf-8")
    assert "CONTRACT" in text
    assert "type=source_identity" in text
    assert "type=structure" in text
    assert "scope=structure_only" in text
