"""Architecture locks for the Claim-Arbiter placement workflow."""

from __future__ import annotations

from pathlib import Path

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import ConsensusClaim

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "aaron_sound_sorter" / "engine"


def claim(
    *,
    family: str,
    sub_family: str,
    source: str,
    strength: float,
    folder_path: str,
    can_override: bool,
    raw_candidate_score: float | None,
    is_real_candidate: bool,
) -> ConsensusClaim:
    return ConsensusClaim(
        family=family,
        sub_family=sub_family,
        strength=strength,
        can_override=can_override,
        reason="test claim",
        source=source,
        raw_candidate_score=raw_candidate_score,
        label=folder_path,
        folder_path=folder_path,
        shared_winner=folder_path,
        shared_candidates=[],
        is_real_candidate=is_real_candidate,
    )


def test_only_family_claim_arbiter_constructs_final_consensus_decision() -> None:
    offenders: list[str] = []
    for path in SRC_ROOT.glob("*.py"):
        if path.name == "family_claim_arbiter.py" or path.name.startswith("._"):
            continue
        text = path.read_text(encoding="utf-8")
        if "ConsensusDecision(" in text:
            offenders.append(str(path.relative_to(SRC_ROOT.parents[1])))
    assert offenders == []


def test_synthetic_human_voice_claim_cannot_steal_instrument_without_candidate() -> None:
    raw_claim = claim(
        family="Instruments",
        sub_family="Brass Woodwinds",
        source="strong_consensus",
        strength=0.78,
        folder_path="Instruments/Brass and Woodwinds/Sax/Loops",
        can_override=False,
        raw_candidate_score=5.0,
        is_real_candidate=True,
    )
    ghost_voice = claim(
        family="FX",
        sub_family="Human and Voice FX",
        source="same_family_role_broad_bucket",
        strength=0.96,
        folder_path="FX/Human and Voice FX",
        can_override=True,
        raw_candidate_score=None,
        is_real_candidate=False,
    )

    decision = FamilyClaimArbiter().adjudicate(
        raw_claim=raw_claim,
        consensus_claims=[ghost_voice],
        eligibility_claims=[],
        facts=SharedAudioFacts(False, False, False, False, False),
    )

    assert decision.folder_path == "Instruments/Brass and Woodwinds/Sax/Loops"
    assert decision.consensus_status == "strong_consensus"


def test_real_candidate_claim_can_override_when_close_enough() -> None:
    raw_claim = claim(
        family="Instruments",
        sub_family="Instrument Loops",
        source="strong_consensus",
        strength=0.50,
        folder_path="Instruments/Instrument Loops/Loops",
        can_override=False,
        raw_candidate_score=9.0,
        is_real_candidate=True,
    )
    real_reed = claim(
        family="Instruments",
        sub_family="Brass Woodwinds",
        source="role_sanity_consensus",
        strength=0.92,
        folder_path="Instruments/Brass and Woodwinds/Loops",
        can_override=True,
        raw_candidate_score=11.0,
        is_real_candidate=True,
    )

    decision = FamilyClaimArbiter().adjudicate(
        raw_claim=raw_claim,
        consensus_claims=[real_reed],
        eligibility_claims=[],
        facts=SharedAudioFacts(False, False, False, False, False),
    )

    assert decision.folder_path == "Instruments/Brass and Woodwinds/Loops"
    assert decision.consensus_status == "role_sanity_consensus"
