"""Regression tests for GUI-taught correction authority."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.claim_contracts import (
    is_human_taught_brain_top_shared_row,
    is_human_taught_rank_one_consensus,
    is_rank_one_concrete_non_sax_instrument_consensus,
)
from aaron_sound_sorter.engine.consensus import ConsensusRunner
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_folder_path


def _teacher_shared_row(path: str, *, brain_rank: int = 1, physics_rank: int = 1) -> dict[str, object]:
    return {
        "label": path,
        "folder_path": path,
        "top_family": path.split("/", 1)[0],
        "brain_rank": brain_rank,
        "physics_rank": physics_rank,
        "combined_rank_score": float(brain_rank + physics_rank),
        "brain_confidence": 1.0,
        "physics_confidence": 0.46,
        "brain_evidence": {
            "human_override_matched": True,
            "human_override_generalized_audio_match": True,
            "human_override_match_kind": "teacher_cloud",
            "human_override_effective_weight": 1203,
        },
        "physics_evidence": {},
        "candidate_role_distance": 0.35,
        "family_compatibility": {"is_family_compatible": True},
    }


def _raw(path: str, *, shared: list[dict[str, object]] | None = None) -> ConsensusClaim:
    return claim_from_folder_path(
        folder_path=path,
        source="strong_consensus",
        reason="raw shared winner",
        shared=shared or [_teacher_shared_row(path)],
        raw_candidate_score=2.0,
        brain_rank=1,
        physics_rank=1,
        shared_winner=path,
        can_override=False,
        strength=0.94,
        is_real_candidate=True,
    )


def _competing(path: str, *, source: str = "mixed_instrument_loop_role_claim") -> ConsensusClaim:
    return claim_from_folder_path(
        folder_path=path,
        source=source,
        reason="competing broad claim",
        shared=[],
        raw_candidate_score=2.0,
        brain_rank=1,
        physics_rank=1,
        shared_winner=path,
        can_override=True,
        strength=0.96,
        is_real_candidate=False,
    )


def _phrase_facts() -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {"primary_shape": "solo_phrase", "confidence": 0.92},
            "physics_subpanels": {
                "flat": {
                    "woodwind_sax_score": 0.74,
                    "synth_pad_score": 0.72,
                    "voice_score": 0.70,
                    "plucked_string_score": 0.68,
                }
            },
        },
    )


def test_human_taught_rank_one_consensus_allows_sax_teacher_support() -> None:
    raw = _raw("Instruments/Woodwinds/Saxophone/Loops")

    assert is_human_taught_rank_one_consensus(raw)
    assert is_rank_one_concrete_non_sax_instrument_consensus(raw)


def test_broad_instrument_loop_claim_cannot_override_taught_rank_one_winner() -> None:
    arbiter = FamilyClaimArbiter()
    raw = _raw("Instruments/Woodwinds/Saxophone/Loops")
    broad = _competing("Instruments/Instrument Loops/Loops")

    assert not arbiter._claim_can_compete(raw, broad, facts=_phrase_facts())

    final = arbiter.adjudicate(
        raw_claim=raw,
        consensus_claims=[broad],
        eligibility_claims=[],
        facts=_phrase_facts(),
    )

    assert final.folder_path == "Instruments/Woodwinds/Saxophone/Loops"
    assert final.consensus_status == "strong_consensus"


def test_conflicted_identity_review_respects_taught_rank_one_winner() -> None:
    arbiter = FamilyClaimArbiter()
    raw = _raw("Instruments/Synths/Synth Pad/Loops")

    final_claim = arbiter._review_conflicted_instrument_identity_leaf(raw, _phrase_facts())

    assert final_claim is raw


def test_raw_winner_contract_respects_taught_rank_one_woodwind_winner() -> None:
    arbiter = FamilyClaimArbiter()
    raw = _raw("Instruments/Woodwinds/Saxophone/Loops")

    replacement = arbiter._raw_winner_contract_stand_down(raw, facts=_phrase_facts())

    assert replacement is None


def test_profile_claim_cannot_cross_family_override_taught_rank_one_fx_winner() -> None:
    arbiter = FamilyClaimArbiter()
    raw = _raw("FX/Human and Voice FX/Spoken Voice/Long FX")
    voice_profile = _competing(
        "Instruments/Voice/Choir/Loops",
        source="profile_candidate_voice_claim",
    )

    assert not arbiter._claim_can_compete(raw, voice_profile, facts=_phrase_facts())


def test_same_family_teacher_candidate_beats_broad_parent_when_physics_corroborates() -> None:
    arbiter = FamilyClaimArbiter()
    runner = ConsensusRunner()
    raw = _raw(
        "Instruments/World and Special Instruments/Harmonica/Loops",
        shared=[_teacher_shared_row("Instruments/Woodwinds/Saxophone/Loops", physics_rank=24)],
    )
    broad = _competing("Instruments/Instrument Loops/Loops")
    taught_claim = runner.human_taught_specific_decision(
        shared=[
            {
                "label": raw.folder_path,
                "folder_path": raw.folder_path,
                "top_family": "Instruments",
                "brain_rank": 3,
                "physics_rank": 4,
                "combined_rank_score": 7.0,
            },
            _teacher_shared_row("Instruments/Woodwinds/Saxophone/Loops", physics_rank=24),
        ],
        winner={
            "label": raw.folder_path,
            "folder_path": raw.folder_path,
            "top_family": "Instruments",
        },
        facts=_phrase_facts(),
    )

    assert taught_claim is not None
    assert taught_claim.source == "human_taught_specific_shared_candidate"
    assert arbiter._claim_can_compete(raw, taught_claim, facts=_phrase_facts())

    final = arbiter.adjudicate(
        raw_claim=raw,
        consensus_claims=[taught_claim, broad],
        eligibility_claims=[],
        facts=_phrase_facts(),
    )

    assert final.folder_path == "Instruments/Woodwinds/Saxophone/Loops"


def test_teacher_candidate_requires_physics_candidate_window() -> None:
    row = _teacher_shared_row("Instruments/Strings/String Loops/Loops", physics_rank=40)

    assert not is_human_taught_brain_top_shared_row(row)
