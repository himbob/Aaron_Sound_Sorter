"""Tests for conservative measured instrument pitch-range guards."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.claim_producers.measured_final_guards import MeasuredFinalGuardClaimProducer
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path
from aaron_sound_sorter.engine.instrument_pitch_ranges import check_instrument_pitch_range


def _raw_claim(path: str, *, strength: float = 0.82):
    return claim_from_folder_path(
        folder_path=path,
        source="strong_consensus",
        reason="test raw",
        shared=[],
        raw_candidate_score=4.0,
        brain_rank=1,
        physics_rank=1,
        shared_winner=path,
        can_override=False,
        strength=strength,
        is_real_candidate=True,
    )


def _broad_instrument_loop_claim():
    return claim_from_folder_path(
        folder_path="Instruments/Instrument Loops/Loops",
        source="final_shape_review_broad_instrument_loop_invariant",
        reason="test broad release",
        shared=[],
        raw_candidate_score=5.0,
        brain_rank=2,
        physics_rank=2,
        shared_winner="Instruments/Instrument Loops/Loops",
        can_override=True,
        strength=0.98,
        is_real_candidate=False,
    )


def _eligibility() -> EligibilityDecision:
    return EligibilityDecision(
        role_name="pitched_reed_or_instrument_loop",
        confidence=0.92,
        allowed_top_families=("Instruments", "_TO_REVIEW"),
        broad_folder_path="Instruments/Brass and Woodwinds/Loops",
        reason="test eligibility",
    )


def _facts(*, f0_hz: float, pitch_confidence: float, voiced_ratio: float) -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {
                "primary_shape": "pitched_phrase",
                "confidence": 0.91,
                "pitched_event_ratio": 0.92,
                "percussive_event_ratio": 0.02,
                "drumlike_frame_ratio": 0.01,
                "sustained_tonal_frame_ratio": 0.88,
            },
            "measured_roles": {
                "detected_parent_role": "pitched_reed_or_instrument_loop",
                "pitched_reed_or_instrument_loop": 0.91,
                "pitched_music_loop": 0.88,
            },
        },
        feature_values_by_name={
            "duration_sec": 4.0,
            "f0_median_hz": f0_hz,
            "pitch_confidence": pitch_confidence,
            "body_pitch_confidence": pitch_confidence,
            "f0_voiced_ratio": voiced_ratio,
        },
    )


def _context(raw_path: str, facts: SharedAudioFacts) -> DecisionContext:
    return DecisionContext(
        raw=_raw_claim(raw_path),
        eligibility=_eligibility(),
        facts=facts,
    )


def test_saxophone_pitch_inside_range_does_not_emit_review_claim() -> None:
    facts = _facts(f0_hz=220.0, pitch_confidence=0.91, voiced_ratio=0.90)
    context = _context("Instruments/Woodwinds/Saxophone/Loops", facts)

    claims = MeasuredFinalGuardClaimProducer().produce_for_claims(context, [])

    assert [claim.source for claim in claims] == []
    check = check_instrument_pitch_range("Instruments/Woodwinds/Saxophone/Loops", facts)
    assert check.status == "inside"


def test_impossible_saxophone_pitch_emits_hard_review_claim() -> None:
    facts = _facts(f0_hz=28.0, pitch_confidence=0.93, voiced_ratio=0.84)
    context = _context("Instruments/Woodwinds/Saxophone/Loops", facts)

    claims = MeasuredFinalGuardClaimProducer().produce_for_claims(context, [])

    assert len(claims) == 1
    assert claims[0].source == "final_instrument_pitch_range_conflict_review"
    assert claims[0].folder_path == "_TO_REVIEW/Instrument Pitch Range Conflict"
    assert "observed 28.0 Hz" in claims[0].reason


def test_low_confidence_out_of_range_pitch_does_not_block_instrument_leaf() -> None:
    facts = _facts(f0_hz=28.0, pitch_confidence=0.42, voiced_ratio=0.84)
    context = _context("Instruments/Woodwinds/Saxophone/Loops", facts)

    claims = MeasuredFinalGuardClaimProducer().produce_for_claims(context, [])

    assert claims == []
    check = check_instrument_pitch_range("Instruments/Woodwinds/Saxophone/Loops", facts)
    assert check.status == "unknown"


def test_hard_pitch_range_review_cannot_be_released_to_broad_loop_claim() -> None:
    raw = _raw_claim("Instruments/Woodwinds/Saxophone/Loops")
    facts = _facts(f0_hz=28.0, pitch_confidence=0.93, voiced_ratio=0.84)
    context = DecisionContext(raw=raw, eligibility=_eligibility(), facts=facts)
    range_review = MeasuredFinalGuardClaimProducer().produce_for_claims(context, [])[0]

    decision = FamilyClaimArbiter().adjudicate(
        raw_claim=raw,
        consensus_claims=[],
        eligibility_claims=[range_review, _broad_instrument_loop_claim()],
        facts=facts,
    )

    assert decision.final_top == "_TO_REVIEW"
    assert decision.folder_path == "_TO_REVIEW/Instrument Pitch Range Conflict"
    assert decision.consensus_status == "final_instrument_pitch_range_conflict_review"
