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


def _bass_harmonic_alias_facts(*, with_owner_support: bool) -> SharedAudioFacts:
    evidence = {
        "shape_vote": {
            "primary_shape": "solo_phrase",
            "confidence": 0.87,
            "pitched_event_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
            "sustained_tonal_frame_ratio": 1.0,
            "low_event_ratio": 0.98,
            "mid_event_ratio": 0.01,
            "high_event_ratio": 0.003,
        },
        "measured_roles": {
            "detected_parent_role": "bass_loop",
            "bass_loop": 1.0,
            "pitched_music_loop": 1.0,
            "evidence": {
                "low_peak_frequency_hz": 64.6,
                "low_total": 0.99,
                "high_total": 0.001,
                "loop_pitched_event_ratio": 1.0,
                "loop_sustained_tonal_frame_ratio": 1.0,
                "loop_non_event_tonal_ratio": 1.0,
            },
        },
        "physics_subpanels": {
            "flat": {
                "bass_synth_score": 0.85,
                "bass_sub_score": 0.77,
                "bass_electric_score": 0.51,
                "low_end_source_score": 0.66,
                "drum_hit_score": 0.02,
                "drum_loop_source_score": 0.11,
                "drum_rim_stick_source_score": 0.18,
                "fx_motion_score": 0.15,
                "fx_transition_authority_score": 0.23,
            }
        },
    }
    if with_owner_support:
        evidence["learned_physics_memory"] = {
            "matched": True,
            "confidence": 0.81,
            "label": "Instruments/Bass/Electric Bass/Loops",
            "target_key": "Instruments/Bass",
            "branch": "Bass",
        }
        evidence["brain_ensemble_vote_1"] = {
            "folder_path": "Instruments/Bass/Electric Bass/Loops",
            "confidence": 1.0,
            "support": 2.0,
        }
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence=evidence,
        feature_values_by_name={
            "duration_sec": 12.0,
            "f0_median_hz": 1575.0,
            "pitch_confidence": 0.97,
            "body_pitch_confidence": 0.97,
            "f0_voiced_ratio": 0.45,
            "low_peak_frequency_hz": 64.6,
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


def test_bass_high_f0_with_memory_and_low_body_is_probable_harmonic_alias() -> None:
    facts = _bass_harmonic_alias_facts(with_owner_support=True)
    context = _context("Instruments/Bass/Electric Bass/Loops", facts)

    claims = MeasuredFinalGuardClaimProducer().produce_for_claims(context, [])

    assert claims == []
    check = check_instrument_pitch_range("Instruments/Bass/Electric Bass/Loops", facts)
    assert check.status == "probable_harmonic_alias"


def test_bass_high_f0_yields_to_normal_yin_with_string_brain_owner() -> None:
    facts = _bass_harmonic_alias_facts(with_owner_support=False)
    facts.evidence["brain_ensemble_vote_1"] = "Instruments/Bass/Electric Bass/Loops"
    facts.feature_values_by_name.update(
        {
            "librosa_yin_f0_median_hz": 70.0,
            "librosa_yin_voiced_ratio": 0.58,
        }
    )
    context = _context("Instruments/Bass/Electric Bass/Loops", facts)

    claims = MeasuredFinalGuardClaimProducer().produce_for_claims(context, [])

    assert claims == []
    check = check_instrument_pitch_range("Instruments/Bass/Electric Bass/Loops", facts)
    assert check.status == "probable_harmonic_alias"


def test_bass_high_f0_without_owner_support_still_reviews() -> None:
    facts = _bass_harmonic_alias_facts(with_owner_support=False)
    context = _context("Instruments/Bass/Electric Bass/Loops", facts)

    claims = MeasuredFinalGuardClaimProducer().produce_for_claims(context, [])

    assert len(claims) == 1
    assert claims[0].source == "final_instrument_pitch_range_conflict_review"


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
