from __future__ import annotations

from aaron_sound_sorter.domain.models import CategoryGuess, SharedAudioFacts, VoterResult
from aaron_sound_sorter.engine.claim_producers.learned_owner_authority import (
    VOICE_OWNER_CLAIM_SOURCE,
    LearnedOwnerAuthorityClaimProducer,
)
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path


def raw_claim(path: str, *, score: float = 8.0):
    return claim_from_folder_path(
        folder_path=path,
        source="strong_consensus",
        reason="test raw",
        shared=[],
        raw_candidate_score=score,
        brain_rank=1,
        physics_rank=1,
        shared_winner=path,
        can_override=False,
        strength=0.75,
        is_real_candidate=True,
    )


def broad_loop_claim():
    return claim_from_folder_path(
        folder_path="Instruments/Instrument Loops/Loops",
        source="final_measured_musical_loop_depth_invariant",
        reason="test broad musical loop",
        shared=[],
        raw_candidate_score=5.0,
        brain_rank=2,
        physics_rank=1,
        shared_winner="Instruments/Instrument Loops/Loops",
        can_override=True,
        strength=0.95,
        is_real_candidate=False,
    )


def eligibility(role: str = "pitched_music_loop") -> EligibilityDecision:
    return EligibilityDecision(
        role_name=role,
        confidence=0.90,
        allowed_top_families=("Instruments", "_TO_REVIEW"),
        broad_folder_path="Instruments/Instrument Loops/Loops",
        reason="test eligibility",
    )


def voice_guess(path: str = "Instruments/Voice/Vocal Loops/Loops") -> CategoryGuess:
    return CategoryGuess(
        label=path,
        folder_path=path,
        top_family=path.split("/", 1)[0],
        score=0.42,
        confidence=0.83,
        rank=1,
        reason="test brain",
        evidence={"support": 2.8},
    )


def physics_guess(path: str) -> CategoryGuess:
    return CategoryGuess(
        label=path,
        folder_path=path,
        top_family=path.split("/", 1)[0],
        score=0.31,
        confidence=0.82,
        rank=1,
        reason="test physics",
        evidence={},
    )


def voice_facts(
    *,
    shape: str = "pitched_repetition_phrase",
    voice_score: float = 0.52,
    drum_hit_score: float = 0.22,
    woodwind_sax_score: float = 0.24,
) -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {
                "primary_shape": shape,
                "confidence": 0.88,
                "pitched_event_ratio": 0.72,
                "percussive_event_ratio": 0.28,
                "drumlike_frame_ratio": 0.20,
            },
            "physics_subpanels": {
                "flat": {
                    "voice_score": voice_score,
                    "human_spoken_voice_score": voice_score,
                    "fx_formant_score": voice_score - 0.04,
                    "drum_hit_score": drum_hit_score,
                    "drum_loop_source_score": 0.18,
                    "drum_kick_source_score": 0.14,
                    "drum_snare_source_score": 0.12,
                    "drum_clap_source_score": 0.12,
                    "compact_struck_tonal_percussion_score": drum_hit_score,
                    "fx_motion_score": 0.20,
                    "fx_transition_authority_score": 0.18,
                    "fx_riser_build_score": 0.12,
                    "woodwind_sax_score": woodwind_sax_score,
                    "reed_wind_score": max(0.18, woodwind_sax_score - 0.10),
                }
            },
        },
        feature_values_by_name={"duration_sec": 3.2},
        feature_count=4,
        feature_vector=(0.1, 0.2, 0.3, 0.4),
    )


def context_with_voice_brain(facts: SharedAudioFacts) -> DecisionContext:
    return DecisionContext(
        raw=raw_claim("Instruments/Instrument Loops/Loops"),
        eligibility=eligibility(),
        facts=facts,
        brain_result=VoterResult(voter_name="brain", guesses=[voice_guess()]),
    )


def test_learned_owner_voice_claim_emits_when_rank_one_brain_and_body_agree() -> None:
    producer = LearnedOwnerAuthorityClaimProducer()

    claims = producer.produce(context_with_voice_brain(voice_facts()))

    assert len(claims) == 1
    assert claims[0].source == VOICE_OWNER_CLAIM_SOURCE
    assert claims[0].folder_path == "Instruments/Voice/Vocal Loops/Loops"
    assert claims[0].is_real_candidate is True


def test_learned_owner_rehomes_internal_human_voice_fx_to_instrument_voice() -> None:
    producer = LearnedOwnerAuthorityClaimProducer()
    context = DecisionContext(
        raw=raw_claim("Instruments/Mixed Musical Loops/Multi Instrument/Loops"),
        eligibility=eligibility(),
        facts=voice_facts(shape="pitched_phrase"),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[voice_guess("FX/Human and Voice FX/Spoken Voice/Long FX")],
        ),
    )

    claims = producer.produce(context)

    assert len(claims) == 1
    assert claims[0].folder_path == "Instruments/Voice/Vocal Loops/Loops"


def test_learned_owner_voice_claim_stands_down_for_hard_drum_body() -> None:
    producer = LearnedOwnerAuthorityClaimProducer()

    claims = producer.produce(context_with_voice_brain(voice_facts(drum_hit_score=0.72)))

    assert claims == []


def test_learned_owner_voice_claim_stands_down_for_non_voice_memory_match() -> None:
    producer = LearnedOwnerAuthorityClaimProducer()
    facts = voice_facts(voice_score=0.86)
    facts.evidence["learned_physics_memory"] = {
        "matched": True,
        "top_family": "Instruments",
        "branch": "Woodwinds",
        "label": "Instruments/Woodwinds/Saxophone/Loops",
        "confidence": 0.94,
    }

    claims = producer.produce(context_with_voice_brain(facts))

    assert claims == []


def test_learned_owner_voice_claim_stands_down_for_measured_sax_physics_branch() -> None:
    producer = LearnedOwnerAuthorityClaimProducer()
    facts = voice_facts(voice_score=0.71, woodwind_sax_score=0.64)
    context = DecisionContext(
        raw=raw_claim("Instruments/Instrument Loops/Loops"),
        eligibility=eligibility(),
        facts=facts,
        brain_result=VoterResult(voter_name="brain", guesses=[voice_guess()]),
        physics_result=VoterResult(
            voter_name="physics",
            guesses=[physics_guess("Instruments/Woodwinds/Saxophone/Loops")],
        ),
    )

    claims = producer.produce(context)

    assert claims == []


def test_arbiter_prefers_learned_voice_owner_over_generic_loop_fallback() -> None:
    producer = LearnedOwnerAuthorityClaimProducer()
    facts = voice_facts()
    raw = raw_claim("Instruments/Instrument Loops/Loops")
    learned_claim = producer.produce(
        DecisionContext(
            raw=raw,
            eligibility=eligibility(),
            facts=facts,
            brain_result=VoterResult(voter_name="brain", guesses=[voice_guess()]),
        )
    )[0]

    final = FamilyClaimArbiter().pick_winner(
        raw_claim=raw,
        claims=[broad_loop_claim(), learned_claim],
        facts=facts,
    )

    assert final.source == VOICE_OWNER_CLAIM_SOURCE
    assert final.folder_path == "Instruments/Voice/Vocal Loops/Loops"
