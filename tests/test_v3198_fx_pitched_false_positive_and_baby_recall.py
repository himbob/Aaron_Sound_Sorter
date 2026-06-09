"""v31.98 locks for FX false positives and baby-lane recall."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts, VoterResult
from aaron_sound_sorter.engine.claim_producers.baby_recall import BabyRecallClaimProducer
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import infer_parent_eligibility
from aaron_sound_sorter.engine.eligibility_decision import EligibilityDecision
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path


def pitched_loop_facts(*, outlier_confirms_siren: bool = False) -> SharedAudioFacts:
    top_outlier = (
        "FX/Designed Noise FX/Siren/Long FX" if outlier_confirms_siren else "Instruments/Brass/Trumpet/One Shots"
    )
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {
                "primary_shape": "pitched_phrase",
                "confidence": 0.98,
                "onset_count": 16.0,
                "pitched_event_ratio": 0.96,
                "sustained_tonal_frame_ratio": 0.96,
                "percussive_event_ratio": 0.0,
                "drumlike_frame_ratio": 0.0,
                "spectral_flatness_mean": 0.12,
            },
            "measured_roles": {
                "pitched_music_loop": 1.0,
                "pitched_music_phrase": 0.95,
                "primary_roles": ["pitched_music_loop"],
            },
            "full_brain_vote_result": {"top_guesses": [{"folder_path": "FX/Designed Noise FX/Siren/Long FX"}]},
            "core_baby_vote_result": {"top_guesses": [{"folder_path": "Instruments/Guitar/Electric Guitar/One Shots"}]},
            "spread_baby_vote_result": {"top_guesses": [{"folder_path": "FX/Designed Noise FX/Siren/Long FX"}]},
            "outlier_baby_vote_result": {"top_guesses": [{"folder_path": top_outlier}]},
        },
    )


def test_strong_pitched_loop_evidence_can_escape_false_fx_siren() -> None:
    raw = claim_from_folder_path(
        folder_path="FX/Designed Noise FX/Siren/Long FX",
        source="strong_consensus",
        reason="raw false positive",
        shared=[],
        raw_candidate_score=5.0,
        brain_rank=4,
        physics_rank=1,
        shared_winner="FX/Designed Noise FX/Siren/Long FX",
        can_override=False,
        strength=0.68,
        is_real_candidate=True,
    )
    instrument = claim_from_folder_path(
        folder_path="Instruments/Instrument Loops/Loops",
        source="shape_sanity_consensus",
        reason="measured pitched music loop",
        shared=[],
        raw_candidate_score=31.0,
        brain_rank=31,
        physics_rank=None,
        shared_winner="Instruments/Instrument Loops/Loops",
        can_override=True,
        strength=0.98,
        is_real_candidate=True,
    )

    measured = pitched_loop_facts(outlier_confirms_siren=False)
    core = DecisionCoreV2()
    claims = core.gather_eligibility_claims(
        raw,
        infer_parent_eligibility(measured),
        measured,
        brain_result=VoterResult(voter_name="brain_full", guesses=[]),
        physics_result=None,
    )
    final = core.arbiter.adjudicate(
        raw_claim=raw,
        consensus_claims=[instrument],
        eligibility_claims=claims,
        facts=measured,
    )

    assert final.folder_path == "Instruments/Instrument Loops/Loops"
    assert final.consensus_status == "final_false_voice_loop_broad_instrument_invariant"


def test_confirmed_concrete_fx_siren_is_not_flattened_to_instrument_loop() -> None:
    raw = claim_from_folder_path(
        folder_path="FX/Designed Noise FX/Siren/Long FX",
        source="strong_consensus",
        reason="real siren",
        shared=[],
        raw_candidate_score=5.0,
        brain_rank=4,
        physics_rank=1,
        shared_winner="FX/Designed Noise FX/Siren/Long FX",
        can_override=False,
        strength=0.68,
        is_real_candidate=True,
    )
    instrument = claim_from_folder_path(
        folder_path="Instruments/Instrument Loops/Loops",
        source="shape_sanity_consensus",
        reason="pitched but real concrete fx",
        shared=[],
        raw_candidate_score=31.0,
        brain_rank=31,
        physics_rank=None,
        shared_winner="Instruments/Instrument Loops/Loops",
        can_override=True,
        strength=0.98,
        is_real_candidate=True,
    )

    final = FamilyClaimArbiter().pick_winner(
        raw_claim=raw,
        claims=[instrument],
        facts=pitched_loop_facts(outlier_confirms_siren=True),
    )

    assert final.folder_path == "FX/Designed Noise FX/Siren/Long FX"


def test_full_and_baby_confirmation_preserves_strong_concrete_fx_siren() -> None:
    raw = claim_from_folder_path(
        folder_path="FX/Designed Noise FX/Siren/Long FX",
        source="strong_consensus",
        reason="real siren",
        shared=[],
        raw_candidate_score=3.0,
        brain_rank=1,
        physics_rank=2,
        shared_winner="FX/Designed Noise FX/Siren/Long FX",
        can_override=False,
        strength=0.80,
        is_real_candidate=True,
    )
    instrument = claim_from_folder_path(
        folder_path="Instruments/Instrument Loops/Loops",
        source="ambiguous_fx_music_loop_broad_bucket",
        reason="pitched but real concrete fx",
        shared=[],
        raw_candidate_score=31.0,
        brain_rank=31,
        physics_rank=None,
        shared_winner="Instruments/Instrument Loops/Loops",
        can_override=True,
        strength=0.98,
        is_real_candidate=False,
    )
    facts = pitched_loop_facts(outlier_confirms_siren=False)

    final = FamilyClaimArbiter().pick_winner(
        raw_claim=raw,
        claims=[instrument],
        facts=facts,
    )

    assert final.folder_path == "FX/Designed Noise FX/Siren/Long FX"


def test_baby_clap_recall_can_rescue_bird_false_positive() -> None:
    raw = claim_from_folder_path(
        folder_path="FX/Animals and Creatures/Bird/One Shots",
        source="strong_consensus",
        reason="physics bird false positive",
        shared=[],
        raw_candidate_score=6.0,
        brain_rank=5,
        physics_rank=1,
        shared_winner="FX/Animals and Creatures/Bird/One Shots",
        can_override=False,
        strength=0.62,
        is_real_candidate=True,
    )
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "shape_vote": {"primary_shape": "single_hit", "confidence": 0.82},
            "measured_roles": {"percussive_one_shot": 0.93, "primary_roles": ["percussive_one_shot"]},
            "spread_baby_vote_result": {
                "diagnostics": {"enabled": True, "lane_name": "spread_baby"},
                "top_guesses": [{"rank": 1, "folder_path": "Drums/Claps Snaps Slaps/Generic Clap/One Shots"}],
            },
        },
    )
    eligibility = EligibilityDecision(
        role_name="percussive_one_shot",
        confidence=0.93,
        broad_folder_path="Drums/Percussion/Generic Percussion/One Shots",
        allowed_top_families=("Drums", "FX", "_TO_REVIEW"),
        blocked_path_fragments=(),
        reason="short percussive hit",
    )
    context = DecisionContext(raw=raw, eligibility=eligibility, facts=facts)

    claims = BabyRecallClaimProducer().produce(context)
    final = FamilyClaimArbiter().pick_winner(raw_claim=raw, claims=claims, facts=facts)

    # Hard rule: identity rescues must not override strong top-family consensus.
    # A future final structure/physics invariant may review or broaden this, but
    # the baby recall identity leaf itself must not win inside pick_winner.
    assert final.folder_path == "FX/Animals and Creatures/Bird/One Shots"


def test_baby_clap_recall_cannot_override_raw_bass_hit() -> None:
    raw = claim_from_folder_path(
        folder_path="Instruments/Bass/808 Bass/One Shots",
        source="strong_consensus",
        reason="raw bass hit",
        shared=[],
        raw_candidate_score=5.0,
        brain_rank=4,
        physics_rank=1,
        shared_winner="Instruments/Bass/808 Bass/One Shots",
        can_override=False,
        strength=0.80,
        is_real_candidate=True,
    )
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "shape_vote": {"primary_shape": "hit_with_tail", "confidence": 0.76},
            "measured_roles": {"percussive_one_shot": 0.90, "primary_roles": ["percussive_one_shot"]},
            "core_baby_vote_result": {
                "diagnostics": {"enabled": True, "lane_name": "core_baby"},
                "top_guesses": [{"rank": 2, "folder_path": "Drums/Claps Snaps Slaps/Generic Clap/One Shots"}],
            },
        },
    )
    eligibility = EligibilityDecision(
        role_name="percussive_one_shot",
        confidence=0.90,
        broad_folder_path="Drums/Percussion/Generic Percussion/One Shots",
        allowed_top_families=("Drums", "FX", "_TO_REVIEW"),
        blocked_path_fragments=(),
        reason="short percussive hit",
    )
    context = DecisionContext(raw=raw, eligibility=eligibility, facts=facts)

    claims = BabyRecallClaimProducer().produce(context)

    assert not any(claim.source == "baby_recall_clap_claim" for claim in claims)


def test_baby_drum_loop_recall_requires_product_or_shared_drum_loop_support() -> None:
    raw = claim_from_folder_path(
        folder_path="FX/Structural and Transitional FX/Risers and Builds/Short Riser/Long FX",
        source="strong_consensus",
        reason="raw riser",
        shared=[
            {
                "folder_path": "FX/Structural and Transitional FX/Risers and Builds/Short Riser/Long FX",
                "combined_rank_score": 2.0,
            }
        ],
        raw_candidate_score=2.0,
        brain_rank=1,
        physics_rank=1,
        shared_winner="FX/Structural and Transitional FX/Risers and Builds/Short Riser/Long FX",
        can_override=False,
        strength=0.90,
        is_real_candidate=True,
    )
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {"primary_shape": "transition_riser", "confidence": 1.0},
            "measured_roles": {"drum_loop": 1.0, "primary_roles": ["drum_loop"]},
            "outlier_baby_vote_result": {
                "diagnostics": {"enabled": True, "lane_name": "outlier_baby"},
                "top_guesses": [{"rank": 2, "folder_path": "Drums/Drum Loops/Full Drum Loops/Loops"}],
            },
        },
    )
    eligibility = EligibilityDecision(
        role_name="drum_loop",
        confidence=0.88,
        broad_folder_path="Drums/Drum Loops/Loops",
        allowed_top_families=("Drums", "_TO_REVIEW"),
        blocked_path_fragments=(),
        reason="measured drum-loop structure",
    )
    context = DecisionContext(raw=raw, eligibility=eligibility, facts=facts)

    claims = BabyRecallClaimProducer().produce(context)

    assert not any(claim.source == "baby_recall_drum_loop_claim" for claim in claims)


def test_single_trumpet_or_flute_baby_row_does_not_force_brass_bucket() -> None:
    raw = claim_from_folder_path(
        folder_path="FX/Designed Noise FX/Siren/Long FX",
        source="strong_consensus",
        reason="raw candidate",
        shared=[],
        raw_candidate_score=5.0,
        brain_rank=4,
        physics_rank=1,
        shared_winner="FX/Designed Noise FX/Siren/Long FX",
        can_override=False,
        strength=0.68,
        is_real_candidate=True,
    )
    facts = pitched_loop_facts(outlier_confirms_siren=False)
    facts.evidence["outlier_baby_vote_result"] = {
        "diagnostics": {"enabled": True, "lane_name": "outlier_baby"},
        "top_guesses": [
            {"rank": 1, "folder_path": "Instruments/Brass/Trumpet/One Shots"},
            {"rank": 3, "folder_path": "Instruments/Woodwinds/Flute/One Shots"},
        ],
    }
    eligibility = EligibilityDecision(
        role_name="pitched_reed_or_instrument_loop",
        confidence=1.0,
        broad_folder_path="Instruments/Woodwinds/Saxophone/Loops",
        allowed_top_families=("Instruments", "_TO_REVIEW"),
        blocked_path_fragments=(),
        reason="generic reed/instrument role",
    )
    context = DecisionContext(raw=raw, eligibility=eligibility, facts=facts)

    claims = BabyRecallClaimProducer().produce(context)

    assert not any(claim.source == "baby_recall_brass_woodwind_claim" for claim in claims)
