"""Tests for isolated profile-candidate claim production."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import CategoryGuess, SharedAudioFacts, VoterResult
from aaron_sound_sorter.engine.claim_producers.profile_candidates import ProfileCandidateClaimProducer
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path


def raw_claim(path: str, score: float = 6.0, shared: list[dict] | None = None):
    return claim_from_folder_path(
        folder_path=path,
        source="test_raw",
        reason="test raw",
        shared=shared or [],
        raw_candidate_score=score,
        brain_rank=3,
        physics_rank=3,
        shared_winner=path,
        can_override=False,
        strength=0.5,
        is_real_candidate=True,
    )


def guess(path: str, rank: int) -> CategoryGuess:
    return CategoryGuess(
        label=path,
        folder_path=path,
        top_family=path.split("/", 1)[0],
        score=0.90,
        confidence=0.90,
        rank=rank,
        reason="test guess",
    )


def eligibility(role: str) -> EligibilityDecision:
    return EligibilityDecision(
        role_name=role,
        confidence=0.92,
        allowed_top_families=("Drums", "Instruments", "FX"),
        broad_folder_path="Instruments/Instrument Loops/Loops",
        reason="test eligibility",
    )


def facts(
    shape: str,
    confidence: float,
    *,
    role_evidence: dict[str, float] | None = None,
    full_lane_top: str | None = None,
) -> SharedAudioFacts:
    evidence = {"shape_vote": {"primary_shape": shape, "confidence": confidence}}
    if role_evidence is not None:
        evidence["measured_roles"] = {"evidence": role_evidence}
    if full_lane_top is not None:
        evidence["full_brain_vote_result"] = {
            "diagnostics": {"enabled": True, "lane_name": "full"},
            "top_guesses": [
                {
                    "folder_path": full_lane_top,
                    "label": full_lane_top,
                    "rank": 1,
                    "top_family": full_lane_top.split("/", 1)[0],
                }
            ],
        }
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence=evidence,
    )


def test_profile_candidate_producer_returns_kick_claim_from_measured_kick_context() -> None:
    producer = ProfileCandidateClaimProducer()
    context = DecisionContext(
        raw=raw_claim("Drums/Percussion/Generic Percussion/One Shots"),
        eligibility=eligibility("low_kick_like_hit"),
        facts=facts("single_hit", 0.92),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess("Drums/Kick Drums/Generic Kick/One Shots", 1)],
        ),
    )

    claims = producer.produce(context)

    assert len(claims) == 1
    assert claims[0].source == "profile_candidate_kick_claim"
    assert claims[0].folder_path == "Drums/Kick Drums/Generic Kick/One Shots"
    assert claims[0].can_override is True


def test_profile_candidate_producer_does_not_promote_reed_without_specific_reed_role() -> None:
    producer = ProfileCandidateClaimProducer()
    context = DecisionContext(
        raw=raw_claim("Instruments/Instrument Loops/Loops", score=4.0),
        eligibility=eligibility("pitched_music_loop"),
        facts=facts("pitched_phrase", 1.0),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess("Instruments/Woodwinds/Saxophone/One Shots", 1)],
        ),
    )

    assert producer.produce(context) == []


def test_profile_candidate_does_not_steal_mallet_bell_leaf_for_reed_rescue() -> None:
    producer = ProfileCandidateClaimProducer()
    context = DecisionContext(
        raw=raw_claim("Instruments/Mallets and Bells/Vibraphone/Loops", score=5.0),
        eligibility=eligibility("pitched_reed_or_instrument_loop"),
        facts=facts(
            "pitched_phrase",
            0.91,
            role_evidence={
                "loop_pitched_event_ratio": 1.0,
                "loop_sustained_tonal_frame_ratio": 0.95,
                "loop_tonal_to_percussive_balance": 0.90,
            },
        ),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess("Instruments/Woodwinds/Saxophone/Loops", 1)],
        ),
    )

    claims = producer.produce(context)

    assert claims == []


def test_profile_candidate_rescues_reed_from_fx_when_full_lane_and_physics_support_instruments() -> None:
    producer = ProfileCandidateClaimProducer()
    context = DecisionContext(
        raw=raw_claim("FX/Designed Noise FX/Beep/One Shots", score=5.0),
        eligibility=eligibility("pitched_reed_or_instrument_phrase"),
        facts=facts(
            "hit_with_tail",
            0.72,
            full_lane_top="Instruments/Woodwinds/Saxophone/One Shots",
        ),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess("FX/Designed Noise FX/Beep/One Shots", 1)],
        ),
        physics_result=VoterResult(
            voter_name="physics",
            guesses=[guess("Instruments/Keys/Wurlitzer/One Shots", 1)],
        ),
    )

    claims = producer.produce(context)

    assert len(claims) == 1
    assert claims[0].source == "profile_candidate_ambiguous_reed_like_parent_claim"
    assert claims[0].folder_path == "Instruments/Instrument Loops/Loops"


def test_profile_candidate_does_not_turn_wrong_instrument_leaf_into_reed_without_reed_witness() -> None:
    producer = ProfileCandidateClaimProducer()
    context = DecisionContext(
        raw=raw_claim("Instruments/Mallets and Bells/Vibraphone/Loops", score=5.0),
        eligibility=eligibility("pitched_music_loop"),
        facts=facts(
            "pitched_phrase",
            0.91,
            role_evidence={
                "loop_pitched_event_ratio": 1.0,
                "loop_sustained_tonal_frame_ratio": 0.95,
                "loop_tonal_to_percussive_balance": 0.90,
            },
        ),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess("Instruments/Synths/Synth Lead/Loops", 1)],
        ),
    )

    assert producer.produce(context) == []


def test_profile_candidate_does_not_steal_string_loop_when_reed_candidate_is_nearby() -> None:
    producer = ProfileCandidateClaimProducer()
    context = DecisionContext(
        raw=raw_claim("Instruments/Strings/String Loops/Loops", score=5.0),
        eligibility=eligibility("pitched_reed_or_instrument_loop"),
        facts=facts(
            "pitched_phrase",
            0.91,
            role_evidence={
                "loop_pitched_event_ratio": 1.0,
                "loop_sustained_tonal_frame_ratio": 0.95,
                "loop_tonal_to_percussive_balance": 0.90,
            },
        ),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess("Instruments/Woodwinds/Saxophone/Loops", 1)],
        ),
    )

    assert producer.produce(context) == []


def test_profile_candidate_does_not_steal_keys_loop_when_reed_candidate_is_nearby() -> None:
    producer = ProfileCandidateClaimProducer()
    context = DecisionContext(
        raw=raw_claim("Instruments/Keys/Keys Loops/Loops", score=5.0),
        eligibility=eligibility("pitched_reed_or_instrument_loop"),
        facts=facts(
            "pitched_phrase",
            0.91,
            role_evidence={
                "loop_pitched_event_ratio": 1.0,
                "loop_sustained_tonal_frame_ratio": 0.95,
                "loop_tonal_to_percussive_balance": 0.90,
            },
        ),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess("Instruments/Woodwinds/Saxophone/Loops", 1)],
        ),
    )

    assert producer.produce(context) == []


def test_profile_candidate_does_not_repromote_sax_when_physics_rejects_clean_keys_loop_decoy() -> None:
    producer = ProfileCandidateClaimProducer()
    context = DecisionContext(
        raw=raw_claim("Instruments/Instrument Loops/Loops", score=5.0),
        eligibility=eligibility("pitched_reed_or_instrument_loop"),
        facts=facts(
            "pitched_phrase",
            0.87,
            role_evidence={
                "loop_pitched_event_ratio": 1.0,
                "loop_sustained_tonal_frame_ratio": 1.0,
                "loop_percussive_event_ratio": 0.0,
                "pitch_confidence": 0.59,
                "f0_voiced_ratio": 0.95,
                "low_event_ratio": 0.406,
                "mid_event_ratio": 0.573,
                "high_event_ratio": 0.021,
                "spectral_flatness_mean": 0.047,
                "presence_ratio_2000_8000hz": 0.015,
                "air_ratio_gt_8000hz": 0.002,
                "body_noise_ratio": 0.17,
                "tail_noise_ratio": 0.22,
            },
        ),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess("Instruments/Woodwinds/Saxophone/One Shots", 1)],
        ),
        physics_result=VoterResult(
            voter_name="physics",
            guesses=[guess("Instruments/Instrument Loops/Loops", 1)],
        ),
    )

    claims = producer.produce(context)

    assert not any(claim.source == "profile_candidate_sax_leaf_claim" for claim in claims)
    assert all(claim.folder_path != "Instruments/Woodwinds/Saxophone/Loops" for claim in claims)


def shared_row(path: str, score: float, *, brain_rank: int = 2, physics_rank: int = 2) -> dict:
    return {
        "folder_path": path,
        "label": path,
        "top_family": path.split("/", 1)[0],
        "combined_rank_score": score,
        "brain_rank": brain_rank,
        "physics_rank": physics_rank,
    }


def test_profile_candidate_emits_synth_claim_against_broad_instrument_loop() -> None:
    producer = ProfileCandidateClaimProducer()
    synth_path = "Instruments/Synths/Synth Pad/Loops"
    context = DecisionContext(
        raw=raw_claim(
            "Instruments/Instrument Loops/Loops",
            score=5.0,
            shared=[shared_row(synth_path, 3.0)],
        ),
        eligibility=eligibility("pitched_music_loop"),
        facts=facts(
            "bass_phrase",
            0.99,
            role_evidence={
                "loop_pitched_event_ratio": 1.0,
                "loop_sustained_tonal_frame_ratio": 0.95,
                "loop_tonal_to_percussive_balance": 0.90,
            },
        ),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess(synth_path, 1)],
        ),
    )

    claims = producer.produce(context)

    assert any(claim.source == "profile_candidate_synth_claim" for claim in claims)
    synth_claim = [claim for claim in claims if claim.source == "profile_candidate_synth_claim"][0]
    assert synth_claim.folder_path == synth_path
    assert synth_claim.is_real_candidate is True


def test_profile_candidate_bass_claim_beats_mallet_leaf_when_low_body_is_clean() -> None:
    producer = ProfileCandidateClaimProducer()
    measured = facts(
        "solo_phrase",
        0.81,
        role_evidence={
            "bass_loop": 0.63,
            "pitched_music_loop": 0.76,
            "low_rhythmic_drum_loop": 0.74,
        },
    )
    measured.evidence["shape_vote"].update(
        {
            "low_event_ratio": 0.98,
            "pitch_confidence": 0.96,
            "pitched_event_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        }
    )
    context = DecisionContext(
        raw=raw_claim("Instruments/Mallets and Bells/Vibraphone/Loops", score=20.0),
        eligibility=eligibility("pitched_music_loop"),
        facts=measured,
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess("Instruments/Bass/Generic Bass/Loops", 1)],
        ),
    )

    claims = producer.produce(context)

    assert any(claim.source == "profile_candidate_bass_claim" for claim in claims)
    bass_claim = [claim for claim in claims if claim.source == "profile_candidate_bass_claim"][0]
    assert bass_claim.folder_path == "Instruments/Bass/Bass Loops"


def test_profile_candidate_promotes_repeated_low_bass_phrase_one_shot_witness_to_loop() -> None:
    producer = ProfileCandidateClaimProducer()
    measured = facts(
        "solo_phrase",
        0.90,
        role_evidence={
            "pitched_music_phrase": 0.89,
            "low_total": 0.99,
        },
    )
    measured.evidence["shape_vote"].update(
        {
            "onset_count": 9.0,
            "onset_span_ratio": 0.70,
            "true_repetition_score": 0.63,
            "low_event_ratio": 0.98,
            "pitch_confidence": 0.92,
            "pitched_event_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        }
    )
    context = DecisionContext(
        raw=raw_claim("FX/Designed Noise FX/Blip/One Shots", score=14.0),
        eligibility=eligibility("pitched_music_phrase"),
        facts=measured,
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess("Instruments/Bass/Electric Bass/One Shots", 1)],
        ),
    )

    claims = producer.produce(context)

    bass_claim = [claim for claim in claims if claim.source == "profile_candidate_bass_claim"][0]
    assert bass_claim.folder_path == "Instruments/Bass/Bass Loops"
    broad_loop_claim = claim_from_folder_path(
        folder_path="Instruments/Instrument Loops/Loops",
        source="candidate_true_bucket_rescue",
        reason="unit test broad fallback",
        shared=[],
        raw_candidate_score=9999.0,
        brain_rank=4,
        physics_rank=10,
        shared_winner="",
        can_override=True,
        strength=0.90,
        is_real_candidate=False,
    )

    decision = FamilyClaimArbiter().adjudicate(
        raw_claim=context.raw,
        consensus_claims=[broad_loop_claim],
        eligibility_claims=[bass_claim],
        facts=measured,
    )

    assert decision.folder_path == "Instruments/Bass/Bass Loops"


def test_profile_candidate_bass_claim_stands_down_for_crowded_mixed_loop() -> None:
    producer = ProfileCandidateClaimProducer()
    measured = facts(
        "bass_phrase",
        0.98,
        role_evidence={
            "bass_loop": 0.63,
            "pitched_music_loop": 0.76,
            "low_rhythmic_drum_loop": 0.74,
        },
    )
    measured.evidence["shape_vote"].update(
        {
            "low_event_ratio": 0.83,
            "pitch_confidence": 0.80,
            "pitched_event_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
        }
    )
    context = DecisionContext(
        raw=raw_claim("Instruments/Mixed Musical Loops/Multi Instrument/Loops", score=20.0),
        eligibility=eligibility("pitched_music_loop"),
        facts=measured,
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[
                guess("Instruments/Synths/Synth Loops/Loops", 1),
                guess("Instruments/Keys/Organ/Loops", 2),
                guess("Instruments/Mixed Musical Loops/Multi Instrument/Loops", 3),
                guess("Instruments/Keys/Piano/Loops", 4),
                guess("Instruments/Bass/Generic Bass/Loops", 5),
            ],
        ),
    )

    claims = producer.produce(context)

    assert all(claim.source != "profile_candidate_bass_claim" for claim in claims)


def test_profile_candidate_mixed_loop_stands_down_for_measured_drum_loop() -> None:
    producer = ProfileCandidateClaimProducer()
    mixed_path = "Instruments/Mixed Musical Loops/Multi Instrument/Loops"
    measured = facts_with_shape_metrics(
        "beat_loop",
        0.96,
        role_evidence={
            "low_rhythmic_drum_loop": 0.64,
            "pitched_music_loop": 1.0,
            "bass_loop": 0.96,
        },
        shape_metrics={
            "onset_count": 12.0,
            "true_repetition_score": 0.58,
            "low_event_ratio": 0.72,
            "high_event_ratio": 0.12,
            "drum_loop_source_score": 0.61,
        },
    )
    context = DecisionContext(
        raw=raw_claim("Instruments/Keys/Processed Keys/Loops", score=5.0),
        eligibility=eligibility("pitched_music_loop"),
        facts=measured,
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[
                guess(mixed_path, 1),
                guess("Drums/Drum Loops/Loops", 4),
            ],
        ),
    )

    claims = producer.produce(context)

    assert all(claim.source != "profile_candidate_mixed_musical_loop_claim" for claim in claims)


def test_profile_candidate_synth_claim_stands_down_for_crowded_mixed_instrument_loop() -> None:
    producer = ProfileCandidateClaimProducer()
    synth_path = "Instruments/Synths/Synth Arp/Loops"
    measured = facts(
        "bass_phrase",
        0.98,
        role_evidence={
            "loop_pitched_event_ratio": 1.0,
            "loop_sustained_tonal_frame_ratio": 1.0,
            "loop_tonal_to_percussive_balance": 1.0,
        },
    )
    measured.evidence["physics_vote_result"] = {
        "top_guesses": [
            {
                "folder_path": "Instruments/Keys/Organ/Loops",
                "label": "Instruments/Keys/Organ/Loops",
                "rank": 1,
            }
        ]
    }
    context = DecisionContext(
        raw=raw_claim(
            synth_path,
            score=19.0,
            shared=[
                shared_row(synth_path, 19.0, brain_rank=12, physics_rank=7),
                shared_row("Instruments/Keys/Organ/Loops", 20.0, brain_rank=18, physics_rank=2),
                shared_row("Instruments/Plucked Strings/Harp/Loops", 22.0, brain_rank=17, physics_rank=5),
                shared_row(
                    "Instruments/Mixed Musical Loops/Multi Instrument/Loops", 24.0, brain_rank=13, physics_rank=11
                ),
            ],
        ),
        eligibility=eligibility("pitched_music_loop"),
        facts=measured,
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess("Instruments/Synths/Synth Loops/Loops", 1)],
        ),
    )

    claims = producer.produce(context)

    assert all(claim.source != "profile_candidate_synth_claim" for claim in claims)


def test_profile_candidate_emits_multiple_claims_so_arbiter_not_branch_order_decides() -> None:
    producer = ProfileCandidateClaimProducer()
    synth_path = "Instruments/Synths/Synth Pad/Loops"
    reed_path = "Instruments/Woodwinds/Saxophone/Loops"
    context = DecisionContext(
        raw=raw_claim(
            "Instruments/Instrument Loops/Loops",
            score=8.0,
            shared=[shared_row(synth_path, 3.0), shared_row(reed_path, 5.0)],
        ),
        eligibility=eligibility("pitched_reed_or_instrument_loop"),
        facts=facts(
            "pitched_phrase",
            0.94,
            role_evidence={
                "loop_pitched_event_ratio": 1.0,
                "loop_sustained_tonal_frame_ratio": 0.95,
                "loop_tonal_to_percussive_balance": 0.90,
            },
        ),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess(synth_path, 1), guess(reed_path, 2)],
        ),
    )

    claims = producer.produce(context)
    sources = {claim.source for claim in claims}

    assert "profile_candidate_synth_claim" in sources
    # Reed should be blocked here because a stronger concrete synth sibling is present.
    assert "profile_candidate_brass_woodwind_claim" not in sources


def test_profile_candidate_broadens_false_voice_when_true_voice_evidence_is_absent() -> None:
    producer = ProfileCandidateClaimProducer()
    context = DecisionContext(
        raw=raw_claim("Instruments/Voice/Vocal One Shots/Loops", score=5.0),
        eligibility=eligibility("pitched_music_loop"),
        facts=facts(
            "transition_riser",
            0.72,
            role_evidence={
                "loop_pitched_event_ratio": 0.95,
                "loop_sustained_tonal_frame_ratio": 0.91,
                "loop_tonal_to_percussive_balance": 0.88,
                "formant_light_voice_identity": 0.0,
            },
        ),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess("Drums/Claps Snaps Slaps/Hand Clap/Loops", 1)],
        ),
    )

    claims = producer.produce(context)

    assert len(claims) == 1
    assert claims[0].source == "profile_candidate_non_voice_instrument_parent_claim"
    assert claims[0].folder_path == "Instruments/Instrument Loops/Loops"
    assert claims[0].is_real_candidate is False


def test_profile_synth_candidate_stands_down_for_clean_designed_tonal_keys_loop() -> None:
    producer = ProfileCandidateClaimProducer()
    measured = facts(
        "designed_tonal_fx",
        0.757,
        role_evidence={"pitched_music_loop": 0.92},
    )
    measured.evidence["shape_vote"].update(
        {
            "mid_event_ratio": 0.66,
            "high_event_ratio": 0.019,
            "pitched_event_ratio": 1.0,
            "sustained_tonal_frame_ratio": 1.0,
            "non_event_tonal_ratio": 1.0,
            "percussive_event_ratio": 0.0,
            "drumlike_frame_ratio": 0.0,
            "spectral_flatness_mean": 0.003,
        }
    )
    measured.evidence["physics_subpanels"] = {
        "flat": {
            "struck_keys_score": 0.56,
            "keys_tonal_decay_score": 0.75,
            "struck_keys_authority_score": 0.46,
            "fx_motion_score": 0.22,
            "fx_transition_authority_score": 0.30,
        }
    }
    synth_path = "Instruments/Synths/Synth Lead/One Shots"
    context = DecisionContext(
        raw=raw_claim(
            "Instruments/Instrument Loops/Loops",
            score=10.0,
            shared=[
                {
                    "folder_path": synth_path,
                    "label": synth_path,
                    "top_family": "Instruments",
                    "combined_rank_score": 7.0,
                }
            ],
        ),
        eligibility=eligibility("pitched_music_loop"),
        facts=measured,
        brain_result=VoterResult(voter_name="brain", guesses=[guess(synth_path, 1)]),
    )

    claims = producer.produce(context)

    assert all(claim.source != "profile_candidate_synth_claim" for claim in claims)


def test_profile_candidate_preserves_direct_terminal_identity_from_both_voters() -> None:
    producer = ProfileCandidateClaimProducer()
    rhodes_path = "Instruments/Keys/Rhodes/Loops"
    context = DecisionContext(
        raw=raw_claim(
            rhodes_path,
            score=2.0,
            shared=[
                {
                    "folder_path": rhodes_path,
                    "label": rhodes_path,
                    "top_family": "Instruments",
                    "combined_rank_score": 2.0,
                    "brain_rank": 1,
                    "physics_rank": 1,
                },
                {
                    "folder_path": "Instruments/Synths/Synth Chord/Loops",
                    "label": "Instruments/Synths/Synth Chord/Loops",
                    "top_family": "Instruments",
                    "combined_rank_score": 10.0,
                    "brain_rank": 5,
                    "physics_rank": 5,
                },
            ],
        ),
        eligibility=eligibility("pitched_music_loop"),
        facts=facts_with_shape_metrics(
            "bass_phrase",
            0.93,
            role_evidence={
                "pitched_music_loop": 0.90,
                "loop_pitched_event_ratio": 1.0,
                "loop_sustained_tonal_frame_ratio": 1.0,
                "loop_tonal_to_percussive_balance": 1.0,
            },
        ),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess(rhodes_path, 1)],
        ),
    )

    claims = producer.produce(context)

    assert all(claim.source != "instrument_sibling_conflict_parent_claim" for claim in claims)


def test_true_voice_claim_beats_instrument_sibling_broadening() -> None:
    from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter

    producer = ProfileCandidateClaimProducer()
    raw_path = "Instruments/Synths/Synth Stab/One Shots"
    voice_path = "Instruments/Voice/Phrase/One Shots"
    context = DecisionContext(
        raw=raw_claim(
            raw_path,
            score=14.0,
            shared=[
                {
                    "folder_path": raw_path,
                    "label": raw_path,
                    "top_family": "Instruments",
                    "combined_rank_score": 14.0,
                    "brain_rank": 4,
                    "physics_rank": 10,
                },
                {
                    "folder_path": voice_path,
                    "label": voice_path,
                    "top_family": "Instruments",
                    "combined_rank_score": 15.0,
                    "brain_rank": 11,
                    "physics_rank": 4,
                },
                {
                    "folder_path": "Instruments/Keys/Wurlitzer/One Shots",
                    "label": "Instruments/Keys/Wurlitzer/One Shots",
                    "top_family": "Instruments",
                    "combined_rank_score": 16.0,
                    "brain_rank": 10,
                    "physics_rank": 6,
                },
            ],
        ),
        eligibility=eligibility("vocal_phrase"),
        facts=facts_with_shape_metrics(
            "vocal_phrase",
            0.80,
            role_evidence={
                "pitched_music_phrase": 0.86,
                "voiced_one_shot": 0.82,
                "formant_light_voice_identity": 0.71,
            },
            shape_metrics={"f0_voiced_ratio": 0.94},
        ),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess("Instruments/Synths/Synth Chord/One Shots", 1)],
        ),
    )

    claims = producer.produce(context)
    final = FamilyClaimArbiter().pick_winner(raw_claim=context.raw, claims=claims, facts=context.facts)

    assert any(claim.source == "profile_candidate_voice_claim" for claim in claims)
    assert any(claim.source == "instrument_sibling_conflict_parent_claim" for claim in claims)
    assert final.source == "profile_candidate_voice_claim"
    assert final.folder_path == voice_path


def test_fx_beep_true_voice_candidate_can_escape_when_measured_voice_is_strong() -> None:
    from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter

    producer = ProfileCandidateClaimProducer()
    raw_path = "FX/Designed Noise FX/Beep/One Shots"
    voice_path = "Instruments/Voice/Phrase/One Shots"
    context = DecisionContext(
        raw=raw_claim(
            raw_path,
            score=6.0,
            shared=[
                {
                    "folder_path": raw_path,
                    "label": raw_path,
                    "top_family": "FX",
                    "combined_rank_score": 6.0,
                    "brain_rank": 1,
                    "physics_rank": 5,
                },
                {
                    "folder_path": voice_path,
                    "label": voice_path,
                    "top_family": "Instruments",
                    "combined_rank_score": 18.0,
                    "brain_rank": 14,
                    "physics_rank": 4,
                },
            ],
        ),
        eligibility=eligibility("voiced_one_shot"),
        facts=facts_with_shape_metrics(
            "hit_with_tail",
            0.72,
            role_evidence={
                "voiced_one_shot": 0.91,
                "formant_light_voice_identity": 0.96,
            },
            shape_metrics={"f0_voiced_ratio": 1.0},
        ),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess(raw_path, 1)],
        ),
    )

    claims = producer.produce(context)
    final = FamilyClaimArbiter().pick_winner(raw_claim=context.raw, claims=claims, facts=context.facts)

    assert any(claim.source == "profile_candidate_voice_claim" for claim in claims)
    assert final.folder_path == voice_path


def test_false_voice_choir_broadens_when_voice_identity_is_absent() -> None:
    producer = ProfileCandidateClaimProducer()
    context = DecisionContext(
        raw=raw_claim("Instruments/Voice/Choir/Loops", score=4.0),
        eligibility=eligibility("pitched_music_loop"),
        facts=facts_with_shape_metrics(
            "vocal_phrase",
            0.95,
            role_evidence={
                "pitched_music_loop": 0.90,
                "pitched_music_phrase": 0.90,
                "vocal_music_phrase": 0.36,
                "formant_light_voice_identity": 0.0,
                "loop_pitched_event_ratio": 1.0,
                "loop_sustained_tonal_frame_ratio": 1.0,
                "loop_tonal_to_percussive_balance": 1.0,
            },
            shape_metrics={
                "f0_voiced_ratio": 1.0,
                "sustain_ratio": 0.69,
                "percussive_event_ratio": 0.0,
                "high_event_ratio": 0.12,
            },
        ),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess("Instruments/Plucked Strings/Banjo/Loops", 1)],
        ),
    )

    claims = producer.produce(context)

    assert any(claim.source == "profile_candidate_non_voice_instrument_parent_claim" for claim in claims)


def test_sustained_non_struck_mallet_winner_broadens_to_instrument_parent() -> None:
    producer = ProfileCandidateClaimProducer()
    mallet_path = "Instruments/Mallets and Bells/Bells and Mallets/Loops"
    context = DecisionContext(
        raw=raw_claim(
            mallet_path,
            score=3.0,
            shared=[
                {
                    "folder_path": mallet_path,
                    "label": mallet_path,
                    "top_family": "Instruments",
                    "combined_rank_score": 3.0,
                    "brain_rank": 2,
                    "physics_rank": 1,
                }
            ],
        ),
        eligibility=eligibility("pitched_music_loop"),
        facts=facts_with_shape_metrics(
            "sustained_pad",
            0.71,
            role_evidence={
                "pitched_music_loop": 0.80,
                "loop_pitched_event_ratio": 0.93,
                "loop_sustained_tonal_frame_ratio": 0.98,
                "loop_tonal_to_percussive_balance": 0.98,
            },
            shape_metrics={
                "f0_voiced_ratio": 0.98,
                "sustain_ratio": 0.68,
                "pitched_event_ratio": 0.93,
                "percussive_event_ratio": 0.0,
                "drumlike_frame_ratio": 0.0,
                "attack_rise_time_norm": 0.03,
                "high_event_ratio": 0.31,
            },
        ),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess(mallet_path, 2)],
        ),
    )

    claims = producer.produce(context)

    assert any(claim.source == "instrument_sibling_conflict_parent_claim" for claim in claims)


def facts_with_shape_metrics(
    shape: str,
    confidence: float,
    *,
    role_evidence: dict[str, float] | None = None,
    shape_metrics: dict[str, float] | None = None,
) -> SharedAudioFacts:
    shape_vote = {"primary_shape": shape, "confidence": confidence}
    if shape_metrics:
        shape_vote.update(shape_metrics)
    evidence = {"shape_vote": shape_vote}
    if role_evidence is not None:
        evidence["measured_roles"] = {**role_evidence, "evidence": role_evidence}
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence=evidence,
    )


def stable_synth_like_metrics() -> dict[str, float]:
    return {
        "high_event_ratio": 0.0,
        "percussive_event_ratio": 0.0,
        "drumlike_frame_ratio": 0.0,
        "attack_rise_time_norm": 0.39,
        "sustain_ratio": 0.78,
        "f0_voiced_ratio": 1.0,
        "pitched_event_ratio": 1.0,
    }


def test_profile_candidate_synth_claim_can_challenge_non_struck_mallet_false_positive() -> None:
    producer = ProfileCandidateClaimProducer()
    synth_path = "Instruments/Synths/Synth Arp/Loops"
    context = DecisionContext(
        raw=raw_claim("Instruments/Mallets and Bells/Bells and Mallets/Loops", score=4.0),
        eligibility=eligibility("pitched_music_loop"),
        facts=facts_with_shape_metrics(
            "vocal_phrase",
            1.0,
            role_evidence={
                "pitched_music_loop": 0.97,
                "pitched_music_phrase": 0.96,
                "loop_pitched_event_ratio": 1.0,
                "loop_sustained_tonal_frame_ratio": 1.0,
                "loop_tonal_to_percussive_balance": 0.95,
            },
            shape_metrics=stable_synth_like_metrics(),
        ),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess(synth_path, 1)],
        ),
    )

    claims = producer.produce(context)

    assert any(claim.source == "profile_candidate_synth_claim" for claim in claims)
    synth_claim = [claim for claim in claims if claim.source == "profile_candidate_synth_claim"][0]
    assert synth_claim.folder_path == synth_path


def test_profile_candidate_synth_claim_can_challenge_non_plucked_string_false_positive() -> None:
    producer = ProfileCandidateClaimProducer()
    synth_path = "Instruments/Synths/Synth Pad/Loops"
    context = DecisionContext(
        raw=raw_claim("Instruments/Plucked Strings/Harp/Loops", score=4.0),
        eligibility=eligibility("pitched_music_loop"),
        facts=facts_with_shape_metrics(
            "pitched_phrase",
            0.91,
            role_evidence={
                "pitched_music_loop": 0.92,
                "pitched_music_phrase": 0.91,
                "loop_pitched_event_ratio": 1.0,
                "loop_sustained_tonal_frame_ratio": 0.94,
                "loop_tonal_to_percussive_balance": 0.92,
            },
            shape_metrics=stable_synth_like_metrics(),
        ),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess(synth_path, 1)],
        ),
    )

    claims = producer.produce(context)

    assert any(claim.source == "profile_candidate_synth_claim" for claim in claims)
    synth_claim = [claim for claim in claims if claim.source == "profile_candidate_synth_claim"][0]
    assert synth_claim.folder_path == synth_path


def test_profile_candidate_synth_claim_stands_down_for_voice_like_solo_phrase() -> None:
    producer = ProfileCandidateClaimProducer()
    synth_path = "Instruments/Synths/Synth Pad/One Shots"
    context = DecisionContext(
        raw=raw_claim(synth_path, score=7.0, shared=[shared_row(synth_path, 9.0, brain_rank=7, physics_rank=23)]),
        eligibility=eligibility("pitched_music_loop"),
        facts=facts_with_shape_metrics(
            "solo_phrase",
            0.87,
            role_evidence={
                "pitched_music_loop": 0.70,
                "pitched_music_phrase": 0.65,
            },
            shape_metrics={
                "synth_tonal_source_score": 0.56,
                "synth_pad_score": 0.68,
                "human_spoken_voice_score": 0.50,
                "voice_choir_score": 0.54,
                "bass_synth_score": 0.64,
                "drum_hit_score": 0.08,
            },
        ),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess(synth_path, 7)],
        ),
    )

    claims = producer.produce(context)

    assert all(claim.source != "profile_candidate_synth_claim" for claim in claims)


def test_profile_candidate_synth_claim_does_not_steal_plucked_string_with_pluck_body() -> None:
    producer = ProfileCandidateClaimProducer()
    synth_path = "Instruments/Synths/Synth Pad/Loops"
    plucked_metrics = stable_synth_like_metrics()
    plucked_metrics.update({"high_event_ratio": 0.22, "percussive_event_ratio": 0.14, "attack_rise_time_norm": 0.12})
    context = DecisionContext(
        raw=raw_claim("Instruments/Plucked Strings/Harp/Loops", score=4.0),
        eligibility=eligibility("pitched_music_loop"),
        facts=facts_with_shape_metrics(
            "pitched_phrase",
            0.91,
            role_evidence={
                "pitched_music_loop": 0.92,
                "loop_pitched_event_ratio": 1.0,
                "loop_sustained_tonal_frame_ratio": 0.94,
                "loop_tonal_to_percussive_balance": 0.92,
            },
            shape_metrics=plucked_metrics,
        ),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess(synth_path, 1)],
        ),
    )

    claims = producer.produce(context)

    assert all(claim.source != "profile_candidate_synth_claim" for claim in claims)


def test_placement_depth_keeps_synth_branch_when_terminal_is_under_supported() -> None:
    from aaron_sound_sorter.engine.placement_depth import PlacementDepthDecider

    decider = PlacementDepthDecider()
    synth_terminal = "Instruments/Synths/Synth Pad/Loops"
    raw = claim_from_folder_path(
        folder_path=synth_terminal,
        source="profile_candidate_synth_claim",
        reason="test synth terminal",
        shared=[],
        raw_candidate_score=6.0,
        brain_rank=1,
        physics_rank=9,
        shared_winner=synth_terminal,
        can_override=True,
        strength=0.91,
        is_real_candidate=True,
    )
    shared = [
        {
            "folder_path": synth_terminal,
            "label": synth_terminal,
            "top_family": "Instruments",
            "combined_rank_score": 6.0,
            "brain_rank": 1,
            "physics_rank": 9,
            "brain_evidence": {"profile_effective_count": 7, "profile_fact_profile_strength": "tentative"},
            "physics_evidence": {"profile_effective_count": 7, "profile_fact_profile_strength": "tentative"},
            "candidate_role_signature": {"pitched_music_loop": 0.94},
        },
        {
            "folder_path": "Instruments/Instrument Loops/Loops",
            "label": "Instruments/Instrument Loops/Loops",
            "top_family": "Instruments",
            "combined_rank_score": 8.0,
            "brain_rank": 4,
            "physics_rank": 4,
            "candidate_role_signature": {"pitched_music_loop": 0.80},
        },
    ]
    facts_obj = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={"measured_roles": {"pitched_music_loop": 0.95}},
    )

    refined = decider.refine_claim(raw_claim=raw, shared=shared, facts=facts_obj)

    assert refined is not None
    assert refined.folder_path == "Instruments/Synths/Synth Loops/Loops"
    assert refined.source == "placement_depth_broad_bucket"


def test_placement_depth_keeps_kick_branch_when_tiny_kick_leaf_broadens() -> None:
    from aaron_sound_sorter.engine.placement_depth import PlacementDepthDecider

    decider = PlacementDepthDecider()
    short_kick = "Drums/Kick Drums/Short Kick/One Shots"
    raw = claim_from_folder_path(
        folder_path=short_kick,
        source="strong_consensus",
        reason="test tiny kick terminal",
        shared=[],
        raw_candidate_score=9.0,
        brain_rank=1,
        physics_rank=2,
        shared_winner=short_kick,
        can_override=False,
        strength=0.95,
        is_real_candidate=True,
    )
    shared = [
        {
            "folder_path": short_kick,
            "label": short_kick,
            "top_family": "Drums",
            "combined_rank_score": 9.0,
            "brain_rank": 1,
            "physics_rank": 2,
            "brain_evidence": {"profile_effective_count": 1, "profile_fact_profile_strength": "tentative"},
            "physics_evidence": {"profile_effective_count": 1, "profile_fact_profile_strength": "tentative"},
            "candidate_role_signature": {"percussive_one_shot": 0.94},
        }
    ]
    facts_obj = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={"measured_roles": {"percussive_one_shot": 1.0}},
    )

    refined = decider.refine_claim(raw_claim=raw, shared=shared, facts=facts_obj)

    assert refined is not None
    assert refined.folder_path == "Drums/Kick Drums/Generic Kick/One Shots"
    assert refined.source == "placement_depth_broad_bucket"


def test_placement_depth_broadens_specific_leaf_when_physics_measures_compound_loop() -> None:
    from aaron_sound_sorter.engine.placement_depth import PlacementDepthDecider

    decider = PlacementDepthDecider()
    guitar_leaf = "Instruments/Guitar/Nylon Guitar/One Shots"
    raw = claim_from_folder_path(
        folder_path=guitar_leaf,
        source="strong_consensus",
        reason="test mixed loop raw leaf",
        shared=[],
        raw_candidate_score=8.0,
        brain_rank=5,
        physics_rank=3,
        shared_winner=guitar_leaf,
        can_override=False,
        strength=0.92,
        is_real_candidate=True,
    )
    shared = [
        {
            "folder_path": guitar_leaf,
            "label": guitar_leaf,
            "top_family": "Instruments",
            "combined_rank_score": 8.0,
            "brain_rank": 5,
            "physics_rank": 3,
            "physics_evidence": {
                "physics_layer_branch": "MixedInstrument",
                "compound_music_prefer_broad_loop": True,
                "profile_effective_count": 80,
            },
            "brain_evidence": {"profile_effective_count": 80},
            "candidate_role_signature": {"pitched_music_loop": 0.86},
        },
        {
            "folder_path": "Instruments/Instrument Loops/Loops",
            "label": "Instruments/Instrument Loops/Loops",
            "top_family": "Instruments",
            "combined_rank_score": 15.0,
            "brain_rank": 14,
            "physics_rank": 1,
            "candidate_role_signature": {"pitched_music_loop": 0.82},
        },
    ]
    facts_obj = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={"measured_roles": {"pitched_music_loop": 0.80}},
    )

    refined = decider.refine_claim(raw_claim=raw, shared=shared, facts=facts_obj)

    assert refined is not None
    assert refined.folder_path == "Instruments/Instrument Loops/Loops"
    assert refined.source == "placement_depth_compound_music_broad_bucket"


def test_profile_candidate_synth_claim_ignores_fx_synth_riser_brain_guess() -> None:
    producer = ProfileCandidateClaimProducer()
    context = DecisionContext(
        raw=raw_claim("Instruments/Voice/Loops", score=7.0),
        eligibility=eligibility("pitched_percussion_loop"),
        facts=facts(
            "bass_phrase",
            0.76,
            role_evidence={
                "loop_pitched_event_ratio": 0.90,
                "loop_sustained_tonal_frame_ratio": 0.90,
                "loop_tonal_to_percussive_balance": 0.70,
            },
        ),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess("FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX", 2)],
        ),
    )

    claims = producer.produce(context)

    assert all(claim.source != "profile_candidate_synth_claim" for claim in claims)


def test_profile_candidate_preserves_sustained_true_voice_identity_body() -> None:
    producer = ProfileCandidateClaimProducer()
    context = DecisionContext(
        raw=raw_claim("Instruments/Voice/Loops", score=7.0),
        eligibility=eligibility("pitched_music_loop"),
        facts=facts(
            "bass_phrase",
            0.76,
            role_evidence={
                "loop_pitched_event_ratio": 0.90,
                "loop_sustained_tonal_frame_ratio": 0.90,
                "loop_tonal_to_percussive_balance": 0.70,
                "formant_light_voice_identity": 0.72,
            },
        ),
        brain_result=VoterResult(
            voter_name="brain",
            guesses=[guess("Instruments/Voice/Loops", 1)],
        ),
    )
    context.facts.evidence["shape_vote"].update(
        {
            "f0_voiced_ratio": 0.88,
            "sustain_ratio": 0.74,
            "percussive_event_ratio": 0.06,
            "high_event_ratio": 0.13,
        }
    )

    claims = producer.produce(context)

    assert all(claim.source != "profile_candidate_non_voice_instrument_parent_claim" for claim in claims)
