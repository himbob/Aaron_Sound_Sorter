"""Source-blind guards from new percussion ZIP probes.

These tests encode failure modes found by sorting neutral-renamed files from
one_shot_percussive_sounds.zip.  The assertions stay at the lower measured
claim/arbiter predicate layer so the fix does not depend on filenames.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.claim_producers.measured_drum_structures import (
    MeasuredDrumStructureClaimProducer,
)
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.eligibility_decision import EligibilityDecision
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path
from aaron_sound_sorter.engine.placement_depth import PlacementDepthDecider


def _raw_fx_blip_claim():
    return claim_from_folder_path(
        folder_path="FX/Designed Noise FX/Beep/One Shots",
        source="strong_consensus",
        reason="test raw candidate",
        shared=[],
        raw_candidate_score=13.0,
        brain_rank=12,
        physics_rank=1,
        shared_winner="FX/Designed Noise FX/Beep/One Shots",
        can_override=False,
        strength=0.20,
        is_real_candidate=True,
    )


def _raw_808_claim():
    return claim_from_folder_path(
        folder_path="Instruments/Bass/808 Bass/One Shots",
        source="strong_consensus",
        reason="test raw candidate",
        shared=[],
        raw_candidate_score=3.0,
        brain_rank=2,
        physics_rank=1,
        shared_winner="Instruments/Bass/808 Bass/One Shots",
        can_override=False,
        strength=0.81,
        is_real_candidate=True,
    )


def _context(raw, facts: SharedAudioFacts) -> DecisionContext:
    return DecisionContext(
        raw=raw,
        eligibility=EligibilityDecision(role_name="percussive_one_shot", confidence=0.80),
        facts=facts,
    )


def _ui_blip_percussion_facts() -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "duration_sec": 0.48,
            "shape_vote": {
                "primary_shape": "ui_blip",
                "confidence": 0.90,
                "onset_count": 1.0,
                "true_repetition_score": 0.0,
                "pitched_event_ratio": 1.0,
                "percussive_event_ratio": 0.0,
                "drumlike_frame_ratio": 0.0,
            },
            "physics_subpanels": {
                "flat": {
                    "role_one_shot_score": 1.0,
                    "compact_struck_tonal_percussion_score": 0.88,
                    "hand_drum_membrane_score": 0.92,
                    "pitched_metal_percussion_score": 0.57,
                    "struck_wood_score": 1.0,
                    "drum_kick_source_score": 0.55,
                    "drum_tom_conga_source_score": 0.36,
                    "drum_rim_stick_source_score": 0.54,
                    "drum_hit_score": 0.34,
                    "onset_percussive_onset_score": 0.55,
                    "fx_blip_beep_score": 0.80,
                    "synth_tonal_source_score": 0.55,
                    "physics_subpanel_clean_tone": 0.70,
                }
            },
        },
    )


def _low_kick_like_bass_phrase_facts() -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "duration_sec": 0.52,
            "shape_vote": {
                "primary_shape": "bass_phrase",
                "confidence": 1.0,
                "onset_count": 3.0,
                "low_event_ratio": 0.999,
                "high_event_ratio": 0.0,
                "pitched_event_ratio": 0.0,
                "percussive_event_ratio": 0.0,
                "drumlike_frame_ratio": 0.0,
            },
            "physics_subpanels": {
                "flat": {
                    "role_one_shot_score": 0.555,
                    "drum_kick_source_score": 0.70,
                    "fx_sub_hit_score": 0.66,
                    "compact_struck_tonal_percussion_score": 0.69,
                    "hand_drum_membrane_score": 0.56,
                    "drum_hit_score": 0.29,
                    "bass_808_score": 0.74,
                    "low_end_source_score": 0.59,
                }
            },
        },
    )


def _clean_tonal_snare_rim_facts() -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "duration_sec": 0.65,
            "shape_vote": {
                "primary_shape": "solo_phrase",
                "confidence": 0.83,
                "onset_count": 1.0,
                "pitched_event_ratio": 1.0,
                "percussive_event_ratio": 0.0,
                "drumlike_frame_ratio": 0.0,
            },
            "physics_subpanels": {
                "flat": {
                    "role_one_shot_score": 0.64,
                    "compact_struck_tonal_percussion_score": 0.68,
                    "hand_drum_membrane_score": 0.86,
                    "struck_wood_score": 0.71,
                    "drum_hit_score": 0.34,
                    "drum_rim_stick_source_score": 0.34,
                    "synth_tonal_source_score": 0.55,
                    "physics_subpanel_clean_tone": 0.70,
                    "struck_keys_score": 0.53,
                }
            },
            "physics_vote_1": {
                "folder_path": "Drums/Rims and Sticks/Rimshot/One Shots",
                "label": "Drums/Rims and Sticks/Rimshot/One Shots",
                "rank": 1,
                "score": 0.20,
                "confidence": 0.72,
            },
        },
    )


def test_ui_blip_shaped_struck_percussion_blocks_clean_tonal_fx_release() -> None:
    facts = _ui_blip_percussion_facts()

    assert FamilyClaimArbiter()._facts_support_decisive_struck_percussion_parent(facts)

    claims = MeasuredDrumStructureClaimProducer().produce(_context(_raw_fx_blip_claim(), facts))
    assert any(claim.source == "final_decisive_struck_percussion_parent_invariant" for claim in claims)


def test_low_kick_like_bass_phrase_gets_measured_kick_claim_before_808() -> None:
    facts = _low_kick_like_bass_phrase_facts()

    claims = MeasuredDrumStructureClaimProducer().produce(_context(_raw_808_claim(), facts))

    assert any(claim.source == "final_measured_kick_one_shot_invariant" for claim in claims)


def test_clean_tonal_snare_rim_does_not_release_to_keys_or_fx() -> None:
    facts = _clean_tonal_snare_rim_facts()

    assert not FamilyClaimArbiter()._facts_support_clean_tonal_non_drum_hit(facts)


def _protected_pitched_hand_drum_facts() -> SharedAudioFacts:
    """Neutralized percussion-ZIP failure shape: pitched, short, struck, not voice."""
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "duration_sec": 0.136,
            "parent_eligibility_v2": {
                "role_name": "protected_percussive_one_shot",
                "confidence": 0.74,
                "allowed_top_families": ["Drums", "_TO_REVIEW"],
                "blocked_path_fragments": ["Instruments", "FX"],
                "broad_folder_path": "Drums/Percussion/Generic Percussion/One Shots",
                "decisive": True,
            },
            "shape_vote": {
                "primary_shape": "solo_phrase",
                "secondary_shape": "single_hit",
                "confidence": 0.94,
                "onset_count": 1.0,
                "pitched_event_ratio": 1.0,
                "percussive_event_ratio": 0.0,
                "drumlike_frame_ratio": 0.0,
                "low_event_ratio": 0.997,
                "attack_rise_time_norm": 0.081,
                "temporal_centroid_ratio": 0.111,
            },
            "measured_roles": {
                "pitched_music_phrase": 0.71,
                "percussive_one_shot": 0.42,
                "primary_roles": ["pitched_music_phrase"],
            },
            "physics_subpanels": {
                "flat": {
                    "role_one_shot_score": 0.854,
                    "compact_struck_tonal_percussion_score": 0.714,
                    "hand_drum_membrane_score": 0.870,
                    "pitched_metal_percussion_score": 0.501,
                    "struck_wood_score": 0.637,
                    "onset_percussive_onset_score": 0.528,
                    "drum_kick_source_score": 0.549,
                    "drum_tom_conga_source_score": 0.360,
                    "drum_rim_stick_source_score": 0.340,
                    "drum_hit_score": 0.340,
                    "voice_score": 0.170,
                    "human_spoken_voice_score": 0.311,
                    "human_breath_mouth_score": 0.210,
                    "physics_subpanel_clean_tone": 0.924,
                    "bass_synth_score": 0.639,
                    "low_end_source_score": 0.500,
                    "physics_subpanel_noisy_air": 0.086,
                }
            },
        },
        feature_values_by_name={"duration_sec": 0.136, "event_count_estimate": 1.0},
    )


def test_protected_percussive_parent_release_blocks_voice_firewall_review() -> None:
    """Parent-protected pitched percussion should return to Drums, not review."""
    facts = _protected_pitched_hand_drum_facts()
    arbiter = FamilyClaimArbiter()
    raw = claim_from_folder_path(
        folder_path="Instruments/Keys/Rhodes/One Shots",
        source="strong_consensus",
        reason="brain and physics shared a wrong clean-tonal instrument leaf",
        shared=[],
        raw_candidate_score=3.0,
        brain_rank=2,
        physics_rank=1,
        shared_winner="Instruments/Keys/Rhodes/One Shots",
        can_override=False,
        strength=0.82,
        is_real_candidate=True,
    )

    assert arbiter._facts_support_protected_percussive_parent_release(facts)
    final = arbiter._protect_measured_voice_from_non_voice_leaf(raw, facts)

    assert final.folder_path.startswith("Drums/")
    assert final.source == "final_protected_percussive_parent_release"


def _protected_noisy_texture_hit_facts() -> SharedAudioFacts:
    """Neutralized percussion-ZIP failure shape: noisy short hit misread as breath/cat FX."""
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "duration_sec": 0.348,
            "parent_eligibility_v2": {
                "role_name": "protected_percussive_one_shot",
                "confidence": 0.74,
                "allowed_top_families": ["Drums", "_TO_REVIEW"],
                "blocked_path_fragments": ["FX", "Instruments"],
                "broad_folder_path": "Drums/Percussion/Generic Percussion/One Shots",
                "decisive": True,
            },
            "shape_vote": {
                "primary_shape": "texture_bed",
                "secondary_shape": "noise_texture",
                "confidence": 0.80,
                "onset_count": 2.0,
                "attack_rise_time_norm": 0.0065,
                "pitched_event_ratio": 1.0,
                "percussive_event_ratio": 0.0,
                "drumlike_frame_ratio": 0.0,
            },
            "physics_subpanels": {
                "flat": {
                    "role_one_shot_score": 0.606,
                    "compact_struck_tonal_percussion_score": 0.636,
                    "hand_drum_membrane_score": 0.504,
                    "pitched_metal_percussion_score": 0.493,
                    "struck_wood_score": 0.635,
                    "onset_percussive_onset_score": 0.738,
                    "drum_kick_source_score": 0.249,
                    "drum_snare_source_score": 0.360,
                    "drum_clap_source_score": 0.340,
                    "drum_rim_stick_source_score": 0.340,
                    "drum_cymbal_source_score": 0.340,
                    "drum_metallic_percussion_source_score": 0.340,
                    "drum_guiro_scrape_source_score": 0.552,
                    "voice_score": 0.895,
                    "human_spoken_voice_score": 0.762,
                    "human_breath_mouth_score": 0.891,
                    "physics_subpanel_noisy_air": 0.907,
                }
            },
        },
        feature_values_by_name={"duration_sec": 0.348, "event_count_estimate": 2.0},
    )


def test_protected_noisy_percussive_parent_release_blocks_voice_fx_review() -> None:
    """Fast noisy protected percussion should not remain in voice/animal conflict review."""
    facts = _protected_noisy_texture_hit_facts()
    arbiter = FamilyClaimArbiter()
    raw = claim_from_folder_path(
        folder_path="FX/Animals and Creatures/Cat/One Shots",
        source="strong_consensus",
        reason="brain and physics shared a noisy animal/voice-like FX leaf",
        shared=[],
        raw_candidate_score=3.0,
        brain_rank=1,
        physics_rank=1,
        shared_winner="FX/Animals and Creatures/Cat/One Shots",
        can_override=False,
        strength=0.82,
        is_real_candidate=True,
    )

    assert arbiter._facts_support_protected_percussive_parent_release(facts)
    final = arbiter._percussive_one_shot_parent_firewall(raw, facts=facts)

    assert final is not None
    assert final.folder_path == "Drums/Percussion/Generic Percussion/One Shots"
    assert final.source == "final_protected_percussive_parent_release"


def test_short_repeated_struck_percussion_gets_parent_protection() -> None:
    """203207-style short pitched flam should be Drums-eligible, not review-only."""
    from aaron_sound_sorter.engine.eligibility import infer_parent_eligibility

    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=False,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "duration_sec": 0.697,
            "event_count_estimate": 5.0,
            "event_rate_hz": 7.98,
            "attack_rise_time_norm": 0.009,
            "temporal_centroid_ratio": 0.333,
            "compact_struck_tonal_percussion_score": 0.525,
            "pitched_metal_percussion_score": 0.320,
            "drum_guiro_scrape_source_score": 0.511,
            "drum_kick_source_score": 0.408,
            "drum_snare_source_score": 0.360,
            "drum_hit_score": 0.340,
            "shape_vote": {
                "primary_shape": "pitched_repetition_phrase",
                "confidence": 0.764,
                "onset_count": 5.0,
                "low_event_ratio": 0.683,
                "mid_event_ratio": 0.118,
                "high_event_ratio": 0.198,
                "pitched_event_ratio": 1.0,
                "percussive_event_ratio": 0.0,
                "drumlike_frame_ratio": 0.0,
                "pitch_confidence": 0.329,
                "f0_voiced_ratio": 1.0,
                "spectral_flatness_mean": 0.396,
                "spectral_entropy_mean": 0.491,
                "sustain_ratio": 0.598,
                "non_event_tonal_ratio": 1.0,
            },
        },
        feature_values_by_name={
            "duration_sec": 0.697,
            "event_count_estimate": 5.0,
            "event_rate_hz": 7.98,
            "attack_rise_time_norm": 0.009,
            "temporal_centroid_ratio": 0.333,
            "spectral_flatness_mean": 0.396,
            "spectral_entropy_mean": 0.491,
            "pitch_confidence": 0.329,
            "f0_voiced_ratio": 1.0,
            "loop_pitched_event_ratio": 1.0,
            "loop_percussive_event_ratio": 0.0,
            "loop_drumlike_frame_ratio": 0.0,
            "loop_sustained_tonal_frame_ratio": 0.598,
            "loop_non_event_tonal_ratio": 1.0,
            "sub_bass_ratio_lt_150hz": 0.40,
            "bass_ratio_150_500hz": 0.28,
            "mid_ratio_500_2000hz": 0.12,
            "presence_ratio_2000_8000hz": 0.12,
            "air_ratio_gt_8000hz": 0.08,
        },
    )

    eligibility = infer_parent_eligibility(facts)

    assert eligibility.role_name == "protected_percussive_one_shot"
    assert eligibility.broad_folder_path.startswith("Drums/")
    assert "Instruments" in eligibility.blocked_path_fragments


def test_protected_percussive_parent_blocks_sax_loop_claim() -> None:
    """102787-style pitched percussion must not emit a sax-loop invariant."""
    from aaron_sound_sorter.engine.claim_producers.measured_instrument_branches import (
        MeasuredInstrumentBranchClaimProducer,
    )

    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "duration_sec": 0.255,
            "parent_eligibility_v2": {
                "role_name": "protected_percussive_one_shot",
                "confidence": 0.74,
                "allowed_top_families": ["Drums", "_TO_REVIEW"],
                "blocked_path_fragments": ["Instruments", "FX"],
                "broad_folder_path": "Drums/Percussion/Generic Percussion/One Shots",
                "decisive": True,
            },
            "shape_vote": {
                "primary_shape": "solo_phrase",
                "confidence": 0.921,
                "onset_count": 1.0,
                "attack_rise_time_norm": 0.100,
                "temporal_centroid_ratio": 0.112,
                "pitched_event_ratio": 1.0,
                "percussive_event_ratio": 0.0,
                "drumlike_frame_ratio": 0.0,
                "pitch_confidence": 0.941,
                "f0_voiced_ratio": 1.0,
                "spectral_flatness_mean": 0.218,
            },
            "physics_subpanels": {
                "flat": {
                    "compact_struck_tonal_percussion_score": 0.718,
                    "hand_drum_membrane_score": 0.855,
                    "pitched_metal_percussion_score": 0.542,
                    "struck_wood_score": 0.846,
                    "drum_kick_source_score": 0.452,
                    "drum_snare_source_score": 0.360,
                    "drum_hit_score": 0.329,
                    "onset_percussive_onset_score": 0.412,
                    "woodwind_sax_score": 0.68,
                    "reed_wind_score": 0.58,
                    "reed_reed_noise_score": 0.421,
                    "reed_formant_envelope_score": 0.760,
                    "synth_tonal_source_score": 0.566,
                    "physics_subpanel_clean_tone": 0.843,
                }
            },
        },
    )
    raw = claim_from_folder_path(
        folder_path="Instruments/Strings Bowed/Cello/One Shots",
        source="strong_consensus",
        reason="test raw candidate",
        shared=[],
        raw_candidate_score=3.0,
        brain_rank=1,
        physics_rank=1,
        shared_winner="Instruments/Strings Bowed/Cello/One Shots",
        can_override=False,
        strength=0.80,
        is_real_candidate=True,
    )
    context = DecisionContext(
        raw=raw,
        eligibility=EligibilityDecision(
            role_name="protected_percussive_one_shot",
            confidence=0.74,
            allowed_top_families=("Drums", "_TO_REVIEW"),
            blocked_path_fragments=("Instruments", "FX"),
            broad_folder_path="Drums/Percussion/Generic Percussion/One Shots",
        ),
        facts=facts,
    )

    claims = MeasuredInstrumentBranchClaimProducer().produce(context)

    assert FamilyClaimArbiter()._facts_support_protected_percussive_parent_release(facts)
    assert not any(claim.source == "final_measured_sax_loop_invariant" for claim in claims)


def test_parent_protected_short_noisy_percussion_claim_emits_before_review() -> None:
    """13913-style short high/noisy struck hit should emit a Drums claim below arbiter."""
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "duration_sec": 0.209,
            "event_count_estimate": 1.0,
            "parent_eligibility_v2": {
                "role_name": "protected_percussive_one_shot",
                "confidence": 0.807,
                "allowed_top_families": ["Drums", "_TO_REVIEW"],
                "blocked_path_fragments": ["FX", "Instruments", "Voice", "Human", "Animals"],
                "broad_folder_path": "Drums/Percussion/Generic Percussion/One Shots",
                "decisive": True,
            },
            "shape_vote": {
                "primary_shape": "texture_bed",
                "secondary_shape": "noise_texture",
                "confidence": 0.91,
                "onset_count": 1.0,
                "attack_rise_time_norm": 0.115,
                "temporal_centroid_ratio": 0.108,
                "high_event_ratio": 0.832,
                "pitched_event_ratio": 1.0,
                "percussive_event_ratio": 0.0,
                "drumlike_frame_ratio": 0.0,
                "pitch_confidence": 0.229,
                "f0_voiced_ratio": 1.0,
            },
            "physics_subpanels": {
                "flat": {
                    "compact_struck_tonal_percussion_score": 0.682,
                    "drum_hit_score": 0.673,
                    "drum_cymbal_source_score": 0.748,
                    "drum_guiro_scrape_source_score": 0.748,
                    "drum_metallic_percussion_source_score": 0.712,
                    "drum_kick_source_score": 0.535,
                    "human_breath_mouth_score": 0.880,
                    "human_spoken_voice_score": 0.716,
                    "voice_score": 0.0,
                }
            },
        },
        feature_values_by_name={
            "duration_sec": 0.209,
            "event_count_estimate": 1.0,
            "attack_rise_time_norm": 0.115,
            "temporal_centroid_ratio": 0.108,
        },
    )
    raw = claim_from_folder_path(
        folder_path="_TO_REVIEW/Measured Role Conflict",
        source="blocked_role_shape_routing_review",
        reason="test raw review",
        shared=[],
        raw_candidate_score=99.0,
        brain_rank=99,
        physics_rank=99,
        shared_winner="_TO_REVIEW/Measured Role Conflict",
        can_override=False,
        strength=0.82,
        is_real_candidate=False,
    )
    context = DecisionContext(
        raw=raw,
        eligibility=EligibilityDecision(
            role_name="protected_percussive_one_shot",
            confidence=0.807,
            allowed_top_families=("Drums", "_TO_REVIEW"),
            blocked_path_fragments=("FX", "Instruments"),
            broad_folder_path="Drums/Percussion/Generic Percussion/One Shots",
        ),
        facts=facts,
    )

    claims = MeasuredDrumStructureClaimProducer().produce(context)

    assert any(claim.source == "final_measured_protected_percussive_parent_claim" for claim in claims)
    assert any(claim.folder_path.startswith("Drums/") for claim in claims)


def test_parent_protected_compact_struck_hit_ignores_false_voice_fx_review() -> None:
    """439764-style compact struck hit should not be reviewed as human/animal FX."""
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "duration_sec": 0.232,
            "event_count_estimate": 1.0,
            "parent_eligibility_v2": {
                "role_name": "protected_percussive_one_shot",
                "confidence": 0.832,
                "allowed_top_families": ["Drums", "_TO_REVIEW"],
                "blocked_path_fragments": ["FX", "Instruments", "Voice", "Human", "Animals"],
                "broad_folder_path": "Drums/Percussion/Generic Percussion/One Shots",
                "decisive": True,
            },
            "measured_roles": {
                "percussive_one_shot": 0.832,
                "voiced_one_shot": 0.595,
                "vocal_music_phrase": 0.0,
            },
            "shape_vote": {
                "primary_shape": "echo_tail_hit",
                "secondary_shape": "hit_with_tail",
                "confidence": 0.80,
                "onset_count": 1.0,
                "attack_rise_time_norm": 0.026,
                "temporal_centroid_ratio": 0.183,
                "tail_ratio": 0.201,
                "low_event_ratio": 0.271,
                "mid_event_ratio": 0.468,
                "high_event_ratio": 0.261,
                "pitched_event_ratio": 1.0,
                "percussive_event_ratio": 0.0,
                "drumlike_frame_ratio": 0.0,
                "pitch_confidence": 0.421,
                "f0_voiced_ratio": 1.0,
            },
            "physics_subpanels": {
                "flat": {
                    "compact_struck_tonal_percussion_score": 0.829,
                    "struck_wood_score": 0.682,
                    "hand_drum_membrane_score": 0.703,
                    "pitched_metal_percussion_score": 0.522,
                    "drum_hit_score": 0.34,
                    "drum_kick_source_score": 0.435,
                    "drum_snare_source_score": 0.36,
                    "drum_tom_conga_source_score": 0.36,
                    "drum_rim_stick_source_score": 0.34,
                    "onset_percussive_onset_score": 0.762,
                    "role_one_shot_score": 0.947,
                    "human_breath_mouth_score": 0.796,
                    "human_spoken_voice_score": 0.746,
                    "voice_score": 0.850,
                    "animal_voice_score": 0.721,
                }
            },
        },
        feature_values_by_name={
            "duration_sec": 0.232,
            "event_count_estimate": 1.0,
            "role_one_shot_score": 0.947,
            "attack_rise_time_norm": 0.026,
            "temporal_centroid_ratio": 0.183,
        },
    )
    raw = claim_from_folder_path(
        folder_path="FX/Human and Voice FX/Spoken Voice/One Shots",
        source="strong_consensus",
        reason="test raw voice-fx false positive",
        shared=[],
        raw_candidate_score=2.0,
        brain_rank=1,
        physics_rank=1,
        shared_winner="FX/Human and Voice FX/Spoken Voice/One Shots",
        can_override=False,
        strength=0.82,
        is_real_candidate=True,
    )
    context = DecisionContext(
        raw=raw,
        eligibility=EligibilityDecision(
            role_name="protected_percussive_one_shot",
            confidence=0.832,
            allowed_top_families=("Drums", "_TO_REVIEW"),
            blocked_path_fragments=("FX", "Instruments", "Voice", "Human", "Animals"),
            broad_folder_path="Drums/Percussion/Generic Percussion/One Shots",
        ),
        facts=facts,
    )

    claims = MeasuredDrumStructureClaimProducer().produce(context)

    assert any(claim.source == "final_measured_protected_percussive_parent_claim" for claim in claims)
    assert any(claim.folder_path.startswith("Drums/") for claim in claims)


def test_compact_struck_tonal_percussion_stays_drum_eligible_not_instrument() -> None:
    """14491-style short tonal struck hit should not become broad Instrument Loops."""
    from aaron_sound_sorter.engine.eligibility import infer_parent_eligibility

    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "duration_sec": 0.526,
            "event_count_estimate": 1.0,
            "attack_rise_time_norm": 0.002,
            "temporal_centroid_ratio": 0.100,
            "tail_energy_ratio": 0.039,
            "compact_struck_tonal_percussion_score": 0.884,
            "hand_drum_membrane_score": 0.916,
            "struck_wood_score": 1.0,
            "pitched_metal_percussion_score": 0.602,
            "role_one_shot_score": 1.0,
            "struck_keys_score": 0.561,
            "struck_keys_authority_score": 0.413,
            "keys_tonal_decay_score": 0.0,
            "synth_tonal_source_score": 0.593,
            "shape_vote": {
                "primary_shape": "solo_phrase",
                "secondary_shape": "ui_blip",
                "confidence": 0.923,
                "onset_count": 1.0,
                "pitched_event_ratio": 1.0,
                "sustained_tonal_frame_ratio": 1.0,
                "non_event_tonal_ratio": 1.0,
                "percussive_event_ratio": 0.0,
                "drumlike_frame_ratio": 0.0,
                "pitch_confidence": 0.866,
                "f0_voiced_ratio": 1.0,
                "attack_rise_time_norm": 0.002,
                "temporal_centroid_ratio": 0.100,
                "spectral_flatness_mean": 0.079,
            },
        },
        feature_values_by_name={
            "duration_sec": 0.526,
            "event_count_estimate": 1.0,
            "attack_rise_time_norm": 0.002,
            "temporal_centroid_ratio": 0.100,
            "tail_energy_ratio": 0.039,
            "pitch_confidence": 0.866,
            "f0_voiced_ratio": 1.0,
            "loop_pitched_event_ratio": 1.0,
            "loop_sustained_tonal_frame_ratio": 1.0,
            "loop_non_event_tonal_ratio": 1.0,
            "loop_percussive_event_ratio": 0.0,
            "loop_drumlike_frame_ratio": 0.0,
            "spectral_flatness_mean": 0.079,
            "sub_bass_ratio_lt_150hz": 0.0,
            "bass_ratio_150_500hz": 0.0,
            "mid_ratio_500_2000hz": 0.94,
            "presence_ratio_2000_8000hz": 0.06,
            "air_ratio_gt_8000hz": 0.0,
        },
    )

    eligibility = infer_parent_eligibility(facts)

    assert eligibility.role_name == "protected_percussive_one_shot"
    assert eligibility.broad_folder_path.startswith("Drums/")
    assert "Instruments" in eligibility.blocked_path_fragments



def _short_low_percussive_hit_facts() -> SharedAudioFacts:
    """105434-style short low pitched hit: strong one-shot, not an instrument loop."""
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=False,
        is_long=False,
        evidence={
            "duration_sec": 0.450,
            "event_count_estimate": 1.0,
            "attack_rise_time_norm": 0.107,
            "temporal_centroid_ratio": 0.377,
            "tail_energy_ratio": 0.494,
            "shape_vote": {
                "primary_shape": "solo_phrase",
                "secondary_shape": "hit_with_tail",
                "confidence": 0.925,
                "onset_count": 1.0,
                "low_event_ratio": 0.999,
                "mid_event_ratio": 0.00002,
                "high_event_ratio": 0.0,
                "pitch_confidence": 0.962,
                "f0_voiced_ratio": 0.0,
                "attack_rise_time_norm": 0.107,
                "temporal_centroid_ratio": 0.377,
                "tail_ratio": 0.494,
            },
            "measured_roles": {
                "bass_loop": 0.0,
                "percussive_one_shot": 0.931,
                "pitched_music_phrase": 0.818,
                "pitched_music_loop": 0.0,
                "evidence": {
                    "low_total": 0.999,
                    "high_total": 0.0,
                    "low_pitched_hit_raw": 0.931,
                    "pitch_confidence": 0.962,
                    "f0_voiced_ratio": 0.0,
                    "attack_rise_time_norm": 0.107,
                    "temporal_centroid_ratio": 0.377,
                    "tail_energy_ratio": 0.494,
                },
            },
        },
        feature_values_by_name={
            "duration_sec": 0.450,
            "event_count_estimate": 1.0,
            "sub_bass_ratio_lt_150hz": 0.80,
            "bass_ratio_150_500hz": 0.19,
            "presence_ratio_2000_8000hz": 0.0,
            "air_ratio_gt_8000hz": 0.0,
            "pitch_confidence": 0.962,
            "f0_voiced_ratio": 0.0,
        },
    )


def test_short_low_percussive_hit_is_low_kick_like_not_instrument_loop() -> None:
    """105434-style low hit should not broaden to Instruments/Instrument Loops."""
    from aaron_sound_sorter.engine.eligibility import infer_parent_eligibility

    facts = _short_low_percussive_hit_facts()
    eligibility = infer_parent_eligibility(facts)

    assert eligibility.role_name == "low_kick_like_hit"
    assert eligibility.broad_folder_path == "Drums/Kick Drums/Generic Kick/One Shots"
    assert "Instruments" in eligibility.blocked_path_fragments


def test_short_low_percussive_hit_blocks_instrument_broadening_at_placement_depth() -> None:
    facts = _short_low_percussive_hit_facts()
    raw = claim_from_folder_path(
        folder_path="Instruments/Bass/Synth Bass/One Shots",
        source="strong_consensus",
        reason="test raw synth-bass false positive",
        shared=[],
        raw_candidate_score=12.0,
        brain_rank=10,
        physics_rank=2,
        shared_winner="Instruments/Bass/Synth Bass/One Shots",
        can_override=False,
        strength=0.25,
        is_real_candidate=True,
    )
    claim = PlacementDepthDecider().refine_claim(raw_claim=raw, shared=[], facts=facts)

    # No shared rows means placement-depth has no work to do in this unit path.
    # The measured predicate itself is the guard used by the real shared-candidate path.
    assert claim is None
    assert PlacementDepthDecider.facts_support_short_low_percussive_drum_parent(facts)


def test_parent_low_kick_like_release_blocks_final_instrument_steal() -> None:
    facts = _short_low_percussive_hit_facts()
    facts.evidence["parent_eligibility_v2"] = {
        "role_name": "low_kick_like_hit",
        "confidence": 0.931,
        "allowed_top_families": ["Drums", "FX", "_TO_REVIEW"],
        "blocked_path_fragments": ["Instruments", "Bass Loops", "Instrument Loops"],
        "broad_folder_path": "Drums/Kick Drums/Generic Kick/One Shots",
        "decisive": True,
    }
    raw = claim_from_folder_path(
        folder_path="Instruments/Bass/Synth Bass/One Shots",
        source="strong_consensus",
        reason="test raw synth-bass false positive",
        shared=[],
        raw_candidate_score=12.0,
        brain_rank=10,
        physics_rank=2,
        shared_winner="Instruments/Bass/Synth Bass/One Shots",
        can_override=False,
        strength=0.25,
        is_real_candidate=True,
    )

    winner = FamilyClaimArbiter()._percussive_one_shot_parent_firewall(raw, facts=facts)

    assert winner is not None
    assert winner.folder_path == "Drums/Kick Drums/Generic Kick/One Shots"
    assert winner.source == "final_parent_low_kick_like_release"



def test_short_repeated_struck_percussion_parent_release_beats_instrument_identity() -> None:
    """205973-style bright repeated struck hit should honor protected drum parent."""
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=False,
        evidence={
            "duration_sec": 0.551,
            "event_count_estimate": 4.0,
            "parent_eligibility_v2": {
                "role_name": "protected_percussive_one_shot",
                "confidence": 0.74,
                "allowed_top_families": ["Drums", "_TO_REVIEW"],
                "blocked_path_fragments": ["Instruments", "FX", "Drum Loops"],
                "broad_folder_path": "Drums/Percussion/Generic Percussion/One Shots",
                "decisive": True,
            },
            "shape_vote": {
                "primary_shape": "pitched_repetition_phrase",
                "confidence": 0.913,
                "onset_count": 4.0,
                "onset_density_hz": 8.61,
                "attack_rise_time_norm": 0.041,
                "temporal_centroid_ratio": 0.077,
                "pitch_confidence": 0.824,
                "low_event_ratio": 0.991,
                "pitched_event_ratio": 1.0,
                "percussive_event_ratio": 0.0,
                "drumlike_frame_ratio": 0.0,
            },
            "physics_subpanels": {
                "flat": {
                    "compact_struck_tonal_percussion_score": 0.689,
                    "struck_wood_score": 0.912,
                    "onset_percussive_onset_score": 0.767,
                    "drum_kick_source_score": 0.561,
                    "drum_metallic_percussion_source_score": 0.523,
                    "physics_subpanel_clean_tone": 0.66,
                }
            },
        },
        feature_values_by_name={
            "duration_sec": 0.551,
            "event_count_estimate": 4.0,
            "spectral_flatness_mean": 0.238,
            "pitch_confidence": 0.824,
            "loop_pitched_event_ratio": 1.0,
            "loop_sustained_tonal_frame_ratio": 1.0,
            "loop_percussive_event_ratio": 0.0,
            "loop_drumlike_frame_ratio": 0.0,
        },
    )
    raw = claim_from_folder_path(
        folder_path="Instruments/Woodwinds/Flute/One Shots",
        source="strong_consensus",
        reason="test raw flute false positive",
        shared=[],
        raw_candidate_score=17.0,
        brain_rank=10,
        physics_rank=7,
        shared_winner="Instruments/Woodwinds/Flute/One Shots",
        can_override=False,
        strength=0.0,
        is_real_candidate=True,
    )

    winner = FamilyClaimArbiter()._percussive_one_shot_parent_firewall(raw, facts=facts)

    assert winner is not None
    assert winner.folder_path.startswith("Drums/")
    assert winner.source == "final_protected_percussive_parent_release"



def test_voice_like_tiny_material_hit_still_honors_protected_percussion_parent() -> None:
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "duration_sec": 0.090,
            "event_count_estimate": 0.0,
            "parent_eligibility_v2": {
                "role_name": "protected_percussive_one_shot",
                "allowed_top_families": ["Drums", "_TO_REVIEW"],
                "blocked_path_fragments": ["Instruments", "FX", "Voice"],
                "broad_folder_path": "Drums/Percussion/Generic Percussion/One Shots",
            },
            "shape_vote": {
                "primary_shape": "solo_phrase",
                "confidence": 0.798,
                "onset_count": 0.0,
                "attack_rise_time_norm": 0.053,
                "temporal_centroid_ratio": 0.214,
                "pitch_confidence": 0.710,
                "f0_voiced_ratio": 1.0,
            },
            "measured_roles": {"voiced_one_shot": 1.0},
            "physics_subpanels": {
                "flat": {
                    "compact_struck_tonal_percussion_score": 0.777,
                    "hand_drum_membrane_score": 0.842,
                    "struck_wood_score": 0.815,
                    "drum_kick_source_score": 0.430,
                    "onset_percussive_onset_score": 0.755,
                    "voice_score": 0.0,
                    "human_spoken_voice_score": 0.0,
                }
            },
        },
        feature_values_by_name={"duration_sec": 0.090, "spectral_flatness_mean": 0.255},
    )
    raw = claim_from_folder_path(
        folder_path="Instruments/Voice/Phrase/One Shots",
        source="strong_consensus",
        reason="test raw voice false positive",
        shared=[],
        raw_candidate_score=12.0,
        brain_rank=8,
        physics_rank=7,
        shared_winner="Instruments/Voice/Phrase/One Shots",
        can_override=False,
        strength=0.25,
        is_real_candidate=True,
    )

    winner = FamilyClaimArbiter()._percussive_one_shot_parent_firewall(raw, facts=facts)

    assert winner is not None
    assert winner.folder_path.startswith("Drums/")


def test_high_cymbal_texture_hit_honors_protected_percussion_parent() -> None:
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=False,
        is_long=False,
        evidence={
            "duration_sec": 0.279,
            "event_count_estimate": 1.0,
            "parent_eligibility_v2": {
                "role_name": "protected_percussive_one_shot",
                "allowed_top_families": ["Drums", "_TO_REVIEW"],
                "blocked_path_fragments": ["Instruments", "FX"],
                "broad_folder_path": "Drums/Percussion/Generic Percussion/One Shots",
            },
            "shape_vote": {
                "primary_shape": "texture_bed",
                "confidence": 0.857,
                "onset_count": 1.0,
                "attack_rise_time_norm": 0.125,
                "high_event_ratio": 0.994,
                "pitch_confidence": 0.513,
                "f0_voiced_ratio": 1.0,
            },
            "physics_subpanels": {
                "flat": {
                    "compact_struck_tonal_percussion_score": 0.546,
                    "pitched_metal_percussion_score": 0.585,
                    "drum_cymbal_source_score": 0.811,
                    "drum_metallic_percussion_source_score": 0.618,
                }
            },
        },
        feature_values_by_name={"duration_sec": 0.279, "spectral_flatness_mean": 0.477},
    )
    raw = claim_from_folder_path(
        folder_path="Instruments/Woodwinds/Flute/One Shots",
        source="strong_consensus",
        reason="test raw flute false positive",
        shared=[],
        raw_candidate_score=17.0,
        brain_rank=10,
        physics_rank=7,
        shared_winner="Instruments/Woodwinds/Flute/One Shots",
        can_override=False,
        strength=0.0,
        is_real_candidate=True,
    )

    winner = FamilyClaimArbiter()._percussive_one_shot_parent_firewall(raw, facts=facts)

    assert winner is not None
    assert winner.folder_path.startswith("Drums/")


def test_bright_noisy_tail_parent_hit_claim_emits_below_arbiter() -> None:
    """15582-style short high noisy cymbal/guiro hit should not remain review-only."""
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=False,
        is_long=False,
        evidence={
            "duration_sec": 0.341,
            "event_count_estimate": 1.0,
            "parent_eligibility_v2": {
                "role_name": "protected_percussive_one_shot",
                "confidence": 0.74,
                "allowed_top_families": ["Drums", "_TO_REVIEW"],
                "blocked_path_fragments": ["FX", "Instruments", "Voice", "Human", "Animals"],
                "broad_folder_path": "Drums/Percussion/Generic Percussion/One Shots",
                "decisive": True,
            },
            "shape_vote": {
                "primary_shape": "texture_bed",
                "secondary_shape": "noise_texture",
                "confidence": 0.937,
                "onset_count": 1.0,
                "high_event_ratio": 0.679,
                "mid_event_ratio": 0.234,
                "low_event_ratio": 0.087,
                "pitched_event_ratio": 0.0,
                "percussive_event_ratio": 1.0,
                "drumlike_frame_ratio": 1.0,
                "pitch_confidence": 0.143,
                "attack_rise_time_norm": 0.307,
                "temporal_centroid_ratio": 0.353,
                "tail_ratio": 0.496,
            },
            "physics_subpanels": {
                "flat": {
                    "role_one_shot_score": 0.545,
                    "compact_struck_tonal_percussion_score": 0.574,
                    "pitched_metal_percussion_score": 0.703,
                    "drum_hit_score": 0.772,
                    "drum_cymbal_source_score": 0.791,
                    "drum_guiro_scrape_source_score": 0.800,
                    "drum_metallic_percussion_source_score": 0.737,
                    "drum_snare_source_score": 0.689,
                    "onset_percussive_onset_score": 0.596,
                    "human_breath_mouth_score": 0.838,
                    "woodwind_sax_score": 0.408,
                    "reed_wind_score": 0.447,
                }
            },
        },
        feature_values_by_name={
            "duration_sec": 0.341,
            "event_count_estimate": 1.0,
            "spectral_flatness_mean": 0.613,
            "tail_energy_ratio": 0.496,
        },
    )
    raw = claim_from_folder_path(
        folder_path="_TO_REVIEW/Measured Role Conflict",
        source="blocked_role_shape_routing_review",
        reason="test raw review",
        shared=[],
        raw_candidate_score=99.0,
        brain_rank=99,
        physics_rank=99,
        shared_winner="_TO_REVIEW/Measured Role Conflict",
        can_override=False,
        strength=0.82,
        is_real_candidate=False,
    )
    context = DecisionContext(
        raw=raw,
        eligibility=EligibilityDecision(
            role_name="protected_percussive_one_shot",
            confidence=0.74,
            allowed_top_families=("Drums", "_TO_REVIEW"),
            blocked_path_fragments=("FX", "Instruments", "Voice", "Human", "Animals"),
            broad_folder_path="Drums/Percussion/Generic Percussion/One Shots",
        ),
        facts=facts,
    )

    claims = MeasuredDrumStructureClaimProducer().produce(context)

    assert any(claim.source == "final_measured_protected_percussive_parent_claim" for claim in claims)
    assert any(claim.folder_path.startswith("Drums/") for claim in claims)


def test_drum_material_voice_decoy_gets_parent_eligibility_before_vocal_review() -> None:
    """439835-style cymbal/guiro hit should not become a voiced-one-shot conflict."""
    from aaron_sound_sorter.engine.eligibility import infer_parent_eligibility

    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "duration_sec": 0.882,
            "event_count_estimate": 1.0,
            "attack_rise_time_norm": 0.0127,
            "temporal_centroid_ratio": 0.213,
            "tail_energy_ratio": 0.207,
            "shape_vote": {
                "primary_shape": "texture_bed",
                "confidence": 0.93,
                "onset_count": 1.0,
                "attack_rise_time_norm": 0.0127,
                "temporal_centroid_ratio": 0.213,
                "pitched_event_ratio": 1.0,
                "percussive_event_ratio": 0.0,
                "drumlike_frame_ratio": 0.0,
            },
            "measured_roles": {
                "voiced_one_shot": 0.720,
                "percussive_one_shot": 0.327,
                "pitched_music_phrase": 0.514,
                "evidence": {"formant_light_voice_identity": 0.0},
            },
            "compact_struck_tonal_percussion_score": 0.816,
            "pitched_metal_percussion_score": 0.717,
            "drum_hit_score": 0.718,
            "drum_cymbal_source_score": 0.793,
            "drum_guiro_scrape_source_score": 0.871,
            "drum_metallic_percussion_source_score": 0.824,
            "onset_percussive_onset_score": 0.848,
            "role_one_shot_score": 0.960,
            "voice_score": 0.960,
            "human_breath_mouth_score": 0.916,
            "human_spoken_voice_score": 0.794,
        },
        feature_values_by_name={
            "duration_sec": 0.882,
            "event_count_estimate": 1.0,
            "attack_rise_time_norm": 0.0127,
            "temporal_centroid_ratio": 0.213,
            "tail_energy_ratio": 0.207,
            "pitch_confidence": 0.527,
            "f0_voiced_ratio": 0.571,
            "spectral_flatness_mean": 0.36,
            "loop_pitched_event_ratio": 1.0,
            "loop_percussive_event_ratio": 0.0,
            "loop_drumlike_frame_ratio": 0.0,
            "loop_sustained_tonal_frame_ratio": 0.42,
            "loop_non_event_tonal_ratio": 0.42,
            "mid_ratio_500_2000hz": 0.45,
            "presence_ratio_2000_8000hz": 0.35,
            "air_ratio_gt_8000hz": 0.10,
        },
    )

    eligibility = infer_parent_eligibility(facts)

    assert eligibility.role_name == "protected_percussive_one_shot"
    assert eligibility.broad_folder_path.startswith("Drums/")
    assert "Instruments" in eligibility.blocked_path_fragments


def test_bright_metallic_voice_decoy_parent_claim_emits_before_review() -> None:
    """439835-style bright metallic hit should emit a measured drum claim below arbiter."""
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "duration_sec": 0.882,
            "event_count_estimate": 1.0,
            "attack_rise_time_norm": 0.0127,
            "temporal_centroid_ratio": 0.213,
            "tail_energy_ratio": 0.207,
            "parent_eligibility_v2": {
                "role_name": "protected_percussive_one_shot",
                "confidence": 0.816,
                "allowed_top_families": ["Drums", "_TO_REVIEW"],
                "blocked_path_fragments": ["FX", "Instruments", "Voice", "Human", "Animals"],
                "broad_folder_path": "Drums/Percussion/Generic Percussion/One Shots",
                "decisive": True,
            },
            "shape_vote": {
                "primary_shape": "texture_bed",
                "secondary_shape": "noise_texture",
                "confidence": 0.828,
                "onset_count": 1.0,
                "attack_rise_time_norm": 0.0127,
                "temporal_centroid_ratio": 0.213,
                "tail_ratio": 0.207,
                "pitched_event_ratio": 1.0,
                "percussive_event_ratio": 0.0,
                "drumlike_frame_ratio": 0.0,
            },
            "measured_roles": {
                "voiced_one_shot": 0.720,
                "percussive_one_shot": 0.327,
                "pitched_music_phrase": 0.514,
                "evidence": {"formant_light_voice_identity": 0.880},
            },
            "physics_subpanels": {
                "flat": {
                    "role_one_shot_score": 0.960,
                    "compact_struck_tonal_percussion_score": 0.816,
                    "pitched_metal_percussion_score": 0.717,
                    "struck_wood_score": 0.658,
                    "hand_drum_membrane_score": 0.514,
                    "drum_hit_score": 0.718,
                    "drum_cymbal_source_score": 0.793,
                    "drum_guiro_scrape_source_score": 0.871,
                    "drum_metallic_percussion_source_score": 0.824,
                    "onset_percussive_onset_score": 0.848,
                    "voice_score": 0.960,
                    "human_breath_mouth_score": 0.916,
                    "human_spoken_voice_score": 0.794,
                }
            },
        },
        feature_values_by_name={
            "duration_sec": 0.882,
            "event_count_estimate": 1.0,
            "attack_rise_time_norm": 0.0127,
            "temporal_centroid_ratio": 0.213,
            "tail_energy_ratio": 0.207,
        },
    )
    raw = claim_from_folder_path(
        folder_path="_TO_REVIEW/Measured Role Conflict",
        source="blocked_role_shape_routing_review",
        reason="test raw review",
        shared=[],
        raw_candidate_score=99.0,
        brain_rank=99,
        physics_rank=99,
        shared_winner="_TO_REVIEW/Measured Role Conflict",
        can_override=False,
        strength=0.82,
        is_real_candidate=False,
    )
    context = DecisionContext(
        raw=raw,
        eligibility=EligibilityDecision(
            role_name="protected_percussive_one_shot",
            confidence=0.816,
            allowed_top_families=("Drums", "_TO_REVIEW"),
            blocked_path_fragments=("FX", "Instruments", "Voice", "Human", "Animals"),
            broad_folder_path="Drums/Percussion/Generic Percussion/One Shots",
        ),
        facts=facts,
    )

    claims = MeasuredDrumStructureClaimProducer().produce(context)

    assert any(claim.source == "final_measured_protected_percussive_parent_claim" for claim in claims)
    assert any(claim.folder_path.startswith("Drums/") for claim in claims)
