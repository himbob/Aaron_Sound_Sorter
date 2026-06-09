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
