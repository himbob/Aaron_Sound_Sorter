from __future__ import annotations

from aaron_sound_sorter.domain.models import ConsensusDecision, SharedAudioFacts
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path, review_claim


def candidate(path: str, score: float) -> dict:
    return {
        "folder_path": path,
        "label": path,
        "top_family": path.split("/", 1)[0],
        "combined_rank_score": score,
        "candidate_role_signature": {},
        "brain_evidence": {"candidate_role_signature": {}},
        "physics_evidence": {"candidate_role_signature": {}},
    }


def raw_fx_with_bass_candidate() -> ConsensusDecision:
    return ConsensusDecision(
        final_label="FX/Impacts and Hits/Short Impact/Long FX",
        final_top="FX",
        folder_path="FX/Impacts and Hits/Short Impact/Long FX",
        consensus_status="strong_consensus",
        reason="synthetic concrete fx raw winner",
        combined_rank_score=10.0,
        shared_candidates=[
            candidate("FX/Impacts and Hits/Short Impact/Long FX", 10.0),
            candidate("Instruments/Bass/Bass Loops", 14.0),
        ],
    )


def bass_loop_eligibility() -> EligibilityDecision:
    return EligibilityDecision(
        role_name="bass_loop",
        confidence=0.96,
        allowed_top_families=("Instruments", "FX", "_TO_REVIEW"),
        blocked_path_fragments=("Drums",),
        broad_folder_path="Instruments/Bass/Bass Loops",
        reason="synthetic low repeated tonal evidence",
    )


def concrete_fx_motion_facts() -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {
                "primary_shape": "beat_loop",
                "confidence": 0.94,
                "shape_scores": [
                    ["beat_loop", 0.94],
                    ["hybrid_fx_motion", 0.63],
                    ["impact_with_tail", 0.60],
                    ["glitch_stutter", 0.59],
                ],
            },
            "measured_roles": {
                "bass_loop": 0.96,
                "pitched_music_loop": 0.98,
                "low_rhythmic_drum_loop": 0.76,
            },
        },
    )


def test_concrete_fx_motion_raw_winner_is_not_stolen_by_bass_loop_rescue() -> None:
    final = DecisionCoreV2().apply_eligibility(
        raw_fx_with_bass_candidate(),
        bass_loop_eligibility(),
        concrete_fx_motion_facts(),
    )

    assert not final.folder_path.startswith("Instruments/Bass")


def test_concrete_fx_drop_raw_blocks_close_broad_instrument_loop_parent() -> None:
    raw = claim_from_folder_path(
        folder_path="FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX",
        source="strong_consensus",
        reason="synthetic concrete drop raw winner",
        shared=[
            candidate(
                "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX", 7.0
            ),
            candidate("Instruments/Instrument Loops/Loops", 11.0),
        ],
        raw_candidate_score=7.0,
        brain_rank=1,
        physics_rank=1,
        shared_winner="FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX",
        can_override=False,
        strength=0.82,
        is_real_candidate=True,
    )
    broad_loop = claim_from_folder_path(
        folder_path="Instruments/Instrument Loops/Loops",
        source="decisive_parent",
        reason="synthetic broad loop fallback",
        shared=raw.shared_candidates,
        raw_candidate_score=11.0,
        brain_rank=2,
        physics_rank=2,
        shared_winner="Instruments/Instrument Loops/Loops",
        can_override=True,
        strength=0.90,
        is_real_candidate=True,
    )

    assert FamilyClaimArbiter()._specific_fx_raw_blocks_broad_instrument_loop_parent(
        raw,
        broad_loop,
        concrete_fx_motion_facts(),
    )


def test_blocked_instrument_review_releases_when_concrete_fx_candidate_is_clearly_better() -> None:
    raw = claim_from_folder_path(
        folder_path="FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/Long FX",
        source="strong_consensus",
        reason="synthetic concrete glitch raw winner",
        shared=[
            candidate("FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/Long FX", 11.0),
            candidate("Instruments/Instrument Loops/Loops", 13.0),
        ],
        raw_candidate_score=11.0,
        brain_rank=1,
        physics_rank=1,
        shared_winner="FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/Long FX",
        can_override=False,
        strength=0.80,
        is_real_candidate=True,
    )
    blocked = claim_from_folder_path(
        folder_path="Instruments/Instrument Loops/Loops",
        source="blocked_role_shape_routing",
        reason="synthetic blocked broad loop",
        shared=raw.shared_candidates,
        raw_candidate_score=13.0,
        brain_rank=2,
        physics_rank=2,
        shared_winner="Instruments/Instrument Loops/Loops",
        can_override=True,
        strength=0.90,
        is_real_candidate=True,
    )

    assert FamilyClaimArbiter()._raw_fx_candidate_decisively_beats_blocked_instrument(raw, blocked)


def test_weak_review_releases_to_fx_when_nearby_voter_window_is_fx_dominant() -> None:
    raw = claim_from_folder_path(
        folder_path="FX/Impacts and Hits/Short Impact/Long FX",
        source="strong_consensus",
        reason="synthetic strong impact raw winner",
        shared=[
            candidate("FX/Impacts and Hits/Short Impact/Long FX", 10.0),
            candidate("FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX", 11.0),
        ],
        raw_candidate_score=10.0,
        brain_rank=1,
        physics_rank=1,
        shared_winner="FX/Impacts and Hits/Short Impact/Long FX",
        can_override=False,
        strength=0.80,
        is_real_candidate=True,
    )
    review = review_claim(
        label="_TO_REVIEW/Shape Conflict",
        reason="synthetic weak shape conflict",
        source="measured_shape_conflict_review",
        shared=raw.shared_candidates,
        winner=raw,
        strength=0.82,
    )

    assert (
        FamilyClaimArbiter()._release_weak_review_to_raw_fx_candidate(
            raw,
            review,
            concrete_fx_motion_facts(),
        )
        == raw
    )
