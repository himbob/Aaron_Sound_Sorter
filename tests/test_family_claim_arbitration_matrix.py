"""Family-claim arbitration regressions.

These tests keep the dangerous final-decision seam table-driven.  They use
synthetic measured roles and candidate rankings, not producer names or stored
WAV files, so future rescue changes cannot silently let one family steal
another.
"""

from __future__ import annotations

import pytest

from aaron_sound_sorter.domain.models import ConsensusDecision, SharedAudioFacts
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import build_human_voice_claim


def candidate(path: str, score: float, *, roles: dict[str, float] | None = None) -> dict:
    """Build one synthetic shared candidate."""
    role_signature = roles or {}
    return {
        "folder_path": path,
        "label": path,
        "top_family": path.split("/", 1)[0],
        "combined_rank_score": score,
        "candidate_role_signature": role_signature,
        "brain_evidence": {"candidate_role_signature": role_signature},
        "physics_evidence": {"candidate_role_signature": role_signature},
    }


def raw(path: str, score: float, candidates: list[dict]) -> ConsensusDecision:
    """Build a synthetic raw consensus decision."""
    return ConsensusDecision(
        final_label=path,
        final_top=path.split("/", 1)[0],
        folder_path=path,
        consensus_status="synthetic_raw",
        reason="synthetic raw decision",
        combined_rank_score=score,
        shared_candidates=candidates,
    )


def eligibility(role: str, broad: str, allowed: tuple[str, ...]) -> EligibilityDecision:
    """Build synthetic parent eligibility."""
    return EligibilityDecision(
        role_name=role,
        confidence=0.80,
        allowed_top_families=allowed,
        blocked_path_fragments=(),
        broad_folder_path=broad,
        reason="synthetic family-claim eligibility",
    )


def facts(shape: str, confidence: float, roles: dict[str, float]) -> SharedAudioFacts:
    """Build synthetic shared facts with measured roles and shape."""
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {"primary_shape": shape, "confidence": confidence},
            "measured_roles": roles,
            "direct_body_view": {"available": True, "measured_roles": roles},
        },
    )


def test_weak_human_voice_claim_cannot_steal_clean_pitched_instrument_loop() -> None:
    """Piano/keys/sax-like loops must not become Human/Voice from weak evidence."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "FX/Human and Voice FX",
            14.0,
            [
                candidate("FX/Human and Voice FX", 14.0),
                candidate("Instruments/Instrument Loops/Loops", 12.0),
            ],
        ),
        eligibility(
            "pitched_music_loop",
            "Instruments/Instrument Loops/Loops",
            ("Instruments", "_TO_REVIEW"),
        ),
        facts(
            "pitched_phrase",
            0.92,
            {
                "pitched_music_loop": 0.90,
                "pitched_music_phrase": 0.86,
                "vocal_music_phrase": 0.20,
                "voiced_one_shot": 0.0,
            },
        ),
    )

    assert final.final_top == "Instruments"
    assert "Human and Voice" not in final.folder_path


def test_decisive_human_voice_candidate_can_still_rescue_processed_vocal_loop() -> None:
    """True processed vocals rescue to Instruments/Voice, not the broad FX bucket."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Instrument Loops/Loops",
            26.0,
            [
                candidate("FX/Human and Voice FX/Spoken Voice/Long FX", 6.0, roles={"vocal_music_phrase": 0.90}),
                candidate("Instruments/Instrument Loops/Loops", 26.0),
                candidate("Drums/Drum Loops/Loops", 44.0),
            ],
        ),
        eligibility(
            "vocal_music_phrase",
            "FX/Human and Voice FX",
            ("FX", "Instruments", "_TO_REVIEW"),
        ),
        facts(
            "vocal_phrase",
            0.94,
            {
                "vocal_music_phrase": 0.88,
                "voiced_one_shot": 0.75,
                "pitched_music_loop": 0.62,
            },
        ),
    )

    assert final.folder_path == "Instruments/Voice/Phrase/One Shots"


@pytest.mark.parametrize(
    ("stable_strength", "candidate_strength", "expected"),
    [
        (0.90, 0.10, False),
        (0.20, 0.90, True),
        (0.20, 0.10, True),
    ],
)
def test_human_voice_claim_requires_positive_evidence_or_clear_rank_win(
    stable_strength: float,
    candidate_strength: float,
    expected: bool,
) -> None:
    """Human/Voice can override only when the claim beats rival families."""
    claim = build_human_voice_claim(
        has_voice_candidate=True,
        voice_candidate_score=7.0,
        raw_score=18.0,
        competitor_score=15.0,
        voice_role_strength=0.20,
        candidate_role_strength=candidate_strength,
        stable_instrument_claim_strength=stable_strength,
    )

    assert claim.can_override is expected


def test_dry_midrange_sax_phrase_gets_sax_loop_parent_without_using_names() -> None:
    """Fast dry midrange reed phrases should not flatten to generic loops."""
    from aaron_sound_sorter.engine.eligibility import infer_parent_eligibility

    feature_values = {
        "sub_bass_ratio_lt_150hz": 0.0,
        "bass_ratio_150_500hz": 0.094777,
        "mid_ratio_500_2000hz": 0.826071,
        "presence_ratio_2000_8000hz": 0.079149,
        "air_ratio_gt_8000hz": 0.000002,
        "pitch_confidence": 0.952024,
        "f0_voiced_ratio": 1.0,
        "formant_like_peak_spacing": 0.0,
        "spectral_flatness_mean": 0.041501,
        "loop_pitched_event_ratio": 1.0,
        "loop_sustained_tonal_frame_ratio": 1.0,
        "loop_non_event_tonal_ratio": 1.0,
        "loop_percussive_event_ratio": 0.0,
        "loop_drumlike_frame_ratio": 0.0,
        "event_rate_hz": 8.227111,
    }
    shared_facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "duration_sec": 12.0,
            "event_count_estimate": 98.0,
            "event_rate_hz": 8.227111,
            "onset_span_ratio": 0.996094,
            "shape_vote": {"primary_shape": "pitched_phrase", "confidence": 1.0},
            "measured_roles": {
                "pitched_music_loop": 1.0,
                "pitched_music_phrase": 1.0,
                "vocal_music_phrase": 0.0,
                "voiced_one_shot": 0.0,
            },
            "feature_values_by_name": feature_values,
        },
        feature_values_by_name=feature_values,
    )

    claim = infer_parent_eligibility(shared_facts)

    assert claim.role_name == "pitched_reed_or_instrument_loop"
    assert claim.broad_folder_path == "Instruments/Woodwinds/Saxophone/Loops"


from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path


def test_candidate_backed_brass_claim_beats_generic_instrument_loop_claim() -> None:
    """A generic broad parent bucket must not swallow close concrete evidence."""
    raw_claim = claim_from_folder_path(
        folder_path="FX/Designed Noise FX/Siren/Long FX",
        source="strong_consensus",
        reason="synthetic raw",
        shared=[],
        raw_candidate_score=10.0,
        brain_rank=7,
        physics_rank=3,
        shared_winner="FX/Designed Noise FX/Siren/Long FX",
        can_override=False,
        strength=0.40,
        is_real_candidate=True,
    )
    brass_claim = claim_from_folder_path(
        folder_path="Instruments/Brass and Woodwinds/Loops",
        source="profile_candidate_brass_woodwind_claim",
        reason="candidate backed brass/woodwind profile",
        shared=[],
        raw_candidate_score=3.0,
        brain_rank=1,
        physics_rank=None,
        shared_winner="Instruments/Woodwinds/Saxophone/One Shots",
        can_override=True,
        strength=0.96,
        is_real_candidate=True,
    )
    broad_claim = claim_from_folder_path(
        folder_path="Instruments/Instrument Loops/Loops",
        source="parent_eligibility_broad_bucket",
        reason="generic parent bucket",
        shared=[],
        raw_candidate_score=10.0,
        brain_rank=7,
        physics_rank=3,
        shared_winner="FX/Designed Noise FX/Siren/Long FX",
        can_override=True,
        strength=1.0,
        is_real_candidate=True,
    )

    winner = FamilyClaimArbiter().pick_winner(
        raw_claim=raw_claim,
        claims=[broad_claim, brass_claim],
    )

    assert winner.sub_family == "Brass Woodwinds"


def test_strong_parent_drum_loop_claim_does_not_erase_review_without_decisive_rank_win() -> None:
    """Review is safest when a broad role claim does not beat a concrete raw candidate."""
    raw_claim = claim_from_folder_path(
        folder_path="FX/Impacts and Hits/Generic Impact/Long FX",
        source="strong_consensus",
        reason="synthetic raw",
        shared=[],
        raw_candidate_score=14.0,
        brain_rank=13,
        physics_rank=1,
        shared_winner="FX/Impacts and Hits/Generic Impact/Long FX",
        can_override=False,
        strength=0.12,
        is_real_candidate=True,
    )
    review = claim_from_folder_path(
        folder_path="_TO_REVIEW/Shape Conflict",
        source="measured_shape_conflict_review",
        reason="shape conflict review",
        shared=[],
        raw_candidate_score=14.0,
        brain_rank=13,
        physics_rank=1,
        shared_winner="FX/Impacts and Hits/Generic Impact/Long FX",
        can_override=True,
        strength=1.0,
        is_real_candidate=False,
    )
    drum_loop = claim_from_folder_path(
        folder_path="Drums/Drum Loops/Loops",
        source="parent_eligibility_broad_bucket",
        reason="measured drum loop parent evidence",
        shared=[],
        raw_candidate_score=14.0,
        brain_rank=13,
        physics_rank=1,
        shared_winner="FX/Impacts and Hits/Generic Impact/Long FX",
        can_override=True,
        strength=0.78,
        is_real_candidate=True,
    )

    winner = FamilyClaimArbiter().pick_winner(
        raw_claim=raw_claim,
        claims=[review, drum_loop],
    )

    assert winner.sub_family == "Measured Role Conflict"
