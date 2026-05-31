"""Follow-up family-claim stability tests from v31.74 red checks.

These are synthetic tests for the architecture seam, not filename rules.  They
lock the three remaining failure classes from Aaron's Mac validation log:
short percussion stolen by Human/Voice, wet sax stolen by FX/Human/Siren, and
sub-heavy bass loops flattened to generic Instrument Loops.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import CategoryGuess, ConsensusDecision, SharedAudioFacts, VoterResult
from aaron_sound_sorter.engine.consensus import ConsensusRunner
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter


def raw_decision(path: str, score: float, candidates: list[dict] | None = None) -> ConsensusDecision:
    """Build a raw synthetic decision."""
    return ConsensusDecision(
        final_label=path,
        final_top=path.split("/", 1)[0],
        folder_path=path,
        consensus_status="synthetic_raw",
        reason="synthetic raw decision",
        combined_rank_score=score,
        shared_candidates=candidates or [],
    )


def candidate(path: str, score: float, *, roles: dict[str, float] | None = None) -> dict:
    """Build one synthetic shared candidate row."""
    role_signature = roles or {}
    return {
        "label": path,
        "folder_path": path,
        "top_family": path.split("/", 1)[0],
        "combined_rank_score": score,
        "brain_rank": int(score),
        "physics_rank": int(score),
        "candidate_role_signature": role_signature,
        "brain_evidence": {"candidate_role_signature": role_signature},
        "physics_evidence": {"candidate_role_signature": role_signature},
    }


def facts_for(
    shape: str, confidence: float, roles: dict[str, float], direct_roles: dict[str, float] | None = None
) -> SharedAudioFacts:
    """Build measured facts with full and direct/body roles."""
    evidence = {
        "shape_vote": {"primary_shape": shape, "confidence": confidence},
        "measured_roles": roles,
        "direct_body_view": {"available": True, "measured_roles": direct_roles or roles},
        "duration_sec": 0.20,
        "event_count_estimate": 1.0,
        "feature_values_by_name": {},
    }
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence=evidence,
        feature_values_by_name={},
    )


def test_short_percussive_hit_is_not_kept_as_synthetic_human_voice() -> None:
    """A short one-event hit may review or go broad Drums, never Human/Voice."""
    final = DecisionCoreV2().apply_eligibility(
        raw_decision("FX/Human and Voice FX", 999.0, []),
        EligibilityDecision(
            role_name="protected_percussive_one_shot",
            confidence=0.74,
            allowed_top_families=("Drums", "_TO_REVIEW"),
            blocked_path_fragments=("FX", "Human", "Voice", "Instruments"),
            broad_folder_path="Drums/Percussion/Generic Percussion/One Shots",
            reason="synthetic protected percussive one-shot",
        ),
        facts_for(
            "hit_with_tail",
            0.75,
            {"voiced_one_shot": 0.95, "percussive_one_shot": 0.27},
        ),
    )

    assert not final.folder_path.startswith("FX/")
    assert "Human" not in final.folder_path and "Voice" not in final.folder_path


def test_stable_pitched_instrument_claim_reviews_siren_false_positive_without_leaf_support() -> None:
    """Wet pitched phrases must not be confidently filed as Siren/Human Voice."""
    final = DecisionCoreV2().apply_eligibility(
        raw_decision(
            "FX/Designed Noise FX/Siren/One Shots",
            7.0,
            [
                candidate("FX/Designed Noise FX/Siren/One Shots", 7.0, roles={"voiced_one_shot": 0.62}),
                candidate("Instruments/Instrument Loops/Loops", 11.0, roles={"pitched_music_phrase": 0.90}),
                candidate("FX/Human and Voice FX/Spoken Voice/Long FX", 33.0, roles={"voiced_one_shot": 0.60}),
            ],
        ),
        EligibilityDecision(
            role_name="pitched_reed_or_instrument_loop",
            confidence=0.78,
            allowed_top_families=("Instruments", "_TO_REVIEW"),
            blocked_path_fragments=("FX", "Human", "Voice", "Drums"),
            broad_folder_path="Instruments/Instrument Loops/Loops",
            reason="synthetic sustained pitched instrument loop",
        ),
        facts_for(
            "pitched_phrase",
            0.80,
            {"pitched_music_phrase": 1.0, "pitched_music_loop": 1.0, "vocal_music_phrase": 0.0},
            {"pitched_music_phrase": 1.0, "pitched_music_loop": 1.0, "vocal_music_phrase": 0.0},
        ),
    )

    assert final.folder_path == "_TO_REVIEW/Measured Role Conflict"
    assert "Human" not in final.folder_path and "Voice" not in final.folder_path
    assert not final.folder_path.startswith("FX/")


def guess(path: str, rank: int, *, roles: dict[str, float] | None = None) -> CategoryGuess:
    """Build a voter guess."""
    role_signature = roles or {}
    return CategoryGuess(
        label=path,
        folder_path=path,
        top_family=path.split("/", 1)[0],
        score=float(rank),
        confidence=0.5,
        rank=rank,
        reason="synthetic",
        evidence={"candidate_role_signature": role_signature},
    )


def test_direct_body_sub_bass_loop_promotes_generic_loop_to_bass_bucket() -> None:
    """A sub-heavy direct/body bass loop should not flatten to Instrument Loops."""
    brain = VoterResult(
        voter_name="brain",
        guesses=[
            guess("Instruments/Bass/Synth Bass/One Shots", 1, roles={"bass_loop": 1.0}),
            guess("Instruments/Instrument Loops/Loops", 2, roles={"pitched_music_loop": 0.95}),
        ],
    )
    physics = VoterResult(
        voter_name="physics",
        guesses=[
            guess("FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/Long FX", 1),
            guess("Instruments/Instrument Loops/Loops", 3, roles={"pitched_music_loop": 0.95}),
        ],
    )
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {"primary_shape": "bass_phrase", "confidence": 0.97},
            "measured_roles": {"pitched_music_loop": 0.76, "bass_loop": 0.63},
            "direct_body_view": {"available": True, "measured_roles": {"bass_loop": 1.0, "pitched_music_loop": 1.0}},
            "feature_values_by_name": {
                "sub_bass_ratio_lt_150hz": 0.94,
                "presence_ratio_2000_8000hz": 0.001,
                "air_ratio_gt_8000hz": 0.0,
                "loop_percussive_event_ratio": 0.0,
                "loop_drumlike_frame_ratio": 0.0,
            },
        },
        feature_values_by_name={
            "sub_bass_ratio_lt_150hz": 0.94,
            "presence_ratio_2000_8000hz": 0.001,
            "air_ratio_gt_8000hz": 0.0,
            "loop_percussive_event_ratio": 0.0,
            "loop_drumlike_frame_ratio": 0.0,
        },
    )

    raw_claim, consensus_claims = ConsensusRunner().choose(brain, physics, facts)
    final = FamilyClaimArbiter().adjudicate(
        raw_claim=raw_claim,
        consensus_claims=consensus_claims,
        eligibility_claims=[],
        facts=facts,
    )

    assert final.folder_path == "Instruments/Bass/Bass Loops"
