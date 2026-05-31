"""Regression locks for vocal shape rescue policy.

These tests prevent a measured vocal one-shot/phrase from being rescued into
arbitrary FX leaves such as motors or machines merely because those leaves are
shape-compatible at the top-family level.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import CategoryGuess, SharedAudioFacts, VoterResult
from aaron_sound_sorter.engine.consensus import ConsensusRunner
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter


def guess(path: str, rank: int, *, signature: dict[str, float] | None = None) -> CategoryGuess:
    """Build a synthetic category guess with optional role signature."""
    return CategoryGuess(
        label=path,
        folder_path=path,
        top_family=path.split("/", 1)[0],
        score=float(rank),
        confidence=max(0.0, 1.0 - rank * 0.02),
        rank=rank,
        reason="synthetic candidate",
        evidence={"candidate_role_signature": signature or {}},
    )


def finalize(brain: VoterResult, physics: VoterResult, facts: SharedAudioFacts):
    """Run raw consensus plus final claim arbitration."""
    raw_claim, consensus_claims = ConsensusRunner().choose(brain, physics, facts)
    return FamilyClaimArbiter().adjudicate(
        raw_claim=raw_claim,
        consensus_claims=consensus_claims,
        eligibility_claims=[],
        facts=facts,
    )


def vocal_one_shot_facts() -> SharedAudioFacts:
    """Build a direct/body vocal one-shot fact set."""
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "shape_vote": {"primary_shape": "vocal_phrase", "confidence": 0.98},
            "parent_eligibility_v2": {
                "role_name": "vocal_phrase",
                "confidence": 0.78,
                "broad_folder_path": "FX/Human and Voice FX",
            },
            "measured_roles": {
                "percussive_one_shot": 1.0,
                "voiced_one_shot": 0.41,
                "evidence": {"formant_light_voice_identity": 0.91},
            },
            "direct_body_view": {
                "available": True,
                "measured_roles": {
                    "voiced_one_shot": 1.0,
                    "percussive_one_shot": 0.24,
                    "evidence": {"formant_light_voice_identity": 0.91},
                },
            },
        },
    )


def test_vocal_phrase_shape_does_not_rescue_into_motor_leaf() -> None:
    """A decisive vocal shape must not choose arbitrary FX/Motor as compatible."""
    brain = VoterResult(
        voter_name="brain",
        guesses=[
            guess("Drums/Percussion/Bells and Metallic Percussion/One Shots", 1),
            guess("FX/Everyday Foley/Machines/Motor/One Shots", 4, signature={"voiced_one_shot": 0.65}),
        ],
    )
    physics = VoterResult(
        voter_name="physics",
        guesses=[
            guess("Drums/Percussion/Generic Percussion/One Shots", 1),
            guess("FX/Everyday Foley/Machines/Motor/One Shots", 5, signature={"voiced_one_shot": 0.65}),
        ],
    )

    decision = finalize(brain, physics, vocal_one_shot_facts())

    assert decision.final_top == "FX"
    assert decision.folder_path == "FX/Hybrid Designed FX"
    assert "Motor" not in decision.folder_path


def test_vocal_brain_profile_reads_direct_body_voice() -> None:
    """Direct/body voice evidence should select the vocal ensemble weight profile."""
    from aaron_sound_sorter.voters.brain_recall import combine_full_and_balanced_brain_votes

    full = VoterResult(
        voter_name="brain_full",
        guesses=[guess("Drums/Percussion/Generic Percussion/One Shots", 1)],
    )
    core = VoterResult(
        voter_name="brain_core_baby",
        guesses=[guess("FX/Human and Voice FX/Spoken Voice/One Shots", 1)],
    )
    spread = VoterResult(
        voter_name="brain_spread_baby",
        guesses=[guess("FX/Human and Voice FX/Spoken Voice/One Shots", 1)],
    )

    result = combine_full_and_balanced_brain_votes(
        full_result=full,
        baby_results={"core_baby": core, "spread_baby": spread},
        facts=vocal_one_shot_facts(),
    )

    assert result.guesses[0].label == "FX/Human and Voice FX/Spoken Voice/One Shots"
    assert result.guesses[0].evidence["brain_ensemble_weight_profile"] == "vocal_core_spread_priority"
