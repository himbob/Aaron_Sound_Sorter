"""Consensus-level regressions for concrete FX candidates.

These tests encode the FX_Aaron2 pattern where a decisive measured top-family
role can synthesize a generic Instrument Loop even though both product voters
only agree on concrete FX categories.  The production sorter must remain
source-name blind; this file uses synthetic candidate rows, not filenames.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import CategoryGuess, SharedAudioFacts, VoterResult
from aaron_sound_sorter.engine.consensus import ConsensusRunner
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter


def guess(path: str, rank: int) -> CategoryGuess:
    """Build one synthetic voter guess."""
    return CategoryGuess(
        label=path,
        folder_path=path,
        top_family=path.split("/", 1)[0],
        score=float(rank),
        confidence=max(0.0, 1.0 - rank * 0.02),
        rank=rank,
        reason="synthetic candidate",
        evidence={"candidate_role_signature": {"pitched_music_loop": 0.15}},
    )


def facts_with_bad_instrument_gate() -> SharedAudioFacts:
    """Build facts where a role gate overcalls Instruments on a concrete FX loop."""
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "dynamic_role_gate": {
                "selected_top_families": ["Instruments"],
                "top_scores": [{"score": 1.0}, {"score": 20.0}],
                "reason": "decisive_top_family_gap",
            },
            "measured_roles": {
                "pitched_music_loop": 0.99,
                "detected_parent_role": "pitched_music_loop",
            },
            "shape_vote": {"primary_shape": "vocal_phrase", "confidence": 1.0},
        },
    )


def finalize_consensus(runner: ConsensusRunner, brain: VoterResult, physics: VoterResult, facts: SharedAudioFacts):
    """Run consensus evidence through the final arbiter."""
    raw_claim, consensus_claims = runner.choose(brain, physics, facts)
    return FamilyClaimArbiter().adjudicate(
        raw_claim=raw_claim,
        consensus_claims=consensus_claims,
        eligibility_claims=[],
        facts=facts,
    )


def test_concrete_fx_consensus_is_not_overwritten_by_generic_instrument_loop() -> None:
    """Concrete FX agreement should beat a synthetic broad Instrument fallback."""
    brain = VoterResult(
        voter_name="brain",
        guesses=[
            guess("FX/Designed Noise FX/Siren/Long FX", 1),
            guess("FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/Long FX", 4),
        ],
    )
    physics = VoterResult(
        voter_name="physics",
        guesses=[
            guess("FX/Designed Noise FX/Siren/Long FX", 2),
            guess("FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/Long FX", 5),
        ],
    )

    decision = finalize_consensus(ConsensusRunner(), brain, physics, facts_with_bad_instrument_gate())

    assert decision.final_top == "FX"
    assert decision.folder_path == "FX/Designed Noise FX/Siren/Long FX"
    assert "concrete FX" in decision.reason


def test_stable_pitched_music_phrase_without_instrument_candidate_goes_to_review() -> None:
    """Role-only Instrument evidence must not synthesize placement over concrete FX voters."""
    brain = VoterResult(
        voter_name="brain",
        guesses=[guess("FX/Designed Noise FX/Alarm/Long FX", 1)],
    )
    physics = VoterResult(
        voter_name="physics",
        guesses=[guess("FX/Designed Noise FX/Alarm/Long FX", 2)],
    )
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "dynamic_role_gate": {
                "selected_top_families": ["Instruments"],
                "top_scores": [{"score": 1.0}, {"score": 20.0}],
                "reason": "decisive_top_family_gap",
            },
            "measured_roles": {
                "pitched_music_loop": 1.0,
                "pitched_music_phrase": 0.87,
                "detected_parent_role": "pitched_music_loop",
            },
            "shape_vote": {"primary_shape": "pitched_phrase", "confidence": 0.86},
        },
    )

    decision = finalize_consensus(ConsensusRunner(), brain, physics, facts)

    assert decision.final_top == "_TO_REVIEW"
    assert decision.folder_path == "_TO_REVIEW/Shape Conflict"
    assert "shape" in decision.reason.lower() or "blocked" in decision.reason.lower()


def test_concrete_fx_gate_stands_down_for_non_fx_music_loop_shape() -> None:
    """A no-FX-compatible music-loop shape must block concrete FX rescue."""
    brain = VoterResult(
        voter_name="brain",
        guesses=[
            guess("FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX", 1),
            guess("FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/Long FX", 4),
        ],
    )
    physics = VoterResult(
        voter_name="physics",
        guesses=[
            guess("FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX", 2),
            guess("FX/Structural and Transitional FX/Risers and Builds/Short Riser/Long FX", 5),
        ],
    )
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "dynamic_role_gate": {
                "selected_top_families": ["Instruments"],
                "top_scores": [{"score": 1.0}, {"score": 24.0}],
                "reason": "decisive_top_family_gap",
            },
            "measured_roles": {
                "pitched_music_loop": 0.72,
                "detected_parent_role": "pitched_music_loop",
            },
            "shape_vote": {"primary_shape": "mixed_instrument_loop", "confidence": 0.78},
            "physics_layer_decision": {
                "fx_role_allows_fx": False,
                "fx_strength": 0.38,
                "fx_conflict": 0.70,
            },
        },
    )

    decision = finalize_consensus(ConsensusRunner(), brain, physics, facts)

    assert decision.final_top == "_TO_REVIEW"
    assert decision.consensus_status != "concrete_fx_gate_override"
