"""v31114 locks for mixed-instrument-loop role fallback.

These tests use only internal voter candidates and measured facts.  They do
not inspect producer filenames or source folders.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import CategoryGuess, SharedAudioFacts, VoterResult
from aaron_sound_sorter.engine.claim_producers.profile_candidates import ProfileCandidateClaimProducer
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path


def guess(path: str, rank: int, *, top_family: str | None = None) -> CategoryGuess:
    """Build one voter candidate."""
    return CategoryGuess(
        label=path,
        folder_path=path,
        top_family=top_family or path.split("/", 1)[0],
        score=float(rank),
        confidence=max(0.0, 1.0 - rank / 30.0),
        rank=rank,
        reason="synthetic internal candidate",
        evidence={},
    )


def shared_row(path: str, score: float, *, brain_rank: int = 8, physics_rank: int = 8) -> dict:
    """Build one shared candidate row."""
    return {
        "folder_path": path,
        "label": path,
        "top_family": path.split("/", 1)[0],
        "combined_rank_score": score,
        "brain_rank": brain_rank,
        "physics_rank": physics_rank,
    }


def raw_claim(path: str, shared: list[dict], *, score: float = 5.0, strength: float = 0.82):
    """Build the raw pre-arbiter claim."""
    return claim_from_folder_path(
        folder_path=path,
        source="strong_consensus",
        reason="synthetic raw winner",
        shared=shared,
        raw_candidate_score=score,
        brain_rank=1,
        physics_rank=3,
        shared_winner=path,
        can_override=False,
        strength=strength,
        is_real_candidate=True,
    )


def eligibility(role: str = "mixed_music_loop", confidence: float = 0.91) -> EligibilityDecision:
    """Build parent eligibility for a measured music loop."""
    return EligibilityDecision(
        role_name=role,
        confidence=confidence,
        allowed_top_families=("Instruments", "_TO_REVIEW"),
        blocked_path_fragments=(),
        broad_folder_path="Instruments/Instrument Loops/Loops",
        reason="synthetic measured loop eligibility",
    )


def facts(shape: str = "mixed_instrument_loop", confidence: float = 0.88) -> SharedAudioFacts:
    """Build measured loop facts with mixed-branch evidence."""
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {
                "primary_shape": shape,
                "confidence": confidence,
                "pitched_event_ratio": 0.86,
                "sustained_tonal_frame_ratio": 0.78,
                "percussive_event_ratio": 0.10,
                "drumlike_frame_ratio": 0.05,
                "onset_count": 12,
            },
            "measured_roles": {
                "mixed_music_loop": 0.88,
                "pitched_music_loop": 0.84,
                "evidence": {
                    "mixed_music_loop": 0.88,
                    "pitched_music_loop": 0.84,
                },
            },
            "physics_layer_branch": "MixedInstrument",
            "instrument_branch_MixedInstrument": 0.74,
        },
    )


def test_mixed_instrument_loop_role_claim_broadens_split_source_identity() -> None:
    """Three instrument branches plus physics broad loop should stay broad."""
    shared = [
        shared_row("Instruments/Woodwinds/Saxophone/Loops", 4.0, brain_rank=1, physics_rank=8),
        shared_row("Instruments/Synths/Synth Pad/Loops", 7.0, brain_rank=3, physics_rank=6),
        shared_row("Instruments/Keys/Electric Piano/Loops", 9.0, brain_rank=5, physics_rank=5),
        shared_row("Instruments/Instrument Loops/Loops", 10.0, brain_rank=10, physics_rank=1),
    ]
    raw = raw_claim("Instruments/Woodwinds/Saxophone/Loops", shared, score=4.0)
    measured = facts()
    producer = ProfileCandidateClaimProducer()
    context = DecisionContext(
        raw=raw,
        eligibility=eligibility(),
        facts=measured,
        brain_result=VoterResult(
            voter_name="brain_full",
            guesses=[
                guess("Instruments/Woodwinds/Saxophone/Loops", 1),
                guess("Instruments/Synths/Synth Pad/Loops", 3),
                guess("Instruments/Keys/Electric Piano/Loops", 5),
            ],
        ),
        physics_result=VoterResult(
            voter_name="physics",
            guesses=[guess("Instruments/Instrument Loops/Loops", 1)],
        ),
    )

    claims = producer.produce(context)
    final = FamilyClaimArbiter().adjudicate(
        raw_claim=raw,
        consensus_claims=[],
        eligibility_claims=claims,
        facts=measured,
    )

    assert any(claim.source == "mixed_instrument_loop_role_claim" for claim in claims)
    assert final.folder_path == "Instruments/Instrument Loops/Loops"
    assert final.consensus_status == "mixed_instrument_loop_role_claim"


def test_mixed_instrument_loop_role_claim_does_not_fire_for_single_source_synth() -> None:
    """A clear single-source synth branch should not be broadened by the new role."""
    synth_path = "Instruments/Synths/Synth Pad/Loops"
    shared = [
        shared_row(synth_path, 3.0, brain_rank=1, physics_rank=2),
        shared_row("Instruments/Instrument Loops/Loops", 8.0, brain_rank=4, physics_rank=1),
    ]
    producer = ProfileCandidateClaimProducer()
    context = DecisionContext(
        raw=raw_claim(synth_path, shared, score=3.0),
        eligibility=eligibility("pitched_music_loop"),
        facts=facts("sustained_pad", 0.91),
        brain_result=VoterResult(voter_name="brain_full", guesses=[guess(synth_path, 1)]),
        physics_result=VoterResult(voter_name="physics", guesses=[guess("Instruments/Instrument Loops/Loops", 1)]),
    )

    claims = producer.produce(context)

    assert all(claim.source != "mixed_instrument_loop_role_claim" for claim in claims)


def test_mixed_instrument_loop_role_claim_preserves_decisive_terminal_identity() -> None:
    """Brain and physics agreement on the same terminal leaf blocks broadening."""
    rhodes_path = "Instruments/Keys/Electric Piano/Loops"
    shared = [
        shared_row(rhodes_path, 2.0, brain_rank=1, physics_rank=1),
        shared_row("Instruments/Synths/Synth Pad/Loops", 13.0, brain_rank=7, physics_rank=7),
        shared_row("Instruments/Woodwinds/Saxophone/Loops", 14.0, brain_rank=8, physics_rank=8),
        shared_row("Instruments/Instrument Loops/Loops", 10.0, brain_rank=9, physics_rank=2),
    ]
    producer = ProfileCandidateClaimProducer()
    context = DecisionContext(
        raw=raw_claim(rhodes_path, shared, score=2.0, strength=0.95),
        eligibility=eligibility(),
        facts=facts(),
        brain_result=VoterResult(
            voter_name="brain_full",
            guesses=[
                guess(rhodes_path, 1),
                guess("Instruments/Synths/Synth Pad/Loops", 7),
                guess("Instruments/Woodwinds/Saxophone/Loops", 8),
            ],
        ),
        physics_result=VoterResult(voter_name="physics", guesses=[guess("Instruments/Instrument Loops/Loops", 1)]),
    )

    claims = producer.produce(context)

    assert all(claim.source != "mixed_instrument_loop_role_claim" for claim in claims)
