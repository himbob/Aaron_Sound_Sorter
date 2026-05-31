"""Compatibility exports for the clean two-voter design.

New code should import from ``aaron_sound_sorter.voters`` and
``aaron_sound_sorter.engine.consensus`` directly. This module only keeps the old
name available during the transition.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.facts import build_shared_audio_facts
from aaron_sound_sorter.domain.models import ConsensusDecision, SharedAudioFacts, VoterResult
from aaron_sound_sorter.domain.policies import BrainVoterPolicy, ConsensusPolicy, PhysicsVoterPolicy
from aaron_sound_sorter.engine.consensus import ConsensusRunner
from aaron_sound_sorter.voters.brain_voter import BrainVoter
from aaron_sound_sorter.voters.physics_voter import PhysicsVoter

REVIEW_BROKEN_OR_TINY = ConsensusPolicy().broken_or_tiny_label
REVIEW_NO_CONSENSUS = ConsensusPolicy().no_consensus_label
REVIEW_WEAK_CONSENSUS = ConsensusPolicy().weak_consensus_label


TWO_VOTER_MANIFEST_FIELDS = [
    "brain_vote_1",
    "brain_vote_2",
    "brain_vote_3",
    "brain_vote_4",
    "brain_vote_5",
    "physics_vote_1",
    "physics_vote_2",
    "physics_vote_3",
    "physics_vote_4",
    "physics_vote_5",
    "shared_winner",
    "shared_winner_brain_rank",
    "shared_winner_physics_rank",
    "shared_winner_combined_rank_score",
    "consensus_status",
    "final_decision_reason",
    "shared_facts_json",
    "vote_trace_json",
]


class TwoVoterTrace:
    """Small transition trace compatible with old preview reporting."""

    def __init__(self, shared_facts, brain_result, physics_result, decision):
        self.shared_facts = shared_facts
        self.brain_result = brain_result
        self.physics_result = physics_result
        self.decision = decision


def run_two_voter_consensus(brain, fingerprint, duration_sec, read_status="ok", top_n=20, consensus_policy=None):
    """Compatibility helper around the clean voter/consensus objects."""
    from pathlib import Path

    import numpy as np

    from aaron_sound_sorter.domain.facts import build_shared_audio_facts
    from aaron_sound_sorter.domain.models import AudioPhysics

    physics = AudioPhysics(
        Path("preview_audio"),
        np.asarray(fingerprint, dtype=np.float32),
        float(duration_sec or 0.0),
        str(read_status or "ok"),
    )
    facts = build_shared_audio_facts(physics)
    brain_result = BrainVoter(BrainVoterPolicy(top_n=top_n)).vote(physics, facts, brain)
    physics_result = PhysicsVoter(PhysicsVoterPolicy(top_n=top_n)).vote(physics, facts, brain)
    decision = ConsensusRunner(policy=consensus_policy).choose(brain_result, physics_result, facts)
    return TwoVoterTrace(facts, brain_result, physics_result, decision)


def two_voter_manifest_row(trace):
    """Return compact manifest columns for a TwoVoterTrace."""
    import json
    from dataclasses import asdict, is_dataclass

    def guess_label(result, index):
        return result.guesses[index].label if len(result.guesses) > index else ""

    def serializable(obj):
        if is_dataclass(obj):
            return asdict(obj)
        if hasattr(obj, "__dict__"):
            return dict(obj.__dict__)
        return str(obj)

    return {
        "brain_vote_1": guess_label(trace.brain_result, 0),
        "brain_vote_2": guess_label(trace.brain_result, 1),
        "brain_vote_3": guess_label(trace.brain_result, 2),
        "brain_vote_4": guess_label(trace.brain_result, 3),
        "brain_vote_5": guess_label(trace.brain_result, 4),
        "physics_vote_1": guess_label(trace.physics_result, 0),
        "physics_vote_2": guess_label(trace.physics_result, 1),
        "physics_vote_3": guess_label(trace.physics_result, 2),
        "physics_vote_4": guess_label(trace.physics_result, 3),
        "physics_vote_5": guess_label(trace.physics_result, 4),
        "shared_winner": trace.decision.shared_winner,
        "shared_winner_brain_rank": "" if trace.decision.brain_rank is None else str(trace.decision.brain_rank),
        "shared_winner_physics_rank": "" if trace.decision.physics_rank is None else str(trace.decision.physics_rank),
        "shared_winner_combined_rank_score": ""
        if trace.decision.combined_rank_score is None
        else f"{trace.decision.combined_rank_score:.3f}",
        "consensus_status": trace.decision.consensus_status,
        "final_decision_reason": trace.decision.reason,
        "shared_facts_json": json.dumps(asdict(trace.shared_facts), sort_keys=True),
        "vote_trace_json": json.dumps(
            {
                "decision": serializable(trace.decision),
                "brain_result": serializable(trace.brain_result),
                "physics_result": serializable(trace.physics_result),
            },
            sort_keys=True,
            default=str,
        ),
    }


def two_voter_error_manifest_row(error):
    """Return manifest columns when two-voter tracing fails."""
    row = {field: "" for field in TWO_VOTER_MANIFEST_FIELDS}
    row["consensus_status"] = "two_voter_trace_error"
    row["final_decision_reason"] = str(error)[:240]
    return row


__all__ = [
    "BrainVoter",
    "ConsensusDecision",
    "ConsensusPolicy",
    "ConsensusRunner",
    "PhysicsVoter",
    "PhysicsVoterPolicy",
    "REVIEW_BROKEN_OR_TINY",
    "REVIEW_NO_CONSENSUS",
    "REVIEW_WEAK_CONSENSUS",
    "SharedAudioFacts",
    "VoterResult",
    "TWO_VOTER_MANIFEST_FIELDS",
    "TwoVoterTrace",
    "build_shared_audio_facts",
    "run_two_voter_consensus",
    "two_voter_error_manifest_row",
    "two_voter_manifest_row",
]
