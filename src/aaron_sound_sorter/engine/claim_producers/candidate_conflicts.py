# SOURCE-NAME BLINDNESS INVARIANT:
# This claim producer may inspect only voter candidates, final folder paths, and
# measured eligibility.  It must never use source filenames or source folders.
"""Candidate-conflict claim producer coordinator for DecisionCoreV2."""

from __future__ import annotations

from aaron_sound_sorter.engine.claim_producers.candidate_raw_conflicts import RawCandidateConflictMixin
from aaron_sound_sorter.engine.claim_producers.candidate_role_conflicts import RoleCandidateConflictMixin
from aaron_sound_sorter.engine.claim_producers.candidate_stability import CandidateStabilityMixin
from aaron_sound_sorter.engine.claim_support import ClaimSupportMixin
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class CandidateConflictClaimProducer(
    RawCandidateConflictMixin,
    RoleCandidateConflictMixin,
    CandidateStabilityMixin,
    ClaimSupportMixin,
):
    """Produce review claims when ranked candidates contradict a broad role."""

    def produce(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return ordered raw-consensus and role-candidate conflict claims."""
        claims: list[ConsensusClaim] = []
        for claim in (
            self.raw_consensus_conflict_claim(context.raw, context.facts),
            self.role_candidate_conflict_claim(context.raw, context.eligibility),
        ):
            if claim is not None:
                claims.append(claim)
        return claims
