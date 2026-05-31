# SOURCE-NAME BLINDNESS INVARIANT:
# This final eligibility producer may inspect measured eligibility and voter
# candidate paths only. It must never inspect source filenames or source paths.
"""Final broad-eligibility fallback claims for DecisionCoreV2."""

from __future__ import annotations

from aaron_sound_sorter.engine.claim_producers.candidate_conflicts import CandidateConflictClaimProducer
from aaron_sound_sorter.engine.claim_producers.instrument_loop_safety import InstrumentLoopSafetyClaimProducer
from aaron_sound_sorter.engine.claim_support import ClaimSupportMixin
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class FinalEligibilityClaimProducer(ClaimSupportMixin):
    """Produce the final measured-parent fallback claims.

    This producer runs after more specific claim producers.  It handles decisive
    parent broadening and measured conflict review without becoming a final
    arbiter.  The :class:`FamilyClaimArbiter` still compares every claim.
    """

    def __init__(
        self,
        instrument_safety: InstrumentLoopSafetyClaimProducer,
        candidate_conflicts: CandidateConflictClaimProducer,
    ) -> None:
        self.instrument_safety = instrument_safety
        self.candidate_conflicts = candidate_conflicts

    def produce(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return final eligibility fallback claims for this context."""
        claims: list[ConsensusClaim] = []
        decisive_claim = self.decisive_parent_claim(context)
        if decisive_claim is not None:
            claims.append(decisive_claim)
        conflict_claim = self.measured_conflict_review_claim(context)
        if conflict_claim is not None:
            claims.append(conflict_claim)
        return claims

    def decisive_parent_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        """Broaden raw placement when decisive parent eligibility disallows it."""
        raw = context.raw
        eligibility = context.eligibility
        if not eligibility.is_decisive:
            return None
        if eligibility.is_path_allowed(raw.folder_path, raw.final_top):
            return None
        if self.instrument_safety.should_skip_decisive_parent_broaden(raw, eligibility, context.facts):
            return None
        return self._broaden_from_raw(
            raw,
            eligibility,
            "raw consensus was incompatible with measured parent eligibility",
        )

    def measured_conflict_review_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        """Review measured conflict roles when raw top support is not stable."""
        raw = context.raw
        eligibility = context.eligibility
        if "conflict" not in str(eligibility.role_name or ""):
            return None
        if not str(eligibility.broad_folder_path).strip("/").startswith("_TO_REVIEW"):
            return None
        if eligibility.is_path_allowed(raw.folder_path, raw.final_top):
            return None
        if self.candidate_conflicts.has_stable_raw_top_support(raw):
            return None
        return self._broaden_from_raw(
            raw,
            eligibility,
            "measured conflict role requires review and raw top family was not decisively supported",
        )
