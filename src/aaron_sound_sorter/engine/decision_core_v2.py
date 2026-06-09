# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""DecisionCoreV2 coordinator for claim-producing final placement.

The raw consensus still finds a candidate neighborhood.  This layer gathers
measured-eligibility claims from small claim producers and sends all evidence to
:class:`FamilyClaimArbiter`.  It should coordinate policy objects, not contain
large decision lattices.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import ConsensusDecision, SharedAudioFacts, VoterResult
from aaron_sound_sorter.engine.claim_producers.baby_recall import BabyRecallClaimProducer
from aaron_sound_sorter.engine.claim_producers.broad_bucket_claims import BroadBucketClaimProducer
from aaron_sound_sorter.engine.claim_producers.candidate_conflicts import CandidateConflictClaimProducer
from aaron_sound_sorter.engine.claim_producers.final_eligibility import FinalEligibilityClaimProducer
from aaron_sound_sorter.engine.claim_producers.instrument_loop_safety import InstrumentLoopSafetyClaimProducer
from aaron_sound_sorter.engine.claim_producers.measured_final_guards import MeasuredFinalGuardClaimProducer
from aaron_sound_sorter.engine.claim_producers.profile_candidates import ProfileCandidateClaimProducer
from aaron_sound_sorter.engine.claim_producers.protocols import ClaimProducer
from aaron_sound_sorter.engine.claim_producers.voice_percussive_buckets import VoicePercussiveBucketClaimProducer
from aaron_sound_sorter.engine.consensus import ConsensusRunner
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.eligibility import EligibilityDecision, infer_parent_eligibility
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_folder_path


class DecisionCoreV2:
    """Coordinate raw consensus, eligibility claims, and final arbitration."""

    def __init__(
        self,
        raw_consensus: ConsensusRunner | None = None,
        arbiter: FamilyClaimArbiter | None = None,
    ) -> None:
        self.raw_consensus = raw_consensus or ConsensusRunner()
        self.arbiter = arbiter or FamilyClaimArbiter()
        self.baby_recall_claim_producer = BabyRecallClaimProducer()
        self.profile_candidate_claim_producer = ProfileCandidateClaimProducer()
        self.measured_bucket_claim_producer = BroadBucketClaimProducer(self)
        self.instrument_safety_claim_producer = InstrumentLoopSafetyClaimProducer()
        self.candidate_conflict_claim_producer = CandidateConflictClaimProducer()
        self.voice_percussive_claim_producer = VoicePercussiveBucketClaimProducer()
        self.final_eligibility_claim_producer = FinalEligibilityClaimProducer(
            instrument_safety=self.instrument_safety_claim_producer,
            candidate_conflicts=self.candidate_conflict_claim_producer,
        )
        self.measured_final_guard_claim_producer = MeasuredFinalGuardClaimProducer()

    def choose(
        self,
        brain_result: VoterResult,
        physics_result: VoterResult,
        facts: SharedAudioFacts,
    ) -> ConsensusDecision:
        """Return a final decision constrained by broad measured eligibility."""
        raw_claim, consensus_claims = self.raw_consensus.choose(brain_result, physics_result, facts)
        eligibility = infer_parent_eligibility(facts)
        self._write_eligibility_debug_evidence(facts, raw_claim, eligibility)
        eligibility_claims = self.gather_eligibility_claims(
            raw_claim,
            eligibility,
            facts,
            brain_result=brain_result,
            physics_result=physics_result,
            consensus_claims=consensus_claims,
        )
        return self.arbiter.adjudicate(
            raw_claim=raw_claim,
            consensus_claims=consensus_claims,
            eligibility_claims=eligibility_claims,
            facts=facts,
        )

    def apply_eligibility(
        self,
        raw: ConsensusClaim | ConsensusDecision,
        eligibility: EligibilityDecision,
        facts: SharedAudioFacts | None = None,
    ) -> ConsensusDecision:
        """Finalize a supplied raw claim through the claim-arbiter path.

        This method exists for direct unit tests that construct a raw decision
        context.  It does not contain a separate legacy decision path.
        """
        raw_claim = self._coerce_raw_claim(raw)
        eligibility_claims = self.gather_eligibility_claims(raw_claim, eligibility, facts)
        return self.arbiter.adjudicate(
            raw_claim=raw_claim,
            consensus_claims=[],
            eligibility_claims=eligibility_claims,
            facts=facts,
        )

    def gather_eligibility_claims(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        facts: SharedAudioFacts | None = None,
        brain_result: VoterResult | None = None,
        physics_result: VoterResult | None = None,
        consensus_claims: list[ConsensusClaim] | None = None,
    ) -> list[ConsensusClaim]:
        """Gather all measured-eligibility claims without final placement.

        Final-guard producers must see consensus claims too, not only later
        eligibility producers.  Several former post-winner guards operate on
        top-family sanity or shape-conflict claims emitted by the raw consensus
        runner.  Feeding those claims through the guard producer keeps the
        cleanup architecture honest: the arbiter receives all guard outcomes as
        normal claims instead of recreating a late mutation layer.
        """
        context = DecisionContext(
            raw=raw,
            eligibility=eligibility,
            facts=facts,
            brain_result=brain_result,
            physics_result=physics_result,
        )
        claims: list[ConsensusClaim] = []
        for producer in self._ordered_claim_producers():
            claims.extend(producer.produce(context))
        guard_seeds = [*(consensus_claims or []), *claims]
        claims.extend(self.measured_final_guard_claim_producer.produce_for_claims(context, guard_seeds))
        return claims

    def _ordered_claim_producers(self) -> tuple[ClaimProducer, ...]:
        """Return claim producers in the legacy arbitration order."""
        return (
            self.profile_candidate_claim_producer,
            self.baby_recall_claim_producer,
            self.measured_bucket_claim_producer,
            self.instrument_safety_claim_producer,
            self.candidate_conflict_claim_producer,
            self.voice_percussive_claim_producer,
            self.final_eligibility_claim_producer,
        )

    @staticmethod
    def _write_eligibility_debug_evidence(
        facts: SharedAudioFacts,
        raw_claim: ConsensusClaim,
        eligibility: EligibilityDecision,
    ) -> None:
        """Attach raw and eligibility evidence to the manifest/debug payload."""
        if not isinstance(facts.evidence, dict):
            return
        facts.evidence["parent_eligibility_v2"] = eligibility.to_report_dict()
        facts.evidence["raw_consensus_before_eligibility_v2"] = {
            "final_label": raw_claim.final_label,
            "final_top": raw_claim.final_top,
            "folder_path": raw_claim.folder_path,
            "consensus_status": raw_claim.source,
            "reason": raw_claim.reason,
        }

    @staticmethod
    def _coerce_raw_claim(raw: ConsensusClaim | ConsensusDecision) -> ConsensusClaim:
        """Convert old test decision contexts into raw claims."""
        if isinstance(raw, ConsensusClaim):
            return raw
        return claim_from_folder_path(
            folder_path=raw.folder_path,
            source=raw.consensus_status,
            reason=raw.reason,
            shared=raw.shared_candidates,
            raw_candidate_score=raw.combined_rank_score,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner,
            can_override=False,
            strength=0.80,
            is_real_candidate=True,
        )

    @staticmethod
    def _baby_row_supports_brass_woodwind(
        path: str,
        rank: int,
        raw: object,
        role_name: str,
        measured_role: str,
        shape_name: str,
        shape_confidence: float,
    ) -> bool:
        """Compatibility seam for older regression tests.

        New code should use :class:`BabyRecallClaimProducer` directly.
        """
        return BabyRecallClaimProducer.row_supports_brass_woodwind(
            path,
            rank,
            raw,
            role_name,
            measured_role,
            shape_name,
            shape_confidence,
        )


# Measured-evidence helper functions moved to decision_helpers.py.
