# SOURCE-NAME BLINDNESS INVARIANT:
# Profile candidate producers may inspect voter output and measured audio facts.
# They must never inspect producer filenames, source folder names, ZIP member
# names, path tokens, or sample-pack labels as classification evidence.
"""Profile-candidate claim producer for DecisionCoreV2.

This coordinator lets real BrainVoter candidates enter family arbitration only
when measured role or shape evidence supports them.  Family-specific policy
lives in small mixins so each musical family can be reviewed independently.
"""

from __future__ import annotations

from aaron_sound_sorter.engine.claim_producers.profile_candidate_drums import ProfileDrumClaimMixin
from aaron_sound_sorter.engine.claim_producers.profile_candidate_fx import ProfileFxClaimMixin
from aaron_sound_sorter.engine.claim_producers.profile_candidate_instruments import ProfileInstrumentClaimMixin
from aaron_sound_sorter.engine.claim_producers.profile_candidate_lookup import ProfileCandidateLookupMixin
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.decision_helpers import (
    _norm_path,
    _shape_confidence_from_facts,
    _shape_vote_from_facts,
)
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class ProfileCandidateClaimProducer(
    ProfileDrumClaimMixin,
    ProfileInstrumentClaimMixin,
    ProfileFxClaimMixin,
    ProfileCandidateLookupMixin,
):
    """Produce measured-compatible claims from real BrainVoter candidates."""

    def produce(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return measured-compatible profile claims for this context.

        Older code returned only the first rescue claim.  That preserved hidden
        branch order and let one family, usually Brass/Woodwinds, become a
        vacuum.  The producer now emits every family-specific claim it can
        justify and lets FamilyClaimArbiter compare them in one place.
        """
        return self.candidate_claims(context)

    def best_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        """Compatibility helper for older unit tests."""
        claims = self.candidate_claims(context)
        return claims[0] if claims else None

    def candidate_claims(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return all measured-compatible profile claims, not just the first."""
        raw = context.raw
        brain_result = context.brain_result
        if brain_result is None:
            return []

        role_name = str(context.eligibility.role_name or "")
        shape_name = _shape_vote_from_facts(context.facts)
        shape_confidence = _shape_confidence_from_facts(context.facts)
        raw_path = _norm_path(raw.folder_path)
        raw_score = float(raw.raw_candidate_score or 9999.0)

        possible_claims = (
            self.kick_claim(
                raw=raw,
                brain_result=brain_result,
                role_name=role_name,
                shape_name=shape_name,
                shape_confidence=shape_confidence,
                raw_score=raw_score,
                facts=context.facts,
            ),
            self.snare_claim(
                raw=raw,
                brain_result=brain_result,
                role_name=role_name,
                shape_name=shape_name,
                shape_confidence=shape_confidence,
                raw_score=raw_score,
                facts=context.facts,
            ),
            self.drum_loop_claim(
                raw=raw,
                brain_result=brain_result,
                role_name=role_name,
                shape_name=shape_name,
                shape_confidence=shape_confidence,
                raw_score=raw_score,
            ),
            self.voice_claim(
                raw=raw,
                brain_result=brain_result,
                role_name=role_name,
                shape_name=shape_name,
                shape_confidence=shape_confidence,
                raw_path=raw_path,
                raw_score=raw_score,
                facts=context.facts,
            ),
            self.synth_claim(
                raw=raw,
                brain_result=brain_result,
                role_name=role_name,
                shape_name=shape_name,
                shape_confidence=shape_confidence,
                raw_path=raw_path,
                raw_score=raw_score,
                facts=context.facts,
            ),
            self.measured_synth_lead_claim(
                raw=raw,
                role_name=role_name,
                shape_name=shape_name,
                shape_confidence=shape_confidence,
                raw_path=raw_path,
                raw_score=raw_score,
                facts=context.facts,
            ),
            self.brass_woodwind_claim(
                raw=raw,
                brain_result=brain_result,
                physics_result=context.physics_result,
                role_name=role_name,
                shape_name=shape_name,
                shape_confidence=shape_confidence,
                raw_path=raw_path,
                raw_score=raw_score,
                facts=context.facts,
            ),
            self.mixed_musical_loop_claim(
                raw=raw,
                brain_result=brain_result,
                physics_result=context.physics_result,
                role_name=role_name,
                shape_name=shape_name,
                shape_confidence=shape_confidence,
                raw_path=raw_path,
                raw_score=raw_score,
                facts=context.facts,
            ),
            self.bass_claim(
                raw=raw,
                brain_result=brain_result,
                role_name=role_name,
                shape_name=shape_name,
                shape_confidence=shape_confidence,
                raw_path=raw_path,
                raw_score=raw_score,
                facts=context.facts,
            ),
            self.non_voice_instrument_parent_claim(
                raw=raw,
                brain_result=brain_result,
                raw_path=raw_path,
                facts=context.facts,
            ),
            self.concrete_instrument_sibling_conflict_parent_claim(
                raw=raw,
                role_name=role_name,
                shape_name=shape_name,
                shape_confidence=shape_confidence,
                raw_path=raw_path,
                raw_score=raw_score,
                facts=context.facts,
            ),
            self.transition_fx_claim(
                raw=raw,
                brain_result=brain_result,
                shape_name=shape_name,
                shape_confidence=shape_confidence,
                raw_path=raw_path,
                raw_score=raw_score,
            ),
        )
        return [claim for claim in possible_claims if claim is not None]
