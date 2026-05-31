"""Claim producers used by the decision core.

A claim producer inspects measured evidence and voter outputs, then returns zero
or more :class:`ConsensusClaim` objects.  Producers do not finalize placement.
"""

from __future__ import annotations

from aaron_sound_sorter.engine.claim_producers.baby_recall import BabyRecallClaimProducer
from aaron_sound_sorter.engine.claim_producers.broad_bucket_claims import BroadBucketClaimProducer
from aaron_sound_sorter.engine.claim_producers.candidate_conflicts import CandidateConflictClaimProducer
from aaron_sound_sorter.engine.claim_producers.final_eligibility import FinalEligibilityClaimProducer
from aaron_sound_sorter.engine.claim_producers.instrument_loop_safety import InstrumentLoopSafetyClaimProducer
from aaron_sound_sorter.engine.claim_producers.measured_buckets import MeasuredBucketClaimProducer
from aaron_sound_sorter.engine.claim_producers.measured_early_adjudication import MeasuredEarlyCandidateAdjudicator
from aaron_sound_sorter.engine.claim_producers.measured_smoke_stability import MeasuredSmokeStabilityClaimProducer
from aaron_sound_sorter.engine.claim_producers.measured_true_bucket import MeasuredTrueBucketClaimProducer
from aaron_sound_sorter.engine.claim_producers.profile_candidates import ProfileCandidateClaimProducer
from aaron_sound_sorter.engine.claim_producers.protocols import ClaimProducer
from aaron_sound_sorter.engine.claim_producers.voice_percussive_buckets import VoicePercussiveBucketClaimProducer

__all__ = [
    "BabyRecallClaimProducer",
    "BroadBucketClaimProducer",
    "CandidateConflictClaimProducer",
    "ClaimProducer",
    "FinalEligibilityClaimProducer",
    "InstrumentLoopSafetyClaimProducer",
    "MeasuredBucketClaimProducer",
    "MeasuredEarlyCandidateAdjudicator",
    "MeasuredSmokeStabilityClaimProducer",
    "MeasuredTrueBucketClaimProducer",
    "ProfileCandidateClaimProducer",
    "VoicePercussiveBucketClaimProducer",
]
