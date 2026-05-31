# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Measured-bucket claim producer coordinator.

The old DecisionCoreV2 held broad measured-bucket rescue logic, true-bucket
conflict resolution, and early candidate adjudication in one giant class.  This
coordinator preserves the legacy order while delegating the policy lattice to
smaller producer classes.
"""

from __future__ import annotations

from aaron_sound_sorter.engine.claim_producers.measured_smoke_stability import (
    MeasuredSmokeStabilityClaimProducer,
)
from aaron_sound_sorter.engine.claim_producers.measured_true_bucket import (
    MeasuredTrueBucketClaimProducer,
)
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class MeasuredBucketClaimProducer:
    """Coordinate measured true-bucket and broad stability claims."""

    def __init__(
        self,
        true_bucket: MeasuredTrueBucketClaimProducer | None = None,
        smoke_stability: MeasuredSmokeStabilityClaimProducer | None = None,
    ) -> None:
        self.true_bucket = true_bucket or MeasuredTrueBucketClaimProducer()
        self.smoke_stability = smoke_stability or MeasuredSmokeStabilityClaimProducer()

    def produce(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return the first measured-bucket claim in legacy order."""
        for producer in (self.true_bucket, self.smoke_stability):
            claims = producer.produce(context)
            if claims:
                return claims
        return []
