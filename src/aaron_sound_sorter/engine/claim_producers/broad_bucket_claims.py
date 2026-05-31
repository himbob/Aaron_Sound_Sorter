# SOURCE-NAME BLINDNESS INVARIANT:
# Broad-bucket claim orchestration may inspect measured facts and voter output.
# It must never inspect producer filenames, source folders, ZIP member names, or sample-pack labels.
"""Compatibility broad-bucket claim producer.

Earlier cleanup bundles exposed :class:`BroadBucketClaimProducer` as the public
coordinator for measured true-bucket and broad measured-bucket claims.  v31.94
renamed the internal coordinator to :class:`MeasuredBucketClaimProducer`, but
older regression tests and local installs still import the broad-bucket name.

Keep this thin wrapper so public tests stay stable while the implementation
continues to live in the smaller measured-bucket modules.
"""

from __future__ import annotations

from typing import Any

from aaron_sound_sorter.engine.claim_producers.measured_buckets import MeasuredBucketClaimProducer
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class BroadBucketClaimProducer:
    """Backward-compatible coordinator for broad measured-bucket claims.

    The optional ``tools`` argument is accepted for compatibility with v31.92
    tests and old call sites.  The current implementation no longer needs a
    DecisionCore tool object because the extracted producers build claims
    directly through shared support helpers.
    """

    def __init__(self, tools: Any | None = None) -> None:
        self.tools = tools
        self._producer = MeasuredBucketClaimProducer()

    def produce(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return broad-bucket claims in the current legacy-safe order."""
        return self._producer.produce(context)
