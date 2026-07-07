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

from aaron_sound_sorter.engine.claim_producers.final_drum_loop import FinalDrumLoopClaimProducer
from aaron_sound_sorter.engine.claim_producers.measured_drum_structures import (
    MeasuredDrumStructureClaimProducer,
)
from aaron_sound_sorter.engine.claim_producers.measured_instrument_branches import (
    MeasuredInstrumentBranchClaimProducer,
)
from aaron_sound_sorter.engine.claim_producers.measured_music_structures import (
    MeasuredMusicStructureClaimProducer,
)
from aaron_sound_sorter.engine.claim_producers.measured_smoke_stability import (
    MeasuredSmokeStabilityClaimProducer,
)
from aaron_sound_sorter.engine.claim_producers.measured_transition_fx import (
    MeasuredTransitionFxClaimProducer,
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
        final_drum_loop: FinalDrumLoopClaimProducer | None = None,
        transition_fx: MeasuredTransitionFxClaimProducer | None = None,
        instrument_branches: MeasuredInstrumentBranchClaimProducer | None = None,
        drum_structures: MeasuredDrumStructureClaimProducer | None = None,
        music_structures: MeasuredMusicStructureClaimProducer | None = None,
    ) -> None:
        self.true_bucket = true_bucket or MeasuredTrueBucketClaimProducer()
        self.smoke_stability = smoke_stability or MeasuredSmokeStabilityClaimProducer()
        self.final_drum_loop = final_drum_loop or FinalDrumLoopClaimProducer()
        self.transition_fx = transition_fx or MeasuredTransitionFxClaimProducer()
        self.instrument_branches = instrument_branches or MeasuredInstrumentBranchClaimProducer()
        self.drum_structures = drum_structures or MeasuredDrumStructureClaimProducer()
        self.music_structures = music_structures or MeasuredMusicStructureClaimProducer()

    def produce(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return measured-bucket claims in architecture-safe priority order.

        True-bucket corrections keep their legacy priority, but measured
        instrument-branch claims may ride alongside them.  That moves branch
        refinement out of late arbiter rescue while preserving broad true-bucket
        claims that existing tests depend on.
        """
        true_claims = self.true_bucket.produce(context)
        if true_claims:
            return (
                true_claims
                + self.drum_structures.produce(context)
                + self.transition_fx.produce(context)
                + self.instrument_branches.produce(context)
                + self.music_structures.produce(context)
            )

        # Parent-protected percussion is a measured role/structure decision.
        # Let it emit before transition/FX shortcuts so a very short noisy
        # struck hit shaped as ``texture_bed`` does not become a role-conflict
        # review merely because a weak FX leaf was also available.
        drum_claims = self.drum_structures.produce(context)
        if any(
            claim.source
            in {
                "final_measured_protected_percussive_parent_claim",
                "final_decisive_struck_percussion_parent_invariant",
            }
            for claim in drum_claims
        ):
            music_claims = self.music_structures.produce(context)
            tonal_stab_claims = [
                claim
                for claim in music_claims
                if claim.source
                in {
                    "final_measured_voice_before_tonal_stab_invariant",
                    "final_measured_tonal_chord_stab_invariant",
                }
            ]
            if tonal_stab_claims:
                return tonal_stab_claims + drum_claims
            return drum_claims

        for producer in (
            self.transition_fx,
            self.final_drum_loop,
            self.instrument_branches,
            self.music_structures,
            self.smoke_stability,
        ):
            claims = producer.produce(context)
            if claims:
                return claims
        if drum_claims:
            return drum_claims
        return []
