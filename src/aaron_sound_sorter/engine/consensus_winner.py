"""Winner preselection for shared Brain/Physics consensus rows."""

from __future__ import annotations

from typing import Any

from aaron_sound_sorter.domain.models import SharedAudioFacts


class ConsensusWinnerPreselector:
    """Choose the raw shared winner before consensus sanity claims are built."""

    def choose(self, shared: list[dict[str, Any]], facts: SharedAudioFacts) -> dict[str, Any]:
        """Return the raw shared row without role/shape preselection.

        The shared-candidate table is already sorted by BrainVoter/PhysicsVoter
        rank agreement.  Earlier builds used measured roles here to replace the
        raw winner before the arbiter saw it.  That recreated the old rule-tree
        failure where ``percussive_one_shot`` could turn bass, boom, or sub-hit
        evidence into Drums.

        Role/shape facts now stay diagnostic until FamilyClaimArbiter compares
        explicit claims.
        """
        del facts
        return shared[0] if shared else {}
