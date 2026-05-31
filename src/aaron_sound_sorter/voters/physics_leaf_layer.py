# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Leaf strategy helper for layered physics scoring."""

from __future__ import annotations

from aaron_sound_sorter.voters.physics_drum_layer import PhysicsDrumLayer
from aaron_sound_sorter.voters.physics_fx_layer import PhysicsFXRoleLayer
from aaron_sound_sorter.voters.physics_instrument_layer import PhysicsInstrumentLayer
from aaron_sound_sorter.voters.physics_layer_types import PhysicsLayerDecision


class PhysicsLeafLayer:
    """Third physics layer: category panels shape leaf authority."""

    def strategy(self, decision: PhysicsLayerDecision) -> str:
        if decision.branch in set(PhysicsDrumLayer.BRANCHES):
            return "profile_leaf_with_drum_branch_safeguard"
        if decision.branch in set(PhysicsInstrumentLayer.BRANCHES):
            return "profile_leaf_with_branch_safeguard"
        if decision.branch in set(PhysicsFXRoleLayer.BRANCHES):
            return "profile_leaf_with_fx_role_safeguard"
        if decision.top_family in {"Drums", "FX", "Instruments"}:
            return "profile_leaf_with_top_family_safeguard"
        return "profile_leaf_only"
