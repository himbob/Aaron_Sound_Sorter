# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Compatibility exports for the layered PhysicsVoter implementation.

The large v31.108 module was split in v31.109 so each layer can be reviewed and
tested independently.  Keep this shim so existing imports keep working.
"""

from __future__ import annotations

from aaron_sound_sorter.voters.layered_physics_scorer import LayeredPhysicsScorer
from aaron_sound_sorter.voters.physics_branch_layer import PhysicsBranchLayer
from aaron_sound_sorter.voters.physics_compound_music_layer import PhysicsCompoundMusicLayer
from aaron_sound_sorter.voters.physics_drum_layer import PhysicsDrumLayer
from aaron_sound_sorter.voters.physics_fx_layer import PhysicsFXRoleLayer
from aaron_sound_sorter.voters.physics_instrument_layer import PhysicsInstrumentLayer
from aaron_sound_sorter.voters.physics_layer_types import PhysicsLayerDecision
from aaron_sound_sorter.voters.physics_leaf_layer import PhysicsLeafLayer
from aaron_sound_sorter.voters.physics_top_family_layer import PhysicsTopFamilyLayer

__all__ = [
    "LayeredPhysicsScorer",
    "PhysicsBranchLayer",
    "PhysicsCompoundMusicLayer",
    "PhysicsDrumLayer",
    "PhysicsFXRoleLayer",
    "PhysicsInstrumentLayer",
    "PhysicsLayerDecision",
    "PhysicsLeafLayer",
    "PhysicsTopFamilyLayer",
]
