"""Reusable audio intelligence primitives for Aaron's product family.

The current sound sorter remains the first product adapter. This package holds
sidecar schemas and trainable evidence-brain interfaces that can later power the
loop analyzer, search browser, training console, and stem-intelligence tools.
"""

from aaron_audio_intelligence.owner_brains import (
    OwnerBrainResult,
    PrototypeOwnerBrain,
    TrainableBrainExample,
)
from aaron_audio_intelligence.physics_memory_brain import PHYSICS_MEMORY_BRAIN_NAME
from aaron_audio_intelligence.shape_memory_brain import (
    SHAPE_MEMORY_BRAIN_NAME,
    SHAPE_STARTER_MEMORY_BRAIN_NAME,
)
from aaron_audio_intelligence.sidecar_schema import (
    AAI_SIDECAR_SCHEMA_VERSION,
    AudioIntelligenceSidecar,
    build_sidecar_from_sort_result,
)
from aaron_audio_intelligence.user_memory_brain import USER_MEMORY_BRAIN_NAME
from aaron_audio_intelligence.voter_memory_brain import VOTER_MEMORY_BRAIN_NAME

__all__ = [
    "AAI_SIDECAR_SCHEMA_VERSION",
    "AudioIntelligenceSidecar",
    "OwnerBrainResult",
    "PHYSICS_MEMORY_BRAIN_NAME",
    "PrototypeOwnerBrain",
    "SHAPE_MEMORY_BRAIN_NAME",
    "SHAPE_STARTER_MEMORY_BRAIN_NAME",
    "TrainableBrainExample",
    "USER_MEMORY_BRAIN_NAME",
    "VOTER_MEMORY_BRAIN_NAME",
    "build_sidecar_from_sort_result",
]
