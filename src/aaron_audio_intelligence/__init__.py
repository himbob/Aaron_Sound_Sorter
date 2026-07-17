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
from aaron_audio_intelligence.sidecar_schema import (
    AAI_SIDECAR_SCHEMA_VERSION,
    AudioIntelligenceSidecar,
    build_sidecar_from_sort_result,
)
from aaron_audio_intelligence.user_memory_brain import USER_MEMORY_BRAIN_NAME

__all__ = [
    "AAI_SIDECAR_SCHEMA_VERSION",
    "AudioIntelligenceSidecar",
    "OwnerBrainResult",
    "PrototypeOwnerBrain",
    "TrainableBrainExample",
    "USER_MEMORY_BRAIN_NAME",
    "build_sidecar_from_sort_result",
]
