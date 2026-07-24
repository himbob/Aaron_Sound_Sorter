"""Shadow-mode neural audio architecture for Aaron Sound Sorter.

This package is deliberately isolated from production folder routing.  It builds
source-name-blind audio embeddings, learned prototypes, uncertainty evidence,
and held-out evaluation reports.  Legacy placement must not import this package
as an authority until the migration gates in the architecture document pass.
"""

from .contracts import (
    EmbeddingRecord,
    EvaluationSplit,
    FusionGateDecision,
    LabeledAudioExample,
    LabelPrediction,
    Prototype,
    PrototypeIndexMetadata,
    TrainerAuditRow,
)
from .lane_selection import gate_fusion_candidate, write_lane_leaderboard
from .prototype_index import PrototypeIndex, PrototypeIndexBuilder
from .trainer_audit import audit_labeled_embeddings, trainer_audit_status_counts

__all__ = [
    "EmbeddingRecord",
    "EvaluationSplit",
    "FusionGateDecision",
    "LabelPrediction",
    "LabeledAudioExample",
    "Prototype",
    "PrototypeIndex",
    "PrototypeIndexBuilder",
    "PrototypeIndexMetadata",
    "TrainerAuditRow",
    "audit_labeled_embeddings",
    "trainer_audit_status_counts",
    "gate_fusion_candidate",
    "write_lane_leaderboard",
]
