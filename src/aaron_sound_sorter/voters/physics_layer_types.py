from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PhysicsLayerDecision:
    """One sample-level physics decision split into readable layers."""

    top_family: str
    top_confidence: float
    branch: str
    branch_confidence: float
    leaf_strategy: str
    evidence: dict[str, Any]
