"""Protocols shared by claim-producing policy classes."""

from __future__ import annotations

from typing import Protocol

from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class ClaimProducer(Protocol):
    """Structural interface for objects that produce eligibility claims."""

    def produce(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return zero or more claims without finalizing placement."""
        ...
