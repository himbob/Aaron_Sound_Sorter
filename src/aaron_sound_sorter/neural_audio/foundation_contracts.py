"""Visible evidence contracts for independent foundation-model lanes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FoundationScore:
    """One labeled score from an independent evidence lane.

    Args:
        label: Model or mapped taxonomy label.
        score: Raw model score; not a calibrated probability.
        margin: Separation from a relevant competing score.
        evidence: Lane-specific diagnostics.

    Side Effects:
        None.
    """

    label: str
    score: float
    margin: float = 0.0
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FoundationLaneEvidence:
    """Versioned output from one independent model or measured lane.

    Args:
        lane_id: Stable lane identifier such as ``clap_prompt`` or ``panns``.
        model_id: Pinned model identity.
        status: ``available``, ``unavailable``, ``disabled``, or ``error``.
        scores: Ranked lane-specific scores.
        message: Compact availability or error detail.

    Side Effects:
        None.
    """

    lane_id: str
    model_id: str
    status: str
    scores: tuple[FoundationScore, ...] = ()
    message: str = ""


@dataclass(frozen=True)
class FoundationEvidencePanel:
    """Per-file foundation evidence with lanes kept separate.

    Args:
        audio_sha256: Content hash used to join evidence without source names.
        taxonomy_version: Canonical taxonomy version.
        brain_version: Aaron prototype brain or index version.
        lanes: Evidence lanes keyed by stable lane ID.
        conflicts: Human-readable broad evidence conflicts.
        suggested_category: Read-only suggestion; never automatic ownership.
        review_reason: Reason this example should be reviewed.

    Side Effects:
        None.
    """

    audio_sha256: str
    taxonomy_version: str
    brain_version: str
    lanes: Mapping[str, FoundationLaneEvidence]
    conflicts: tuple[str, ...] = ()
    suggested_category: str = ""
    review_reason: str = ""

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping without fusing lane scores."""
        return asdict(self)
