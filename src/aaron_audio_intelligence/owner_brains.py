"""Trainable owner-brain primitives.

Owner brains are not folder mappers. They answer whether a role or family is
allowed to own a sound, using positive and negative examples. The initial
implementation is intentionally small and dependency-free so it can run beside
the existing sorter as read-only evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class TrainableBrainExample:
    """One weighted training example for a trait, shape, or owner brain.

    Args:
        example_id: Stable ID for reporting and rollback.
        label: Brain-local label or owner name represented by the example.
        vector: Numeric evidence vector. The vector is produced from measured
            audio features, not source names.
        polarity: ``"positive"`` teaches ownership; ``"negative"`` teaches a
            counterexample that should reduce ownership.
        weight: Human corrections may use larger weights than weak automatic
            examples. Weight must be positive.
        metadata: Optional provenance such as correction run ID or training tree
            label. Metadata is diagnostic only.
    """

    example_id: str
    label: str
    vector: tuple[float, ...]
    polarity: str = "positive"
    weight: float = 1.0
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the example shape and learning weight."""
        if self.polarity not in {"positive", "negative"}:
            raise ValueError(f"polarity must be positive or negative, got {self.polarity!r}")
        if self.weight <= 0:
            raise ValueError("weight must be positive")
        if not self.vector:
            raise ValueError("vector must contain at least one feature")


@dataclass(frozen=True)
class OwnerBrainResult:
    """Read-only ownership score from a trainable owner brain.

    Args:
        owner_name: Brain or owner identifier.
        score: Signed raw ownership score in roughly ``[-1, 1]``.
        confidence: Calibrated confidence in ``[0, 1]``.
        positive_evidence: Human-readable positive support strings.
        negative_evidence: Human-readable counterexample support strings.
        allowed_final_tops: Top folders this owner can authorize.
        blocked_final_tops: Top folders this owner should resist.
        training_match_ids: Matched training/correction example IDs.
        reason: Compact explanation for sidecars and manifests.
    """

    owner_name: str
    score: float
    confidence: float
    positive_evidence: tuple[str, ...] = field(default_factory=tuple)
    negative_evidence: tuple[str, ...] = field(default_factory=tuple)
    allowed_final_tops: tuple[str, ...] = field(default_factory=tuple)
    blocked_final_tops: tuple[str, ...] = field(default_factory=tuple)
    training_match_ids: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation for sidecars."""
        return {
            "owner_name": self.owner_name,
            "score": round(float(self.score), 6),
            "confidence": round(float(self.confidence), 6),
            "positive_evidence": list(self.positive_evidence),
            "negative_evidence": list(self.negative_evidence),
            "allowed_final_tops": list(self.allowed_final_tops),
            "blocked_final_tops": list(self.blocked_final_tops),
            "training_match_ids": list(self.training_match_ids),
            "reason": self.reason,
        }


class OwnerBrain(Protocol):
    """Protocol for read-only owner evidence producers."""

    @property
    def owner_name(self) -> str:
        """Return the stable owner-brain name."""
        ...

    def score_vector(self, vector: Sequence[float]) -> OwnerBrainResult:
        """Score one measured feature vector without changing final routing."""
        ...


@dataclass(frozen=True)
class PrototypeOwnerBrain:
    """Prototype-based trainable owner brain with positive and negative support.

    This is a deliberately simple first brain type. It lets GUI corrections and
    curated training examples become weighted evidence without introducing a
    heavyweight ML dependency. Later implementations can swap in embeddings,
    logistic heads, nearest-neighbor indexes, or calibrated classifiers while
    preserving the same result contract.
    """

    owner_name: str
    examples: tuple[TrainableBrainExample, ...]
    allowed_final_tops: tuple[str, ...] = field(default_factory=tuple)
    blocked_final_tops: tuple[str, ...] = field(default_factory=tuple)
    min_confidence_floor: float = 0.05

    def score_vector(self, vector: Sequence[float]) -> OwnerBrainResult:
        """Return read-only ownership evidence for one measured vector."""
        query = tuple(float(value) for value in vector)
        if not query:
            raise ValueError("vector must contain at least one feature")
        positive_matches = self._ranked_matches(query, "positive")
        negative_matches = self._ranked_matches(query, "negative")
        positive_score = weighted_similarity(positive_matches)
        negative_score = weighted_similarity(negative_matches)
        raw_score = positive_score - negative_score
        confidence = clamp01(abs(raw_score))
        confidence = max(self.min_confidence_floor if positive_matches or negative_matches else 0.0, confidence)
        positive_ids = tuple(example.example_id for _sim, example in positive_matches[:3])
        negative_ids = tuple(example.example_id for _sim, example in negative_matches[:3])
        reason = self._reason(raw_score, positive_ids, negative_ids)
        return OwnerBrainResult(
            owner_name=self.owner_name,
            score=raw_score,
            confidence=confidence,
            positive_evidence=tuple(format_match(sim, example) for sim, example in positive_matches[:3]),
            negative_evidence=tuple(format_match(sim, example) for sim, example in negative_matches[:3]),
            allowed_final_tops=self.allowed_final_tops,
            blocked_final_tops=self.blocked_final_tops,
            training_match_ids=positive_ids + negative_ids,
            reason=reason,
        )

    def _ranked_matches(
        self,
        query: tuple[float, ...],
        polarity: str,
    ) -> list[tuple[float, TrainableBrainExample]]:
        """Return same-polarity examples ranked by weighted cosine similarity."""
        matches: list[tuple[float, TrainableBrainExample]] = []
        for example in self.examples:
            if example.polarity != polarity:
                continue
            sim = cosine_similarity(query, example.vector) * example.weight
            matches.append((sim, example))
        matches.sort(key=lambda row: row[0], reverse=True)
        return matches

    def _reason(
        self,
        raw_score: float,
        positive_ids: tuple[str, ...],
        negative_ids: tuple[str, ...],
    ) -> str:
        """Create a compact diagnostic reason for sidecars."""
        if raw_score > 0:
            return f"{self.owner_name} positive prototypes outweighed negatives"
        if raw_score < 0:
            return f"{self.owner_name} negative prototypes outweighed positives"
        if positive_ids or negative_ids:
            return f"{self.owner_name} positive and negative prototypes tied"
        return f"{self.owner_name} has no prototypes"


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for two same-length numeric vectors."""
    if len(left) != len(right):
        raise ValueError(f"vector length mismatch: {len(left)} != {len(right)}")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def weighted_similarity(matches: Sequence[tuple[float, TrainableBrainExample]]) -> float:
    """Return the strongest weighted similarity from a ranked match list."""
    if not matches:
        return 0.0
    return float(matches[0][0])


def clamp01(value: float) -> float:
    """Clamp a float into ``[0, 1]``."""
    return max(0.0, min(1.0, float(value)))


def format_match(similarity: float, example: TrainableBrainExample) -> str:
    """Format one prototype match for diagnostics."""
    return f"{example.example_id}:{example.label}:{similarity:.3f}"
