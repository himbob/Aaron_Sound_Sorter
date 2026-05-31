"""Base voter contract and common ranking helper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from aaron_sound_sorter.domain.models import AudioPhysics, CategoryGuess, SharedAudioFacts, VoterResult


def finite_score(value: Any, *, lower_score_is_better: bool) -> float:
    """Return a sortable finite score."""
    try:
        number = float(value)
        if number == number and number not in (float("inf"), float("-inf")):
            return number
    except Exception:
        pass
    return float("inf") if lower_score_is_better else float("-inf")


class Voter(ABC):
    """Base contract for category voters."""

    voter_name: str = "voter"

    def __init__(self, top_n: int = 100) -> None:
        self.top_n = max(1, int(top_n or 100))

    @abstractmethod
    def vote(self, physics: AudioPhysics, facts: SharedAudioFacts, brain: dict[str, Any]) -> VoterResult:
        """Return a ranked list of category guesses."""
        raise NotImplementedError

    def review_result(self, label: str, reason: str) -> VoterResult:
        """Return a standard single review result."""
        return VoterResult(
            voter_name=self.voter_name,
            guesses=[
                CategoryGuess(
                    label=label,
                    folder_path=label,
                    top_family="_TO_REVIEW",
                    score=0.0,
                    confidence=1.0,
                    rank=1,
                    reason=reason,
                    evidence={"voter": self.voter_name},
                )
            ],
            diagnostics={"review_reason": reason},
        )

    def ranked_guesses(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        label_to_folder: dict[str, str],
        label_to_top: dict[str, str],
        lower_score_is_better: bool,
        default_reason: str,
    ) -> list[CategoryGuess]:
        """Build normalized CategoryGuess objects from score rows."""
        ordered = sorted(
            [dict(row) for row in rows if str(row.get("label", ""))],
            key=lambda row: (
                finite_score(row.get("score", 0.0), lower_score_is_better=lower_score_is_better)
                if lower_score_is_better
                else -finite_score(row.get("score", 0.0), lower_score_is_better=lower_score_is_better),
                str(row.get("label", "")),
            ),
        )
        guesses: list[CategoryGuess] = []
        for rank, row in enumerate(ordered[: self.top_n], start=1):
            label = str(row["label"])
            confidence = clamp01(row.get("confidence", 0.0))
            evidence = row.get("evidence", {})
            guesses.append(
                CategoryGuess(
                    label=label,
                    folder_path=str(label_to_folder.get(label, label)),
                    top_family=str(label_to_top.get(label, "")) or "_TO_REVIEW",
                    score=float(finite_score(row.get("score", 0.0), lower_score_is_better=lower_score_is_better)),
                    confidence=confidence,
                    rank=rank,
                    reason=str(row.get("reason", default_reason)),
                    evidence=evidence if isinstance(evidence, dict) else {"value": evidence},
                )
            )
        return guesses


def clamp01(value: Any) -> float:
    """Clamp a value to 0..1."""
    try:
        number = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, number))
