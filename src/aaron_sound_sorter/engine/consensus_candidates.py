"""Shared candidate table construction for Brain/Physics consensus."""

from __future__ import annotations

from typing import Any

from aaron_sound_sorter.domain.models import CategoryGuess, VoterResult
from aaron_sound_sorter.engine.consensus_support import first_finite


class SharedCandidateBuilder:
    """Build comparable rows for categories both product voters ranked."""

    def build(self, brain_result: VoterResult, physics_result: VoterResult) -> list[dict[str, Any]]:
        """Return shared category candidates sorted by combined rank strength."""
        brain_by_label = {guess.label: guess for guess in brain_result.guesses}
        physics_by_label = {guess.label: guess for guess in physics_result.guesses}
        rows = [
            self.row(label, brain_by_label[label], physics_by_label[label])
            for label in sorted(set(brain_by_label).intersection(physics_by_label))
        ]
        rows.sort(
            key=lambda row: (
                float(row["combined_rank_score"]),
                -(float(row["brain_confidence"]) + float(row["physics_confidence"])),
                str(row["label"]),
            )
        )
        return rows

    @staticmethod
    def row(label: str, brain_guess: CategoryGuess, physics_guess: CategoryGuess) -> dict[str, Any]:
        """Build one comparable shared-candidate row."""
        brain_evidence = brain_guess.evidence if isinstance(brain_guess.evidence, dict) else {}
        physics_evidence = physics_guess.evidence if isinstance(physics_guess.evidence, dict) else {}
        role_distance = first_finite(
            physics_evidence.get("candidate_role_distance"),
            brain_evidence.get("candidate_role_distance"),
        )
        return {
            "label": label,
            "folder_path": brain_guess.folder_path or physics_guess.folder_path,
            "top_family": brain_guess.top_family
            if brain_guess.top_family != "_TO_REVIEW"
            else physics_guess.top_family,
            "brain_rank": brain_guess.rank,
            "physics_rank": physics_guess.rank,
            "combined_rank_score": float(brain_guess.rank + physics_guess.rank),
            "brain_score": brain_guess.score,
            "physics_score": physics_guess.score,
            "brain_confidence": brain_guess.confidence,
            "physics_confidence": physics_guess.confidence,
            "brain_evidence": brain_evidence,
            "physics_evidence": physics_evidence,
            "candidate_role_signature": (
                physics_evidence.get("candidate_role_signature") or brain_evidence.get("candidate_role_signature") or {}
            ),
            "candidate_role_distance": role_distance,
            "family_compatibility": (
                physics_evidence.get("family_compatibility") or brain_evidence.get("family_compatibility") or {}
            ),
        }
