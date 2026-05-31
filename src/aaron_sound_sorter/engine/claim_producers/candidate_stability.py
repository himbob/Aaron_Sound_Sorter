# SOURCE-NAME BLINDNESS INVARIANT:
# This claim producer may inspect only voter candidates, final folder paths, and
# measured eligibility.  It must never use source filenames or source folders.
"""Candidate stability helpers for raw-top support checks."""

from __future__ import annotations

from aaron_sound_sorter.engine.decision_helpers import _norm_path
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class CandidateStabilityMixin:
    """Evaluate whether the raw top family is stable enough to keep."""

    @staticmethod
    def _safe_raw_score(raw: ConsensusClaim) -> float:
        """Return raw combined score as a numeric value."""
        try:
            return float(raw.combined_rank_score or 9999.0)
        except Exception:
            return 9999.0

    def has_stable_raw_top_support(self, raw: ConsensusClaim) -> bool:
        """Return True when candidates support the raw top family decisively."""
        if raw.final_top == "_TO_REVIEW":
            return False
        raw_score = self._safe_raw_score(raw)
        best_same = self._best_candidate_score(raw, include_top={raw.final_top}, include_fragments=())
        best_other = self._best_other_top_candidate_score(raw)
        if best_same is None:
            best_same = raw_score
        if best_same > raw_score + 2.5:
            return False
        if best_other is None:
            return True
        return best_same <= best_other - 1.5

    @staticmethod
    def _best_other_top_candidate_score(raw: ConsensusClaim) -> float | None:
        """Return best score outside the raw top family."""
        best_other: float | None = None
        for candidate in raw.shared_candidates or []:
            folder_path = _norm_path(str(candidate.get("folder_path") or candidate.get("label") or ""))
            top_family = str(candidate.get("top_family") or folder_path.split("/", 1)[0].title())
            if not top_family or top_family in {raw.final_top, "_TO_REVIEW"}:
                continue
            try:
                score = float(candidate.get("combined_rank_score", 9999.0) or 9999.0)
            except Exception:
                score = 9999.0
            if best_other is None or score < best_other:
                best_other = score
        return best_other
