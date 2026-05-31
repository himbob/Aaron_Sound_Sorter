# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Candidate lookup helpers shared by claim producers."""

from __future__ import annotations

from aaron_sound_sorter.engine.decision_helpers import (
    _candidate_combined_score,
    _norm_path,
    _path_has_any,
)
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class ClaimCandidateLookupMixin:
    """Find shared-candidate support for claim builders."""

    def _best_candidate(
        self,
        raw: ConsensusClaim,
        *,
        include_top: set[str],
        include_fragments: tuple[str, ...],
        exclude_fragments: tuple[str, ...] = (),
    ) -> tuple[float, str] | None:
        """Return the best matching candidate score/path pair."""
        best: tuple[float, str] | None = None
        for candidate in raw.shared_candidates or []:
            folder_path = str(candidate.get("folder_path") or candidate.get("label") or "")
            low_path = _norm_path(folder_path)
            top_family = str(candidate.get("top_family") or folder_path.split("/", 1)[0])
            if top_family not in include_top:
                continue
            if include_fragments and not _path_has_any(low_path, include_fragments):
                continue
            if exclude_fragments and _path_has_any(low_path, exclude_fragments):
                continue
            score = self._safe_candidate_score(candidate)
            if best is None or score < best[0]:
                best = (score, folder_path)
        return best

    @staticmethod
    def _candidate_score_for_path(raw: ConsensusClaim, folder_path: str) -> float | None:
        """Return the best shared-candidate score for a target folder path."""
        wanted = _norm_path(folder_path)
        best_score: float | None = None
        for candidate in raw.shared_candidates or []:
            candidate_path = _norm_path(str(candidate.get("folder_path") or candidate.get("label") or ""))
            if not (candidate_path == wanted or candidate_path.startswith(wanted + "/")):
                continue
            score = _candidate_combined_score(candidate)
            if best_score is None or score < best_score:
                best_score = score
        return best_score

    @staticmethod
    def _candidate_score_for_broad_target(raw: ConsensusClaim, folder_path: str) -> float | None:
        """Return candidate support for broad resolver buckets."""
        normalized = _norm_path(folder_path)
        if "bass loops" in normalized:
            fragments = ("bass", "808", "sub bass", "synth bass", "electric bass", "upright bass")
            top_family = "Instruments"
        elif "brass and woodwinds" in normalized or "woodwinds" in normalized or "saxophone" in normalized:
            fragments = (
                "brass",
                "woodwind",
                "woodwinds",
                "sax",
                "saxophone",
                "flute",
                "clarinet",
                "horn",
                "trumpet",
                "trombone",
                "reed",
            )
            top_family = "Instruments"
        elif "human and voice" in normalized:
            fragments = ("human and voice", "voice", "vocal", "vox", "spoken", "choir", "breath", "crowd")
            top_family = "FX"
        else:
            return None
        best_score: float | None = None
        for candidate in raw.shared_candidates or []:
            candidate_path = _norm_path(str(candidate.get("folder_path") or candidate.get("label") or ""))
            candidate_top = str(candidate.get("top_family") or candidate_path.split("/", 1)[0]).lower()
            if candidate_top != top_family.lower() or not _path_has_any(candidate_path, fragments):
                continue
            score = _candidate_combined_score(candidate)
            if best_score is None or score < best_score:
                best_score = score
        return best_score

    @staticmethod
    def _path_has_real_candidate(raw: ConsensusClaim, folder_path: str) -> bool:
        """Return True when a target folder exists in the shared candidate list."""
        return ClaimCandidateLookupMixin._candidate_score_for_path(raw, folder_path) is not None

    def _has_close_candidate(
        self,
        raw: ConsensusClaim,
        *,
        include_top: set[str],
        include_fragments: tuple[str, ...],
        max_gap: float,
        against_top: str,
    ) -> bool:
        """Return True when a filtered candidate is close to the comparison family."""
        best_target = self._best_candidate_score(raw, include_top=include_top, include_fragments=include_fragments)
        if best_target is None:
            return False
        best_against = self._best_candidate_score(raw, include_top={against_top}, include_fragments=())
        if best_against is None:
            try:
                best_against = float(raw.combined_rank_score or 9999.0)
            except Exception:
                best_against = 9999.0
        return best_target <= best_against + max_gap

    def _best_candidate_score(
        self,
        raw: ConsensusClaim,
        *,
        include_top: set[str],
        include_fragments: tuple[str, ...],
    ) -> float | None:
        """Return the best score for candidates matching a family/path filter."""
        best: float | None = None
        for candidate in raw.shared_candidates or []:
            folder_path = _norm_path(str(candidate.get("folder_path") or candidate.get("label") or ""))
            top_family = str(candidate.get("top_family") or folder_path.split("/", 1)[0].title())
            if top_family not in include_top:
                continue
            if include_fragments and not _path_has_any(folder_path, include_fragments):
                continue
            score = self._safe_candidate_score(candidate)
            if best is None or score < best:
                best = score
        return best

    @staticmethod
    def _safe_candidate_score(candidate: dict) -> float:
        """Return a numeric combined candidate score."""
        try:
            return float(candidate.get("combined_rank_score", 9999.0) or 9999.0)
        except Exception:
            return 9999.0
