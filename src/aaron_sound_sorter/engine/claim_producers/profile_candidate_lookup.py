# SOURCE-NAME BLINDNESS INVARIANT:
# Profile candidate helpers inspect voter output and measured audio facts only.
# They must never inspect source filenames or source folders.
"""Shared lookup helpers for profile-candidate claim producers."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import CategoryGuess, VoterResult
from aaron_sound_sorter.engine.decision_helpers import _norm_path, _path_has_any
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class ProfileCandidateLookupMixin:
    """Find matching brain and shared candidate rows."""

    @staticmethod
    def best_brain_guess(
        brain_result: VoterResult,
        *,
        fragments: tuple[str, ...],
        max_rank: int,
        include_top: set[str] | None = None,
    ) -> CategoryGuess | None:
        """Return the highest-ranked BrainVoter guess matching folder fragments.

        ``include_top`` prevents family-specific profile producers from matching
        words that happen to appear in another top family.  For example, an
        Instruments/Synth claim must not use an FX/Synth Riser candidate as its
        evidence, because that lets a vocal or mixed instrument loop jump into
        transition FX while the code reports the move as a Synth instrument
        claim.
        """
        matches: list[CategoryGuess] = []
        for guess in brain_result.guesses:
            if guess.rank > max_rank:
                continue
            if include_top is not None and str(guess.top_family or "") not in include_top:
                continue
            candidate_path = _norm_path(str(guess.folder_path or guess.label))
            if _path_has_any(candidate_path, fragments):
                matches.append(guess)
        if not matches:
            return None
        matches.sort(key=lambda guess: (guess.rank, float(guess.score), str(guess.folder_path or guess.label)))
        return matches[0]

    @staticmethod
    def best_shared_candidate(
        raw: ConsensusClaim,
        *,
        include_top: set[str] | None = None,
        include_fragments: tuple[str, ...] = (),
    ) -> tuple[float, dict] | None:
        """Return the best shared candidate row matching family/path filters."""
        rows: list[tuple[float, dict]] = []
        for row in raw.shared_candidates:
            path = _norm_path(str(row.get("folder_path") or row.get("label") or ""))
            top = str(row.get("top_family") or path.split("/", 1)[0].title())
            if include_top is not None and top not in include_top:
                continue
            if include_fragments and not _path_has_any(path, include_fragments):
                continue
            try:
                score = float(row.get("combined_rank_score", 9999.0) or 9999.0)
            except Exception:
                score = 9999.0
            rows.append((score, row))
        if not rows:
            return None
        rows.sort(key=lambda item: item[0])
        return rows[0]
