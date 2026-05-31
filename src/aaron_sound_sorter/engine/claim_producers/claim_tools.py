# SOURCE-NAME BLINDNESS INVARIANT:
# Claim tool adapters may inspect candidate paths already produced by voters,
# but they must never inspect producer filenames, source folders, or ZIP member
# names as classification evidence.
"""Shared claim-building protocol and adapter base for claim producers."""

from __future__ import annotations

from typing import Protocol

from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class BroadBucketClaimTools(Protocol):
    """Small adapter protocol supplied by ``DecisionCoreV2``.

    Producers own the claim policy.  The core still owns low-level candidate
    search and claim construction until those helpers are extracted into a
    reusable toolkit.
    """

    def _best_candidate(
        self,
        raw: ConsensusClaim,
        *,
        include_top: set[str],
        include_fragments: tuple[str, ...],
        exclude_fragments: tuple[str, ...] = (),
    ) -> tuple[float, str] | None:
        """Return the best shared candidate matching broad path filters."""

    def _best_candidate_score(
        self,
        raw: ConsensusClaim,
        *,
        include_top: set[str],
        include_fragments: tuple[str, ...],
    ) -> float | None:
        """Return only the best matching shared-candidate score."""

    def _broaden_from_raw(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        prefix: str,
    ) -> ConsensusClaim:
        """Build a broad-folder claim from the measured eligibility target."""

    def _candidate_score_for_broad_target(self, raw: ConsensusClaim, folder_path: str) -> float:
        """Return a score for a broad synthetic target folder."""

    def _has_stable_raw_top_support(self, raw: ConsensusClaim) -> bool:
        """Return True when the raw top has enough stable support to protect it."""

    def _has_voice_candidate_with_role(self, raw: ConsensusClaim) -> bool:
        """Return True when candidate evidence supports an actual voice path."""

    def _redirect_from_raw(
        self,
        raw: ConsensusClaim,
        folder_path: str,
        reason: str,
    ) -> ConsensusClaim:
        """Build a redirect claim from the raw context."""

    def _review_from_raw(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        reason: str,
    ) -> ConsensusClaim:
        """Build a review claim from the raw context."""


class BroadBucketClaimProducerBase:
    """Base class that keeps producer policy separate from claim construction."""

    def __init__(self, tools: BroadBucketClaimTools) -> None:
        self.tools = tools

    def _best_candidate(
        self,
        raw: ConsensusClaim,
        include_top: set[str],
        include_fragments: tuple[str, ...],
        exclude_fragments: tuple[str, ...] = (),
    ) -> tuple[float, str] | None:
        return self.tools._best_candidate(
            raw,
            include_top=include_top,
            include_fragments=include_fragments,
            exclude_fragments=exclude_fragments,
        )

    def _best_candidate_score(
        self,
        raw: ConsensusClaim,
        include_top: set[str],
        include_fragments: tuple[str, ...],
    ) -> float | None:
        return self.tools._best_candidate_score(
            raw,
            include_top=include_top,
            include_fragments=include_fragments,
        )

    def _broaden_from_raw(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        prefix: str,
    ) -> ConsensusClaim:
        return self.tools._broaden_from_raw(raw, eligibility, prefix)

    def _candidate_score_for_broad_target(self, raw: ConsensusClaim, folder_path: str) -> float:
        return self.tools._candidate_score_for_broad_target(raw, folder_path)

    def _has_stable_raw_top_support(self, raw: ConsensusClaim) -> bool:
        return self.tools._has_stable_raw_top_support(raw)

    def _has_voice_candidate_with_role(self, raw: ConsensusClaim) -> bool:
        return self.tools._has_voice_candidate_with_role(raw)

    def _redirect_from_raw(
        self,
        raw: ConsensusClaim,
        folder_path: str,
        reason: str,
    ) -> ConsensusClaim:
        return self.tools._redirect_from_raw(raw, folder_path, reason)

    def _review_from_raw(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        reason: str,
    ) -> ConsensusClaim:
        return self.tools._review_from_raw(raw, eligibility, reason)
