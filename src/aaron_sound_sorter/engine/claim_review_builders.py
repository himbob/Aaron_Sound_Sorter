# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Redirect and review claim construction helpers."""

from __future__ import annotations

from aaron_sound_sorter.engine.decision_helpers import _path_has_any
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import (
    ConsensusClaim,
    claim_from_folder_path,
    review_claim,
)


class ClaimReviewBuilderMixin:
    """Build redirect and review claims from raw consensus evidence."""

    def _redirect_from_raw(
        self,
        raw: ConsensusClaim,
        folder_path: str,
        reason_prefix: str,
    ) -> ConsensusClaim:
        """Return a non-review safety redirect to a real broad/near bucket."""
        clean_path = str(folder_path or "_TO_REVIEW/Measured Role Conflict").strip("/")
        lower_reason = reason_prefix.lower()
        # True musical/speech voice belongs under Instruments/Voice.  FX/Human
        # and Voice is no longer used as a positive voice rescue bucket because
        # it creates a split taxonomy and sends obvious vocal material into FX.
        # Non-voice formant/designed FX should use explicit FX/Formant or FX
        # designed categories, not the broad Human/Voice fallback.
        if clean_path.lower() == "fx/human and voice fx" and (
            "voice" in lower_reason or "vocal" in lower_reason or "human/voice" in lower_reason
        ):
            clean_path = "Instruments/Voice/Phrase/One Shots"
        source = "candidate_true_bucket_rescue"
        if _path_has_any(clean_path, ("human and voice", "voice", "vocal")) and (
            "human/voice" in lower_reason or "voice" in lower_reason or "vocal" in lower_reason
        ):
            source = "human_voice_true_bucket_rescue"
        strength = (
            0.90 if _path_has_any(clean_path, ("drum loops", "instrument loops", "brass and woodwinds")) else 0.86
        )
        if source == "human_voice_true_bucket_rescue":
            strength = 0.92
        direct_score = self._candidate_score_for_path(raw, clean_path)
        broad_score = self._candidate_score_for_broad_target(raw, clean_path)
        return claim_from_folder_path(
            folder_path=clean_path,
            source=source,
            reason=f"{reason_prefix}; raw={raw.folder_path}; fallback={clean_path}",
            shared=raw.shared_candidates,
            raw_candidate_score=direct_score or broad_score,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner,
            can_override=True,
            strength=strength,
            is_real_candidate=(direct_score or broad_score) is not None,
        )

    def _review_from_raw(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        conflict_reason: str,
    ) -> ConsensusClaim:
        """Return a review decision that preserves raw/candidate evidence."""
        reason = (
            f"{conflict_reason}: role={eligibility.role_name}, "
            f"confidence={eligibility.confidence:.2f}; raw={raw.folder_path}; "
            "candidate evidence conflicted with broad role forcing"
        )
        return review_claim(
            label="_TO_REVIEW/Measured Role Conflict",
            reason=reason,
            source="role_candidate_conflict_review",
            shared=raw.shared_candidates,
            winner=raw,
        )

    def _review_from_raw_with_role(
        self,
        raw: ConsensusClaim,
        role_name: str,
        conflict_reason: str,
    ) -> ConsensusClaim:
        """Return a review decision when no decisive eligibility object exists."""
        reason = (
            f"{conflict_reason}: role={role_name}; raw={raw.folder_path}; "
            "candidate evidence conflicted with raw consensus winner"
        )
        return review_claim(
            label="_TO_REVIEW/Measured Role Conflict",
            reason=reason,
            source="raw_candidate_conflict_review",
            shared=raw.shared_candidates,
            winner=raw,
        )
