# SOURCE-NAME BLINDNESS INVARIANT:
# This claim producer may inspect measured eligibility and voter candidates.
# It must never use source filenames or source-folder tokens.
"""Voice, percussive, and weak-review broad bucket claims."""

from __future__ import annotations

from aaron_sound_sorter.engine.claim_support import ClaimSupportMixin
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.decision_helpers import _candidate_role_strength, _norm_path
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class VoicePercussiveBucketClaimProducer(ClaimSupportMixin):
    """Produce broad voice/percussive buckets without finalizing placement.

    The logic here is intentionally broad-bucket policy.  It does not try to
    identify exact voice, drum, or FX leaves.
    """

    def produce(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return ordered voice/percussive/weak-review broadening claims."""
        raw = context.raw
        eligibility = context.eligibility
        claims: list[ConsensusClaim] = []
        for claim in (
            self.weak_review_parent_claim(raw, eligibility),
            self.ambiguous_percussive_fx_claim(raw, eligibility),
            self.non_voice_instrument_voice_claim(raw, eligibility),
            self.voice_dominates_generic_loop_claim(raw, eligibility),
        ):
            if claim is not None:
                claims.append(claim)
        return claims

    def weak_review_parent_claim(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
    ) -> ConsensusClaim | None:
        """Use decisive parent evidence to broaden weak review outcomes."""
        if raw.final_top != "_TO_REVIEW":
            return None
        if eligibility.confidence < 0.74:
            return None
        broad_path = eligibility.broad_folder_path.strip("/")
        if not broad_path or broad_path.startswith("_TO_REVIEW"):
            return None
        low_reason = str(raw.reason or "").lower()
        if self._reason_contains_hard_conflict(low_reason):
            return None
        if not self._reason_is_weak_review(low_reason):
            return None
        return self._broaden_from_raw(
            raw,
            eligibility,
            "raw consensus chose weak review but measured parent eligibility was decisive",
        )

    @staticmethod
    def _reason_contains_hard_conflict(reason: str) -> bool:
        """Return True for review reasons that must stay review."""
        return any(
            fragment in reason for fragment in ("broken", "tiny", "read failure", "measured role conflict", "integrity")
        )

    @staticmethod
    def _reason_is_weak_review(reason: str) -> bool:
        """Return True for review reasons that may be broadened by parent evidence."""
        return any(
            fragment in reason
            for fragment in (
                "no strong voter consensus",
                "best shared category existed but was too weak",
                "ambiguous",
                "too weak",
            )
        )

    def ambiguous_percussive_fx_claim(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
    ) -> ConsensusClaim | None:
        """Prefer a broad drum bucket for ambiguous percussive FX one-shots."""
        if eligibility.role_name != "percussive_one_shot" or eligibility.confidence < 0.70:
            return None
        if raw.final_top != "FX":
            return None
        raw_path = _norm_path(raw.folder_path)
        if not self._is_ambiguous_percussive_fx_path(raw_path):
            return None
        best_drum_rank = self._best_percussive_drum_rank(raw)
        if best_drum_rank is None:
            return None
        raw_rank = float(raw.combined_rank_score or 9999.0)
        if best_drum_rank > raw_rank + 8.0:
            return None
        drum_eligibility = EligibilityDecision(
            role_name="percussive_one_shot",
            confidence=eligibility.confidence,
            allowed_top_families=("Drums", "_TO_REVIEW"),
            blocked_path_fragments=(),
            broad_folder_path="Drums/Percussion/Generic Percussion/One Shots",
            reason="measured percussive one-shot had comparable shared Drums support; broadening away from ambiguous FX leaf",
        )
        return self._broaden_from_raw(
            raw, drum_eligibility, "ambiguous FX one-shot leaf was not safer than broad Drums"
        )

    @staticmethod
    def _is_ambiguous_percussive_fx_path(raw_path: str) -> bool:
        """Return True for FX leaves that can steal percussive one-shots."""
        return any(
            fragment in raw_path
            for fragment in (
                "keys coins",
                "coins",
                "keys/",
                "small objects",
                "glitch",
                "stutter",
                "blip",
                "beep",
                "animals",
                "dog",
                "cat",
                "bird",
                "human and voice",
                "breath",
            )
        )

    @staticmethod
    def _best_percussive_drum_rank(raw: ConsensusClaim) -> float | None:
        """Return best shared Drum candidate with percussive one-shot support."""
        best_drum_rank: float | None = None
        for candidate in raw.shared_candidates or []:
            folder_path = str(candidate.get("folder_path") or candidate.get("label") or "")
            if not _norm_path(folder_path).startswith("drums/"):
                continue
            candidate_role_strength = _candidate_role_strength(candidate, "percussive_one_shot")
            if candidate_role_strength < 0.70:
                continue
            try:
                candidate_rank = float(candidate.get("combined_rank_score", 9999.0) or 9999.0)
            except Exception:
                candidate_rank = 9999.0
            if best_drum_rank is None or candidate_rank < best_drum_rank:
                best_drum_rank = candidate_rank
        return best_drum_rank

    def non_voice_instrument_voice_claim(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
    ) -> ConsensusClaim | None:
        """Keep measured vocal phrases out of generic non-voice buckets."""
        if eligibility.role_name not in {"vocal_phrase", "vocal_one_shot", "vocal_music_phrase", "voiced_one_shot"}:
            return None
        raw_path = _norm_path(raw.folder_path)
        if "voice" in raw_path or "vocal" in raw_path or "human and voice" in raw_path:
            return None
        if raw.final_top not in {"Instruments", "FX"}:
            return None
        if eligibility.confidence < 0.72:
            return None
        return self._broaden_from_raw(
            raw,
            eligibility,
            "measured vocal role was safer than a non-voice broad/instrument leaf",
        )

    def voice_dominates_generic_loop_claim(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
    ) -> ConsensusClaim | None:
        """Recover obvious voice candidates from generic loop fallbacks."""
        if raw.final_top != "Instruments":
            return None
        raw_path = _norm_path(raw.folder_path)
        if "voice" in raw_path or "vocal" in raw_path or "human" in raw_path:
            return None
        if not ("instrument loops" in raw_path or "bass" in raw_path or "mixed musical loops" in raw_path):
            return None
        if eligibility.role_name not in {"vocal_phrase", "vocal_one_shot"}:
            return None
        raw_rank = self._safe_raw_score(raw)
        best_voice_rank = self._best_voice_rank(raw)
        if best_voice_rank is None:
            return None
        has_voice_role_support = self._has_voice_candidate_with_role(raw)
        required_win = 4.0 if has_voice_role_support else 8.0
        if best_voice_rank > raw_rank - required_win:
            return None
        voice_eligibility = EligibilityDecision(
            role_name=eligibility.role_name
            if eligibility.role_name in {"vocal_phrase", "vocal_one_shot", "vocal_music_phrase", "voiced_one_shot"}
            else "vocal_phrase",
            confidence=max(eligibility.confidence, 0.74),
            allowed_top_families=("FX", "Instruments", "_TO_REVIEW"),
            blocked_path_fragments=(),
            broad_folder_path="FX/Human and Voice FX",
            reason="shared Human/Voice candidate dominated generic instrument-loop fallback",
        )
        return self._broaden_from_raw(
            raw,
            voice_eligibility,
            "human/voice shared candidate was safer than generic instrument-loop fallback",
        )

    @staticmethod
    def _safe_raw_score(raw: ConsensusClaim) -> float:
        """Return raw combined score as a numeric value."""
        try:
            return float(raw.combined_rank_score or 9999.0)
        except Exception:
            return 9999.0

    @staticmethod
    def _best_voice_rank(raw: ConsensusClaim) -> float | None:
        """Return best shared Human/Voice candidate rank."""
        best_voice_rank: float | None = None
        for candidate in raw.shared_candidates or []:
            folder_path = _norm_path(str(candidate.get("folder_path") or candidate.get("label") or ""))
            if "human and voice" not in folder_path and "voice" not in folder_path and "vocal" not in folder_path:
                continue
            try:
                candidate_rank = float(candidate.get("combined_rank_score", 9999.0) or 9999.0)
            except Exception:
                candidate_rank = 9999.0
            if best_voice_rank is None or candidate_rank < best_voice_rank:
                best_voice_rank = candidate_rank
        return best_voice_rank
