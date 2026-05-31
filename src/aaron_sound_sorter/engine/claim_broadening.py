# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Broad eligibility claim construction helpers."""

from __future__ import annotations

from aaron_sound_sorter.engine.decision_helpers import (
    _candidate_role_strength,
    _is_concrete_fx_path,
    _norm_path,
    _path_has_any,
)
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_folder_path, review_claim


class ClaimBroadeningMixin:
    """Build broad eligibility claims while preserving candidate evidence."""

    def _broad_claim_can_override(
        self,
        raw: ConsensusClaim,
        broad_path: str,
        eligibility: EligibilityDecision,
    ) -> bool:
        """Return whether a broad eligibility claim may override raw."""
        normalized = _norm_path(broad_path)
        if normalized.startswith("_to_review"):
            return True
        if normalized.startswith("drums/") and "drum loops" in normalized:
            if eligibility.role_name == "pitched_percussion_loop" and _path_has_any(
                raw.folder_path, ("human and voice", "voice", "vocal")
            ):
                return False
        if _path_has_any(normalized, ("human and voice", "voice", "vocal")):
            broad_top = normalized.split("/", 1)[0]
            raw_path = _norm_path(raw.folder_path)
            if raw.final_top.lower() == broad_top:
                return bool(
                    _path_has_any(raw_path, ("human and voice", "voice", "vocal"))
                    or self._has_voice_candidate_with_role(raw)
                )
            return self._has_voice_candidate_with_role(raw)
        return True

    @staticmethod
    def _has_voice_candidate_with_role(raw: ConsensusClaim) -> bool:
        """Return True for a real nearby Human/Voice candidate with role support."""
        for candidate in raw.shared_candidates or []:
            candidate_path = _norm_path(str(candidate.get("folder_path") or candidate.get("label") or ""))
            if not _path_has_any(
                candidate_path, ("human and voice", "voice", "vocal", "vox", "choir", "spoken", "breath", "crowd")
            ):
                continue
            role_support = max(
                _candidate_role_strength(candidate, "vocal_music_phrase"),
                _candidate_role_strength(candidate, "vocal_phrase"),
                _candidate_role_strength(candidate, "vocal_one_shot"),
                _candidate_role_strength(candidate, "voiced_one_shot"),
            )
            if role_support >= 0.75:
                return True
        return False

    def _broaden_from_raw(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        prefix: str,
    ) -> ConsensusClaim:
        """Return a broad fallback decision while preserving raw evidence."""
        broad_path = eligibility.broad_folder_path.strip("/") or "_TO_REVIEW/Measured Role Conflict"
        if broad_path.lower() == "fx/human and voice fx" and eligibility.role_name in {
            "vocal_phrase",
            "vocal_one_shot",
            "vocal_music_phrase",
            "voiced_one_shot",
        }:
            broad_path = "Instruments/Voice/Phrase/One Shots"
        top = broad_path.split("/", 1)[0]
        status = "parent_eligibility_broad_bucket"
        if top == "_TO_REVIEW":
            status = "parent_eligibility_review"
        reason = (
            f"{prefix}: role={eligibility.role_name}, "
            f"confidence={eligibility.confidence:.2f}; raw={raw.folder_path}; "
            f"fallback={broad_path}; {eligibility.reason}"
        )
        candidate_score = self._candidate_score_for_path(raw, broad_path)
        if candidate_score is None:
            candidate_score = self._candidate_score_for_broad_target(raw, broad_path)
        has_real_candidate = candidate_score is not None
        broad_strength = eligibility.confidence
        broad_path_normalized = _norm_path(broad_path)
        raw_score = raw.raw_candidate_score
        if (
            candidate_score is not None
            and raw_score is not None
            and ("instrument loops" in broad_path_normalized or "human and voice" in broad_path_normalized)
            and candidate_score > raw_score + 4.0
        ):
            candidate_score = None
            has_real_candidate = False
        if (
            candidate_score is not None
            and raw_score is not None
            and _path_has_any(broad_path_normalized, ("brass", "woodwind", "sax", "reed"))
            and candidate_score > raw_score + 10.0
        ):
            candidate_score = None
            has_real_candidate = False
        if (
            raw.final_top != "Instruments"
            and not has_real_candidate
            and _path_has_any(broad_path_normalized, ("brass", "woodwind", "sax", "reed"))
        ):
            broad_path = "Instruments/Instrument Loops/Loops"
            broad_path_normalized = _norm_path(broad_path)
            top = "Instruments"
        protected_claim = self._maybe_keep_raw_over_generic_instrument(
            raw=raw,
            eligibility=eligibility,
            prefix=prefix,
            broad_path=broad_path,
            broad_path_normalized=broad_path_normalized,
            has_real_candidate=has_real_candidate,
        )
        if protected_claim is not None:
            return protected_claim
        if has_real_candidate and top != "_TO_REVIEW":
            broad_strength = max(broad_strength, 0.90)
        if (
            top == "Drums"
            and "drum loops" in broad_path_normalized
            and eligibility.role_name
            in {
                "drum_loop",
                "percussive_drum_loop",
                "bright_drum_loop",
                "low_rhythmic_drum_loop",
            }
        ):
            broad_strength = max(broad_strength, 0.90)
            if (
                candidate_score is not None
                and raw.raw_candidate_score is not None
                and candidate_score > raw.raw_candidate_score + 4.0
            ):
                candidate_score = None
                has_real_candidate = False
        reason = (
            f"{prefix}: role={eligibility.role_name}, "
            f"confidence={eligibility.confidence:.2f}; raw={raw.folder_path}; "
            f"fallback={broad_path}; {eligibility.reason}"
        )
        return claim_from_folder_path(
            folder_path=broad_path,
            source=status,
            reason=reason,
            shared=raw.shared_candidates,
            raw_candidate_score=candidate_score,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner,
            can_override=self._broad_claim_can_override(raw, broad_path, eligibility),
            strength=broad_strength,
            is_real_candidate=has_real_candidate,
        )

    def _maybe_keep_raw_over_generic_instrument(
        self,
        *,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        prefix: str,
        broad_path: str,
        broad_path_normalized: str,
        has_real_candidate: bool,
    ) -> ConsensusClaim | None:
        """Protect concrete raw FX/review cases from unsupported Instrument Loops."""
        if not (
            broad_path.split("/", 1)[0] == "Instruments"
            and "instrument loops" in broad_path_normalized
            and raw.final_top != "Instruments"
            and not has_real_candidate
            and not (raw.final_top == "FX" and "human and voice" in _norm_path(raw.folder_path))
        ):
            return None
        sustained_loop_role = eligibility.role_name in {
            "pitched_reed_or_instrument_loop",
            "pitched_reed_or_instrument_phrase",
            "pitched_music_loop",
            "pitched_music_phrase",
            "mixed_music_loop",
            "clean_sustained_tonal_instrument_loop",
        }
        if raw.final_top == "_TO_REVIEW" or (
            _is_concrete_fx_path(raw.folder_path) and not (sustained_loop_role and eligibility.confidence >= 0.74)
        ):
            return claim_from_folder_path(
                folder_path=raw.folder_path,
                source=raw.source,
                reason=raw.reason,
                shared=raw.shared_candidates,
                raw_candidate_score=raw.raw_candidate_score,
                brain_rank=raw.brain_rank,
                physics_rank=raw.physics_rank,
                shared_winner=raw.shared_winner,
                can_override=True,
                strength=max(raw.strength, 0.80),
                is_real_candidate=raw.final_top != "_TO_REVIEW",
            )
        if sustained_loop_role and eligibility.confidence >= 0.74:
            return claim_from_folder_path(
                folder_path=broad_path,
                source="sustained_pitched_instrument_broad_bucket",
                reason=(
                    f"{prefix}: role={eligibility.role_name}, "
                    f"confidence={eligibility.confidence:.2f}; raw={raw.folder_path}; "
                    f"fallback={broad_path}; measured sustained pitched instrument evidence was safer than review"
                ),
                shared=raw.shared_candidates,
                raw_candidate_score=None,
                brain_rank=raw.brain_rank,
                physics_rank=raw.physics_rank,
                shared_winner=raw.shared_winner,
                can_override=True,
                strength=eligibility.confidence,
                is_real_candidate=False,
            )
        return review_claim(
            label="_TO_REVIEW/Measured Role Conflict",
            reason=(
                f"{prefix}: role={eligibility.role_name}, "
                f"confidence={eligibility.confidence:.2f}; raw={raw.folder_path}; "
                f"fallback={broad_path}; generic Instrument Loops had no close real candidate support"
            ),
            source="parent_eligibility_review",
            shared=raw.shared_candidates,
            winner=raw,
        )
