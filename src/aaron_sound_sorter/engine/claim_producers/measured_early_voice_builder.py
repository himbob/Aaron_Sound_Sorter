# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Human/Voice candidate evidence builder for early adjudication."""

from __future__ import annotations

from aaron_sound_sorter.engine.claim_producers.measured_early_types import (
    NONVOICE_FRAGMENTS,
    VOICE_FRAGMENTS,
    EarlyAdjudicationContext,
    VoiceCandidateEvidence,
)
from aaron_sound_sorter.engine.decision_helpers import (
    _candidate_role_strength,
    _direct_body_role_strength_from_facts,
    _norm_path,
    _path_has_any,
    _role_strength_from_facts,
    _stable_instrument_claim_strength,
)
from aaron_sound_sorter.engine.family_claims import build_human_voice_claim


class VoiceEvidenceBuilderMixin:
    """Build shared Human/Voice evidence for downstream voice guards."""

    def _build_voice_candidate_evidence(self, context: EarlyAdjudicationContext) -> VoiceCandidateEvidence:
        """Collect ranked-candidate and measured-role evidence for voice guards."""
        best_voice_candidate = self._best_candidate(
            context.raw,
            include_top={"FX", "Instruments"},
            include_fragments=VOICE_FRAGMENTS,
        )
        best_voice = best_voice_candidate[0] if best_voice_candidate is not None else None
        best_voice_path = best_voice_candidate[1] if best_voice_candidate is not None else ""

        best_voice_role_strength = 0.0
        best_voice_candidate_role_strength = 0.0
        for candidate in context.raw.shared_candidates or []:
            folder = _norm_path(str(candidate.get("folder_path") or candidate.get("label") or ""))
            if not _path_has_any(folder, VOICE_FRAGMENTS):
                continue
            candidate_voice_strength = max(
                _candidate_role_strength(candidate, "vocal_music_phrase"),
                _candidate_role_strength(candidate, "vocal_phrase"),
                _candidate_role_strength(candidate, "vocal_one_shot"),
                _candidate_role_strength(candidate, "voiced_one_shot"),
                _candidate_role_strength(candidate, "pitched_music_phrase"),
            )
            best_voice_role_strength = max(best_voice_role_strength, candidate_voice_strength)
            if best_voice_path and _norm_path(best_voice_path) == folder:
                best_voice_candidate_role_strength = max(
                    best_voice_candidate_role_strength,
                    candidate_voice_strength,
                )

        best_nonvoice = self._best_candidate(
            context.raw,
            include_top={"FX", "Drums", "Instruments"},
            include_fragments=NONVOICE_FRAGMENTS,
        )
        best_drum_loop = self._best_candidate(
            context.raw,
            include_top={"Drums"},
            include_fragments=(
                "drum loop",
                "drum loops",
                "break",
                "breaks",
                "kick",
                "percussion",
                "conga",
                "bongo",
                "tabla",
                "triangle",
                "wood block",
                "wood blocks",
                "metallic percussion",
            ),
        )
        best_inst_any = self._best_candidate_score(
            context.raw,
            include_top={"Instruments"},
            include_fragments=(),
        )
        best_fx_any = self._best_candidate_score(
            context.raw,
            include_top={"FX"},
            include_fragments=(),
        )

        measured_voice_strength = max(
            _role_strength_from_facts(context.facts, "voiced_one_shot"),
            _role_strength_from_facts(context.facts, "vocal_music_phrase"),
        )
        direct_voice_strength = max(
            _direct_body_role_strength_from_facts(context.facts, "voiced_one_shot"),
            _direct_body_role_strength_from_facts(context.facts, "vocal_music_phrase"),
        )
        best_competitor = min(
            [
                score
                for score in (
                    best_inst_any,
                    best_drum_loop[0] if best_drum_loop is not None else None,
                    best_nonvoice[0] if best_nonvoice is not None else None,
                )
                if score is not None
            ],
            default=None,
        )

        voice_candidate_is_actual_voice_path = bool(
            best_voice is not None
            and _path_has_any(
                _norm_path(best_voice_path),
                ("human and voice", "voice", "vocal", "vox", "spoken", "choir", "breath", "crowd"),
            )
        )
        voice_candidate_has_role_support = bool(
            best_voice_role_strength >= 0.75 or best_voice_candidate_role_strength >= 0.75
        )
        measured_voice_claim_strength = max(measured_voice_strength, direct_voice_strength)
        stable_instrument_claim_strength = _stable_instrument_claim_strength(
            context.role,
            context.measured_role,
            context.shape,
            context.shape_conf,
            context.facts,
        )
        human_voice_claim = build_human_voice_claim(
            has_voice_candidate=voice_candidate_is_actual_voice_path,
            voice_candidate_score=best_voice,
            raw_score=context.raw_score,
            competitor_score=best_competitor,
            voice_role_strength=measured_voice_claim_strength,
            candidate_role_strength=max(best_voice_role_strength, best_voice_candidate_role_strength),
            stable_instrument_claim_strength=stable_instrument_claim_strength,
        )
        direct_one_shot_voice_claim = bool(
            voice_candidate_is_actual_voice_path
            and best_voice is not None
            and not bool(getattr(context.facts, "is_loop_like", False))
            and _direct_body_role_strength_from_facts(context.facts, "voiced_one_shot") >= 0.88
            and (context.raw_score >= 999.0 or best_voice <= context.raw_score + 25.0)
        )
        measured_voice_claim_strength = max(measured_voice_strength, direct_voice_strength)
        return VoiceCandidateEvidence(
            best_voice=best_voice,
            best_voice_path=best_voice_path,
            best_voice_role_strength=best_voice_role_strength,
            best_voice_candidate_role_strength=best_voice_candidate_role_strength,
            best_nonvoice=best_nonvoice,
            best_drum_loop=best_drum_loop,
            best_inst_any=best_inst_any,
            best_fx_any=best_fx_any,
            measured_voice_strength=measured_voice_strength,
            direct_voice_strength=direct_voice_strength,
            best_competitor=best_competitor,
            voice_candidate_is_actual_voice_path=voice_candidate_is_actual_voice_path,
            voice_candidate_has_role_support=voice_candidate_has_role_support,
            measured_voice_claim_strength=measured_voice_claim_strength,
            voice_has_positive_audio_evidence=bool(
                voice_candidate_has_role_support or measured_voice_claim_strength >= 0.82
            ),
            positive_voice_claim=human_voice_claim.can_override,
            direct_one_shot_voice_claim=direct_one_shot_voice_claim,
        )
