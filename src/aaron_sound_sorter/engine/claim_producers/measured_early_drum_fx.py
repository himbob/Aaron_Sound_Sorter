# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Drum-loop and non-voice FX rescue guards for early adjudication."""

from __future__ import annotations

from aaron_sound_sorter.engine.claim_producers.measured_early_types import (
    ABSTRACT_TONE_FX_FRAGMENTS,
    CONCRETE_FX_FRAGMENTS,
    TONAL_FX_FRAGMENTS,
    EarlyAdjudicationContext,
    VoiceCandidateEvidence,
)
from aaron_sound_sorter.engine.decision_helpers import (
    _direct_body_role_strength_from_facts,
    _has_drum_loop_structure_support,
    _norm_path,
    _path_has_any,
    _role_strength_from_facts,
    _shape_blocks_percussion_loop_rescue,
)
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class DrumFxRescueMixin:
    """Handle drum-loop rescues and Human/Voice-to-FX corrections."""

    def _adjudicate_drum_and_fx_rescues(
        self,
        context: EarlyAdjudicationContext,
        evidence: VoiceCandidateEvidence,
    ) -> ConsensusClaim | None:
        decision = self._maybe_redirect_low_rhythmic_drum_loop(context)
        if decision is not None:
            return decision
        decision = self._maybe_review_vocal_loop_vs_drum_loop(context, evidence)
        if decision is not None:
            return decision
        decision = self._maybe_rescue_drum_loop_true_bucket(context, evidence)
        if decision is not None:
            return decision
        decision = self._maybe_rescue_pitched_percussion_loop(context, evidence)
        if decision is not None:
            return decision
        decision = self._maybe_rescue_raw_human_voice_fx_false_positive(context, evidence)
        if decision is not None:
            return decision
        return self._maybe_rescue_tonal_metallic_fx(context, evidence)

    @staticmethod
    def _low_rhythmic_drum_strength(context: EarlyAdjudicationContext) -> float:
        return max(
            _role_strength_from_facts(context.facts, "low_rhythmic_drum_loop"),
            _role_strength_from_facts(context.facts, "percussive_drum_loop"),
            _direct_body_role_strength_from_facts(context.facts, "low_rhythmic_drum_loop"),
            _direct_body_role_strength_from_facts(context.facts, "percussive_drum_loop"),
        )

    def _maybe_redirect_low_rhythmic_drum_loop(
        self,
        context: EarlyAdjudicationContext,
    ) -> ConsensusClaim | None:
        if (
            context.role == "drum_loop"
            and self._low_rhythmic_drum_strength(context) >= 0.75
            and context.shape != "vocal_phrase"
        ):
            return self._redirect_from_raw(
                context.raw,
                "Drums/Drum Loops/Loops",
                "low-rhythmic direct/full body evidence beat bass/FX loop reading",
            )
        return None

    def _maybe_review_vocal_loop_vs_drum_loop(
        self,
        context: EarlyAdjudicationContext,
        evidence: VoiceCandidateEvidence,
    ) -> ConsensusClaim | None:
        if context.role != "drum_loop" or evidence.best_voice is None:
            return None
        best_drum_score = (
            evidence.best_drum_loop[0]
            if evidence.best_drum_loop is not None
            else self._best_candidate_score(context.raw, include_top={"Drums"}, include_fragments=())
        )
        voice_is_structural = bool(
            evidence.best_voice_candidate_role_strength >= 0.70
            or evidence.measured_voice_strength >= 0.72
            or evidence.direct_voice_strength >= 0.72
        )
        voice_blocks_generic_instrument = bool(
            voice_is_structural
            and context.raw.final_top == "Instruments"
            and evidence.best_inst_any is not None
            and evidence.best_voice <= evidence.best_inst_any + 2.0
            and (best_drum_score is None or best_drum_score >= evidence.best_voice + 6.0)
        )
        if (voice_is_structural or voice_blocks_generic_instrument) and (
            best_drum_score is None or evidence.best_voice <= best_drum_score - 3.0
        ):
            return self._review_from_raw(
                context.raw,
                context.eligibility,
                "vocal loop conflict: drum-loop role had stronger structural human/voice candidate evidence",
            )
        return None

    def _maybe_rescue_drum_loop_true_bucket(
        self,
        context: EarlyAdjudicationContext,
        evidence: VoiceCandidateEvidence,
    ) -> ConsensusClaim | None:
        if evidence.best_drum_loop is None:
            return None
        drum_score, drum_path = evidence.best_drum_loop
        raw_wrong_for_drum = bool(
            (
                context.raw.final_top == "Instruments"
                and _path_has_any(context.raw_path, ("bass loops", "instrument loops", "mixed musical loops"))
            )
            or (
                context.raw.final_top == "Instruments"
                and _path_has_any(context.raw_path, ("bass", "808", "sub bass", "synth bass"))
                and _path_has_any(context.raw_path, ("one shot", "one shots"))
                and _has_drum_loop_structure_support(
                    context.role,
                    context.measured_role,
                    context.shape,
                    context.shape_conf,
                )
            )
            or (
                context.raw.final_top == "FX"
                and _path_has_any(context.raw_path, ("human and voice", "voice", "vocal", "breath", "crowd"))
            )
        )
        drum_structure_support = bool(
            _has_drum_loop_structure_support(context.role, context.measured_role, context.shape, context.shape_conf)
            or (context.shape == "bass_phrase" and drum_score <= context.raw_score + 4.0)
        )
        phrase_like_non_drum = bool(
            context.shape in {"vocal_phrase", "pitched_phrase", "sustained_pad"} and not drum_structure_support
        )
        if not raw_wrong_for_drum or phrase_like_non_drum:
            return None
        against = (
            evidence.best_inst_any
            if context.raw.final_top == "Instruments"
            else (evidence.best_voice if evidence.best_voice is not None else evidence.best_fx_any)
        )
        if against is None:
            against = context.raw_score
        if drum_score <= against + 5.0:
            reason = (
                "percussion-loop true-bucket rescue"
                if _path_has_any(
                    _norm_path(drum_path),
                    ("percussion", "conga", "bongo", "tabla", "triangle", "wood block", "metallic"),
                )
                else "kick/drum-loop true-bucket rescue"
            )
            return self._redirect_from_raw(
                context.raw,
                "Drums/Drum Loops/Loops",
                f"{reason}: close drum/percussion candidate beat unsafe generic/FX family",
            )
        return None

    def _maybe_rescue_pitched_percussion_loop(
        self,
        context: EarlyAdjudicationContext,
        evidence: VoiceCandidateEvidence,
    ) -> ConsensusClaim | None:
        if context.role != "pitched_percussion_loop" or evidence.best_drum_loop is None:
            return None
        if _shape_blocks_percussion_loop_rescue(context.shape, context.shape_conf):
            if context.raw.final_top in {"Instruments", "FX"} and not context.raw.folder_path.startswith("_TO_REVIEW"):
                return context.raw
            return None
        return self._broaden_from_raw(
            context.raw,
            context.eligibility,
            "percussion-loop true-bucket rescue: measured pitched_percussion_loop overruled generic instrument/FX candidates",
        )

    def _maybe_rescue_raw_human_voice_fx_false_positive(
        self,
        context: EarlyAdjudicationContext,
        evidence: VoiceCandidateEvidence,
    ) -> ConsensusClaim | None:
        if not (
            context.raw.final_top == "FX"
            and "human and voice" in context.raw_path
            and evidence.best_nonvoice is not None
        ):
            return None
        nonvoice_score, nonvoice_path = evidence.best_nonvoice
        nonvoice_low = _norm_path(nonvoice_path)
        phrase_like = context.shape in {"vocal_phrase", "pitched_phrase"} and context.shape_conf >= 0.72
        nonvoice_is_abstract_tone = _path_has_any(nonvoice_low, ABSTRACT_TONE_FX_FRAGMENTS)
        nonvoice_is_concrete_fx = _path_has_any(nonvoice_low, CONCRETE_FX_FRAGMENTS)
        if phrase_like and nonvoice_is_abstract_tone and not nonvoice_is_concrete_fx:
            return self._redirect_from_raw(
                context.raw,
                "Instruments/Brass and Woodwinds/Loops",
                "reed-like phrase rescue: phrase-shaped Human/Voice false positive had only abstract beep/siren FX candidates",
            )
        if nonvoice_low.startswith("fx/") and (
            evidence.best_voice is None or nonvoice_score <= evidence.best_voice - 2.0
        ):
            return self._redirect_from_raw(
                context.raw,
                nonvoice_path,
                "non-voice FX true-bucket rescue: stronger non-voice FX candidate beat Human/Voice FX",
            )
        return None

    def _maybe_rescue_tonal_metallic_fx(
        self,
        context: EarlyAdjudicationContext,
        evidence: VoiceCandidateEvidence,
    ) -> ConsensusClaim | None:
        del evidence
        if not (
            context.raw.final_top == "Instruments"
            and _path_has_any(context.raw_path, ("instrument loops", "mixed musical loops"))
        ):
            return None
        anchor_fx = self._best_candidate(
            context.raw,
            include_top={"FX"},
            include_fragments=("bell", "bells", "chime", "chimes", "hybrid designed", "designed tonal", "metallic"),
        )
        best_tonal_fx = self._best_candidate(
            context.raw,
            include_top={"FX"},
            include_fragments=TONAL_FX_FRAGMENTS,
        )
        if anchor_fx is None or best_tonal_fx is None:
            return None
        anchor_score, _anchor_path = anchor_fx
        fx_score, fx_path = best_tonal_fx
        if anchor_score <= context.raw_score + 5.0 and fx_score <= context.raw_score + 5.0:
            return self._redirect_from_raw(
                context.raw,
                fx_path,
                "tonal FX true-bucket rescue: close tonal/designed FX candidate beat generic Instrument Loop",
            )
        return None
