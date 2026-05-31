# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Reed and vocal false-positive conflict guards.

The rules here are intentionally conservative copies of the legacy measured
adjudication lattice.  They are isolated so sax/voice false positives can be
reviewed without digging through drum-loop rescue code.
"""

from __future__ import annotations

from aaron_sound_sorter.engine.claim_producers.measured_early_types import (
    ABSTRACT_TONE_FX_FRAGMENTS,
    CONCRETE_FX_FRAGMENTS,
    EarlyAdjudicationContext,
    VoiceCandidateEvidence,
)
from aaron_sound_sorter.engine.decision_helpers import (
    _norm_path,
    _path_has_any,
    _voice_identity_from_facts,
)
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class ReedVoiceConflictMixin:
    """Handle reed over-narrowing and vocal false-positive conflicts."""

    def _adjudicate_reed_and_voice_conflicts(
        self,
        context: EarlyAdjudicationContext,
        evidence: VoiceCandidateEvidence,
    ) -> ConsensusClaim | None:
        decision = self._maybe_review_weak_reed_over_narrowing(context)
        if decision is not None:
            return decision
        return self._maybe_handle_voice_false_positive_conflict(context, evidence)

    def _maybe_review_weak_reed_over_narrowing(
        self,
        context: EarlyAdjudicationContext,
    ) -> ConsensusClaim | None:
        if context.role not in {"pitched_reed_or_instrument_loop", "pitched_reed_or_instrument_phrase"}:
            return None
        broad_path = _norm_path(context.eligibility.broad_folder_path)
        best_reed = self._best_candidate_score(
            context.raw,
            include_top={"Instruments"},
            include_fragments=("brass", "woodwind", "sax", "reed"),
        )
        if not _path_has_any(broad_path, ("brass", "woodwind", "sax", "reed")):
            return None
        if context.raw.final_top == "Instruments":
            return None

        sax_claim = "woodwinds/saxophone" in broad_path or "saxophone" in broad_path
        raw_fx_false_positive = bool(
            context.raw.final_top == "FX"
            and _path_has_any(context.raw_path, ("human and voice", "voice", "vocal", "siren", "alarm", "beep"))
        )
        if sax_claim and raw_fx_false_positive:
            if best_reed is None or best_reed > context.raw_score + 10.0:
                return self._review_from_raw(
                    context.raw,
                    context.eligibility,
                    "reed/woodwind over-narrowing conflict: sax-like measured role lacked close reed candidate support",
                )
            return self._broaden_from_raw(
                context.raw,
                context.eligibility,
                "measured sax/woodwind claim beat non-reed FX or Human/Voice false positive",
            )
        if best_reed is None or best_reed > context.raw_score + 10.0:
            return self._review_from_raw(
                context.raw,
                context.eligibility,
                "reed/woodwind over-narrowing conflict: weak reed-like role could not safely override raw non-reed candidate evidence",
            )
        return None

    def _maybe_handle_voice_false_positive_conflict(
        self,
        context: EarlyAdjudicationContext,
        evidence: VoiceCandidateEvidence,
    ) -> ConsensusClaim | None:
        if context.role not in {"vocal_phrase", "vocal_one_shot", "voiced_one_shot", "vocal_music_phrase"}:
            return None

        voice_identity = _voice_identity_from_facts(context.facts)
        strong_measured_voice_broad = bool(
            context.eligibility.confidence >= 0.78
            and context.shape == "vocal_phrase"
            and context.shape_conf >= 0.78
            and max(evidence.measured_voice_strength, evidence.direct_voice_strength) >= 0.78
            and voice_identity >= 0.68
            and context.measured_role
            not in {
                "pitched_reed_or_instrument_loop",
                "pitched_reed_or_instrument_phrase",
                "bass_loop",
                "drum_loop",
                "low_rhythmic_drum_loop",
                "percussive_drum_loop",
            }
        )
        if evidence.positive_voice_claim or strong_measured_voice_broad:
            reason = (
                "vocal true-bucket rescue: positive Human/Voice claim beat unsafe non-voice family"
                if evidence.positive_voice_claim
                else "measured vocal phrase/one-shot evidence beat unsafe non-voice candidate set"
            )
            return self._redirect_from_raw(context.raw, "FX/Human and Voice FX", reason)

        if evidence.best_nonvoice is None:
            return None
        nonvoice_score, nonvoice_path = evidence.best_nonvoice
        if evidence.best_voice is not None and nonvoice_score > evidence.best_voice - 2.0:
            return None

        nonvoice_low = _norm_path(nonvoice_path)
        phrase_like = context.shape in {"vocal_phrase", "pitched_phrase"} and context.shape_conf >= 0.72
        abstract_tone_fx = _path_has_any(nonvoice_low, ABSTRACT_TONE_FX_FRAGMENTS)
        concrete_fx = _path_has_any(nonvoice_low, CONCRETE_FX_FRAGMENTS)
        if context.raw.final_top == "FX" and "human and voice" in context.raw_path and nonvoice_low.startswith("fx/"):
            if phrase_like and abstract_tone_fx and not concrete_fx:
                return self._redirect_from_raw(
                    context.raw,
                    "Instruments/Brass and Woodwinds/Loops",
                    "reed-like phrase rescue: phrase-shaped Human/Voice false positive had only abstract beep/siren FX candidates",
                )
            return self._redirect_from_raw(
                context.raw,
                nonvoice_path,
                "non-voice FX true-bucket rescue: stronger non-voice FX candidate beat Human/Voice FX",
            )
        return self._review_from_raw(
            context.raw,
            context.eligibility,
            "voice false-positive conflict: broad vocal role had clearly stronger non-voice candidate evidence",
        )
