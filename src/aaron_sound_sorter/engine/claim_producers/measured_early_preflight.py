# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Preflight measured-role guards for early adjudication."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.claim_producers.measured_early_types import (
    PROTECTED_HIT_ROLES,
    EarlyAdjudicationContext,
)
from aaron_sound_sorter.engine.decision_helpers import (
    _direct_body_role_strength_from_facts,
    _norm_path,
    _path_has_any,
    _role_strength_from_facts,
)
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class EarlyPreflightMixin:
    """Build shared context and run first-refusal safety guards."""

    def _build_early_context(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        measured_role: str,
        shape: str,
        shape_conf: float,
        facts: SharedAudioFacts | None,
    ) -> EarlyAdjudicationContext:
        """Collect common scores used by the extracted guard sections."""
        try:
            raw_score = float(raw.combined_rank_score or 9999.0)
        except Exception:
            raw_score = 9999.0
        return EarlyAdjudicationContext(
            raw=raw,
            eligibility=eligibility,
            measured_role=measured_role,
            shape=shape,
            shape_conf=shape_conf,
            facts=facts,
            raw_path=_norm_path(raw.folder_path),
            role=str(eligibility.role_name or ""),
            raw_score=raw_score,
            direct_voice_strength=max(
                _direct_body_role_strength_from_facts(facts, "voiced_one_shot"),
                _direct_body_role_strength_from_facts(facts, "vocal_music_phrase"),
            ),
            full_voice_strength=max(
                _role_strength_from_facts(facts, "voiced_one_shot"),
                _role_strength_from_facts(facts, "vocal_music_phrase"),
            ),
            bass_loop_strength=max(
                _role_strength_from_facts(facts, "bass_loop"),
                _direct_body_role_strength_from_facts(facts, "bass_loop"),
            ),
        )

    def _adjudicate_preflight_claims(self, context: EarlyAdjudicationContext) -> ConsensusClaim | None:
        """Protect obvious drum, bass, and voice claims before deeper fallbacks."""
        raw = context.raw
        early_best_drum_loop = self._best_candidate(
            raw,
            include_top={"Drums"},
            include_fragments=("drum loop", "drum loops", "break", "breaks"),
        )
        early_drum_loop_strength = max(
            _role_strength_from_facts(context.facts, "low_rhythmic_drum_loop"),
            _role_strength_from_facts(context.facts, "percussive_drum_loop"),
            _direct_body_role_strength_from_facts(context.facts, "low_rhythmic_drum_loop"),
            _direct_body_role_strength_from_facts(context.facts, "percussive_drum_loop"),
        )
        if (
            early_best_drum_loop is not None
            and early_best_drum_loop[0] <= 4.0
            and early_drum_loop_strength >= 0.55
            and not (context.shape == "bass_phrase" and context.bass_loop_strength >= early_drum_loop_strength + 0.12)
        ):
            return self._redirect_from_raw(
                raw,
                "Drums/Drum Loops/Loops",
                "positive drum-loop claim beat later bass/voice rescue branches",
            )

        if (
            raw.final_top == "Instruments"
            and _path_has_any(context.raw_path, ("bass loops",))
            and context.shape == "bass_phrase"
            and context.bass_loop_strength >= 0.70
        ):
            return raw

        if (
            raw.final_top == "FX"
            and "human and voice" in context.raw_path
            and context.bass_loop_strength >= 0.86
            and context.shape == "bass_phrase"
            and context.shape_conf >= 0.82
            and max(context.full_voice_strength, context.direct_voice_strength) < 0.92
        ):
            return self._redirect_from_raw(
                raw,
                "Instruments/Bass/Bass Loops",
                "bass-loop claim blocked weak Human/Voice false positive",
            )

        # Pattern 2 Fix: Unify voice threshold to 0.75
        if (
            raw.final_top == "FX"
            and "human and voice" in context.raw_path
            and max(context.full_voice_strength, context.direct_voice_strength) >= 0.75
        ):
            direct_pitched_strength = max(
                _direct_body_role_strength_from_facts(context.facts, "pitched_music_phrase"),
                _direct_body_role_strength_from_facts(context.facts, "pitched_music_loop"),
            )
            vocal_phrase_strength = max(
                _role_strength_from_facts(context.facts, "vocal_music_phrase"),
                _direct_body_role_strength_from_facts(context.facts, "vocal_music_phrase"),
            )
            if context.role not in PROTECTED_HIT_ROLES and not (
                direct_pitched_strength >= 0.84 and vocal_phrase_strength < 0.55
            ):
                return raw

        # Pattern 2 Fix: Unify voice threshold to 0.75
        if (
            context.direct_voice_strength >= 0.75
            and raw.final_top == "Drums"
            and _path_has_any(context.raw_path, ("percussion", "rim", "clap", "snare", "tom", "cymbal"))
        ):
            return self._redirect_from_raw(
                raw,
                "FX/Human and Voice FX",
                "direct/body voice evidence beat tail-heavy drum one-shot reading",
            )
        return None
