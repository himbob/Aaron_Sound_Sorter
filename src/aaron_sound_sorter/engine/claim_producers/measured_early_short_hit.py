# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Short-hit safety guard for measured early adjudication."""

from __future__ import annotations

from aaron_sound_sorter.engine.claim_producers.measured_early_types import (
    DRUM_HIT_EXCLUDES,
    DRUM_HIT_FRAGMENTS,
    VOICE_FRAGMENTS,
    EarlyAdjudicationContext,
)
from aaron_sound_sorter.engine.decision_helpers import (
    _direct_body_role_strength_from_facts,
    _feature_number_from_facts,
    _is_measured_loop_phrase_context,
    _norm_path,
    _path_has_any,
    _role_strength_from_facts,
    _shape_metric_from_facts,
)
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class ShortHitGuardMixin:
    """Keep short hit-shaped files out of unsafe loop or instrument rescues."""

    def _adjudicate_short_hit_guard(self, context: EarlyAdjudicationContext) -> ConsensusClaim | None:
        """Apply the legacy short-hit demotion guard without changing thresholds."""
        if not self._is_short_hit_guard_active(context):
            return None
        if not self._raw_is_suspicious_short_hit_target(context):
            return None
        if context.raw.final_top == "Drums" and _path_has_any(context.raw_path, ("kick", "kick drums")):
            return context.raw

        voice_decision = self._maybe_keep_short_hit_voice(context)
        if voice_decision is not None:
            return voice_decision
        return self._maybe_route_short_hit_to_drum(context)

    @staticmethod
    def _is_short_hit_guard_active(context: EarlyAdjudicationContext) -> bool:
        return bool(
            context.shape in {"bass_phrase", "hit_with_tail"}
            and context.shape_conf >= 0.72
            and not _is_measured_loop_phrase_context(context.shape, context.role, context.measured_role)
        )

    @staticmethod
    def _raw_is_suspicious_short_hit_target(context: EarlyAdjudicationContext) -> bool:
        raw_is_suspicious_one_shot = (
            context.shape == "hit_with_tail"
            or _path_has_any(context.raw_path, ("one shots", "one shot"))
            or (
                context.raw.final_top == "Instruments"
                and _path_has_any(context.raw_path, ("instrument loops", "mixed musical loops"))
            )
        )
        return bool(raw_is_suspicious_one_shot and not _path_has_any(context.raw_path, ("drum loops", "looped")))

    def _maybe_keep_short_hit_voice(self, context: EarlyAdjudicationContext) -> ConsensusClaim | None:
        short_hit_voice_candidate = self._best_candidate(
            context.raw,
            include_top={"FX", "Instruments"},
            include_fragments=VOICE_FRAGMENTS,
        )
        short_hit_voice_strength = max(
            _role_strength_from_facts(context.facts, "voiced_one_shot"),
            _direct_body_role_strength_from_facts(context.facts, "voiced_one_shot"),
            _role_strength_from_facts(context.facts, "vocal_music_phrase"),
            _direct_body_role_strength_from_facts(context.facts, "vocal_music_phrase"),
        )
        if short_hit_voice_candidate is None or short_hit_voice_strength < 0.70:
            return None

        voice_score, _voice_path = short_hit_voice_candidate
        if self._bright_single_transient_blocks_voice_rescue(context):
            return None
        competing_drum_for_voice = self._best_candidate(
            context.raw,
            include_top={"Drums"},
            include_fragments=DRUM_HIT_FRAGMENTS,
            exclude_fragments=DRUM_HIT_EXCLUDES,
        )
        drum_dominates_voice = bool(
            competing_drum_for_voice is not None and voice_score > competing_drum_for_voice[0] + 18.0
        )
        if drum_dominates_voice:
            return None
        if voice_score <= context.raw_score + 25.0 or context.raw_score >= 999.0:
            return self._redirect_from_raw(
                context.raw,
                "FX/Human and Voice FX",
                "positive voice one-shot claim blocked short-hit percussion demotion",
            )
        return self._review_from_raw(
            context.raw,
            context.eligibility,
            "voice one-shot claim blocked short-hit percussion demotion but candidate support was weak",
        )

    def _maybe_route_short_hit_to_drum(self, context: EarlyAdjudicationContext) -> ConsensusClaim | None:
        best_drum_hit = self._best_candidate(
            context.raw,
            include_top={"Drums"},
            include_fragments=DRUM_HIT_FRAGMENTS,
            exclude_fragments=DRUM_HIT_EXCLUDES,
        )
        best_kick_hit = self._best_candidate(
            context.raw,
            include_top={"Drums"},
            include_fragments=("kick",),
            exclude_fragments=DRUM_HIT_EXCLUDES,
        )
        best_inst_hit_score = self._best_candidate_score(
            context.raw,
            include_top={"Instruments"},
            include_fragments=(),
        )
        compare_score = best_inst_hit_score if best_inst_hit_score is not None else context.raw_score
        if best_kick_hit is not None and self._bass_shaped_one_shot_has_kick_evidence(
            context=context,
            kick_score=best_kick_hit[0],
            compare_score=compare_score,
        ):
            return self._redirect_from_raw(
                context.raw,
                "Drums/Kick Drums/Generic Kick/One Shots",
                "short-hit guard: low/bass-shaped one-shot had close kick candidate evidence",
            )
        if best_drum_hit is not None:
            drum_score, drum_path = best_drum_hit
            drum_margin = 10.0 if context.shape == "hit_with_tail" else 2.5
            kick_margin = 6.0 if context.shape == "bass_phrase" else 3.0
            if drum_score <= compare_score + drum_margin:
                if best_kick_hit is not None and best_kick_hit[0] <= min(drum_score + 3.0, compare_score + kick_margin):
                    return self._redirect_from_raw(
                        context.raw,
                        "Drums/Kick Drums/Generic Kick/One Shots",
                        "short-hit guard: low/bass-shaped one-shot had close kick candidate evidence",
                    )
                if _path_has_any(_norm_path(drum_path), ("snare", "clap", "rim", "hat", "cymbal", "tom")):
                    return self._redirect_from_raw(
                        context.raw,
                        drum_path,
                        "short-hit guard: measured hit shape had close drum one-shot candidate evidence",
                    )
                return self._redirect_from_raw(
                    context.raw,
                    "Drums/Percussion/Generic Percussion/One Shots",
                    "short-hit guard: measured hit shape had close percussion candidate evidence",
                )
        raw_path = _norm_path(context.raw.folder_path)
        if context.raw.final_top == "Instruments" and _path_has_any(raw_path, ("bass", "808", "sub bass")):
            return None
        if context.raw.final_top == "Instruments":
            return self._review_from_raw(
                context.raw,
                context.eligibility,
                "short-hit guard: measured hit shape prevented unsafe Instrument Loop/Bass one-shot rescue without enough drum evidence",
            )
        if context.raw.final_top == "FX" and self._bright_single_transient_blocks_voice_rescue(context):
            return self._review_from_raw(
                context.raw,
                context.eligibility,
                "short-hit guard: bright single transient blocked unsafe voice/FX certainty without enough drum evidence",
            )
        return None

    @staticmethod
    def _bright_single_transient_blocks_voice_rescue(context: EarlyAdjudicationContext) -> bool:
        """Return True when a voice-like formant read is probably a bright hit."""
        if context.shape not in {"hit_with_tail", "single_hit"} or context.shape_conf < 0.70:
            return False
        raw_path = _norm_path(context.raw.folder_path)
        if _path_has_any(raw_path, ("voice", "vocal", "human and voice", "spoken", "choir")):
            return False
        high_event = max(
            _shape_metric_from_facts(context.facts, "high_event_ratio"),
            _feature_number_from_facts(context.facts, "high_total"),
        )
        non_event_tonal = _shape_metric_from_facts(context.facts, "non_event_tonal_ratio")
        event_count = max(
            _shape_metric_from_facts(context.facts, "onset_count"),
            _feature_number_from_facts(context.facts, "event_count_estimate"),
        )
        return bool(high_event >= 0.60 and non_event_tonal <= 0.20 and (event_count <= 2.0 or event_count == 0.0))

    @staticmethod
    def _bass_shaped_one_shot_has_kick_evidence(
        *,
        context: EarlyAdjudicationContext,
        kick_score: float,
        compare_score: float,
    ) -> bool:
        """Return True for very short low hits where Kick is the safer parent."""
        if context.shape != "bass_phrase" or context.shape_conf < 0.88:
            return False
        low_total = _feature_number_from_facts(context.facts, "low_total")
        high_total = _feature_number_from_facts(context.facts, "high_total")
        attack = _shape_metric_from_facts(context.facts, "attack_rise_time_norm")
        event_count = max(
            _shape_metric_from_facts(context.facts, "onset_count"),
            _feature_number_from_facts(context.facts, "event_count_estimate"),
        )
        tail = _feature_number_from_facts(context.facts, "tail_energy_ratio")
        return bool(
            low_total >= 0.90
            and high_total <= 0.08
            and attack <= 0.05
            and (event_count <= 3.0 or event_count == 0.0)
            and tail <= 0.12
            and kick_score <= compare_score + 16.0
        )
