# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Human/Voice rescue and review guards for early adjudication."""

from __future__ import annotations

from aaron_sound_sorter.engine.claim_producers.measured_early_types import (
    VOICE_ROLE_NAMES,
    EarlyAdjudicationContext,
    VoiceCandidateEvidence,
)
from aaron_sound_sorter.engine.decision_helpers import (
    _direct_body_role_strength_from_facts,
    _feature_number_from_facts,
    _has_drum_loop_structure_support,
    _norm_path,
    _path_has_any,
    _role_strength_from_facts,
)
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


def _physics_subpanel_score(facts, name: str) -> float:
    """Read a source-name-blind physics subpanel score from shared facts."""
    if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
        return 0.0
    block = facts.evidence.get("physics_subpanels", {})
    if not isinstance(block, dict):
        return 0.0
    flat = block.get("flat", {})
    if not isinstance(flat, dict):
        return 0.0
    try:
        return float(flat.get(name, 0.0) or 0.0)
    except Exception:
        return 0.0


def _strong_single_hit_human_voice_panel(context: EarlyAdjudicationContext) -> bool:
    """True for short voiced human shots that candidate distance misreads."""
    if context.role not in VOICE_ROLE_NAMES:
        return False
    if context.shape not in {"single_hit", "hit_with_tail", "short_hit", "vocal_phrase"}:
        return False
    voice_score = _physics_subpanel_score(context.facts, "voice_score")
    human_spoken = _physics_subpanel_score(context.facts, "human_spoken_voice_score")
    human_breath = _physics_subpanel_score(context.facts, "human_breath_mouth_score")
    choir_score = _physics_subpanel_score(context.facts, "voice_choir_score")
    formant_score = _physics_subpanel_score(context.facts, "fx_formant_score")
    drum_hit = _physics_subpanel_score(context.facts, "drum_hit_score")
    drum_loop = _physics_subpanel_score(context.facts, "drum_loop_source_score")
    direct_human = max(human_spoken, human_breath)
    broad_voice = max(voice_score, direct_human, choir_score, formant_score)
    return bool(
        context.shape_conf >= 0.72
        and broad_voice >= 0.74
        and direct_human >= 0.70
        and drum_hit <= 0.48
        and drum_loop <= 0.30
        and not bool(getattr(context.facts, "is_loop_like", False))
    )


class VoiceClaimGuardMixin:
    """Apply Human/Voice rescue guards in legacy branch order."""

    def _adjudicate_voice_claims(
        self,
        context: EarlyAdjudicationContext,
        evidence: VoiceCandidateEvidence,
    ) -> ConsensusClaim | None:
        """Run voice-specific early adjudication branches in legacy order."""
        decision = self._maybe_review_short_single_event_voice(context, evidence)
        if decision is not None:
            return decision
        decision = self._maybe_redirect_weak_voice_false_positive(context, evidence)
        if decision is not None:
            return decision
        decision = self._maybe_redirect_decisive_voice_candidate(context, evidence)
        if decision is not None:
            return decision
        decision = self._maybe_redirect_direct_body_voice(context, evidence)
        if decision is not None:
            return decision
        return self._maybe_redirect_positive_voice_role(context, evidence)

    def _maybe_review_short_single_event_voice(
        self,
        context: EarlyAdjudicationContext,
        evidence: VoiceCandidateEvidence,
    ) -> ConsensusClaim | None:
        duration_sec = _feature_number_from_facts(context.facts, "duration_sec")
        event_count = _feature_number_from_facts(context.facts, "event_count_estimate")
        short_single_event = bool(
            duration_sec > 0.0
            and duration_sec <= 0.35
            and event_count <= 2.0
            and context.shape in {"hit_with_tail", "single_hit", "short_hit"}
        )
        if (
            short_single_event
            and context.raw.final_top == "FX"
            and "human and voice" in context.raw_path
            and not evidence.positive_voice_claim
            and not evidence.direct_one_shot_voice_claim
        ):
            return self._review_from_raw(
                context.raw,
                context.eligibility,
                "short single-event voice-like reading lacked a positive Human/Voice claim",
            )
        return None

    def _maybe_redirect_weak_voice_false_positive(
        self,
        context: EarlyAdjudicationContext,
        evidence: VoiceCandidateEvidence,
    ) -> ConsensusClaim | None:
        instrument_phrase_strength = max(
            _role_strength_from_facts(context.facts, "pitched_music_phrase"),
            _role_strength_from_facts(context.facts, "pitched_music_loop"),
            _direct_body_role_strength_from_facts(context.facts, "pitched_music_phrase"),
            _direct_body_role_strength_from_facts(context.facts, "pitched_music_loop"),
        )
        weak_voice_false_positive = bool(
            context.raw.final_top == "FX"
            and "human and voice" in context.raw_path
            and instrument_phrase_strength >= 0.84
            and not evidence.positive_voice_claim
            and not evidence.direct_one_shot_voice_claim
        )
        if not weak_voice_false_positive:
            return None
        best_instrument = self._best_candidate(
            context.raw,
            include_top={"Instruments"},
            include_fragments=(
                "instrument loops",
                "brass",
                "woodwind",
                "sax",
                "trumpet",
                "piano",
                "keys",
                "rhodes",
                "guitar",
                "strings",
                "synth",
            ),
        )
        if best_instrument is not None:
            _instrument_score, instrument_path = best_instrument
            safe_path = instrument_path
            if not _path_has_any(
                _norm_path(safe_path),
                (
                    "brass",
                    "woodwind",
                    "sax",
                    "instrument loops",
                    "piano",
                    "keys",
                    "rhodes",
                    "guitar",
                    "strings",
                    "synth",
                ),
            ):
                safe_path = "Instruments/Instrument Loops/Loops"
            return self._redirect_from_raw(
                context.raw,
                safe_path,
                "stable pitched instrument claim blocked weak Human/Voice false positive",
            )
        return self._redirect_from_raw(
            context.raw,
            "Instruments/Instrument Loops/Loops",
            "stable pitched instrument claim blocked weak Human/Voice false positive",
        )

    def _maybe_redirect_decisive_voice_candidate(
        self,
        context: EarlyAdjudicationContext,
        evidence: VoiceCandidateEvidence,
    ) -> ConsensusClaim | None:
        best_drum_for_voice_check = evidence.best_drum_loop[0] if evidence.best_drum_loop is not None else None
        voice_beats_raw = evidence.best_voice is not None and evidence.best_voice <= context.raw_score - 8.0
        voice_beats_drum = bool(
            evidence.best_voice is not None
            and (best_drum_for_voice_check is None or evidence.best_voice <= best_drum_for_voice_check - 8.0)
        )
        voice_beats_nonvoice = bool(
            evidence.best_voice is not None
            and (evidence.best_nonvoice is None or evidence.best_voice <= evidence.best_nonvoice[0] - 6.0)
        )
        decisive_voice_candidate = bool(
            evidence.positive_voice_claim and voice_beats_raw and voice_beats_drum and voice_beats_nonvoice
        )
        if decisive_voice_candidate and context.raw.final_top in {"Instruments", "Drums", "FX"}:
            return self._redirect_from_raw(
                context.raw,
                "FX/Human and Voice FX",
                "decisive Human/Voice candidate evidence beat false drum/instrument loop rescue",
            )
        return None

    def _maybe_redirect_direct_body_voice(
        self,
        context: EarlyAdjudicationContext,
        evidence: VoiceCandidateEvidence,
    ) -> ConsensusClaim | None:
        strong_single_hit_voice_panel = _strong_single_hit_human_voice_panel(context)
        if (
            max(evidence.measured_voice_strength, evidence.direct_voice_strength) < 0.72
            and not strong_single_hit_voice_panel
        ):
            return None
        raw_is_drum_or_generic = bool(
            context.raw.final_top == "Drums"
            or _path_has_any(context.raw_path, ("percussion", "rim", "clap", "instrument loops", "mixed musical loops"))
        )
        strong_measured_vocal_shape = bool(
            context.shape == "vocal_phrase"
            and context.shape_conf >= 0.88
            and max(evidence.measured_voice_strength, evidence.direct_voice_strength) >= 0.84
            and (
                context.role in VOICE_ROLE_NAMES
                or context.measured_role in VOICE_ROLE_NAMES
                or evidence.positive_voice_claim
            )
        )
        strong_direct_vocal_hit = bool(
            context.shape in {"hit_with_tail", "single_hit"}
            and evidence.direct_voice_strength >= 0.90
            and not _has_drum_loop_structure_support(
                context.role,
                context.measured_role,
                context.shape,
                context.shape_conf,
            )
        )
        if (raw_is_drum_or_generic or strong_single_hit_voice_panel) and (
            evidence.positive_voice_claim
            or evidence.direct_one_shot_voice_claim
            or strong_measured_vocal_shape
            or strong_direct_vocal_hit
            or strong_single_hit_voice_panel
        ):
            return self._redirect_from_raw(
                context.raw,
                "FX/Human and Voice FX",
                "positive measured Human/Voice claim beat tail-heavy percussive or generic-loop reading",
            )
        return None

    def _maybe_redirect_positive_voice_role(
        self,
        context: EarlyAdjudicationContext,
        evidence: VoiceCandidateEvidence,
    ) -> ConsensusClaim | None:
        if (
            context.role in VOICE_ROLE_NAMES
            and evidence.direct_voice_strength >= 0.85
            and context.shape == "vocal_phrase"
            and context.shape_conf >= 0.88
            and (evidence.positive_voice_claim or evidence.direct_one_shot_voice_claim)
        ):
            return self._redirect_from_raw(
                context.raw,
                "FX/Human and Voice FX",
                "positive Human/Voice claim beat tail-heavy non-voice candidate set",
            )
        return None
