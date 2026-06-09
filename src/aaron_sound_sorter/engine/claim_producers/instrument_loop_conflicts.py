# SOURCE-NAME BLINDNESS INVARIANT:
# This claim producer may inspect voter output and measured audio facts only.
# It must never inspect producer filenames, source folder names, ZIP member
# names, path tokens, or sample-pack labels as classification evidence.
"""Tonal and ambiguous-FX instrument-loop safety claims."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.decision_helpers import (
    _direct_body_role_strength_from_facts,
    _feature_number_from_facts,
    _measured_role_from_facts,
    _norm_path,
    _path_has_any,
    _role_strength_from_facts,
    _shape_confidence_from_facts,
    _shape_metric_from_facts,
)
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_folder_path


class InstrumentLoopConflictMixin:
    """Handle tonal conflict loops and ambiguous FX false positives."""

    def tonal_conflict_loop_claim(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        """Resolve high-tonal pitched-percussion conflicts to broad Instruments."""
        if eligibility.role_name != "pitched_percussion_conflict_loop":
            return None
        if raw.final_top == "Instruments" and _path_has_any(
            _norm_path(raw.folder_path), ("instrument loops", "mixed musical loops")
        ):
            return None
        if not self._has_high_tonal_non_drum_loop_body(facts):
            return None
        # Pattern 1 Fix: Honor drum detection—don't override if measured_role is drum-like
        measured_role = _measured_role_from_facts(facts)
        if measured_role in {"low_rhythmic_drum_loop", "percussive_drum_loop"}:
            measured_strength = max(
                _role_strength_from_facts(facts, measured_role),
                _direct_body_role_strength_from_facts(facts, measured_role),
            )
            if measured_strength >= 0.70:
                return None  # Honor drum detection; don't override
        broad_eligibility = EligibilityDecision(
            role_name="pitched_music_loop",
            confidence=max(float(eligibility.confidence or 0.0), 0.78),
            allowed_top_families=("Instruments", "_TO_REVIEW"),
            blocked_path_fragments=(
                "FX",
                "Human",
                "Voice",
                "Animals",
                "Dog",
                "Bird",
                "Cat",
                "Coins",
                "Machines",
                "Motor",
            ),
            broad_folder_path="Instruments/Instrument Loops/Loops",
            reason="high-tonal non-drum loop body resolved conservative pitched-percussion conflict",
        )
        return self._broaden_from_raw(
            raw,
            broad_eligibility,
            "pitched conflict loop had stronger sustained tonal instrument evidence than drum or FX evidence",
        )

    @staticmethod
    def _has_high_tonal_non_drum_loop_body(facts: SharedAudioFacts | None) -> bool:
        """Return True for the tonal non-drum body used by the legacy guard."""
        shape_confidence = _shape_confidence_from_facts(facts)
        pitched_ratio = _shape_metric_from_facts(facts, "pitched_event_ratio")
        percussive_ratio = _shape_metric_from_facts(facts, "percussive_event_ratio")
        drumlike_ratio = _shape_metric_from_facts(facts, "drumlike_frame_ratio")
        pitch_confidence = _shape_metric_from_facts(facts, "pitch_confidence")
        sustain_ratio = _shape_metric_from_facts(facts, "sustain_ratio")
        return (
            shape_confidence >= 0.78
            and pitched_ratio >= 0.85
            and percussive_ratio <= 0.15
            and drumlike_ratio <= 0.10
            and pitch_confidence >= 0.70
            and sustain_ratio >= 0.45
        )

    def ambiguous_fx_music_loop_claim(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        """Move measured music loops out of ambiguous FX leaf false positives."""
        role_name = str(eligibility.role_name or "")
        if role_name not in {"mixed_music_loop", "pitched_music_loop", "pitched_music_phrase"}:
            return None
        if raw.final_top != "FX":
            return None
        raw_path = _norm_path(raw.folder_path)
        if not self._raw_fx_leaf_is_ambiguous_music_false_positive(raw_path):
            return None
        raw_score = float(raw.raw_candidate_score or 9999.0)
        if raw_score < 5.0 and _path_has_any(raw_path, ("siren", "alarm", "beep", "glitch", "stutter")):
            return None
        if (
            raw_score <= 8.0
            and _path_has_any(raw_path, ("siren", "alarm", "beep", "glitch", "stutter"))
            and self._full_brain_top_is_concrete_fx(facts)
        ):
            return None
        if float(eligibility.confidence or 0.0) < 0.60:
            return None
        if not self._has_music_loop_structure(facts):
            return None
        return claim_from_folder_path(
            folder_path="Instruments/Instrument Loops/Loops",
            source="ambiguous_fx_music_loop_broad_bucket",
            reason=(
                "measured pitched/mixed loop structure beat an ambiguous FX leaf; "
                f"role={role_name}, confidence={float(eligibility.confidence or 0.0):.2f}; "
                f"raw={raw.folder_path}"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=self._candidate_score_for_broad_target(raw, "Instruments/Instrument Loops/Loops"),
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner,
            can_override=True,
            strength=max(0.78, float(eligibility.confidence or 0.0)),
            is_real_candidate=False,
        )

    @staticmethod
    def _raw_fx_leaf_is_ambiguous_music_false_positive(raw_path: str) -> bool:
        """Return True for FX leaves that often steal musical loops."""
        if _path_has_any(raw_path, ("riser", "build", "downlifter", "drop", "whoosh", "sweep", "impact", "boom")):
            return False
        return _path_has_any(
            raw_path,
            (
                "human",
                "voice",
                "vocal",
                "crowd",
                "spoken",
                "keys coins",
                "coins",
                "small objects",
                "animals",
                "dog",
                "cat",
                "bird",
                "siren",
                "alarm",
                "beep",
                "machine",
                "machines",
                "motor",
                "engine",
                "ambience",
                "water",
                "wind",
            ),
        )

    @staticmethod
    def _full_brain_top_is_concrete_fx(facts: SharedAudioFacts | None) -> bool:
        """Return True when the full brain's top lane still sees concrete FX."""
        if facts is None or not isinstance(facts.evidence, dict):
            return False
        result = facts.evidence.get("full_brain_vote_result")
        if not isinstance(result, dict):
            return False
        guesses = result.get("top_guesses")
        if not isinstance(guesses, list) or not guesses or not isinstance(guesses[0], dict):
            return False
        path = _norm_path(str(guesses[0].get("folder_path") or guesses[0].get("label") or ""))
        return path.startswith("fx/") and _path_has_any(
            path,
            ("siren", "alarm", "beep", "glitch", "stutter", "machine", "motor", "engine"),
        )

    @staticmethod
    def _has_music_loop_structure(facts: SharedAudioFacts | None) -> bool:
        """Return True for measured pitched or mixed-music loop structure."""
        shape_confidence = _shape_confidence_from_facts(facts)
        pitched_ratio = max(
            _shape_metric_from_facts(facts, "pitched_event_ratio"),
            _feature_number_from_facts(facts, "librosa_tonal_confidence"),
        )
        percussive_ratio = max(
            _shape_metric_from_facts(facts, "percussive_event_ratio"),
            _feature_number_from_facts(facts, "librosa_percussive_confidence"),
        )
        pitch_confidence = max(
            _shape_metric_from_facts(facts, "pitch_confidence"),
            _feature_number_from_facts(facts, "librosa_tonal_confidence"),
        )
        onset_count = max(
            _shape_metric_from_facts(facts, "onset_count"),
            _feature_number_from_facts(facts, "librosa_onset_event_count"),
        )
        loop_confidence = _feature_number_from_facts(facts, "librosa_loop_confidence")
        third_party_music_loop = bool(
            loop_confidence >= 0.62 and onset_count >= 2.0 and pitch_confidence >= 0.35 and percussive_ratio <= 0.60
        )
        return (
            third_party_music_loop
            or shape_confidence >= 0.68
            and onset_count >= 3.0
            and (pitched_ratio >= 0.50 or pitch_confidence >= 0.30)
            and (pitched_ratio + percussive_ratio) >= 0.80
        )
