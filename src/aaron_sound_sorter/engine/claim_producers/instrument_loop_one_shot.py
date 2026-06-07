# SOURCE-NAME BLINDNESS INVARIANT:
# This claim producer may inspect voter output and measured audio facts only.
# It must never inspect producer filenames, source folder names, ZIP member
# names, path tokens, or sample-pack labels as classification evidence.
"""Instrument one-shot leaf to loop-bucket safety claims."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.decision_helpers import (
    LOOP_PHRASE_ROLES,
    _norm_path,
    _path_has_any,
    _shape_confidence_from_facts,
    _shape_metric_from_facts,
    _shape_vote_from_facts,
)
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class InstrumentOneShotLoopMixin:
    """Broaden instrument one-shot leaves when measured structure is a loop."""

    def one_shot_leaf_loop_claim(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        """Broaden instrument one-shot leaves when measured structure is a loop."""
        role_name = str(eligibility.role_name or "")
        if role_name not in LOOP_PHRASE_ROLES and role_name != "pitched_percussion_conflict_loop":
            return None
        # Pattern 4 Fix: Don't broaden drum one-shots—honor drum role
        if role_name in {"percussive_drum_one_shot", "struck_drum_one_shot", "protected_percussive_one_shot"}:
            return None
        raw_path = _norm_path(raw.folder_path)
        if not self._raw_path_is_loopish_one_shot_leaf(raw, raw_path):
            return None
        if not self._measured_loop_structure_is_strong(facts):
            return None
        broad_folder_path, broad_reason = self._broad_instrument_loop_target(raw_path, facts)
        broad_eligibility = EligibilityDecision(
            role_name=role_name,
            confidence=max(float(eligibility.confidence or 0.0), 0.88),
            allowed_top_families=("Instruments", "_TO_REVIEW"),
            blocked_path_fragments=(),
            broad_folder_path=broad_folder_path,
            reason=broad_reason,
        )
        return self._broaden_from_raw(
            raw,
            broad_eligibility,
            "instrument one-shot leaf conflicted with measured loop structure",
        )

    @staticmethod
    def _raw_path_is_loopish_one_shot_leaf(raw: ConsensusClaim, raw_path: str) -> bool:
        """Return True when raw is an Instrument one-shot leaf, not a safe broad loop."""
        if raw.final_top != "Instruments":
            return False
        if _path_has_any(raw_path, ("instrument loops", "mixed musical loops", "brass and woodwinds", "bass loops")):
            return False
        if not _path_has_any(raw_path, ("one shots", "one shot")):
            return False
        return not _path_has_any(raw_path, ("voice", "vocal", "vox", "choir", "breath", "mouth"))

    @staticmethod
    def _measured_loop_structure_is_strong(facts: SharedAudioFacts | None) -> bool:
        """Return True when measured shape supports broad loop placement."""
        shape_confidence = _shape_confidence_from_facts(facts)
        shape_name = _shape_vote_from_facts(facts)
        pitched_ratio = _shape_metric_from_facts(facts, "pitched_event_ratio")
        percussive_ratio = _shape_metric_from_facts(facts, "percussive_event_ratio")
        sustain_ratio = _shape_metric_from_facts(facts, "sustain_ratio")
        onset_count = _shape_metric_from_facts(facts, "onset_count")
        repeated_tonal_hit = (
            shape_name == "hit_with_tail"
            and shape_confidence >= 0.70
            and onset_count >= 2.8
            and pitched_ratio >= 0.80
            and sustain_ratio >= 0.70
            and percussive_ratio <= 0.12
        )
        if shape_confidence < 0.74 and not repeated_tonal_hit:
            return False
        return not (pitched_ratio < 0.55 and sustain_ratio < 0.45 and onset_count < 4.0 and not repeated_tonal_hit)

    @staticmethod
    def _broad_instrument_loop_target(
        raw_path: str,
        facts: SharedAudioFacts | None,
    ) -> tuple[str, str]:
        """Return the safest broad Instrument loop folder for a one-shot leaf."""
        shape_name = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        brass_or_reed_leaf = _path_has_any(
            raw_path,
            (
                "sax",
                "saxophone",
                "woodwind",
                "woodwinds",
                "brass",
                "trumpet",
                "trombone",
                "horn",
                "flute",
                "clarinet",
                "reed",
            ),
        )
        strong_phrase_shape = shape_name in {"pitched_phrase", "sustained_pad", "vocal_phrase", "bass_phrase"}
        if brass_or_reed_leaf and strong_phrase_shape and shape_confidence >= 0.82:
            return (
                "Instruments/Brass and Woodwinds/Loops",
                "measured loop/phrase structure was safer than a source-specific brass/woodwind one-shot leaf",
            )
        return (
            "Instruments/Instrument Loops/Loops",
            "measured loop/phrase structure was safer than a source-specific instrument one-shot leaf",
        )
