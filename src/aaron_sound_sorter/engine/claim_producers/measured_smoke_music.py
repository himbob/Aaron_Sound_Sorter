# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Bass and pitched-music smoke-stability rescues."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.decision_helpers import (
    _direct_body_role_strength_from_facts,
    _is_concrete_fx_path,
    _norm_path,
    _path_has_any,
    _role_strength_from_facts,
)
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import ConsensusClaim

PITCHED_ROLES = {
    "pitched_music_loop",
    "pitched_music_phrase",
    "pitched_reed_or_instrument_loop",
    "pitched_reed_or_instrument_phrase",
    "mixed_music_loop",
    "clean_sustained_tonal_instrument_loop",
}
PITCHED_SHAPES = {"pitched_phrase", "bass_phrase", "sustained_pad"}


class SmokeMusicRescueMixin:
    """Recover clear measured musical structures from review-prone paths."""

    def _maybe_rescue_bass_smoke(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        broad_path: str,
        role: str,
        shape: str,
        shape_conf: float,
    ) -> ConsensusClaim | None:
        if (
            role == "bass_loop"
            and shape == "bass_phrase"
            and shape_conf >= 0.86
            and broad_path
            and "bass" in broad_path
        ):
            raw_path = _norm_path(raw.folder_path)
            if raw.final_top == "Instruments" and _path_has_any(
                raw_path, ("instrument loops", "mixed musical loops", "multi instrument")
            ):
                return None
            return self._broaden_from_raw(
                raw,
                eligibility,
                "measured bass-loop structure beat unsafe non-bass family",
            )
        return None

    def _maybe_rescue_pitched_music_smoke(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        raw_path: str,
        role: str,
        measured_role: str,
        shape: str,
        shape_conf: float,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        if not (role in PITCHED_ROLES or (shape in PITCHED_SHAPES and shape_conf >= 0.72)):
            return None
        if raw.final_top == "Instruments":
            return self._maybe_broaden_instrument_one_shot(raw, raw_path, shape, shape_conf)
        if raw.final_top == "Drums" and _path_has_any(raw_path, ("drum loop", "drum loops", "percussion")):
            return None
        if self._should_keep_human_voice_fx(raw, raw_path, role, measured_role, shape, facts):
            return None
        if _is_concrete_fx_path(raw_path):
            transition_like_fx = _path_has_any(
                raw_path,
                ("riser", "build", "sweep", "whoosh", "drop", "downlifter", "transition"),
            )
            stable_instrument_role = role in {
                "pitched_reed_or_instrument_loop",
                "pitched_reed_or_instrument_phrase",
                "clean_sustained_tonal_instrument_loop",
            }
            if not stable_instrument_role and not (transition_like_fx and shape in PITCHED_SHAPES):
                return raw
        return self._redirect_from_raw(
            raw,
            "Instruments/Instrument Loops/Loops",
            "measured pitched musical structure kept curated smoke example out of review",
        )

    def _maybe_broaden_instrument_one_shot(
        self,
        raw: ConsensusClaim,
        raw_path: str,
        shape: str,
        shape_conf: float,
    ) -> ConsensusClaim | None:
        if (
            shape in PITCHED_SHAPES
            and shape_conf >= 0.72
            and _path_has_any(raw_path, ("one shots", "one shot"))
            and not _path_has_any(raw_path, ("bass", "808", "drum", "percussion", "voice", "vocal", "human and voice"))
        ):
            broad_loop_path = "Instruments/Instrument Loops/Loops"
            if _path_has_any(
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
            ):
                broad_loop_path = "Instruments/Brass and Woodwinds/Loops"
            return self._redirect_from_raw(
                raw,
                broad_loop_path,
                "measured pitched/bass phrase shape prevented exact one-shot leaf from replacing broad loop bucket",
            )
        return None

    @staticmethod
    def _should_keep_human_voice_fx(
        raw: ConsensusClaim,
        raw_path: str,
        role: str,
        measured_role: str,
        shape: str,
        facts: SharedAudioFacts | None,
    ) -> bool:
        if not (raw.final_top == "FX" and _path_has_any(raw_path, ("human and voice", "voice", "vocal"))):
            return False
        direct_pitched_strength = max(
            _direct_body_role_strength_from_facts(facts, "pitched_music_phrase"),
            _direct_body_role_strength_from_facts(facts, "pitched_music_loop"),
        )
        measured_vocal_strength = max(
            _role_strength_from_facts(facts, "vocal_music_phrase"),
            _direct_body_role_strength_from_facts(facts, "vocal_music_phrase"),
        )
        return bool(
            role not in {"protected_percussive_one_shot", "percussive_one_shot", "low_kick_like_hit"}
            and measured_role
            not in {
                "pitched_music_loop",
                "pitched_music_phrase",
                "pitched_reed_or_instrument_loop",
                "pitched_reed_or_instrument_phrase",
            }
            and shape not in PITCHED_SHAPES
            and not (direct_pitched_strength >= 0.84 and measured_vocal_strength < 0.55)
        )
