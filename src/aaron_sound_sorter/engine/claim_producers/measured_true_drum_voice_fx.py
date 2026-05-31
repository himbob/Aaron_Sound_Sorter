# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Drum, voice, and tonal-FX rescues for measured true-bucket policy."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.decision_helpers import (
    _has_drum_loop_structure_support,
    _norm_path,
    _path_has_any,
    _shape_confidence_from_facts,
    _shape_vote_from_facts,
)
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class MeasuredTrueDrumVoiceFxMixin:
    """Handle drum rescues, Human/Voice protection, and tonal FX movement."""

    def _maybe_keep_pitched_percussion_as_instrument(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        raw_path: str,
        measured_role: str,
    ) -> ConsensusClaim | None:
        if (
            eligibility.role_name == "pitched_percussion_loop"
            and measured_role in {"pitched_music_loop", "pitched_music_phrase", "vocal_music_phrase", "bass_loop"}
            and raw.final_top == "Instruments"
            and _path_has_any(raw_path, ("instrument loops", "mixed musical loops", "bass"))
        ):
            return raw
        return None

    def _maybe_rescue_drum_loop(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        raw_path: str,
        measured_role: str,
        shape: str,
        shape_conf: float,
    ) -> ConsensusClaim | None:
        drum_rescue_allowed = _has_drum_loop_structure_support(
            eligibility.role_name,
            measured_role,
            shape,
            shape_conf,
        )
        if not drum_rescue_allowed:
            return None
        best_drum_loop = self._best_candidate(
            raw,
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
        if best_drum_loop is None:
            return None
        drum_score, drum_path = best_drum_loop
        drum_path_low = _norm_path(drum_path)
        drum_rescue_reason = (
            "percussion-loop true-bucket rescue"
            if _path_has_any(
                drum_path_low,
                ("percussion", "conga", "bongo", "tabla", "triangle", "wood block", "wood blocks", "metallic"),
            )
            else "kick/drum-loop true-bucket rescue"
        )
        raw_is_wrong_loop = bool(
            raw.final_top == "Instruments"
            and _path_has_any(raw_path, ("bass loops", "instrument loops", "mixed musical loops"))
        )
        raw_is_human_voice_fx = raw.final_top == "FX" and "human and voice" in raw_path
        best_wrong_instrument = self._best_candidate_score(raw, include_top={"Instruments"}, include_fragments=())
        best_voice = self._best_candidate_score(
            raw,
            include_top={"FX", "Instruments"},
            include_fragments=("human and voice", "voice", "vocal", "vox", "choir", "breath", "crowd"),
        )
        if raw_is_wrong_loop and (best_wrong_instrument is None or drum_score <= best_wrong_instrument + 5.0):
            return self._redirect_from_raw(
                raw,
                "Drums/Drum Loops/Loops",
                f"{drum_rescue_reason}: close drum/percussion candidate beat unsafe instrument-loop family",
            )
        if raw_is_human_voice_fx and (best_voice is None or drum_score <= best_voice + 4.0):
            return self._redirect_from_raw(
                raw,
                "Drums/Drum Loops/Loops",
                f"{drum_rescue_reason}: close drum/percussion candidate beat unsafe Human/Voice FX family",
            )
        return None

    @staticmethod
    def _maybe_keep_measured_human_voice(
        raw: ConsensusClaim,
        raw_path: str,
        eligibility: EligibilityDecision,
        measured_role: str,
    ) -> ConsensusClaim | None:
        if raw.final_top == "FX" and "human and voice" in raw_path:
            voice_roles = {"vocal_phrase", "vocal_one_shot", "voiced_one_shot", "vocal_music_phrase"}
            protected_hit_roles = {"protected_percussive_one_shot", "percussive_one_shot", "low_kick_like_hit"}
            if eligibility.role_name not in protected_hit_roles and (
                eligibility.role_name in voice_roles or measured_role in voice_roles
            ):
                return raw
        return None

    def _maybe_rescue_instrument_hit_from_fx(
        self,
        raw: ConsensusClaim,
    ) -> ConsensusClaim | None:
        if raw.final_top != "FX":
            return None
        best_inst_hit = self._best_candidate(
            raw,
            include_top={"Instruments"},
            include_fragments=(
                "keys",
                "rhodes",
                "electric piano",
                "piano",
                "organ",
                "guitar",
                "strum",
                "chord",
                "pluck",
                "synth",
                "strings",
            ),
        )
        if best_inst_hit is None:
            return None
        inst_score, inst_path = best_inst_hit
        try:
            raw_score = float(raw.combined_rank_score or 9999.0)
        except Exception:
            raw_score = 9999.0
        if inst_score <= raw_score - 4.0:
            return self._redirect_from_raw(
                raw,
                inst_path,
                "instrument true-bucket rescue: instrument candidate evidence strongly beat unsafe FX one-shot family",
            )
        return None

    def _maybe_rescue_nonvoice_fx_from_human_voice(
        self,
        raw: ConsensusClaim,
        raw_path: str,
    ) -> ConsensusClaim | None:
        if not (raw.final_top == "FX" and "human and voice" in raw_path):
            return None
        best_nonvoice_fx = self._best_candidate(
            raw,
            include_top={"FX"},
            include_fragments=(
                "whoosh",
                "swoosh",
                "swish",
                "sweep",
                "seagull",
                "seaguls",
                "bird",
                "animal",
                "clank",
                "metallic",
                "bell",
                "chime",
                "hybrid designed",
                "designed noise",
                "riser",
                "build",
                "transition",
            ),
            exclude_fragments=("human and voice", "voice", "vocal", "crowd", "breath"),
        )
        best_voice = self._best_candidate_score(
            raw,
            include_top={"FX", "Instruments"},
            include_fragments=("human and voice", "voice", "vocal", "vox", "choir", "breath", "crowd"),
        )
        if best_nonvoice_fx is not None:
            nonvoice_score, nonvoice_path = best_nonvoice_fx
            if best_voice is None or nonvoice_score <= best_voice - 2.0:
                return self._redirect_from_raw(
                    raw,
                    nonvoice_path,
                    "non-voice FX true-bucket rescue: stronger non-voice FX candidate beat Human/Voice FX",
                )
        return None

    def _maybe_rescue_tonal_transition_fx(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        raw_path: str,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        if not (
            eligibility.role_name in {"pitched_music_loop", "pitched_music_phrase", "mixed_music_loop"}
            and raw.final_top == "Instruments"
            and _path_has_any(raw_path, ("instrument loops", "mixed musical loops"))
        ):
            return None
        shape = _shape_vote_from_facts(facts)
        shape_conf = _shape_confidence_from_facts(facts)
        if not (shape in {"transition_riser", "transition_drop"} and shape_conf >= 0.86):
            return None
        best_tonal_fx = self._best_candidate(
            raw,
            include_top={"FX"},
            include_fragments=(
                "bell",
                "bells",
                "chime",
                "chimes",
                "riser",
                "build",
                "sweep",
                "whoosh",
                "hybrid designed",
                "designed tonal",
                "metallic",
            ),
        )
        best_inst = self._best_candidate_score(raw, include_top={"Instruments"}, include_fragments=())
        if best_tonal_fx is not None:
            fx_score, fx_path = best_tonal_fx
            if best_inst is None or fx_score <= best_inst - 2.0:
                return self._redirect_from_raw(
                    raw,
                    fx_path,
                    "tonal transition FX true-bucket rescue: transition shape and stronger FX candidate beat generic Instrument Loop",
                )
        return None
