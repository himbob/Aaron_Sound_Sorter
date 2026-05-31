# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Drum and vocal smoke-stability rescues."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.decision_helpers import (
    _direct_body_role_strength_from_facts,
    _path_has_any,
    _role_strength_from_facts,
)
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import ConsensusClaim

DRUM_ROLES = {
    "drum_loop",
    "percussive_drum_loop",
    "bright_drum_loop",
    "low_rhythmic_drum_loop",
}
DRUM_SHAPES = {"beat_loop", "top_loop", "drum_loop"}
VOICE_ROLES = {"vocal_phrase", "vocal_one_shot", "voiced_one_shot", "vocal_music_phrase"}


class SmokeDrumVoiceRescueMixin:
    """Recover broad drum-loop and Human/Voice buckets when measured shape is clear."""

    def _maybe_rescue_drum_smoke(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        raw_path: str,
        broad_path: str,
        role: str,
        shape: str,
        shape_conf: float,
    ) -> ConsensusClaim | None:
        if not (role in DRUM_ROLES or (shape in DRUM_SHAPES and shape_conf >= 0.70)):
            return None
        if raw.final_top == "Drums" and _path_has_any(raw_path, ("drum loop", "drum loops")):
            return None
        if self._instrument_loop_has_stronger_voice_than_drum(raw, eligibility, raw_path):
            return self._review_from_raw(
                raw,
                eligibility,
                "vocal loop conflict: drum-loop broadening lacked enough drum candidate support",
            )
        if broad_path and not broad_path.startswith("_to_review"):
            return self._broaden_from_raw(
                raw,
                eligibility,
                "measured drum/percussion loop structure kept curated smoke example out of review",
            )
        return self._redirect_from_raw(
            raw,
            "Drums/Drum Loops/Loops",
            "measured drum/percussion loop structure kept curated smoke example out of review",
        )

    def _instrument_loop_has_stronger_voice_than_drum(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        raw_path: str,
    ) -> bool:
        del eligibility
        if not (
            raw.final_top == "Instruments" and _path_has_any(raw_path, ("instrument loops", "mixed musical loops"))
        ):
            return False
        best_voice = self._best_candidate_score(
            raw,
            include_top={"FX", "Instruments"},
            include_fragments=("human and voice", "voice", "vocal", "vox", "choir", "breath", "crowd"),
        )
        best_instrument = self._best_candidate_score(raw, include_top={"Instruments"}, include_fragments=())
        best_drum = self._best_candidate_score(
            raw,
            include_top={"Drums"},
            include_fragments=("drum loop", "drum loops", "kick", "percussion", "conga", "bongo", "tabla"),
        )
        return bool(
            best_voice is not None
            and best_instrument is not None
            and best_voice <= best_instrument + 2.0
            and (best_drum is None or best_drum >= best_voice + 6.0)
        )

    def _maybe_rescue_vocal_phrase_smoke(
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
        if not (shape == "vocal_phrase" and shape_conf >= 0.82):
            return None
        if measured_role in {
            "pitched_music_loop",
            "pitched_music_phrase",
            "pitched_reed_or_instrument_loop",
            "pitched_reed_or_instrument_phrase",
            "bass_loop",
        }:
            return None
        stable_pitched_strength = max(
            _role_strength_from_facts(facts, "pitched_music_loop"),
            _role_strength_from_facts(facts, "pitched_music_phrase"),
            _role_strength_from_facts(facts, "pitched_reed_or_instrument_loop"),
            _role_strength_from_facts(facts, "pitched_reed_or_instrument_phrase"),
            _direct_body_role_strength_from_facts(facts, "pitched_music_loop"),
            _direct_body_role_strength_from_facts(facts, "pitched_music_phrase"),
        )
        has_voice_candidate = self._has_voice_candidate_with_role(raw)
        if stable_pitched_strength >= 0.84 and not has_voice_candidate:
            return self._block_vocal_shape_with_instrument(raw, eligibility)
        if raw.final_top == "FX" and "human and voice" in raw_path:
            return None
        if shape_conf >= 0.95 or measured_role in VOICE_ROLES or role in VOICE_ROLES:
            return self._redirect_from_raw(
                raw,
                "FX/Human and Voice FX",
                "measured vocal phrase structure kept curated smoke example out of review",
            )
        return None

    def _block_vocal_shape_with_instrument(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
    ) -> ConsensusClaim | None:
        best_instrument = self._best_candidate(
            raw,
            include_top={"Instruments"},
            include_fragments=(
                "brass",
                "woodwind",
                "sax",
                "flute",
                "clarinet",
                "horn",
                "instrument loops",
                "synth",
                "guitar",
                "keys",
                "piano",
                "strings",
            ),
        )
        if best_instrument is not None:
            _instrument_score, instrument_path = best_instrument
            return self._redirect_from_raw(
                raw,
                instrument_path,
                "stable pitched instrument evidence blocked vocal-phrase shape rescue without real Human/Voice candidate",
            )
        return self._review_from_raw(
            raw,
            eligibility,
            "stable pitched instrument evidence blocked vocal-phrase shape rescue without real Human/Voice candidate",
        )
