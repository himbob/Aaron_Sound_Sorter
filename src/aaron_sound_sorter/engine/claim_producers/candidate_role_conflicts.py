# SOURCE-NAME BLINDNESS INVARIANT:
# This claim producer may inspect only voter candidates, final folder paths, and
# measured eligibility.  It must never use source filenames or source folders.
"""Measured-role versus ranked-candidate conflict guards."""

from __future__ import annotations

from aaron_sound_sorter.engine.decision_helpers import _norm_path, _path_has_any
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class RoleCandidateConflictMixin:
    """Review dangerous measured-role and candidate contradictions."""

    def role_candidate_conflict_claim(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
    ) -> ConsensusClaim | None:
        """Send dangerous measured-role/candidate contradictions to review."""
        if raw.final_top == "_TO_REVIEW":
            return None
        if eligibility.is_path_allowed(raw.folder_path, raw.final_top) and raw.final_top != "Drums":
            return None
        role = str(eligibility.role_name or "")
        raw_path = _norm_path(raw.folder_path)
        for claim in (
            self._bass_or_pitched_loop_vs_drum_claim(raw, eligibility, role, raw_path),
            self._instrument_loop_vs_tonal_fx_claim(raw, eligibility, role, raw_path),
            self._reed_role_conflict_claim(raw, eligibility, role, raw_path),
            self._percussive_one_shot_conflict_claim(raw, eligibility, role, raw_path),
            self._drum_loop_voice_conflict_claim(raw, eligibility, role, raw_path),
            self._pitched_loop_percussion_conflict_claim(raw, eligibility, role, raw_path),
            self._voice_false_positive_conflict_claim(raw, eligibility, role, raw_path),
        ):
            if claim is not None:
                return claim
        return None

    def _bass_or_pitched_loop_vs_drum_claim(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        role: str,
        raw_path: str,
    ) -> ConsensusClaim | None:
        del raw_path
        if not (
            role in {"bass_loop", "pitched_music_loop", "mixed_music_loop"}
            and raw.final_top == "Instruments"
            and _path_has_any(_norm_path(raw.folder_path), ("bass loops", "instrument loops", "mixed musical loops"))
        ):
            return None
        kick_or_drum_loop = self._has_close_candidate(
            raw,
            include_top={"Drums"},
            include_fragments=("kick", "drum loop", "drum loops", "break", "breaks"),
            max_gap=6.0,
            against_top="Instruments",
        )
        if not kick_or_drum_loop:
            return None
        return self._review_from_raw(
            raw,
            eligibility,
            "kick/drum-loop conflict: bass or pitched-loop role had close kick/drum-loop candidate evidence",
        )

    def _instrument_loop_vs_tonal_fx_claim(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        role: str,
        raw_path: str,
    ) -> ConsensusClaim | None:
        if not (
            role in {"pitched_music_loop", "pitched_music_phrase", "mixed_music_loop"}
            and raw.final_top == "Instruments"
            and _path_has_any(raw_path, ("instrument loops", "mixed musical loops"))
        ):
            return None
        tonal_fx = self._has_close_candidate(
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
            max_gap=6.0,
            against_top="Instruments",
        )
        if not tonal_fx:
            return None
        return self._review_from_raw(
            raw,
            eligibility,
            "tonal FX conflict: generic pitched Instrument Loop had close designed/evolving FX candidate evidence",
        )

    def _reed_role_conflict_claim(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        role: str,
        raw_path: str,
    ) -> ConsensusClaim | None:
        if role not in {"pitched_reed_or_instrument_loop", "pitched_reed_or_instrument_phrase"}:
            return None
        broad_path = _norm_path(eligibility.broad_folder_path)
        if not _path_has_any(broad_path, ("brass", "woodwind", "sax", "reed")):
            return None
        raw_is_exact_reed = raw.final_top == "Instruments" and _path_has_any(
            raw_path, ("brass", "woodwind", "sax", "reed")
        )
        raw_is_safe_generic = raw.final_top == "Instruments" and _path_has_any(
            raw_path, ("instrument loops", "mixed musical loops")
        )
        if raw_is_exact_reed or raw_is_safe_generic:
            return None
        if ("woodwinds/saxophone" in broad_path or "saxophone" in broad_path) and (
            raw.final_top == "FX"
            and _path_has_any(raw_path, ("human and voice", "voice", "vocal", "siren", "alarm", "beep"))
        ):
            best_reed = self._best_candidate_score(
                raw,
                include_top={"Instruments"},
                include_fragments=("brass", "woodwind", "sax", "reed"),
            )
            raw_score = float(raw.raw_candidate_score or 9999.0)
            if best_reed is None or best_reed > raw_score + 10.0:
                return self._review_from_raw(
                    raw,
                    eligibility,
                    "reed/woodwind over-narrowing conflict: sax-like measured role lacked close reed candidate support",
                )
            return self._broaden_from_raw(
                raw,
                eligibility,
                "measured sax/woodwind claim beat non-reed FX or Human/Voice false positive",
            )
        return self._review_from_raw(
            raw,
            eligibility,
            "reed/woodwind over-narrowing conflict: weak reed-like role could not safely override raw non-reed candidate evidence",
        )

    def _percussive_one_shot_conflict_claim(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        role: str,
        raw_path: str,
    ) -> ConsensusClaim | None:
        if role not in {"protected_percussive_one_shot", "percussive_one_shot", "low_kick_like_hit"}:
            return None
        instrument_hit = self._has_close_candidate(
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
                "violin",
                "cello",
            ),
            max_gap=18.0,
            against_top=raw.final_top,
        )
        if raw.final_top == "FX" and instrument_hit:
            return self._review_from_raw(
                raw,
                eligibility,
                "instrument one-shot conflict: percussive role/raw FX winner had stronger harmonic instrument candidate evidence",
            )
        return self._drum_raw_percussive_conflict_claim(raw, eligibility, raw_path)

    def _drum_raw_percussive_conflict_claim(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        raw_path: str,
    ) -> ConsensusClaim | None:
        fx_motion = self._has_close_candidate(
            raw,
            include_top={"FX"},
            include_fragments=(
                "whoosh",
                "swoosh",
                "swish",
                "sweep",
                "riser",
                "downlifter",
                "downlifter",
                "drop",
                "transition",
            ),
            max_gap=22.0,
            against_top="Drums",
        )
        if raw.final_top == "Drums" and fx_motion:
            return self._review_from_raw(
                raw,
                eligibility,
                "fx motion conflict: percussive one-shot role had close transition/woosh FX candidate",
            )
        pitched_inst = self._has_close_candidate(
            raw,
            include_top={"Instruments"},
            include_fragments=(
                "guitar",
                "strum",
                "chord",
                "keys",
                "piano",
                "rhodes",
                "organ",
                "synth",
                "strings",
                "violin",
                "cello",
                "brass",
                "woodwind",
                "trumpet",
                "sax",
            ),
            max_gap=5.0,
            against_top="Drums",
        )
        if (
            raw.final_top == "Drums"
            and pitched_inst
            and not _path_has_any(raw_path, ("kick", "snare", "clap", "rim", "stick", "sidestick", "hat", "cymbal"))
        ):
            return self._review_from_raw(
                raw,
                eligibility,
                "pitched instrument one-shot conflict: percussive role had close harmonic instrument candidate",
            )
        return None

    def _drum_loop_voice_conflict_claim(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        role: str,
        raw_path: str,
    ) -> ConsensusClaim | None:
        if role != "drum_loop":
            return None
        if raw.final_top == "Drums" and _path_has_any(raw_path, ("drum loop", "drum loops")):
            return None
        vocal_candidate = self._has_close_candidate(
            raw,
            include_top={"FX", "Instruments"},
            include_fragments=("human and voice", "voice", "vocal", "vox", "choir", "spoken", "crowd"),
            max_gap=4.0,
            against_top="Drums",
        )
        if vocal_candidate and raw.final_top == "Drums" and not _path_has_any(raw_path, ("drum loop", "drum loops")):
            return self._review_from_raw(
                raw,
                eligibility,
                "vocal loop conflict: drum-loop role had close human/voice candidate",
            )
        return None

    def _pitched_loop_percussion_conflict_claim(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        role: str,
        raw_path: str,
    ) -> ConsensusClaim | None:
        if not (role in {"pitched_music_loop", "mixed_music_loop"} and raw.final_top == "Instruments"):
            return None
        pitched_perc = self._has_close_candidate(
            raw,
            include_top={"Drums"},
            include_fragments=("percussion", "bongo", "conga", "tabla", "triangle", "mallet", "kalimba", "drum loop"),
            max_gap=6.0,
            against_top="Instruments",
        )
        if pitched_perc and _path_has_any(raw_path, ("instrument loops", "mixed musical loops")):
            return self._review_from_raw(
                raw,
                eligibility,
                "pitched percussion conflict: pitched loop role had close percussion/drum candidate",
            )
        return None

    def _voice_false_positive_conflict_claim(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        role: str,
        raw_path: str,
    ) -> ConsensusClaim | None:
        if role not in {"vocal_phrase", "vocal_one_shot", "voiced_one_shot"}:
            return None
        if "human and voice" not in _norm_path(eligibility.broad_folder_path):
            return None
        if raw.final_top not in {"Instruments", "FX"}:
            return None
        best_voice = self._best_candidate_score(
            raw,
            include_top={"FX", "Instruments"},
            include_fragments=("human and voice", "voice", "vocal", "vox", "choir", "spoken", "crowd"),
        )
        best_nonvoice = self._best_candidate_score(
            raw,
            include_top={"FX", "Instruments", "Drums"},
            include_fragments=(
                "siren",
                "alarm",
                "beep",
                "blip",
                "chime",
                "bell",
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
                "hybrid designed",
                "trumpet",
                "rhodes",
                "keys",
                "guitar",
                "percussion",
                "conga",
                "tabla",
            ),
        )
        raw_human_voice = "human and voice" in raw_path
        if best_nonvoice is not None and raw_human_voice and (best_voice is None or best_nonvoice <= best_voice - 3.0):
            return self._review_from_raw(
                raw,
                eligibility,
                "voice false-positive conflict: Human/Voice raw winner had clearly stronger non-voice FX/instrument candidate evidence",
            )
        return None
