# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Early keep-raw safety guards for measured true-bucket policy."""

from __future__ import annotations

from aaron_sound_sorter.engine.decision_helpers import _path_has_any
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class MeasuredTrueSafetyMixin:
    """Keep stable raw winners before stronger rescue logic runs."""

    def _maybe_keep_stable_drum_loop(
        self,
        raw: ConsensusClaim,
        raw_path: str,
    ) -> ConsensusClaim | None:
        """Keep a supported raw Drum Loop from being broadened to Instruments."""
        if not (raw.final_top == "Drums" and _path_has_any(raw_path, ("drum loop", "drum loops"))):
            return None
        best_drum = self._best_candidate_score(
            raw,
            include_top={"Drums"},
            include_fragments=("drum", "drum loop", "drum loops"),
        )
        best_bass = self._best_candidate_score(
            raw,
            include_top={"Instruments"},
            include_fragments=("bass", "808", "sub bass", "synth bass"),
        )
        if best_drum is not None and (best_bass is None or best_drum <= best_bass):
            return raw
        return None

    @staticmethod
    def _maybe_keep_pitched_instrument_over_voice(
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        raw_path: str,
        measured_role: str,
    ) -> ConsensusClaim | None:
        """Keep safe Instrument Loops from a vocal-shaped false fallback."""
        if (
            raw.final_top == "Instruments"
            and _path_has_any(raw_path, ("instrument loops", "mixed musical loops", "bass"))
            and eligibility.role_name in {"vocal_phrase", "vocal_one_shot", "voiced_one_shot", "vocal_music_phrase"}
            and measured_role
            in {
                "pitched_music_loop",
                "pitched_music_phrase",
                "pitched_reed_or_instrument_loop",
                "pitched_reed_or_instrument_phrase",
                "bass_loop",
            }
        ):
            return raw
        return None
