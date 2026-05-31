# SOURCE-NAME BLINDNESS INVARIANT:
# This claim producer may inspect only voter candidates, final folder paths, and
# measured eligibility.  It must never use source filenames or source folders.
"""Raw-consensus candidate conflict guards."""

from __future__ import annotations

from aaron_sound_sorter.engine.decision_helpers import _norm_path, _path_has_any
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class RawCandidateConflictMixin:
    """Review strong raw winners when nearby candidates contradict the family."""

    def raw_consensus_conflict_claim(self, raw: ConsensusClaim) -> ConsensusClaim | None:
        """Review raw winners that have clearly stronger contradicting evidence."""
        if raw.final_top == "_TO_REVIEW":
            return None
        raw_path = _norm_path(raw.folder_path)
        if raw.final_top == "FX" and "human and voice" in raw_path:
            return self._human_voice_raw_conflict(raw)
        if raw.final_top == "FX" and _path_has_any(raw_path, ("slam", "impact", "hit")):
            return self._fx_impact_raw_conflict(raw)
        if raw.final_top == "Drums" and _path_has_any(raw_path, ("drum loops", "drum loop")):
            return self._drum_loop_bass_raw_conflict(raw)
        return None

    def _human_voice_raw_conflict(self, raw: ConsensusClaim) -> ConsensusClaim | None:
        best_voice = self._best_candidate_score(
            raw,
            include_top={"FX", "Instruments"},
            include_fragments=("human and voice", "voice", "vocal", "vox", "choir", "breath", "crowd"),
        )
        best_nonvoice = self._best_candidate_score(
            raw,
            include_top={"FX", "Drums", "Instruments"},
            include_fragments=(
                "drum loop",
                "drum loops",
                "break",
                "kick",
                "snare",
                "hat",
                "percussion",
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
                "keys",
                "rhodes",
                "piano",
            ),
        )
        if best_nonvoice is not None and (best_voice is None or best_nonvoice <= best_voice - 3.0):
            return self._review_from_raw_with_role(
                raw,
                "raw_human_voice_nonvoice_candidate_conflict",
                "human/voice false-positive conflict: raw Human/Voice FX winner had clearly stronger non-voice candidate evidence",
            )
        return None

    def _fx_impact_raw_conflict(self, raw: ConsensusClaim) -> ConsensusClaim | None:
        best_drum = self._best_candidate_score(
            raw,
            include_top={"Drums"},
            include_fragments=("drum", "snare", "ride", "hat", "cymbal", "percussion", "kick", "tom", "rim", "loop"),
        )
        best_fx = self._best_candidate_score(raw, include_top={"FX"}, include_fragments=())
        raw_score = self._safe_raw_score(raw)
        fx_score = best_fx if best_fx is not None else raw_score
        if best_drum is not None and best_drum <= fx_score - 1.0:
            return self._review_from_raw_with_role(
                raw,
                "raw_fx_drum_candidate_conflict",
                "drum/FX conflict: raw FX impact/slam winner had clearly stronger drum/percussion candidate evidence",
            )
        return None

    def _drum_loop_bass_raw_conflict(self, raw: ConsensusClaim) -> ConsensusClaim | None:
        best_bass = self._best_candidate_score(
            raw,
            include_top={"Instruments"},
            include_fragments=("bass", "808", "sub bass", "synth bass"),
        )
        best_drum = self._best_candidate_score(raw, include_top={"Drums"}, include_fragments=())
        if best_bass is not None and (best_drum is None or best_bass <= best_drum + 4.0):
            return self._review_from_raw_with_role(
                raw,
                "raw_drum_bass_candidate_conflict",
                "bass/drum-loop conflict: raw drum-loop winner had close bass-instrument candidate evidence",
            )
        return None
