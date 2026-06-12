# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Bass and concrete-FX rescues for measured true-bucket policy."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.decision_helpers import (
    _norm_path,
    _path_has_any,
    _shape_confidence_from_facts,
    _shape_vote_from_facts,
)
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class MeasuredTrueBassFxMixin:
    """Handle bass-loop, concrete FX, and tonal-alert rescues."""

    def _maybe_rescue_bass_loop(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        raw_path: str,
        measured_role: str,
        shape: str,
        shape_conf: float,
        facts: SharedAudioFacts | None = None,
    ) -> ConsensusClaim | None:
        """Redirect unsafe generic/FX winners to Bass Loops when well supported."""
        bass_rescue_role = (
            eligibility.role_name == "bass_loop"
            or (measured_role == "bass_loop" and eligibility.confidence >= 0.78)
            or (shape == "bass_phrase" and shape_conf >= 0.86 and eligibility.role_name == "bass_loop")
        )
        if not (bass_rescue_role and eligibility.confidence >= 0.78):
            return None
        if raw.final_top == "Drums" and shape in {"beat_loop", "top_loop", "drum_loop"} and shape_conf >= 0.80:
            return None
        if self._raw_concrete_fx_should_not_become_bass_loop(raw, raw_path, facts):
            return None
        best_bass = self._best_candidate(
            raw,
            include_top={"Instruments"},
            include_fragments=("bass", "808", "sub bass", "synth bass"),
        )
        best_nonbass_inst = self._best_candidate(
            raw,
            include_top={"Instruments"},
            include_fragments=(
                "guitar",
                "rhodes",
                "piano",
                "keys",
                "organ",
                "synth lead",
                "strings",
                "brass",
                "woodwind",
                "instrument loops",
            ),
            exclude_fragments=("bass", "808", "sub bass", "synth bass"),
        )
        if best_bass is None:
            return None
        bass_score, _bass_path = best_bass
        nonbass_score = best_nonbass_inst[0] if best_nonbass_inst is not None else None
        try:
            raw_score = float(raw.raw_candidate_score if raw.raw_candidate_score is not None else 9999.0)
        except Exception:
            raw_score = 9999.0
        raw_is_unsafe = bool(
            raw.final_top == "FX"
            or (raw.final_top == "Instruments" and _path_has_any(raw_path, ("instrument loops", "mixed musical loops")))
            or raw.final_top == "Drums"
        )
        bass_is_supported = bool(
            bass_score <= raw_score + 8.0 and (nonbass_score is None or bass_score <= nonbass_score + 3.0)
        )
        if raw_is_unsafe and bass_is_supported:
            return self._redirect_from_raw(
                raw,
                "Instruments/Bass/Bass Loops",
                "bass-loop true-bucket rescue: measured bass_loop plus Bass candidate support beat unsafe generic/FX family",
            )
        return None

    def _raw_concrete_fx_should_not_become_bass_loop(
        self,
        raw: ConsensusClaim,
        raw_path: str,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when concrete FX evidence should block bass-loop rescue.

        Low, tonal, repeated designed effects can look like bass loops in the
        measured role layer.  If the raw consensus already found a concrete FX
        family and the shape layer also reports transition/impact/glitch/FX
        motion, do not let the broad bass-loop rescue steal it into Instruments.
        """
        if raw.final_top != "FX":
            return False
        concrete_raw_fx = _path_has_any(
            raw_path,
            (
                "impact",
                "impacts",
                "hit",
                "whoosh",
                "sweep",
                "riser",
                "build",
                "drop",
                "downlifter",
                "reverse",
                "glitch",
                "stutter",
                "hybrid designed",
                "designed noise",
                "structural and transitional",
            ),
        )
        if not concrete_raw_fx:
            return False
        shape_vote = {}
        if facts is not None and isinstance(getattr(facts, "evidence", None), dict):
            candidate = facts.evidence.get("shape_vote", {})
            if isinstance(candidate, dict):
                shape_vote = candidate
        shape_scores_raw = shape_vote.get("shape_scores", ()) if isinstance(shape_vote, dict) else ()
        shape_scores: dict[str, float] = {}
        if isinstance(shape_scores_raw, (list, tuple)):
            for item in shape_scores_raw:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    try:
                        shape_scores[str(item[0])] = float(item[1])
                    except Exception:
                        pass
        primary_shape = str(shape_vote.get("primary_shape") or "") if isinstance(shape_vote, dict) else ""
        fx_shape_score = max(
            [
                shape_scores.get("hybrid_fx_motion", 0.0),
                shape_scores.get("impact_with_tail", 0.0),
                shape_scores.get("hit_with_tail", 0.0),
                shape_scores.get("glitch_stutter", 0.0),
                shape_scores.get("transition_riser", 0.0),
                shape_scores.get("transition_drop", 0.0),
                shape_scores.get("reverse_swell", 0.0),
                shape_scores.get("whoosh_sweep", 0.0),
                0.64
                if primary_shape
                in {
                    "hybrid_fx_motion",
                    "impact_with_tail",
                    "hit_with_tail",
                    "glitch_stutter",
                    "transition_riser",
                    "transition_drop",
                    "reverse_swell",
                    "whoosh_sweep",
                }
                else 0.0,
            ]
        )
        if fx_shape_score >= 0.55:
            return True
        # Even if the shape scorer overcalls a low beat loop, keep a concrete FX
        # raw winner when the raw FX candidate is substantially stronger than a
        # broad bass rescue candidate.  The arbiter can still review later if the
        # evidence is genuinely contradictory.
        try:
            raw_score = float(raw.raw_candidate_score if raw.raw_candidate_score is not None else 9999.0)
        except Exception:
            raw_score = 9999.0
        best_bass = self._best_candidate(
            raw,
            include_top={"Instruments"},
            include_fragments=("bass", "808", "sub bass", "synth bass"),
        )
        bass_score = best_bass[0] if best_bass is not None else 9999.0
        return bool(raw_score + 3.0 <= bass_score)

    def _maybe_rescue_concrete_fx_over_generic_instrument(
        self,
        raw: ConsensusClaim,
        raw_path: str,
    ) -> ConsensusClaim | None:
        """Keep concrete FX leaves from being washed into Instrument Loops."""
        if not (
            raw.final_top == "Instruments" and _path_has_any(raw_path, ("instrument loops", "mixed musical loops"))
        ):
            return None
        best_concrete_fx = self._best_candidate(
            raw,
            include_top={"FX"},
            include_fragments=(
                "siren",
                "alarm",
                "glitch",
                "stutter",
                "designed noise",
                "hybrid designed",
                "whoosh",
                "swoosh",
                "swish",
                "sweep",
                "riser",
                "build",
                "drop",
                "downlifter",
                "transition",
            ),
        )
        best_real_instrument = self._best_candidate_score(
            raw,
            include_top={"Instruments"},
            include_fragments=(),
        )
        if best_concrete_fx is not None and best_real_instrument is None:
            _fx_score, fx_path = best_concrete_fx
            return self._redirect_from_raw(
                raw,
                fx_path,
                "concrete FX true-bucket rescue: FX candidates survived while generic Instrument Loop was only a synthetic broad fallback",
            )
        return None

    def _maybe_rescue_tonal_alert_to_instrument(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        """Protect clean musical loops from weak tonal-alert/siren readings."""
        if eligibility.role_name != "fx_tonal_alert_or_siren":
            return None
        shape = _shape_vote_from_facts(facts)
        shape_conf = _shape_confidence_from_facts(facts)
        transition_shape = shape in {"transition_riser", "transition_drop"} and shape_conf >= 0.84
        best_inst = self._best_candidate(
            raw,
            include_top={"Instruments"},
            include_fragments=(
                "instrument loops",
                "synth",
                "lead",
                "guitar",
                "piano",
                "keys",
                "rhodes",
                "strings",
                "cello",
                "brass",
                "woodwind",
            ),
        )
        best_fx = self._best_candidate_score(raw, include_top={"FX"}, include_fragments=())
        if transition_shape or best_inst is None:
            return None
        inst_score, inst_path = best_inst
        if best_fx is None or inst_score <= best_fx + 5.0:
            safe_path = inst_path
            if not _path_has_any(
                _norm_path(safe_path),
                ("instrument loops", "synth", "guitar", "piano", "keys", "rhodes", "strings", "brass", "woodwind"),
            ):
                safe_path = "Instruments/Instrument Loops/Loops"
            return self._redirect_from_raw(
                raw,
                safe_path,
                "musical-loop safeguard: tonal-alert role lacked transition shape and instrument candidate support beat unsafe FX alarm/riser family",
            )
        return None
