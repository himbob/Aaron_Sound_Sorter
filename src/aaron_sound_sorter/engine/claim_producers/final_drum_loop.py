# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Measured drum-loop claim producer.

This producer moves the broad Drum Loops repair out of the final arbiter rescue
chain and into measured claim generation.  The claim remains broad on purpose:
shape/role evidence may say "this is a drum loop", but it must not promote a
specific kick, snare, hat, or percussion identity.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import CategoryGuess, SharedAudioFacts, VoterResult
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.decision_helpers import (
    _feature_number_from_facts,
    _has_drum_loop_structure_support,
    _measured_role_from_facts,
    _norm_path,
    _organic_percussion_loop_has_identity_conflict,
    _organic_percussion_loop_has_shape_support,
    _percussive_loop_pressure_from_facts,
    _shape_confidence_from_facts,
    _shape_metric_from_facts,
    _shape_vote_from_facts,
)
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_folder_path
from aaron_sound_sorter.voters.scoring_tools import role_strength

DRUM_LOOP_FRAGMENTS = ("drum loops", "drum loop", "break", "breaks")
DRUM_FAMILY_FRAGMENTS = (
    "drum",
    "kick",
    "snare",
    "tom",
    "percussion",
    "clap",
    "hat",
    "cymbal",
)
DRUM_LOOP_SHAPES = {"beat_loop", "drum_loop", "top_loop", "bass_phrase", "repeated_phrase_loop"}
TONAL_PHRASE_SHAPES = {
    "bass_phrase",
    "repeated_phrase_loop",
    "pitched_repetition_phrase",
    "pitched_phrase",
    "pitched_phrase_shape",
    "sustained_pad",
    "vocal_phrase",
}


class FinalDrumLoopClaimProducer:
    """Emit one broad Drum Loops claim when measured structure is decisive."""

    def produce(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return a measured drum-loop claim, or no claim when support is weak."""
        claim = self._produce_claim(context)
        return [] if claim is None else [claim]

    def _produce_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        raw = context.raw
        raw_path = _norm_path(raw.folder_path or raw.label)
        if raw.family == "Drums" and "drum loop" in raw_path:
            return None
        if not self._facts_support_drum_loop_claim(context):
            return None
        organic_percussion_strength = _percussive_loop_pressure_from_facts(context.facts)
        authoritative_organic_percussion_loop = bool(
            organic_percussion_strength >= 0.58
            and _organic_percussion_loop_has_shape_support(context.facts)
            and not _organic_percussion_loop_has_identity_conflict(context.facts, organic_percussion_strength)
        )
        claim_strength = 0.99 if authoritative_organic_percussion_loop else 0.94
        return claim_from_folder_path(
            folder_path="Drums/Drum Loops/Loops",
            source="measured_drum_loop_claim",
            reason=(
                "measured drum-loop claim: shape/role evidence and voter candidate "
                "support identified a broad drum loop before final arbitration"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=raw.raw_candidate_score,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner or raw.folder_path,
            can_override=True,
            strength=max(claim_strength, raw.strength),
            is_real_candidate=authoritative_organic_percussion_loop,
        )

    def _facts_support_drum_loop_claim(self, context: DecisionContext) -> bool:
        facts = context.facts
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False
        roles = facts.evidence.get("measured_roles", {})
        if not isinstance(roles, dict):
            roles = {}
        role = str(context.eligibility.role_name or "")
        measured_role = _measured_role_from_facts(facts)
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        drum_loop_strength = max(
            role_strength(roles, "drum_loop"),
            role_strength(roles, "low_rhythmic_drum_loop"),
            role_strength(roles, "percussive_drum_loop"),
            role_strength(roles, "bright_drum_loop"),
        )
        structure_support = _has_drum_loop_structure_support(role, measured_role, shape, shape_confidence)

        non_voice = (
            max(
                role_strength(roles, "vocal_music_phrase"),
                role_strength(roles, "vocal_phrase"),
                role_strength(roles, "voiced_one_shot"),
            )
            <= 0.35
        )
        if not non_voice:
            return False

        pulse = self._shape_number(facts, "pulse_regularity")
        onset_count = self._shape_number(facts, "onset_count")
        low_event = self._shape_number(facts, "low_event_ratio")
        high_event = self._shape_number(facts, "high_event_ratio")
        percussive_event = self._shape_number(facts, "percussive_event_ratio")
        drumlike_event = self._shape_number(facts, "drumlike_frame_ratio")
        sustained_tonal = self._shape_number(facts, "sustained_tonal_frame_ratio")
        non_event_tonal = self._shape_number(facts, "non_event_tonal_ratio")
        pitched_event = self._shape_number(facts, "pitched_event_ratio")
        pitch_confidence = self._shape_number(facts, "pitch_confidence")
        candidate_supported = self._has_candidate_support(context, DRUM_LOOP_FRAGMENTS, top_family="Drums")
        drum_family_supported = self._has_candidate_support(context, DRUM_FAMILY_FRAGMENTS, top_family="Drums")
        layer = self._physics_layer(facts)
        layer_drum_branch = str(layer.get("drum_branch_selected") or "") if isinstance(layer, dict) else ""
        layer_drum_confidence = (
            self._safe_float(layer.get("drum_branch_selected_confidence")) if isinstance(layer, dict) else 0.0
        )
        bright_hat_or_cymbal_loop_authority = bool(
            shape
            in {
                "pitched_repetition_phrase",
                "repeated_phrase_loop",
                "top_loop",
                "beat_loop",
            }
            and shape_confidence >= 0.72
            and onset_count >= 8.0
            and high_event >= 0.78
            and low_event <= 0.08
            and self._shape_number(facts, "true_repetition_score") >= 0.62
            and self._measured_score(
                facts,
                "drum_shaker_tambourine_source_score",
                "drum_cymbal_source_score",
                "drum_metallic_percussion_source_score",
            )
            >= 0.72
            and self._measured_score(facts, "drum_loop_source_score") >= 0.48
            and self._measured_score(
                facts,
                "synth_tonal_source_score",
                "synth_lead_score",
                "synth_pad_score",
                "synth_chord_score",
            )
            < 0.58
        )
        organic_percussion_loop_pressure = _percussive_loop_pressure_from_facts(facts)
        organic_percussion_loop_authority = bool(
            organic_percussion_loop_pressure >= 0.58
            and onset_count >= 8.0
            and self._shape_number(facts, "true_repetition_score") >= 0.42
            and _organic_percussion_loop_has_shape_support(facts)
            and not _organic_percussion_loop_has_identity_conflict(facts, organic_percussion_loop_pressure)
            and (
                drum_family_supported
                or (layer_drum_branch in {"DrumLoop", "ShakerTambourine", "ScrapeGuiro", "TomOrConga"})
            )
        )
        if not (structure_support or bright_hat_or_cymbal_loop_authority or organic_percussion_loop_authority):
            return False
        if candidate_supported or (layer_drum_branch == "DrumLoop" and layer_drum_confidence >= 0.50):
            drum_loop_strength = max(
                drum_loop_strength,
                self._measured_score(facts, "drum_loop_source_score"),
                self._measured_score(facts, "rhythmic_break_loop_score"),
            )

        bass_identity = self._measured_score(
            facts,
            "bass_synth_score",
            "bass_sub_score",
            "bass_electric_score",
            "low_end_source_score",
            "bass_808_score",
            "instruments_bass_generic_bass_one_shots_score",
            "instruments_bass_synth_bass_one_shots_score",
            "instruments_bass_sub_bass_one_shots_score",
        )
        weak_drum_loop_source = (
            max(
                self._measured_score(facts, "drum_loop_source_score"),
                self._measured_score(facts, "rhythmic_break_loop_score"),
            )
            <= 0.44
        )
        clean_low_tonal_bass_loop = bool(
            shape in {"bass_phrase", "beat_loop", "pitched_repetition_phrase"}
            and shape_confidence >= 0.88
            and pitched_event >= 0.90
            and sustained_tonal >= 0.86
            and non_event_tonal >= 0.86
            and low_event >= 0.88
            and high_event <= 0.08
            and percussive_event <= 0.08
            and drumlike_event <= 0.08
            and pitch_confidence >= 0.70
            and bass_identity >= 0.62
            and (shape == "bass_phrase" or weak_drum_loop_source)
        )
        if clean_low_tonal_bass_loop:
            return False
        strong_pitched_nonpercussive_loop = bool(
            shape in TONAL_PHRASE_SHAPES
            and shape_confidence >= 0.70
            and pitched_event >= 0.50
            and sustained_tonal >= 0.50
            and percussive_event <= 0.24
            and drumlike_event <= 0.26
            and drum_loop_strength < 0.70
            and not candidate_supported
            and not bright_hat_or_cymbal_loop_authority
            and not organic_percussion_loop_authority
            and not (layer_drum_branch == "DrumLoop" and layer_drum_confidence >= 0.50)
        )
        if strong_pitched_nonpercussive_loop:
            return False
        if self._raw_fx_loop_shape_is_designed_fx_decoy(context, facts):
            return False
        measured_loop = bool(
            drum_loop_strength >= 0.76
            and shape in DRUM_LOOP_SHAPES
            and shape_confidence >= 0.72
            and (pulse >= 0.24 or onset_count >= 3.0 or drum_loop_strength >= 0.88)
        )
        broad_music_loop_eligibility = role in {"mixed_music_loop", "pitched_music_loop", "pitched_music_phrase"}
        low_confidence_drum_loop = bool(
            drum_loop_strength >= 0.28
            and shape in DRUM_LOOP_SHAPES
            and shape_confidence >= 0.72
            and onset_count >= 8.0
            and (pulse >= 0.16 or percussive_event >= 0.08 or candidate_supported)
            and self._shape_number(facts, "f0_voiced_ratio") <= 0.70
            and not (
                broad_music_loop_eligibility
                and drum_loop_strength < 0.70
                and role_strength(roles, "drum_loop") < 0.50
                and role_strength(roles, "low_rhythmic_drum_loop") < 0.50
            )
        )
        decisive_measured_role = bool(role_strength(roles, "drum_loop") >= 0.82 and shape == "drum_loop")
        return bool(
            bright_hat_or_cymbal_loop_authority
            and drum_family_supported
            or measured_loop
            and candidate_supported
            or low_confidence_drum_loop
            and (candidate_supported or drum_family_supported)
            or organic_percussion_loop_authority
            or decisive_measured_role
        )

    def _raw_fx_loop_shape_is_designed_fx_decoy(self, context: DecisionContext, facts: SharedAudioFacts) -> bool:
        """Return True when a raw FX loop was falsely claimed as a drum loop.

        Fast lasers, chopped alarms, and stutter FX often look like top/beat
        loops because they have many sharp high-band events.  They should not
        become Drum Loops unless there is real drum-loop authority: regular
        pulse, strong drum material, or decisive drum-family support.
        """
        raw = context.raw
        if raw.family != "FX":
            return False
        shape = _shape_vote_from_facts(facts)
        shape_confidence = _shape_confidence_from_facts(facts)
        if shape not in {"top_loop", "beat_loop", "repeated_phrase_loop"} or shape_confidence < 0.84:
            return False
        designed_shape = max(
            self._shape_score(facts, "designed_low_fx"),
            self._shape_score(facts, "designed_tonal_fx"),
            self._shape_score(facts, "designed_motion_fx_loop"),
            self._shape_score(facts, "glitch_stutter"),
            self._shape_score(facts, "siren_alarm_tone"),
        )
        fx_design = max(
            self._measured_score(facts, "fx_glitch_stutter_score"),
            self._measured_score(facts, "fx_siren_score"),
            self._measured_score(facts, "fx_alarm_score"),
            self._measured_score(facts, "fx_formant_score"),
            self._measured_score(facts, "fx_radio_electrical_score"),
            self._measured_score(facts, "fx_motion_score"),
        )
        if max(designed_shape, fx_design) < 0.66:
            return False

        pulse = self._shape_number(facts, "pulse_regularity")
        drum_material = max(
            self._measured_score(facts, "drum_kick_source_score"),
            self._measured_score(facts, "drum_snare_source_score"),
            self._measured_score(facts, "drum_closed_hat_source_score"),
            self._measured_score(facts, "drum_cymbal_source_score"),
            self._measured_score(facts, "drum_shaker_tambourine_source_score"),
            self._measured_score(facts, "drum_hit_score"),
        )
        drum_loop_source = self._measured_score(facts, "drum_loop_source_score")
        # rhythmic_break_loop_score mostly measures repeated transient activity;
        # lasers/stutters can trigger it.  Require the source/material drum loop
        # score for the real-drum counter, not rhythmic repetition alone.
        real_drum_loop_counter = bool(
            pulse >= 0.24
            or (
                drum_material >= 0.74
                and drum_loop_source >= 0.62
                and self._has_candidate_support(context, DRUM_LOOP_FRAGMENTS, top_family="Drums")
            )
        )
        if real_drum_loop_counter:
            return False
        return bool(
            pulse <= 0.16
            and drum_material < 0.74
            and drum_loop_source <= 0.70
            and self._shape_number(facts, "onset_count") >= 8.0
        )

    @staticmethod
    def _shape_number(facts: SharedAudioFacts, metric_name: str) -> float:
        return max(_shape_metric_from_facts(facts, metric_name), _feature_number_from_facts(facts, metric_name))

    @staticmethod
    def _shape_score(facts: SharedAudioFacts, shape_name: str) -> float:
        if not isinstance(getattr(facts, "evidence", None), dict):
            return 0.0
        container = facts.evidence.get("shape_vote")
        scores = container.get("shape_scores") if isinstance(container, dict) else None
        if not isinstance(scores, (list, tuple)):
            return 0.0
        for item in scores:
            if isinstance(item, (list, tuple)) and len(item) >= 2 and str(item[0]) == shape_name:
                return min(1.0, max(0.0, FinalDrumLoopClaimProducer._safe_float(item[1])))
        return 0.0

    @staticmethod
    def _physics_layer(facts: SharedAudioFacts) -> dict:
        layer = facts.evidence.get("physics_layer_decision") if isinstance(facts.evidence, dict) else None
        return layer if isinstance(layer, dict) else {}

    @staticmethod
    def _measured_score(facts: SharedAudioFacts, *names: str) -> float:
        if not isinstance(facts.evidence, dict):
            return 0.0
        best = 0.0
        containers = [facts.evidence]
        subpanels = facts.evidence.get("physics_subpanels")
        if isinstance(subpanels, dict) and isinstance(subpanels.get("flat"), dict):
            containers.append(subpanels["flat"])
        roles = facts.evidence.get("measured_roles")
        if isinstance(roles, dict) and isinstance(roles.get("evidence"), dict):
            containers.append(roles["evidence"])
        direct = facts.evidence.get("direct_body_view")
        if isinstance(direct, dict):
            direct_roles = direct.get("measured_roles")
            if isinstance(direct_roles, dict) and isinstance(direct_roles.get("evidence"), dict):
                containers.append(direct_roles["evidence"])
        for container in containers:
            for name in names:
                best = max(best, FinalDrumLoopClaimProducer._safe_float(container.get(name)))
        return min(1.0, max(0.0, best))

    @staticmethod
    def _has_candidate_support(
        context: DecisionContext,
        fragments: tuple[str, ...],
        *,
        top_family: str,
    ) -> bool:
        for row in context.raw.shared_candidates or []:
            if FinalDrumLoopClaimProducer._row_matches(row, fragments, top_family, max_score=48.0):
                return True
        for result in (context.brain_result, context.physics_result):
            if FinalDrumLoopClaimProducer._result_has_guess(result, fragments, top_family):
                return True
        return False

    @staticmethod
    def _result_has_guess(result: VoterResult | None, fragments: tuple[str, ...], top_family: str) -> bool:
        if result is None:
            return False
        for guess in result.guesses[:24]:
            if FinalDrumLoopClaimProducer._guess_matches(guess, fragments, top_family):
                return True
        return False

    @staticmethod
    def _guess_matches(guess: CategoryGuess, fragments: tuple[str, ...], top_family: str) -> bool:
        path = _norm_path(getattr(guess, "folder_path", "") or getattr(guess, "label", ""))
        family = str(getattr(guess, "top_family", "") or "").lower()
        wanted = top_family.lower()
        if family and family != wanted:
            return False
        if not family and not path.startswith(f"{wanted}/"):
            return False
        if not any(fragment in path for fragment in fragments):
            return False
        rank = int(getattr(guess, "rank", 9999) or 9999)
        score = FinalDrumLoopClaimProducer._safe_float(getattr(guess, "score", rank + 1.0), rank + 1.0)
        return rank <= 14 or score <= 4.75

    @staticmethod
    def _row_matches(row: object, fragments: tuple[str, ...], top_family: str, *, max_score: float) -> bool:
        if not isinstance(row, dict):
            return False
        path = _norm_path(row.get("folder_path") or row.get("label") or "")
        wanted = top_family.lower()
        row_top = str(row.get("top_family") or "").lower()
        if row_top and row_top != wanted:
            return False
        if not row_top and not path.startswith(f"{wanted}/"):
            return False
        if not any(fragment in path for fragment in fragments):
            return False
        score = FinalDrumLoopClaimProducer._safe_float(row.get("combined_rank_score"), 9999.0)
        brain_rank = FinalDrumLoopClaimProducer._safe_int(row.get("brain_rank"), 9999)
        physics_rank = FinalDrumLoopClaimProducer._safe_int(row.get("physics_rank"), 9999)
        return score <= max_score or brain_rank <= 14 or physics_rank <= 12

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value if value is not None else default)
        except Exception:
            return float(default)

    @staticmethod
    def _safe_int(value: object, default: int = 0) -> int:
        try:
            return int(value if value is not None else default)
        except Exception:
            return int(default)
