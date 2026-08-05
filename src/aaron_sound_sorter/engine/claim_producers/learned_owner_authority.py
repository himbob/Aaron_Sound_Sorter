# SOURCE-NAME BLINDNESS INVARIANT:
# Learned-owner claims may inspect internal voter outputs, learned-memory
# diagnostics, and measured audio facts. They must never inspect source
# filenames, source folders, ZIP member names, or sample-pack labels.
"""Learned owner-authority claim producer.

This producer is the bridge between the newer trainable memory brains and the
older static arbitration shell.  It lets a learned owner lane make a concrete
source-family claim only after measured audio facts validate the body of the
sound.  Static code stays in the role of a safety contract instead of becoming a
replacement classifier.
"""

from __future__ import annotations

from dataclasses import dataclass

from aaron_sound_sorter.domain.models import CategoryGuess, SharedAudioFacts
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.decision_helpers import (
    _direct_body_role_strength_from_facts,
    _direct_voice_source_score_from_facts,
    _feature_number_from_facts,
    _role_strength_from_facts,
    _shape_confidence_from_facts,
    _shape_metric_from_facts,
    _shape_vote_from_facts,
    _voice_claim_has_tonal_instrument_conflict,
)
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_folder_path
from aaron_sound_sorter.engine.learned_memory_contracts import (
    EXACT_HUMAN_TEACHER_OWNER_CLAIM_SOURCE,
    LearnedMemoryMatch,
    exact_human_teacher_consensus,
    has_non_voice_memory_match,
    is_voice_category_path,
    iter_learned_memory_matches,
    safe_float,
)

LEARNED_OWNER_CLAIM_SOURCE = "learned_owner_body_claim"
VOICE_OWNER_CLAIM_SOURCE = "learned_owner_voice_body_claim"
LEARNED_OWNER_CLAIM_SOURCES = frozenset(
    {
        LEARNED_OWNER_CLAIM_SOURCE,
        VOICE_OWNER_CLAIM_SOURCE,
        EXACT_HUMAN_TEACHER_OWNER_CLAIM_SOURCE,
    }
)

VOICE_BODY_SHAPES = {
    "vocal_phrase",
    "vocal_one_shot",
    "pitched_phrase",
    "pitched_phrase_shape",
    "pitched_repetition_phrase",
    "repeated_phrase_loop",
    "bass_phrase",
    "solo_phrase",
    "sustained_pad",
    "designed_low_fx",
    "designed_tonal_fx",
}


@dataclass(frozen=True)
class LearnedOwnerCandidate:
    """One source-blind learned owner proposal.

    Args:
        target_path: Public folder path the learned owner proposes.
        evidence_path: Internal voter or memory label that supplied the owner.
        confidence: Normalized learned-lane confidence.
        support: Optional support score from the product brain ensemble.
        score: Optional distance/score from the learned lane.
        origin: Diagnostic origin of the learned claim.
        exact_human_teacher: Whether both independent memory lanes recovered
            the same near-identical GUI correction.

    Side Effects:
        None.

    Raises:
        No intentional exceptions.

    Important Constraints:
        ``evidence_path`` is internal taxonomy/voter metadata. It is not an
        input audio filename or source folder.
    """

    target_path: str
    evidence_path: str
    confidence: float
    support: float
    score: float
    origin: str
    exact_human_teacher: bool = False


class LearnedOwnerAuthorityClaimProducer:
    """Emit learned owner claims after measured body validation.

    Important Constraints:
        The producer does not train, mutate brains, or finalize placement.
        It emits normal ``ConsensusClaim`` objects for the arbiter to
        compare. It may only speak for matched trainable memory lanes; raw
        brain rank-one candidates belong to ordinary brain arbitration.
    """

    def produce(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return trainable owner claims for this decision context.

        Args:
            context: Immutable decision inputs for the current audio file.

        Returns:
            A list containing zero or one learned owner claim.

        Side Effects:
            None.

        Raises:
            No intentional exceptions.
        """
        owner_candidate = self.owner_candidate(context)
        if owner_candidate is None:
            return []
        claim_source = self._claim_source(owner_candidate)
        return [
            claim_from_folder_path(
                folder_path=owner_candidate.target_path,
                source=claim_source,
                reason=(
                    (
                        "exact human teacher owner claim: both independent memory lanes "
                        "recovered the same near-identical GUI correction and measured "
                        "body facts rejected catastrophic family contradiction"
                        if owner_candidate.exact_human_teacher
                        else "learned owner claim: matched trainable memory evidence was "
                        "validated by measured body facts before broad static fallback"
                    )
                    + f"; origin={owner_candidate.origin}; evidence={owner_candidate.evidence_path}"
                ),
                shared=context.raw.shared_candidates,
                raw_candidate_score=self._claim_score(context.raw, owner_candidate),
                brain_rank=1,
                physics_rank=None,
                shared_winner=owner_candidate.target_path,
                can_override=True,
                strength=self._claim_strength(owner_candidate),
                is_real_candidate=True,
            )
        ]

    def owner_candidate(self, context: DecisionContext) -> LearnedOwnerCandidate | None:
        """Return the best trainable owner candidate authorized by body facts.

        Args:
            context: Decision context with raw claim, facts, and voter outputs.

        Returns:
            A learned owner candidate, or ``None`` when memory is missing,
            weak, or contradicted by measured top-family evidence.

        Side Effects:
            None.

        Raises:
            No intentional exceptions.
        """
        candidate = self._best_owner_candidate_from_learned_lanes(context)
        if candidate is None:
            return None
        if self._learned_memory_conflict_blocks_candidate(context, candidate):
            return None
        if not self._facts_support_owner_body(context, candidate):
            return None
        if candidate.target_path.startswith("Instruments/Voice") and self._measured_non_voice_instrument_conflicts(
            context,
            candidate,
        ):
            return None
        return candidate

    def voice_owner_candidate(self, context: DecisionContext) -> LearnedOwnerCandidate | None:
        """Return a Voice owner candidate when learned evidence and body agree.

        Args:
            context: Decision context with raw claim, facts, and voter outputs.

        Returns:
            A learned Voice candidate, or ``None`` when measured body evidence
            does not authorize Voice ownership.

        Side Effects:
            None.

        Raises:
            No intentional exceptions.
        """
        candidate = self.owner_candidate(context)
        if candidate is None or not candidate.target_path.startswith("Instruments/Voice"):
            return None
        return candidate

    def _best_owner_candidate_from_learned_lanes(
        self,
        context: DecisionContext,
    ) -> LearnedOwnerCandidate | None:
        candidates = self._candidates_from_memory_facts(context)
        if not candidates:
            return None
        exact_teacher = exact_human_teacher_consensus(context.facts)
        if exact_teacher is not None:
            exact_target = self._target_path_for_exact_teacher(context.facts, exact_teacher.label)
            if exact_target is not None:
                return LearnedOwnerCandidate(
                    target_path=exact_target,
                    evidence_path=exact_teacher.label,
                    confidence=exact_teacher.confidence,
                    support=float(exact_teacher.effective_weight),
                    score=exact_teacher.nearest_distance,
                    origin="dual_memory_exact_human_teacher",
                    exact_human_teacher=True,
                )
        candidates.sort(key=lambda candidate: (-candidate.confidence, -candidate.support, candidate.score))
        return candidates[0]

    def _candidate_from_memory_facts(self, context: DecisionContext) -> LearnedOwnerCandidate | None:
        candidates = [
            candidate
            for candidate in self._candidates_from_memory_facts(context)
            if candidate.target_path.startswith("Instruments/Voice")
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda candidate: (-candidate.confidence, -candidate.support, candidate.score))
        return candidates[0]

    def _candidates_from_memory_facts(self, context: DecisionContext) -> list[LearnedOwnerCandidate]:
        facts = context.facts
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return []
        candidates: list[LearnedOwnerCandidate] = []
        for memory in iter_learned_memory_matches(facts, minimum_confidence=0.72):
            target_path = self._target_path_for_memory(facts, memory)
            if target_path is None:
                continue
            candidates.append(
                LearnedOwnerCandidate(
                    target_path=target_path,
                    evidence_path=memory.label,
                    confidence=memory.confidence,
                    support=float(memory.effective_weight),
                    score=memory.nearest_distance,
                    origin=memory.origin,
                )
            )
        return candidates

    def _learned_non_voice_memory_conflicts(self, context: DecisionContext) -> bool:
        """Return True when learned memory has a closer non-voice owner match."""
        return has_non_voice_memory_match(context.facts, minimum_confidence=0.86)

    def _learned_memory_conflict_blocks_candidate(
        self,
        context: DecisionContext,
        candidate: LearnedOwnerCandidate,
    ) -> bool:
        """Return True when another memory lane clearly owns another source."""
        if candidate.confidence >= 0.94 or candidate.support >= 512.0:
            return False
        if candidate.target_path.startswith("Instruments/Voice"):
            return self._learned_non_voice_memory_conflicts(context)
        return False

    def _measured_non_voice_instrument_conflicts(
        self,
        context: DecisionContext,
        candidate: LearnedOwnerCandidate,
    ) -> bool:
        """Return True when Physics owns a comparable non-voice instrument body.

        Brain labels are useful owner evidence, but they are not enough to
        overrule a concrete PhysicsVoter instrument branch.  This keeps learned
        Voice from stealing sax/keys/guitar/bass neighbors while still allowing
        direct human-trained memory matches to speak strongly.
        """
        if candidate.origin in {"voter_memory", "physics_memory"} and candidate.confidence >= 0.90:
            return False
        physics_guess = self._top_physics_guess(context)
        if physics_guess is None or str(physics_guess.top_family) != "Instruments":
            return False
        physics_path = str(physics_guess.folder_path or physics_guess.label or "")
        if not physics_path or is_voice_category_path(physics_path):
            return False
        facts = context.facts
        if facts is None:
            return False
        voice_body = self._voice_body_score(facts)
        non_voice_body = max(
            self._score(facts, "woodwind_sax_score"),
            self._score(facts, "reed_wind_score"),
            self._score(facts, "reed_wind_authority_score"),
            self._score(facts, "plucked_string_score"),
            self._score(facts, "plucked_string_authority_score"),
            self._score(facts, "struck_keys_score"),
            self._score(facts, "struck_keys_authority_score"),
            self._score(facts, "keys_tonal_decay_score"),
            self._score(facts, "instrument_bass_source_score"),
            self._score(facts, "bass_loop_source_score"),
            self._score(facts, "synth_tonal_source_score"),
            self._score(facts, "bowed_string_score"),
            self._score(facts, "string_violin_score"),
            self._score(facts, "string_cello_score"),
        )
        shape = _shape_vote_from_facts(facts)
        if shape in {"vocal_phrase", "vocal_one_shot"}:
            return False
        if non_voice_body < 0.58:
            return False
        return bool(voice_body <= non_voice_body + 0.12 and voice_body < 0.82)

    def _facts_support_voice_owner_body(
        self,
        facts: SharedAudioFacts | None,
        candidate: LearnedOwnerCandidate,
    ) -> bool:
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        confidence = _shape_confidence_from_facts(facts)
        if not self._shape_authorizes_voice_owner_body(facts, candidate, shape, confidence):
            return False
        voice_body = self._voice_body_score(facts)
        hard_drum = max(
            self._score(facts, "drum_hit_score"),
            self._score(facts, "drum_loop_source_score"),
            self._score(facts, "drum_kick_source_score"),
            self._score(facts, "drum_snare_source_score"),
            self._score(facts, "drum_clap_source_score"),
            self._score(facts, "compact_struck_tonal_percussion_score"),
        )
        transition_fx = max(
            self._score(facts, "fx_motion_score"),
            self._score(facts, "fx_transition_authority_score"),
            self._score(facts, "fx_riser_build_score"),
            self._score(facts, "fx_whoosh_sweep_score"),
        )
        percussive_ratio = _shape_metric_from_facts(facts, "percussive_event_ratio")
        drumlike_ratio = _shape_metric_from_facts(facts, "drumlike_frame_ratio")
        if hard_drum >= 0.68:
            return False
        if hard_drum >= 0.58 and max(percussive_ratio, drumlike_ratio) >= 0.40:
            return False
        if transition_fx >= 0.68 and shape not in {"vocal_phrase", "vocal_one_shot"}:
            return False
        vocal_role = max(
            _role_strength_from_facts(facts, "vocal_music_phrase"),
            _role_strength_from_facts(facts, "voiced_one_shot"),
            _direct_body_role_strength_from_facts(facts, "vocal_music_phrase"),
            _direct_body_role_strength_from_facts(facts, "voiced_one_shot"),
        )
        if _direct_voice_source_score_from_facts(facts) < 0.62 and vocal_role < 0.40:
            return False
        if _voice_claim_has_tonal_instrument_conflict(facts):
            return False
        return bool(voice_body >= 0.40 and percussive_ratio <= 0.72 and drumlike_ratio <= 0.72)

    def _shape_authorizes_voice_owner_body(
        self,
        facts: SharedAudioFacts,
        candidate: LearnedOwnerCandidate,
        shape: str,
        confidence: float,
    ) -> bool:
        """Return whether shape evidence may coexist with learned Voice memory.

        Strong learned memory owns source identity. ShapeVoter only describes
        structure, and processed rap/vocal loops can look alert-like because
        pitch contour, filtering, or repeats resemble sirens. Keep those
        teachable only when direct voice evidence is present and measured
        drum/transition evidence does not dominate.
        """
        if shape in VOICE_BODY_SHAPES and confidence >= 0.62:
            return True
        strong_memory = candidate.confidence >= 0.90 or candidate.support >= 512.0
        if not strong_memory or confidence < 0.58:
            return False
        if shape not in {"siren_alarm_tone", "hybrid_fx_motion", "designed_tonal_fx", "designed_low_fx"}:
            return False
        voice_body = self._voice_body_score(facts)
        direct_voice = max(
            _direct_voice_source_score_from_facts(facts),
            self._score(facts, "human_spoken_voice_score"),
            self._score(facts, "human_breath_mouth_score"),
        )
        hard_drum = max(
            self._score(facts, "drum_hit_score"),
            self._score(facts, "drum_loop_source_score"),
            self._score(facts, "drum_kick_source_score"),
            self._score(facts, "drum_snare_source_score"),
            self._score(facts, "drum_clap_source_score"),
        )
        transition_fx = max(
            self._score(facts, "fx_motion_score"),
            self._score(facts, "fx_transition_authority_score"),
            self._score(facts, "fx_riser_build_score"),
            self._score(facts, "fx_drop_downlifter_score"),
            self._score(facts, "fx_whoosh_sweep_score"),
        )
        return bool(voice_body >= 0.46 and direct_voice >= 0.62 and hard_drum < 0.62 and transition_fx < 0.62)

    def _facts_support_owner_body(
        self,
        context: DecisionContext,
        candidate: LearnedOwnerCandidate,
    ) -> bool:
        """Return whether measured facts authorize a learned owner target.

        This is intentionally a top-family/body contract, not a static leaf
        classifier. The learned memory owns the label. Code only blocks obvious
        catastrophic family mismatches.
        """
        facts = context.facts
        if facts is None:
            return False
        target_top = candidate.target_path.split("/", 1)[0]
        if candidate.target_path.startswith("Instruments/Voice"):
            return self._facts_support_voice_owner_body(facts, candidate)
        if target_top == "Drums":
            return self._facts_support_drum_owner_body(facts, candidate)
        if target_top == "Instruments":
            return self._facts_support_instrument_owner_body(facts, candidate)
        if target_top == "FX":
            return self._facts_support_fx_owner_body(facts, candidate)
        return False

    def _facts_support_drum_owner_body(
        self,
        facts: SharedAudioFacts,
        candidate: LearnedOwnerCandidate,
    ) -> bool:
        drum_body = max(
            self._score(facts, "drum_hit_score"),
            self._score(facts, "drum_loop_source_score"),
            self._score(facts, "drum_kick_source_score"),
            self._score(facts, "drum_snare_source_score"),
            self._score(facts, "drum_clap_source_score"),
            self._score(facts, "drum_tom_conga_source_score"),
            self._score(facts, "drum_metallic_percussion_source_score"),
            self._score(facts, "compact_struck_tonal_percussion_score"),
        )
        shape = _shape_vote_from_facts(facts)
        confidence = _shape_confidence_from_facts(facts)
        percussive_ratio = _shape_metric_from_facts(facts, "percussive_event_ratio")
        drumlike_ratio = _shape_metric_from_facts(facts, "drumlike_frame_ratio")
        strong_memory = candidate.confidence >= 0.90 or candidate.support >= 512.0
        return bool(
            drum_body >= (0.46 if strong_memory else 0.58)
            or (
                shape in {"single_hit", "hit_with_tail", "beat_loop", "top_loop", "drum_loop"}
                and confidence >= 0.70
                and max(percussive_ratio, drumlike_ratio) >= (0.22 if strong_memory else 0.34)
            )
        )

    def _facts_support_instrument_owner_body(
        self,
        facts: SharedAudioFacts,
        candidate: LearnedOwnerCandidate,
    ) -> bool:
        shape = _shape_vote_from_facts(facts)
        confidence = _shape_confidence_from_facts(facts)
        pitched_ratio = _shape_metric_from_facts(facts, "pitched_event_ratio")
        tonal_ratio = _shape_metric_from_facts(facts, "sustained_tonal_frame_ratio")
        percussive_ratio = _shape_metric_from_facts(facts, "percussive_event_ratio")
        drumlike_ratio = _shape_metric_from_facts(facts, "drumlike_frame_ratio")
        instrument_body = max(
            self._score(facts, "woodwind_sax_score"),
            self._score(facts, "reed_wind_score"),
            self._score(facts, "reed_wind_authority_score"),
            self._score(facts, "plucked_string_score"),
            self._score(facts, "plucked_string_authority_score"),
            self._score(facts, "struck_keys_score"),
            self._score(facts, "struck_keys_authority_score"),
            self._score(facts, "synth_tonal_source_score"),
            self._score(facts, "bass_synth_score"),
            self._score(facts, "bass_electric_score"),
            self._score(facts, "bowed_string_score"),
            self._score(facts, "pitched_mallet_instrument_score"),
        )
        voice_body = self._voice_body_score(facts)
        hard_drum = max(
            self._score(facts, "drum_hit_score"),
            self._score(facts, "drum_loop_source_score"),
            self._score(facts, "drum_kick_source_score"),
            self._score(facts, "compact_struck_tonal_percussion_score"),
        )
        transition_fx = max(
            self._score(facts, "fx_motion_score"),
            self._score(facts, "fx_transition_authority_score"),
            self._score(facts, "fx_riser_build_score"),
            self._score(facts, "fx_whoosh_sweep_score"),
        )
        strong_memory = candidate.confidence >= 0.90 or candidate.support >= 512.0
        if hard_drum >= 0.78 and max(percussive_ratio, drumlike_ratio) >= 0.48:
            return False
        if transition_fx >= 0.82 and shape in {"transition_riser", "transition_drop", "whoosh_sweep"}:
            return False
        if voice_body >= 0.70 and instrument_body < voice_body - 0.16:
            return False
        return bool(
            instrument_body >= (0.34 if strong_memory else 0.48)
            or (
                shape
                in {
                    "pitched_phrase",
                    "pitched_phrase_shape",
                    "pitched_repetition_phrase",
                    "repeated_phrase_loop",
                    "bass_phrase",
                    "solo_phrase",
                    "sustained_pad",
                    "vocal_phrase",
                    "designed_tonal_fx",
                }
                and confidence >= 0.62
                and max(pitched_ratio, tonal_ratio) >= (0.28 if strong_memory else 0.44)
                and max(percussive_ratio, drumlike_ratio) <= 0.80
            )
        )

    def _facts_support_fx_owner_body(
        self,
        facts: SharedAudioFacts,
        candidate: LearnedOwnerCandidate,
    ) -> bool:
        shape = _shape_vote_from_facts(facts)
        confidence = _shape_confidence_from_facts(facts)
        fx_body = max(
            self._score(facts, "fx_motion_score"),
            self._score(facts, "fx_transition_authority_score"),
            self._score(facts, "fx_riser_build_score"),
            self._score(facts, "fx_drop_downlifter_score"),
            self._score(facts, "fx_whoosh_sweep_score"),
            self._score(facts, "fx_reverse_score"),
            self._score(facts, "fx_impact_score"),
            self._score(facts, "fx_blip_beep_score"),
            self._score(facts, "fx_glitch_stutter_score"),
            self._score(facts, "fx_formant_score"),
            self._score(facts, "fx_foley_material_score"),
            self._score(facts, "texture_bed_score"),
        )
        hard_drum = max(
            self._score(facts, "drum_hit_score"),
            self._score(facts, "drum_loop_source_score"),
            self._score(facts, "drum_kick_source_score"),
            self._score(facts, "drum_snare_source_score"),
        )
        strong_memory = candidate.confidence >= 0.90 or candidate.support >= 512.0
        if hard_drum >= 0.82 and shape in {"single_hit", "beat_loop", "top_loop", "drum_loop"}:
            return False
        return bool(
            fx_body >= (0.34 if strong_memory else 0.52)
            or (
                shape
                in {
                    "transition_riser",
                    "transition_drop",
                    "whoosh_sweep",
                    "reverse_swell",
                    "hit_with_tail",
                    "ui_blip",
                    "designed_low_fx",
                    "designed_tonal_fx",
                    "texture_bed",
                    "foley_action",
                }
                and confidence >= (0.58 if strong_memory else 0.70)
            )
        )

    def _voice_body_score(self, facts: SharedAudioFacts) -> float:
        """Return the strongest measured voice/formant body score."""
        return max(
            self._score(facts, "voice_score"),
            self._score(facts, "human_spoken_voice_score"),
            self._score(facts, "human_breath_mouth_score"),
            self._score(facts, "voice_choir_score"),
            self._score(facts, "fx_formant_score"),
            self._score(facts, "formant_fx_score"),
        )

    def _target_path_for_exact_teacher(
        self,
        facts: SharedAudioFacts | None,
        label: str,
    ) -> str | None:
        """Return the exact supervised folder for dual-lane memory.

        ``facts`` remains part of the private method contract because callers
        supply it alongside every learned-memory lookup. It is deliberately
        not used to rewrite a human-approved label into another family.
        """
        del facts
        normalized_label = str(label or "").strip("/")
        if not normalized_label or normalized_label.lower().startswith("_to_review"):
            return None
        if normalized_label.split("/", 1)[0] not in {"Drums", "Instruments", "FX"}:
            return None
        return normalized_label

    def _target_path_for_memory(
        self,
        facts: SharedAudioFacts | None,
        memory: LearnedMemoryMatch,
    ) -> str | None:
        """Return the exact supervised folder owned by a memory row."""
        del facts
        if memory.confidence < 0.72:
            return None
        normalized_label = str(memory.label or "").strip("/")
        if not normalized_label or normalized_label.lower().startswith("_to_review"):
            return None
        target_top = normalized_label.split("/", 1)[0]
        if target_top not in {"Drums", "Instruments", "FX"}:
            return None
        return normalized_label

    @staticmethod
    def _claim_source(candidate: LearnedOwnerCandidate) -> str:
        """Return the stable source id for a learned owner claim."""
        if candidate.exact_human_teacher:
            return EXACT_HUMAN_TEACHER_OWNER_CLAIM_SOURCE
        if candidate.target_path.startswith("Instruments/Voice"):
            return VOICE_OWNER_CLAIM_SOURCE
        return LEARNED_OWNER_CLAIM_SOURCE

    @staticmethod
    def _claim_strength(candidate: LearnedOwnerCandidate) -> float:
        support_bonus = 0.03 if candidate.support >= 2.20 else 0.0
        memory_bonus = 0.03 if candidate.support >= 512.0 else 0.0
        return min(0.99, max(0.92, candidate.confidence + 0.16 + support_bonus + memory_bonus))

    @staticmethod
    def _claim_score(raw: ConsensusClaim, candidate: LearnedOwnerCandidate) -> float:
        raw_score = safe_float(raw.raw_candidate_score, 9999.0)
        if candidate.score < 10.0:
            return min(raw_score, candidate.score)
        return raw_score

    @staticmethod
    def _top_physics_guess(context: DecisionContext) -> CategoryGuess | None:
        result = context.physics_result
        if result is None or not getattr(result, "guesses", None):
            return None
        return result.guesses[0]

    @staticmethod
    def _score(facts: SharedAudioFacts, key: str) -> float:
        best = max(_feature_number_from_facts(facts, key), _shape_metric_from_facts(facts, key))
        evidence = facts.evidence if isinstance(getattr(facts, "evidence", None), dict) else {}
        panels = evidence.get("physics_subpanels", {}) if isinstance(evidence, dict) else {}
        flat = panels.get("flat", {}) if isinstance(panels, dict) else {}
        if isinstance(flat, dict):
            best = max(best, safe_float(flat.get(key)))
        return best
