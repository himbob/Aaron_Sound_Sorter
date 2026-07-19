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
from typing import Any

from aaron_sound_sorter.domain.models import CategoryGuess, SharedAudioFacts
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.decision_helpers import (
    _feature_number_from_facts,
    _norm_path,
    _shape_confidence_from_facts,
    _shape_metric_from_facts,
    _shape_vote_from_facts,
)
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_folder_path

VOICE_OWNER_CLAIM_SOURCE = "learned_owner_voice_body_claim"

VOICE_PATH_FRAGMENTS = (
    "voice",
    "vocal",
    "vocals",
    "spoken",
    "choir",
    "breath",
    "mouth",
    "human and voice",
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


class LearnedOwnerAuthorityClaimProducer:
    """Emit learned owner claims after measured body validation.

    Args:
        None.

    Side Effects:
        None.

    Important Constraints:
        The producer does not train, mutate brains, or finalize placement.  It
        emits normal ``ConsensusClaim`` objects for the arbiter to compare.
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
        voice_candidate = self.voice_owner_candidate(context)
        if voice_candidate is None:
            return []
        return [
            claim_from_folder_path(
                folder_path=voice_candidate.target_path,
                source=VOICE_OWNER_CLAIM_SOURCE,
                reason=(
                    "learned owner claim: rank-one Voice memory/brain evidence "
                    "was validated by measured voice/formant body facts before "
                    f"broad role fallback; origin={voice_candidate.origin}; "
                    f"evidence={voice_candidate.evidence_path}"
                ),
                shared=context.raw.shared_candidates,
                raw_candidate_score=self._claim_score(context.raw, voice_candidate),
                brain_rank=1,
                physics_rank=None,
                shared_winner=voice_candidate.target_path,
                can_override=True,
                strength=self._claim_strength(voice_candidate),
                is_real_candidate=True,
            )
        ]

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
        facts = context.facts
        candidate = self._best_voice_candidate_from_learned_lanes(context)
        if candidate is None:
            return None
        if self._learned_non_voice_memory_conflicts(context):
            return None
        if not self._facts_support_voice_owner_body(facts):
            return None
        if self._measured_non_voice_instrument_conflicts(context, candidate):
            return None
        return candidate

    def _best_voice_candidate_from_learned_lanes(
        self,
        context: DecisionContext,
    ) -> LearnedOwnerCandidate | None:
        candidates: list[LearnedOwnerCandidate] = []
        from_brain = self._candidate_from_brain_result(context)
        if from_brain is not None:
            candidates.append(from_brain)
        from_ensemble = self._candidate_from_ensemble_facts(context)
        if from_ensemble is not None:
            candidates.append(from_ensemble)
        from_memory = self._candidate_from_memory_facts(context)
        if from_memory is not None:
            candidates.append(from_memory)
        if not candidates:
            return None
        candidates.sort(key=lambda candidate: (-candidate.confidence, -candidate.support, candidate.score))
        return candidates[0]

    def _candidate_from_brain_result(self, context: DecisionContext) -> LearnedOwnerCandidate | None:
        result = context.brain_result
        if result is None or not getattr(result, "guesses", None):
            return None
        top_guess = result.guesses[0]
        if not self._guess_path_is_voice(top_guess):
            return None
        confidence = safe_float(getattr(top_guess, "confidence", 0.0))
        support = self._guess_support(top_guess)
        score = safe_float(getattr(top_guess, "score", 9999.0), 9999.0)
        if confidence < 0.70 and support < 2.20 and score > 0.70:
            return None
        return LearnedOwnerCandidate(
            target_path=self._voice_target_path(
                context.facts,
                str(top_guess.folder_path or top_guess.label or ""),
            ),
            evidence_path=str(top_guess.folder_path or top_guess.label or ""),
            confidence=max(confidence, 0.74 if support >= 2.20 else 0.0),
            support=support,
            score=score,
            origin="brain_result_rank_one",
        )

    def _candidate_from_ensemble_facts(self, context: DecisionContext) -> LearnedOwnerCandidate | None:
        facts = context.facts
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return None
        ensemble = facts.evidence.get("brain_ensemble_vote_result")
        guesses = ensemble.get("top_guesses") if isinstance(ensemble, dict) else []
        if not isinstance(guesses, list) or not guesses or not isinstance(guesses[0], dict):
            return None
        top_guess = guesses[0]
        top_path = self._row_path(top_guess)
        if not self._path_is_voice(top_path):
            return None
        support = safe_float(top_guess.get("support", top_guess.get("ensemble_support", 0.0)))
        confidence = safe_float(top_guess.get("confidence"))
        score = safe_float(top_guess.get("score", top_guess.get("ensemble_score", 9999.0)), 9999.0)
        non_voice_support = self._nearest_non_voice_support(guesses[1:6])
        strong_margin = bool(
            support >= 2.20
            and (non_voice_support <= 0.0 or support >= non_voice_support + 0.65 or support >= non_voice_support * 1.35)
        )
        if confidence < 0.70 and not strong_margin and score > 0.70:
            return None
        return LearnedOwnerCandidate(
            target_path=self._voice_target_path(facts, top_path),
            evidence_path=top_path,
            confidence=max(confidence, 0.76 if strong_margin else 0.0),
            support=support,
            score=score,
            origin="brain_ensemble_rank_one",
        )

    def _candidate_from_memory_facts(self, context: DecisionContext) -> LearnedOwnerCandidate | None:
        facts = context.facts
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return None
        candidates: list[LearnedOwnerCandidate] = []
        for key, origin in (
            ("learned_voter_memory", "voter_memory"),
            ("learned_physics_memory", "physics_memory"),
        ):
            memory = facts.evidence.get(key)
            if not isinstance(memory, dict) or not bool(memory.get("matched")):
                continue
            label = str(memory.get("label", ""))
            top_family = str(memory.get("top_family", ""))
            if top_family not in {"Instruments", "FX"} or not self._path_is_voice(label):
                continue
            confidence = safe_float(memory.get("confidence"))
            if confidence < 0.72:
                continue
            candidates.append(
                LearnedOwnerCandidate(
                    target_path=self._voice_target_path(facts, label),
                    evidence_path=label,
                    confidence=confidence,
                    support=float(safe_int(memory.get("effective_weight"))),
                    score=safe_float(memory.get("nearest_distance"), 9999.0),
                    origin=origin,
                )
            )
        if not candidates:
            return None
        candidates.sort(key=lambda candidate: (-candidate.confidence, -candidate.support, candidate.score))
        return candidates[0]

    def _learned_non_voice_memory_conflicts(self, context: DecisionContext) -> bool:
        """Return True when learned memory has a closer non-voice owner match."""
        facts = context.facts
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return False
        for key in ("learned_voter_memory", "learned_physics_memory"):
            memory = facts.evidence.get(key)
            if not isinstance(memory, dict) or not bool(memory.get("matched")):
                continue
            confidence = safe_float(memory.get("confidence"))
            if confidence < 0.86:
                continue
            label = str(memory.get("label", ""))
            top_family = str(memory.get("top_family", ""))
            branch = str(memory.get("branch", ""))
            role = str(memory.get("role", ""))
            if self._memory_target_is_voice(label=label, top_family=top_family, branch=branch, role=role):
                continue
            if top_family in {"Instruments", "Drums", "FX"} or branch or role:
                return True
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
        if not physics_path or self._path_is_voice(physics_path):
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

    def _facts_support_voice_owner_body(self, facts: SharedAudioFacts | None) -> bool:
        if facts is None:
            return False
        shape = _shape_vote_from_facts(facts)
        confidence = _shape_confidence_from_facts(facts)
        if shape not in VOICE_BODY_SHAPES or confidence < 0.62:
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
        return bool(voice_body >= 0.40 and percussive_ratio <= 0.72 and drumlike_ratio <= 0.72)

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

    def _voice_target_path(self, facts: SharedAudioFacts | None, evidence_path: str) -> str:
        normalized = _norm_path(evidence_path)
        if normalized.startswith("instruments/voice"):
            return str(evidence_path).strip("/")
        duration = _feature_number_from_facts(facts, "duration_sec")
        onset_count = _shape_metric_from_facts(facts, "onset_count")
        shape = _shape_vote_from_facts(facts)
        loopish = bool(
            getattr(facts, "is_loop_like", False)
            or duration >= 1.35
            or onset_count >= 4.0
            or shape in {"pitched_repetition_phrase", "repeated_phrase_loop", "bass_phrase"}
        )
        if loopish:
            return "Instruments/Voice/Vocal Loops/Loops"
        return "Instruments/Voice/Phrase/One Shots"

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
    def _guess_path_is_voice(guess: CategoryGuess) -> bool:
        return LearnedOwnerAuthorityClaimProducer._path_is_voice(str(guess.folder_path or guess.label or ""))

    @staticmethod
    def _guess_support(guess: CategoryGuess) -> float:
        evidence = getattr(guess, "evidence", {}) or {}
        if not isinstance(evidence, dict):
            return 0.0
        return max(
            safe_float(evidence.get("support")),
            safe_float(evidence.get("ensemble_support")),
            safe_float(evidence.get("human_override_effective_weight")),
        )

    @staticmethod
    def _top_physics_guess(context: DecisionContext) -> CategoryGuess | None:
        result = context.physics_result
        if result is None or not getattr(result, "guesses", None):
            return None
        return result.guesses[0]

    @staticmethod
    def _nearest_non_voice_support(rows: list[object]) -> float:
        best = 0.0
        for row in rows:
            if not isinstance(row, dict):
                continue
            if LearnedOwnerAuthorityClaimProducer._path_is_voice(LearnedOwnerAuthorityClaimProducer._row_path(row)):
                continue
            best = max(best, safe_float(row.get("support", row.get("ensemble_support", 0.0))))
        return best

    @staticmethod
    def _row_path(row: dict[str, Any]) -> str:
        return str(row.get("folder_path") or row.get("path") or row.get("label") or "").strip("/")

    @staticmethod
    def _path_is_voice(path: str) -> bool:
        normalized = _norm_path(path)
        if normalized.startswith("instruments/voice") or normalized.startswith("fx/human and voice fx"):
            return True
        return any(fragment in normalized for fragment in VOICE_PATH_FRAGMENTS)

    @staticmethod
    def _memory_target_is_voice(*, label: str, top_family: str, branch: str, role: str) -> bool:
        normalized_label = _norm_path(label)
        normalized_role = str(role or "").lower()
        normalized_branch = str(branch or "").lower()
        if LearnedOwnerAuthorityClaimProducer._path_is_voice(normalized_label):
            return True
        if str(top_family) == "Instruments" and normalized_branch == "voice":
            return True
        if "voice" in normalized_role or "vocal" in normalized_role:
            return True
        if str(top_family) == "FX" and (
            normalized_branch in {"formantfx", "humancreaturefx"}
            or "human_voice" in normalized_role
            or "formant" in normalized_role
        ):
            return True
        return False

    @staticmethod
    def _score(facts: SharedAudioFacts, key: str) -> float:
        best = max(_feature_number_from_facts(facts, key), _shape_metric_from_facts(facts, key))
        evidence = facts.evidence if isinstance(getattr(facts, "evidence", None), dict) else {}
        panels = evidence.get("physics_subpanels", {}) if isinstance(evidence, dict) else {}
        flat = panels.get("flat", {}) if isinstance(panels, dict) else {}
        if isinstance(flat, dict):
            best = max(best, safe_float(flat.get(key)))
        return best


def safe_float(value: object, default: float = 0.0) -> float:
    """Return ``value`` as a finite float, or ``default``.

    Args:
        value: Object to coerce.
        default: Fallback value when coercion fails.

    Returns:
        Finite float value.

    Side Effects:
        None.
    """
    try:
        number = float(value)  # type: ignore[arg-type]
    except Exception:
        return default
    return number if number == number else default


def safe_int(value: object, default: int = 0) -> int:
    """Return ``value`` as an integer, or ``default`` when parsing fails."""
    try:
        return int(float(value))  # type: ignore[arg-type]
    except Exception:
        return default
