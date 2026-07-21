# SOURCE-NAME BLINDNESS INVARIANT:
# This claim producer may inspect only raw/claim paths, voter candidate metadata,
# and measured audio facts. It must never inspect source filenames, source folder
# names, ZIP member names, path tokens, or sample-pack labels as classification evidence.
"""Measured final-guard claims produced before arbitration.

This module downshifts source-blind final safety guards out of
``FamilyClaimArbiter.adjudicate``.  The guard predicates still live on the
arbiter during this transition so we do not duplicate a large lattice or risk a
one-pass rewrite.  The important architectural change is that review/rehome
claims now enter the normal claim competition instead of mutating the winner
after arbitration.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.decision_helpers import _feature_number_from_facts
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_folder_path, review_claim
from aaron_sound_sorter.engine.instrument_pitch_ranges import check_instrument_pitch_range


class MeasuredFinalGuardClaimProducer:
    """Emit measured safety-guard claims before final arbitration.

    This is a deliberately small transition adapter.  It applies selected
    source-blind guard predicates to the raw claim and to claims already emitted
    by lower producers, then returns only the changed claims.  Keeping this as a
    claim producer removes late winner mutation without moving the policy lattice
    sideways into another giant file.
    """

    def __init__(self, guard: FamilyClaimArbiter | None = None) -> None:
        self.guard = guard or FamilyClaimArbiter()

    def produce_for_claims(
        self,
        context: DecisionContext,
        existing_claims: list[ConsensusClaim],
    ) -> list[ConsensusClaim]:
        """Return guard claims derived from raw plus existing lower claims."""
        facts = context.facts
        seeds = [context.raw, *existing_claims]
        guard_claims: list[ConsensusClaim] = []
        seen: set[tuple[str, str, str]] = set()
        for seed in seeds:
            changed = self._guarded_claim(seed, facts, raw_claim=context.raw)
            if changed is None:
                continue
            changed = self._preserve_existing_voice_one_shot_structure(
                changed,
                existing_claims=existing_claims,
                facts=facts,
            )
            key = (changed.family, changed.folder_path or changed.label, changed.source)
            if key in seen:
                continue
            seen.add(key)
            guard_claims.append(changed)
        return guard_claims

    def _preserve_existing_voice_one_shot_structure(
        self,
        changed: ConsensusClaim,
        *,
        existing_claims: list[ConsensusClaim],
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim:
        """Do not let a broad measured voice guard erase one-shot structure.

        The measured voice guard is allowed to keep true voice under
        ``Instruments/Voice`` when raw consensus is pulled toward an abstract
        alarm/synth/FX lane.  It must not upgrade a measured single-event voice
        phrase into ``Vocal Loops`` when an earlier lower-layer claim already
        captured the safer one-shot depth.
        """
        path = str(changed.folder_path or changed.label or "").replace("\\", "/").lower()
        if changed.source != "final_measured_voice_invariant":
            return changed
        if "vocal loops" not in path and not path.endswith("/loops"):
            return changed
        if not self._facts_support_voice_one_shot_structure(facts):
            return changed

        voice_one_shot_claims = [
            claim
            for claim in existing_claims
            if claim.family == "Instruments"
            and "voice" in str(claim.folder_path or claim.label or "").lower()
            and "one shot" in str(claim.folder_path or claim.label or "").lower()
            and claim.can_override
        ]
        if not voice_one_shot_claims:
            return changed
        voice_one_shot_claims.sort(key=lambda claim: (-claim.strength, str(claim.source)))
        witness = voice_one_shot_claims[0]
        return claim_from_folder_path(
            folder_path="Instruments/Voice/Phrase/One Shots",
            source="final_short_true_voice_one_shot_invariant",
            reason=(
                "measured final-guard claim preserved single-event Voice one-shot "
                "structure instead of broad Vocal Loops"
            ),
            shared=changed.shared_candidates or witness.shared_candidates,
            raw_candidate_score=changed.raw_candidate_score,
            brain_rank=changed.brain_rank,
            physics_rank=changed.physics_rank,
            shared_winner=changed.shared_winner or witness.shared_winner or witness.folder_path,
            can_override=True,
            strength=max(0.94, changed.strength, witness.strength),
            is_real_candidate=changed.is_real_candidate,
        )

    @staticmethod
    def _facts_support_voice_one_shot_structure(facts: SharedAudioFacts | None) -> bool:
        """Return True when measured facts describe a single-event voice phrase."""
        if facts is None:
            return False
        duration = _feature_number_from_facts(facts, "duration_sec")
        if duration <= 0.0 or duration > 1.80:
            return False
        if not (getattr(facts, "is_single_event_like", False) or getattr(facts, "is_short_hit_like", False)):
            return False
        evidence = getattr(facts, "evidence", {})
        roles = evidence.get("measured_roles", {}) if isinstance(evidence, dict) else {}
        direct = evidence.get("direct_body_view", {}) if isinstance(evidence, dict) else {}
        direct_roles = direct.get("measured_roles", {}) if isinstance(direct, dict) else {}

        def role_value(source: object, name: str) -> float:
            if not isinstance(source, dict):
                return 0.0
            try:
                return float(source.get(name, 0.0) or 0.0)
            except Exception:
                return 0.0

        percussive_short_hit_pressure = max(
            role_value(roles, "percussive_one_shot"),
            role_value(direct_roles, "percussive_one_shot"),
        )
        loop_confidence = role_value(evidence, "librosa_loop_confidence") if isinstance(evidence, dict) else 0.0
        librosa_onsets = role_value(evidence, "librosa_onset_event_count") if isinstance(evidence, dict) else 0.0
        if loop_confidence >= 0.58 and librosa_onsets >= 3.0:
            return False
        shape = ""
        if isinstance(evidence, dict):
            shape_vote = evidence.get("shape_vote", {})
            if isinstance(shape_vote, dict):
                shape = str(shape_vote.get("primary_shape", ""))
        if shape in {"single_hit", "hit_with_tail", "echo_tail_hit"} and percussive_short_hit_pressure >= 0.25:
            return False

        voice_strength = max(
            role_value(roles, "voiced_one_shot"),
            role_value(roles, "vocal_one_shot"),
            role_value(roles, "vocal_phrase"),
            role_value(roles, "vocal_music_phrase"),
            role_value(direct_roles, "voiced_one_shot"),
            role_value(direct_roles, "vocal_one_shot"),
            role_value(direct_roles, "vocal_phrase"),
            role_value(direct_roles, "vocal_music_phrase"),
        )
        return voice_strength >= 0.72

    def _guarded_claim(
        self,
        seed: ConsensusClaim,
        facts: SharedAudioFacts | None,
        *,
        raw_claim: ConsensusClaim,
    ) -> ConsensusClaim | None:
        """Apply downshifted final guards to one candidate claim.

        The order mirrors the former late-winner guard order while making each
        outcome enter normal claim competition before arbitration.  The final
        consensus firewall still runs after candidate-local guards so cross-family
        final claims cannot bypass strong raw committee consensus.
        """
        pitch_range_review = self._review_impossible_instrument_pitch_range(seed, facts)
        if pitch_range_review is not None:
            return pitch_range_review
        claim = seed
        for mutator in (
            self.guard._release_shape_review_to_broad_bucket,
            self.guard._release_clean_tonal_non_drum_hit_to_fx,
            self.guard._protect_clean_tonal_instrument_phrase_from_drum_leaf,
            self.guard._rehome_true_voice_from_fx_bucket,
            self.guard._protect_short_true_voice_one_shot_from_loop_bucket,
            self.guard._review_pitched_music_hit_stolen_by_drum_leaf,
            self.guard._protect_measured_voice_from_non_voice_leaf,
            self.guard._protect_compound_music_loop_from_specific_instrument_leaf,
            self.guard._protect_clean_pitched_instrument_loop_from_fx_leaf,
            self.guard._review_conflicted_instrument_identity_leaf,
            self.guard._review_uncertain_generic_loop_release,
            self.guard._review_non_transition_pitched_loop_stolen_by_fx_leaf,
            self.guard._review_percussive_voice_or_animal_fx_conflict,
            self.guard._protect_measured_mixed_loop_from_sax_overreach,
            self.guard._review_voice_claim_with_non_voice_instrument_pressure,
        ):
            before = claim
            claim = mutator(claim, facts)
            if (
                mutator == self.guard._rehome_true_voice_from_fx_bucket
                and claim.source == "true_voice_instrument_rehome"
                and self._facts_have_short_percussive_hit_pressure(facts)
            ):
                claim = before
        claim = self.guard._enforce_final_invariant_consensus_firewall(raw_claim, claim, facts)
        if self._same_claim(seed, claim):
            return None
        return claim

    @staticmethod
    def _review_impossible_instrument_pitch_range(
        seed: ConsensusClaim,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        """Review narrow instrument leaves that fail hard pitch-range proof.

        This is a measured safety guard, not an identity scorer.  It only
        activates when a proposed internal instrument leaf has confident F0
        evidence outside a conservative hard range.  Broad instrument parents
        and low-confidence pitch estimates are left alone.
        """
        if seed.family != "Instruments" or seed.is_review:
            return None
        folder_path = seed.folder_path or seed.label
        check = check_instrument_pitch_range(folder_path, facts)
        if not check.is_hard_violation or check.rule is None or check.evidence is None:
            return None
        reason = (
            "measured pitch-range guard reviewed an impossible narrow "
            f"{check.rule.family_name} leaf: observed {check.evidence.observed_hz:.1f} Hz "
            f"from {check.evidence.source} with confidence {check.evidence.confidence:.2f} "
            f"and voiced ratio {check.evidence.voiced_ratio:.2f}; hard allowed range is "
            f"{check.rule.hard_min_hz:.1f}-{check.rule.hard_max_hz:.1f} Hz"
        )
        return review_claim(
            label="_TO_REVIEW/Instrument Pitch Range Conflict",
            source="final_instrument_pitch_range_conflict_review",
            reason=reason,
            shared=seed.shared_candidates,
            winner=seed,
            strength=max(0.99, seed.strength),
        )

    @staticmethod
    def _facts_have_short_percussive_hit_pressure(facts: SharedAudioFacts | None) -> bool:
        """Return True when a short hit has enough percussive role pressure to block voice rehome."""
        if facts is None:
            return False
        evidence = getattr(facts, "evidence", {})
        if not isinstance(evidence, dict):
            return False
        shape_vote = evidence.get("shape_vote", {})
        shape = str(shape_vote.get("primary_shape", "")) if isinstance(shape_vote, dict) else ""
        if shape not in {"single_hit", "hit_with_tail", "echo_tail_hit"}:
            return False
        roles = evidence.get("measured_roles", {})
        direct = evidence.get("direct_body_view", {})
        direct_roles = direct.get("measured_roles", {}) if isinstance(direct, dict) else {}

        def role_value(source: object, name: str) -> float:
            if not isinstance(source, dict):
                return 0.0
            try:
                return float(source.get(name, 0.0) or 0.0)
            except Exception:
                return 0.0

        pressure = max(
            role_value(roles, "percussive_one_shot"),
            role_value(direct_roles, "percussive_one_shot"),
        )
        return bool(pressure >= 0.25)

    @staticmethod
    def _same_claim(left: ConsensusClaim, right: ConsensusClaim) -> bool:
        """Return True when a guard did not change the routing claim."""
        return bool(
            left.family == right.family
            and left.sub_family == right.sub_family
            and (left.folder_path or left.label) == (right.folder_path or right.label)
            and left.source == right.source
            and left.reason == right.reason
        )
