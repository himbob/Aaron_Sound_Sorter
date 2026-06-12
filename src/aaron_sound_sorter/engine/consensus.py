"""Consensus runner for the two-voter product sorter."""

from __future__ import annotations

from typing import Any

from aaron_sound_sorter.domain.models import CategoryGuess, SharedAudioFacts, VoterResult
from aaron_sound_sorter.domain.policies import ConsensusPolicy
from aaron_sound_sorter.engine.concrete_fx_gate import ConcreteFxGateProtector
from aaron_sound_sorter.engine.consensus_candidates import SharedCandidateBuilder
from aaron_sound_sorter.engine.consensus_support import (
    _best_broad_instrument_loop_row,
    _candidate_voice_fit,
    _direct_body_roles,
    _drum_loop_evidence_strength,
    _has_near_human_voice_candidate,
    _is_human_voice_candidate_row,
    _role_value,
    _shape_name_and_confidence,
    _should_promote_direct_body_bass_loop,
    _strong_vocal_shape_true_bucket_evidence,
    label_depth,
    role_signature_from_row,
)
from aaron_sound_sorter.engine.consensus_winner import ConsensusWinnerPreselector
from aaron_sound_sorter.engine.family_claims import (
    ConsensusClaim,
    claim_from_candidate_row,
    claim_from_folder_path,
    review_claim,
)
from aaron_sound_sorter.engine.placement_depth import PlacementDepthDecider, synthetic_broad_bucket
from aaron_sound_sorter.voters.scoring_tools import compatible_tops_for_role, detected_parent_role_name, role_strength
from aaron_sound_sorter.voters.shape_voter import PITCHED_PHRASE_SHAPE, shape_compatible_tops

VOICE_PHRASE_SHAPES = {"vocal_phrase", PITCHED_PHRASE_SHAPE}


def is_voice_phrase_shape(shape_name: str) -> bool:
    """Return True for current and legacy voice-like phrase shape names."""
    return str(shape_name or "") in VOICE_PHRASE_SHAPES


def _fact_number(facts: SharedAudioFacts, key: str) -> float:
    """Read a numeric measured fact from evidence or feature_values_by_name."""
    evidence = facts.evidence if isinstance(getattr(facts, "evidence", None), dict) else {}
    for source in (evidence, getattr(facts, "feature_values_by_name", {}) or {}):
        if isinstance(source, dict) and key in source:
            try:
                return float(source.get(key) or 0.0)
            except Exception:
                return 0.0
    return 0.0


def _subpanel_number(facts: SharedAudioFacts, key: str) -> float:
    evidence = facts.evidence if isinstance(getattr(facts, "evidence", None), dict) else {}
    panels = evidence.get("physics_subpanels", {}) if isinstance(evidence, dict) else {}
    flat = panels.get("flat", {}) if isinstance(panels, dict) else {}
    try:
        return float(flat.get(key, 0.0) or 0.0) if isinstance(flat, dict) else 0.0
    except Exception:
        return 0.0


def _shape_texture_conflict_is_really_short_percussion(
    *,
    facts: SharedAudioFacts,
    winner: dict[str, Any],
    primary_shape: str,
    confidence: float,
) -> bool:
    """Return True when ShapeVoter's texture label should not steal from Drums.

    Some very short cymbal/guiro/metal hits have a noisy tail and get a
    ``texture_bed``/``noise_texture`` primary shape.  If measured role and
    drum-material panels already say short percussion, ShapeVoter should stand
    down instead of emitting an FX/Textural shape-sanity claim.
    """
    if str(winner.get("top_family", "")) != "Drums":
        return False
    if primary_shape not in {"texture_bed", "noise_texture"}:
        return False
    if confidence < 0.82:
        return False
    evidence = facts.evidence if isinstance(getattr(facts, "evidence", None), dict) else {}
    roles = evidence.get("measured_roles", {}) if isinstance(evidence, dict) else {}
    percussive = max(
        _role_value(roles if isinstance(roles, dict) else {}, "percussive_one_shot"),
        _role_value(roles if isinstance(roles, dict) else {}, "protected_percussive_one_shot"),
        _role_value(roles if isinstance(roles, dict) else {}, "low_kick_like_hit"),
    )
    duration = _fact_number(facts, "duration_sec")
    events = max(_fact_number(facts, "event_count_estimate"), _fact_number(facts, "onset_count"))
    attack = _fact_number(facts, "attack_rise_time_norm")
    temporal = _fact_number(facts, "temporal_centroid_ratio")
    compact = _subpanel_number(facts, "compact_struck_tonal_percussion_score")
    drum_branch = max(
        _subpanel_number(facts, "drum_hit_score"),
        _subpanel_number(facts, "drum_cymbal_source_score"),
        _subpanel_number(facts, "drum_guiro_scrape_source_score"),
        _subpanel_number(facts, "drum_metallic_percussion_source_score"),
        _subpanel_number(facts, "drum_kick_source_score"),
        _subpanel_number(facts, "drum_snare_source_score"),
        _subpanel_number(facts, "drum_clap_source_score"),
        _subpanel_number(facts, "drum_tom_conga_source_score"),
        _subpanel_number(facts, "drum_rim_stick_source_score"),
    )
    return bool(
        percussive >= 0.72
        and 0.0 < duration <= 0.65
        and 0.0 < events <= 3.0
        and attack <= 0.14
        and temporal <= 0.30
        and compact >= 0.62
        and drum_branch >= 0.58
    )


def _measured_voice_source_strength(facts: SharedAudioFacts) -> float:
    """Return low-level Human/Voice source strength without using filenames."""
    evidence = facts.evidence if isinstance(facts.evidence, dict) else {}
    values = []
    for key in ("human_spoken_voice_score", "voice_score", "voice_vocal_source_score", "human_voice_score"):
        try:
            values.append(float(evidence.get(key, 0.0) or 0.0))
        except (TypeError, ValueError):
            pass
    flat = (
        evidence.get("physics_subpanels", {}).get("flat", {})
        if isinstance(evidence.get("physics_subpanels"), dict)
        else {}
    )
    if isinstance(flat, dict):
        for key in ("human_spoken_voice_score", "voice_score", "voice_vocal_source_score", "human_voice_score"):
            try:
                values.append(float(flat.get(key, 0.0) or 0.0))
            except (TypeError, ValueError):
                pass
    return max(values) if values else 0.0


class ConsensusRunner:
    """Choose the best shared category from BrainVoter and PhysicsVoter."""

    def __init__(self, policy: ConsensusPolicy | None = None) -> None:
        self.policy = policy or ConsensusPolicy()
        self.depth_decider = PlacementDepthDecider()
        self.shared_candidate_builder = SharedCandidateBuilder()
        self.winner_preselector = ConsensusWinnerPreselector()
        self.concrete_fx_gate = ConcreteFxGateProtector(self.policy)

    def choose(
        self,
        brain_result: VoterResult,
        physics_result: VoterResult,
        facts: SharedAudioFacts,
    ) -> tuple[ConsensusClaim, list[ConsensusClaim]]:
        """Return the raw winner claim plus competing consensus claims."""
        if facts.is_broken_or_tiny:
            raw_review = self.review_decision(
                label=self.policy.broken_or_tiny_label,
                reason="shared audio facts marked the file broken or tiny",
                status="broken_or_tiny_review",
            )
            return raw_review, [raw_review]
        shared = self.shared_candidate_builder.build(brain_result, physics_result)
        if not shared:
            raw_review = self.review_decision(
                label=self.policy.no_consensus_label,
                reason="brain and physics had no shared category in their ranked lists",
                status="no_voter_consensus",
            )
            return raw_review, [raw_review]
        winner = self.winner_preselector.choose(shared, facts)
        raw_claim = claim_from_candidate_row(
            row=winner,
            source="strong_consensus",
            reason="brain and physics shared the winning category",
            shared=shared,
            can_override=False,
        )
        claims: list[ConsensusClaim] = []
        primary_claims = [
            self.top_family_sanity_decision(shared, winner, facts),
            self.role_sanity_decision(shared, winner, facts, brain_result, physics_result),
            self.shape_sanity_decision(shared, winner, facts),
        ]
        for claim in primary_claims:
            if claim is not None:
                claims.append(claim)
                depth_claim = self.depth_decider.refine_claim(raw_claim=claim, shared=shared, facts=facts)
                if depth_claim is not None:
                    claims.append(depth_claim)
        raw_depth_claim = self.depth_decider.refine_claim(raw_claim=raw_claim, shared=shared, facts=facts)
        if raw_depth_claim is not None:
            claims.append(raw_depth_claim)
        if float(winner["combined_rank_score"]) > self.policy.max_combined_rank_score:
            claims.append(
                self.review_decision(
                    label=self.policy.weak_consensus_label,
                    reason=(
                        "best shared category existed but was too weak: "
                        f"combined_rank_score={float(winner['combined_rank_score']):.1f}, "
                        f"limit={self.policy.max_combined_rank_score:.1f}"
                    ),
                    status="weak_voter_consensus",
                    shared=shared,
                    winner=winner,
                )
            )
        return raw_claim, claims

    def top_family_sanity_decision(
        self,
        shared: list[dict[str, Any]],
        winner: dict[str, Any],
        facts: SharedAudioFacts,
    ) -> ConsensusClaim | None:
        """Use a decisive measured top-family gate as a legality constraint.

        This is the general fix for tiny terminal categories: when the measured
        role/top-family audit is overwhelming, a broad family can be enforced
        before terminal labels compete.  Raw voter rankings remain visible in
        the manifest.
        """
        gate = facts.evidence.get("dynamic_role_gate", {}) if isinstance(facts.evidence, dict) else {}
        if not isinstance(gate, dict) or not winner:
            return None
        selected = gate.get("selected_top_families", [])
        if not isinstance(selected, list) or len(selected) != 1:
            return None
        selected_top = str(selected[0])
        if not selected_top or str(winner.get("top_family", "")) == selected_top:
            return None

        # Do not let a possibly noisy top-family gate override a very clear
        # transition shape that already supports the raw FX winner.
        shape = facts.evidence.get("shape_vote", {}) if isinstance(facts.evidence, dict) else {}
        if isinstance(shape, dict):
            primary_shape = str(shape.get("primary_shape", ""))
            try:
                shape_confidence = float(shape.get("confidence", 0.0) or 0.0)
            except Exception:
                shape_confidence = 0.0
            if shape_confidence >= 0.85 and primary_shape in {"transition_riser", "transition_drop"}:
                if str(winner.get("top_family", "")) in shape_compatible_tops(primary_shape):
                    return None

        if not self._dynamic_top_gate_is_decisive(gate):
            return None

        # Require independent structural support before enforcing the diagnostic
        # top-family gate.  This prevents a noisy top gate from moving real
        # vocals or drum loops away from a ShapeVoter-compatible winner.
        shape_supported = False
        parent_role = "unknown"
        parent_strength = 0.0
        if isinstance(shape, dict):
            primary_shape = str(shape.get("primary_shape", ""))
            try:
                shape_confidence = float(shape.get("confidence", 0.0) or 0.0)
            except Exception:
                shape_confidence = 0.0
            shape_tops = shape_compatible_tops(primary_shape)
            if shape_confidence >= 0.70 and shape_tops and selected_top in shape_tops:
                shape_supported = True
        roles = facts.evidence.get("measured_roles", {}) if isinstance(facts.evidence, dict) else {}
        role_supported = False
        if isinstance(roles, dict):
            parent_role = detected_parent_role_name(roles)
            parent_strength = role_strength(roles, parent_role) if parent_role != "unknown" else 0.0
            role_tops = compatible_tops_for_role(parent_role)
            if parent_strength >= self.policy.role_sanity_min_strength and role_tops:
                if selected_top in role_tops and str(winner.get("top_family", "")) not in role_tops:
                    role_supported = True
        if not (shape_supported or role_supported):
            return None

        concrete_fx_decision = self.concrete_fx_gate.override(
            selected_top=selected_top,
            shared=shared,
            winner=winner,
            facts=facts,
        )
        if concrete_fx_decision is not None:
            return concrete_fx_decision

        compatible_shared = [
            row
            for row in shared
            if str(row.get("top_family", "")) == selected_top and self.role_sanity_has_raw_support(row)
        ]
        if not compatible_shared:
            broad_decision = self.broad_measured_role_decision(
                top_family=selected_top,
                parent_role=parent_role,
                parent_strength=parent_strength,
                shared=shared,
                status="top_family_sanity_broad_bucket",
                reason=(
                    "raw shared winner conflicted with decisive measured top family "
                    f"{selected_top}; stopped at a broad measured-role bucket because no shared "
                    "terminal candidate in that family had enough raw support"
                ),
            )
            if broad_decision is not None:
                return broad_decision
            return self.review_decision(
                label=self.policy.role_conflict_label,
                reason=(
                    "raw shared winner conflicted with decisive measured top family "
                    f"{selected_top}, and no shared candidate in that family had enough raw support"
                ),
                status="measured_top_family_conflict_review",
                shared=shared,
                winner=winner,
            )
        compatible_shared.sort(
            key=lambda row: (
                label_depth(row),
                float(row.get("combined_rank_score", 9999.0)),
                int(row.get("physics_rank", 9999)),
                int(row.get("brain_rank", 9999)),
                str(row.get("label", "")),
            )
        )
        rescued = compatible_shared[0]
        return claim_from_candidate_row(
            row=rescued,
            source="top_family_sanity_consensus",
            reason=(
                "raw shared winner conflicted with decisive measured top family "
                f"{selected_top}; chose nearest shared candidate inside that family"
            ),
            shared=shared,
            can_override=True,
            strength=max(0.85, shape_confidence if shape_supported else 0.0, parent_strength),
        )

    @staticmethod
    def _dynamic_top_gate_is_decisive(gate: dict[str, Any]) -> bool:
        """Return True when the top-family diagnostic has a large margin."""
        scores = gate.get("top_scores", [])
        if not isinstance(scores, list) or len(scores) < 2:
            reason = str(gate.get("reason", ""))
            return "decisive_top_family_gap" in reason
        try:
            ordered = sorted(float(row.get("score", 999999.0)) for row in scores if isinstance(row, dict))
        except Exception:
            return False
        if len(ordered) < 2:
            return False
        best, second = ordered[0], ordered[1]
        return bool((second - best) >= 8.0 or second / max(best, 0.001) >= 4.0)

    def shape_sanity_decision(
        self,
        shared: list[dict[str, Any]],
        winner: dict[str, Any],
        facts: SharedAudioFacts,
    ) -> ConsensusClaim | None:
        """Use ShapeVoter as a structural sanity check, not a sorter.

        ShapeVoter never chooses a destination label directly.  It only says
        whether the existing shared winner is structurally plausible.  When a
        compatible shared candidate already has raw voter support, consensus may
        choose that candidate.  Otherwise the file goes to review with the shape
        evidence visible in the manifest.
        """
        shape = facts.evidence.get("shape_vote", {}) if isinstance(facts.evidence, dict) else {}
        if not isinstance(shape, dict) or not winner:
            return None
        primary = str(shape.get("primary_shape", ""))
        try:
            confidence = float(shape.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        if confidence < self.policy.shape_sanity_min_confidence:
            return None
        compatible_tops = shape_compatible_tops(primary)
        if not compatible_tops:
            return None
        if _shape_texture_conflict_is_really_short_percussion(
            facts=facts,
            winner=winner,
            primary_shape=primary,
            confidence=confidence,
        ):
            return None

        if is_voice_phrase_shape(primary) and confidence >= 0.75:
            evidence = facts.evidence if isinstance(facts.evidence, dict) else {}
            roles = evidence.get("measured_roles", {}) if isinstance(evidence, dict) else {}
            direct_roles = _direct_body_roles(facts)
            measured_voice = max(
                _role_value(roles if isinstance(roles, dict) else {}, "vocal_music_phrase"),
                _role_value(roles if isinstance(roles, dict) else {}, "vocal_phrase"),
                _role_value(roles if isinstance(roles, dict) else {}, "vocal_one_shot"),
                _role_value(roles if isinstance(roles, dict) else {}, "voiced_one_shot"),
                _role_value(direct_roles, "vocal_music_phrase"),
                _role_value(direct_roles, "vocal_phrase"),
                _role_value(direct_roles, "vocal_one_shot"),
                _role_value(direct_roles, "voiced_one_shot"),
            )
            winner_is_voice = _is_human_voice_candidate_row(winner)
            winner_top = str(winner.get("top_family", ""))
            strong_vocal_shape = bool(
                measured_voice >= 0.84 or _strong_vocal_shape_true_bucket_evidence(facts, confidence)
            )
            if strong_vocal_shape and winner_top == "FX" and not winner_is_voice:
                return claim_from_folder_path(
                    folder_path="Instruments/Voice/Phrase/One Shots",
                    source="shape_vocal_true_bucket_rescue",
                    reason=("measured voice role plus voice-like phrase shape blocked a non-voice FX winner"),
                    shared=shared,
                    raw_candidate_score=None,
                    brain_rank=None,
                    physics_rank=None,
                    shared_winner="Instruments/Voice/Phrase/One Shots",
                    can_override=True,
                    strength=max(0.90, confidence),
                    is_real_candidate=False,
                )

        # Bass-phrase shape is not allowed to steal a real drum loop.  Many
        # full drum loops have a dominant low/kick body and measure as
        # bass_phrase even though the temporal role is still a loop.  Let the
        # role/arbiter path handle those instead of producing a Synth Bass claim.
        if primary == "bass_phrase":
            drum_strength = _drum_loop_evidence_strength(facts)
            has_drum_loop_candidate = any(
                str(row.get("top_family", "")) == "Drums"
                and "drum loop" in str(row.get("folder_path") or row.get("label") or "").lower()
                and float(row.get("combined_rank_score", 9999.0) or 9999.0) <= 32.0
                for row in shared
            )
            if drum_strength >= 0.30 or has_drum_loop_candidate:
                return None

        if str(winner.get("top_family", "")) in compatible_tops:
            if (
                is_voice_phrase_shape(primary)
                and not _is_human_voice_candidate_row(winner)
                and _strong_vocal_shape_true_bucket_evidence(facts, confidence)
            ):
                return claim_from_folder_path(
                    folder_path="FX/Human and Voice FX",
                    source="shape_vocal_true_bucket_rescue",
                    reason=(
                        "measured vocal phrase shape plus direct/body or parent-role voice evidence "
                        "blocked a non-voice compatible winner"
                    ),
                    shared=shared,
                    raw_candidate_score=None,
                    brain_rank=None,
                    physics_rank=None,
                    shared_winner="FX/Human and Voice FX",
                    can_override=True,
                    strength=max(0.90, confidence),
                    is_real_candidate=False,
                )
            return None

        # High-confidence transition shapes are allowed to protect FX from a
        # misleading drum-loop parent role.  This is a shape legality check,
        # not a category-name rescue.
        shape = facts.evidence.get("shape_vote", {}) if isinstance(facts.evidence, dict) else {}
        if isinstance(shape, dict):
            primary_shape = str(shape.get("primary_shape", ""))
            try:
                shape_confidence = float(shape.get("confidence", 0.0) or 0.0)
            except Exception:
                shape_confidence = 0.0
            if shape_confidence >= 0.85 and primary_shape in {"transition_riser", "transition_drop"}:
                shape_tops = shape_compatible_tops(primary_shape)
                if str(winner.get("top_family", "")) in shape_tops:
                    return None

        compatible_shared = [
            row
            for row in shared
            if str(row.get("top_family", "")) in compatible_tops
            and float(row.get("combined_rank_score", 9999.0)) <= self.policy.shape_sanity_max_combined_rank_score
        ]

        if is_voice_phrase_shape(primary):
            voice_shared = [row for row in compatible_shared if _is_human_voice_candidate_row(row)]
            if voice_shared:
                voice_shared.sort(
                    key=lambda row: (
                        -_candidate_voice_fit(row),
                        float(row.get("combined_rank_score", 9999.0)),
                        int(row.get("physics_rank", 9999)),
                        int(row.get("brain_rank", 9999)),
                        str(row.get("label", "")),
                    )
                )
                rescued = voice_shared[0]
                return claim_from_candidate_row(
                    row=rescued,
                    source="shape_vocal_candidate_rescue",
                    reason=(
                        "raw shared winner conflicted with measured vocal phrase shape "
                        f"{primary}={confidence:.2f}; chose nearest Human/Voice shared candidate"
                    ),
                    shared=shared,
                    can_override=True,
                    strength=confidence,
                )
            if _strong_vocal_shape_true_bucket_evidence(facts, confidence):
                return claim_from_folder_path(
                    folder_path="FX/Human and Voice FX",
                    source="shape_vocal_true_bucket_rescue",
                    reason=(
                        "measured vocal phrase shape plus direct/body or parent-role voice evidence "
                        "blocked arbitrary FX leaf rescue"
                    ),
                    shared=shared,
                    raw_candidate_score=None,
                    brain_rank=None,
                    physics_rank=None,
                    shared_winner="FX/Human and Voice FX",
                    can_override=True,
                    strength=max(0.90, confidence),
                    is_real_candidate=False,
                )
            return self.review_decision(
                label=self.policy.shape_conflict_label,
                reason=(
                    "raw shared winner conflicted with measured vocal phrase shape "
                    f"{primary}={confidence:.2f} and no Human/Voice candidate or decisive voice evidence was present"
                ),
                status="measured_shape_conflict_review",
                shared=shared,
                winner=winner,
            )

        if compatible_shared:
            if primary == "pitched_repetition_phrase" and _measured_voice_source_strength(facts) >= 0.66:
                voice_shared = [row for row in compatible_shared if _is_human_voice_candidate_row(row)]
                if voice_shared:
                    voice_shared.sort(
                        key=lambda row: (
                            -_candidate_voice_fit(row),
                            float(row.get("combined_rank_score", 9999.0)),
                            int(row.get("physics_rank", 9999)),
                            int(row.get("brain_rank", 9999)),
                            str(row.get("label", "")),
                        )
                    )
                    rescued = voice_shared[0]
                    return claim_from_candidate_row(
                        row=rescued,
                        source="shape_pitched_repetition_voice_identity_guard",
                        reason=(
                            "measured pitched-repetition shape kept its low-level Human/Voice source "
                            "identity instead of broad Instrument Loops"
                        ),
                        shared=shared,
                        can_override=True,
                        strength=max(confidence, 0.90),
                    )
            broad_loop = _best_broad_instrument_loop_row(compatible_shared)
            if broad_loop is not None and primary in {"pitched_phrase", "sustained_pad"}:
                # v31.90: ShapeVoter is a structural witness, not a leaf sorter.
                # A pitched phrase/held pad shape should not narrow to Sax/Guitar/
                # Synth one-shot leaves when the voter set already contains the
                # broad Instrument Loops parent.  That keeps mixed sax/brass,
                # keys, bells, strings, and pads out of brittle source leaves.
                return claim_from_candidate_row(
                    row=broad_loop,
                    source="shape_sanity_consensus",
                    reason=(
                        "raw shared winner conflicted with measured temporal shape "
                        f"{primary}={confidence:.2f}; chose broad Instrument Loops "
                        "instead of a source-specific one-shot leaf"
                    ),
                    shared=shared,
                    can_override=True,
                    strength=confidence,
                )
            compatible_shared.sort(
                key=lambda row: (
                    label_depth(row) if primary in {"beat_loop", "top_loop"} else 9999,
                    float(row.get("combined_rank_score", 9999.0)),
                    int(row.get("physics_rank", 9999)),
                    int(row.get("brain_rank", 9999)),
                    str(row.get("label", "")),
                )
            )
            rescued = compatible_shared[0]
            return claim_from_candidate_row(
                row=rescued,
                source="shape_sanity_consensus",
                reason=(
                    "raw shared winner conflicted with measured temporal shape "
                    f"{primary}={confidence:.2f}; chose nearest shared compatible family"
                ),
                shared=shared,
                can_override=True,
                strength=confidence,
            )
        return self.review_decision(
            label=self.policy.shape_conflict_label,
            reason=(
                "raw shared winner conflicted with measured temporal shape "
                f"{primary}={confidence:.2f} and no compatible shared candidate had enough raw support"
            ),
            status="measured_shape_conflict_review",
            shared=shared,
            winner=winner,
        )

    def role_sanity_decision(
        self,
        shared: list[dict[str, Any]],
        winner: dict[str, Any],
        facts: SharedAudioFacts,
        brain_result: VoterResult,
        physics_result: VoterResult,
    ) -> ConsensusClaim | None:
        """Use measured parent role as an explicit broad-family sanity check."""
        roles = facts.evidence.get("measured_roles", {}) if isinstance(facts.evidence, dict) else {}
        if not isinstance(roles, dict) or not winner:
            return None
        parent_role = detected_parent_role_name(roles)
        parent_strength = role_strength(roles, parent_role) if parent_role != "unknown" else 0.0
        if _should_promote_direct_body_bass_loop(facts, parent_role):
            parent_role = "bass_loop"
            parent_strength = max(parent_strength, _role_value(_direct_body_roles(facts), "bass_loop"))
        if parent_strength < self.policy.role_sanity_min_strength:
            return None
        compatible_tops = compatible_tops_for_role(parent_role)
        if not compatible_tops:
            return None
        winner_role_fit = role_strength(role_signature_from_row(winner), parent_role)

        # Bass-loop roles can appear in full drum loops because the kick/bass body
        # dominates the spectrum.  Do not let a measured bass_loop role demote a
        # shared Drums/Drum Loops winner when the ranked candidates themselves
        # strongly agree on Drum Loops.  This keeps role sanity as a guardrail,
        # not a cross-family override hammer.
        winner_path = str(winner.get("folder_path") or winner.get("label") or "")
        if (
            parent_role == "bass_loop"
            and str(winner.get("top_family", "")) == "Drums"
            and "drum loop" in winner_path.lower()
        ):
            drum_loop_role_strength = max(
                _role_value(roles, "low_rhythmic_drum_loop"),
                _role_value(roles, "percussive_drum_loop"),
                _role_value(_direct_body_roles(facts), "low_rhythmic_drum_loop"),
                _role_value(_direct_body_roles(facts), "percussive_drum_loop"),
            )
            strong_drum_loop_candidate = any(
                str(row.get("top_family", "")) == "Drums"
                and "drum loop" in str(row.get("folder_path") or row.get("label") or "").lower()
                and float(row.get("combined_rank_score", 9999.0) or 9999.0) <= 4.0
                for row in shared
            )
            if drum_loop_role_strength >= 0.25 or strong_drum_loop_candidate:
                return None

        if str(winner.get("top_family", "")) in compatible_tops:
            broad_decision = self.same_family_role_bucket_decision(
                shared=shared,
                winner=winner,
                facts=facts,
                parent_role=parent_role,
                parent_strength=parent_strength,
                winner_role_fit=winner_role_fit,
            )
            if broad_decision is not None:
                return broad_decision
            return None

        # High-confidence transition shapes are allowed to protect FX from a
        # misleading drum-loop parent role.  This is a shape legality check,
        # not a category-name rescue.
        shape = facts.evidence.get("shape_vote", {}) if isinstance(facts.evidence, dict) else {}
        if isinstance(shape, dict):
            primary_shape = str(shape.get("primary_shape", ""))
            try:
                shape_confidence = float(shape.get("confidence", 0.0) or 0.0)
            except Exception:
                shape_confidence = 0.0
            if shape_confidence >= 0.85 and primary_shape in {"transition_riser", "transition_drop"}:
                shape_tops = shape_compatible_tops(primary_shape)
                if str(winner.get("top_family", "")) in shape_tops:
                    return None

        if parent_role == "bass_loop":
            low_rhythmic_strength = max(
                _role_value(roles, "low_rhythmic_drum_loop"),
                _role_value(roles, "percussive_drum_loop"),
                _role_value(_direct_body_roles(facts), "low_rhythmic_drum_loop"),
                _role_value(_direct_body_roles(facts), "percussive_drum_loop"),
            )
            has_direct_brain_bass_support = any(
                guess.top_family == "Instruments"
                and guess.rank <= 6
                and any(
                    fragment in str(guess.folder_path or guess.label).lower()
                    for fragment in ("bass", "808", "sub bass", "synth bass")
                )
                for guess in brain_result.guesses
            )
            if low_rhythmic_strength >= 0.75 and not has_direct_brain_bass_support:
                drum_loop_decision = self.broad_measured_role_decision(
                    top_family="Drums",
                    parent_role="low_rhythmic_drum_loop",
                    parent_strength=low_rhythmic_strength,
                    shared=shared,
                    status="role_claim_arbitration_broad_bucket",
                    reason=(
                        "measured bass_loop conflicted with a stronger low-rhythmic drum-loop claim "
                        "and no direct BrainVoter bass support was present; chose Drum Loops broad bucket"
                    ),
                )
                if drum_loop_decision is not None:
                    return drum_loop_decision

        compatible_shared = [
            row
            for row in shared
            if str(row.get("top_family", "")) in compatible_tops and self.role_sanity_has_raw_support(row)
        ]

        generic_pitched_loop_role = parent_role in {
            "pitched_music_loop",
            "pitched_music_phrase",
            "clean_sustained_tonal_instrument_loop",
            "mixed_music_loop",
        }
        if generic_pitched_loop_role and str(winner.get("top_family", "")) == "FX":
            winner_path_low = str(winner.get("folder_path") or winner.get("label") or "").lower()
            concrete_fx_fragments = ("siren", "alarm", "beep", "glitch", "stutter", "machine", "motor", "engine")
            brain_top_path = ""
            if brain_result.guesses:
                brain_top_path = str(brain_result.guesses[0].folder_path or brain_result.guesses[0].label or "").lower()
            try:
                winner_score = float(winner.get("combined_rank_score", 9999.0) or 9999.0)
            except (TypeError, ValueError):
                winner_score = 9999.0
            if (
                winner_score <= 8.0
                and any(fragment in winner_path_low for fragment in concrete_fx_fragments)
                and brain_top_path.startswith("fx/")
                and any(fragment in brain_top_path for fragment in concrete_fx_fragments)
            ):
                return None
        if generic_pitched_loop_role and "Instruments" in compatible_tops:
            broad_instrument = _best_broad_instrument_loop_row(shared)
            if broad_instrument is not None:
                return claim_from_candidate_row(
                    row=broad_instrument,
                    source="role_sanity_generic_pitched_broad_instrument_loop",
                    reason=(
                        "generic pitched/mixed loop role chose broad Instrument Loops "
                        "instead of an over-specific sax/guitar/synth/one-shot leaf"
                    ),
                    shared=shared,
                    can_override=True,
                    strength=max(0.82, min(0.96, parent_strength)),
                )
            broad_loop_claim = self.broad_measured_role_decision(
                top_family="Instruments",
                parent_role="pitched_music_loop",
                parent_strength=parent_strength,
                shared=shared,
                status="role_sanity_generic_pitched_broad_instrument_loop",
                reason=(
                    "generic pitched/mixed loop role stopped at broad Instrument Loops "
                    "because no safe broad shared row was available"
                ),
            )
            if broad_loop_claim is not None:
                return broad_loop_claim

        # Bass-loop broad bucket is a real product bucket, not a terminal leaf.
        # If measured audio strongly says bass_loop and BrainVoter has a direct
        # Bass/808/sub/synth-bass candidate near the top, do not let a generic
        # Instrument Loops shared row flatten the result.  This uses candidate
        # category labels, not source filenames.
        if parent_role == "bass_loop" and parent_strength >= 0.90:
            has_brain_bass_support = any(
                guess.top_family == "Instruments"
                and guess.rank <= 6
                and any(
                    fragment in str(guess.folder_path or guess.label).lower()
                    for fragment in ("bass", "808", "sub bass", "synth bass")
                )
                for guess in brain_result.guesses
            )
            if has_brain_bass_support:
                broad_bass = self.broad_measured_role_decision(
                    top_family="Instruments",
                    parent_role="bass_loop",
                    parent_strength=parent_strength,
                    shared=shared,
                    status="role_sanity_broad_bucket",
                    reason=(
                        "measured bass_loop plus direct BrainVoter Bass support; stopped at Bass Loops broad bucket "
                        "instead of generic Instrument Loops"
                    ),
                )
                if broad_bass is not None:
                    return broad_bass

        if compatible_shared:
            prefer_broad_loop_bucket = parent_role in {
                "bright_drum_loop",
                "percussive_drum_loop",
                "low_rhythmic_drum_loop",
            }
            compatible_shared.sort(
                key=lambda row: (
                    label_depth(row) if prefer_broad_loop_bucket else 9999,
                    float(row.get("combined_rank_score", 9999.0)),
                    -role_strength(role_signature_from_row(row), parent_role),
                    int(row.get("physics_rank", 9999)),
                    int(row.get("brain_rank", 9999)),
                    str(row.get("label", "")),
                )
            )
            rescued = compatible_shared[0]
            profile_leaf = self.prefer_profile_leaf_with_broad_support(
                rescued,
                facts,
                brain_result,
                physics_result,
                parent_role,
                parent_strength,
            )
            if profile_leaf is not None:
                return profile_leaf
            return claim_from_candidate_row(
                row=rescued,
                source="role_sanity_consensus",
                reason=(
                    "raw shared winner conflicted with measured parent role "
                    f"{parent_role}={parent_strength:.2f}; chose nearest shared compatible family"
                ),
                shared=shared,
                can_override=True,
                strength=parent_strength,
            )
        broad_decision = self.broad_measured_role_decision(
            top_family=str(compatible_tops[0]),
            parent_role=parent_role,
            parent_strength=parent_strength,
            shared=shared,
            status="role_sanity_broad_bucket",
            reason=(
                "raw shared winner conflicted with measured parent role "
                f"{parent_role}={parent_strength:.2f}; stopped at a broad measured-role bucket "
                "because no compatible terminal candidate had enough raw support"
            ),
        )
        if broad_decision is not None:
            return broad_decision
        return self.review_decision(
            label=self.policy.role_conflict_label,
            reason=(
                "raw shared winner conflicted with measured parent role "
                f"{parent_role}={parent_strength:.2f} and no compatible shared candidate had enough raw support"
            ),
            status="measured_role_conflict_review",
            shared=shared,
            winner=winner,
        )

    def same_family_role_bucket_decision(
        self,
        *,
        shared: list[dict[str, Any]],
        winner: dict[str, Any],
        facts: SharedAudioFacts,
        parent_role: str,
        parent_strength: float,
        winner_role_fit: float,
    ) -> ConsensusClaim | None:
        """Stop at a broad role bucket when a same-family leaf contradicts role evidence."""
        if parent_role == "percussive_one_shot":
            role_evidence = {}
            roles = facts.evidence.get("measured_roles", {}) if isinstance(facts.evidence, dict) else {}
            if isinstance(roles, dict) and isinstance(roles.get("evidence", {}), dict):
                role_evidence = roles.get("evidence", {})
            try:
                low_pitched_hit = float(role_evidence.get("low_pitched_hit_raw", 0.0) or 0.0)
            except Exception:
                low_pitched_hit = 0.0
            if parent_strength >= 0.90 and low_pitched_hit >= 0.80 and str(winner.get("top_family", "")) == "Drums":
                return claim_from_candidate_row(
                    row=winner,
                    source="role_sanity_consensus",
                    reason=(
                        "measured low pitched one-shot role supported the shared Drums candidate: "
                        f"{parent_role}={parent_strength:.2f}, low_pitched_hit={low_pitched_hit:.2f}"
                    ),
                    shared=shared,
                    can_override=True,
                )
            return None
        if parent_role == "bass_loop":
            if parent_strength >= 0.90 and str(winner.get("top_family", "")) == "Instruments":
                return self.broad_measured_role_decision(
                    top_family="Instruments",
                    parent_role=parent_role,
                    parent_strength=parent_strength,
                    shared=shared,
                    status="same_family_role_broad_bucket",
                    reason=(
                        "raw shared winner stayed in Instruments but measured role was a decisive "
                        f"bass loop: {parent_role}={parent_strength:.2f}; stopped at a bass broad bucket"
                    ),
                )
            return None
        if parent_role not in {"voiced_one_shot", "vocal_music_phrase"}:
            return None
        if parent_strength < 0.70:
            return None
        primary_shape, shape_confidence = _shape_name_and_confidence(facts)
        if not (is_voice_phrase_shape(primary_shape) or primary_shape in {"hit_with_tail", "single_hit"}):
            return None
        if is_voice_phrase_shape(primary_shape) and shape_confidence < 0.75:
            return None
        if not is_voice_phrase_shape(primary_shape) and parent_strength < 0.74:
            return None
        if str(winner.get("top_family", "")) not in {"FX", "Instruments"}:
            return None
        try:
            winner_score = float(winner.get("combined_rank_score", 9999.0) or 9999.0)
        except Exception:
            winner_score = 9999.0
        # A measured voiced_one_shot score alone is not enough to synthesize a
        # Human/Voice bucket.  Reeds, pitched percussion, animal-like tonal hits,
        # and some keys can all look voice-like in the full or direct/body view.
        # Broad Human/Voice therefore needs one of three positive claims:
        #   1. a nearby Human/Voice candidate with role support,
        #   2. an extremely clear vocal-phrase shape, or
        #   3. a same-family candidate whose role signature itself is voice-like.
        # The third case keeps old bird/siren/FX false-positive tests honest
        # without reopening the piano/sax voice-sink failure.
        has_positive_voice_candidate = _has_near_human_voice_candidate(shared, winner_score)
        strong_measured_vocal_shape = bool(
            parent_role == "voiced_one_shot"
            and is_voice_phrase_shape(primary_shape)
            and shape_confidence >= 0.90
            and parent_strength >= 0.90
        )
        formant_hit_voice_claim = bool(
            parent_role == "voiced_one_shot"
            and primary_shape == "hit_with_tail"
            and parent_strength >= 0.84
            and winner_role_fit >= 0.75
        )
        if parent_role == "voiced_one_shot" and not (
            has_positive_voice_candidate or strong_measured_vocal_shape or formant_hit_voice_claim
        ):
            return None
        return self.broad_measured_role_decision(
            top_family=str(winner.get("top_family", "")),
            parent_role=parent_role,
            parent_strength=parent_strength,
            shared=shared,
            status="same_family_role_broad_bucket",
            reason=(
                "raw shared winner stayed in a compatible top family, and a positive Human/Voice claim "
                f"supported a broad voice bucket: {parent_role}={parent_strength:.2f}, "
                f"candidate_fit={winner_role_fit:.2f}"
            ),
            can_override=has_positive_voice_candidate,
        )

    @staticmethod
    def broad_measured_role_decision(
        *,
        top_family: str,
        parent_role: str,
        parent_strength: float,
        shared: list[dict[str, Any]],
        status: str,
        reason: str,
        can_override: bool = True,
    ) -> ConsensusClaim | None:
        """Return a conservative broad measured-role bucket when terminal evidence is unsafe."""
        if parent_strength < 0.70 or parent_role == "unknown":
            return None
        broad = synthetic_broad_bucket(top_family=top_family, parent_role=parent_role)
        if broad is None:
            return None
        return claim_from_folder_path(
            folder_path=str(broad["folder_path"]),
            source=status,
            reason=reason,
            shared=shared,
            raw_candidate_score=None,
            brain_rank=None,
            physics_rank=None,
            shared_winner=str(broad["label"]),
            can_override=can_override,
            strength=parent_strength,
            is_real_candidate=False,
        )

    def prefer_profile_leaf_with_broad_support(
        self,
        broad_row: dict[str, Any],
        facts: SharedAudioFacts,
        brain_result: VoterResult,
        physics_result: VoterResult,
        parent_role: str,
        parent_strength: float,
    ) -> ConsensusClaim | None:
        """Allow one voter to choose a close leaf when the other supports its broad family.

        This handles bass identity only: PhysicsVoter may support the broad
        instrument-loop family while BrainVoter has a strong profile-compatible
        bass leaf.  The decision status makes that bridge explicit instead of
        pretending it was strict two-label consensus.
        """
        if parent_role != "bass_loop" or parent_strength < 0.90:
            return None
        broad_top = str(broad_row.get("top_family", ""))
        broad_signature = role_signature_from_row(broad_row)
        broad_fit = role_strength(broad_signature, parent_role)
        physics_family_rank = self.best_family_rank(physics_result, broad_top)
        if physics_family_rank > 7:
            return None

        candidates: list[CategoryGuess] = []
        for guess in brain_result.guesses:
            if guess.top_family != broad_top or guess.rank > 4:
                continue
            evidence = guess.evidence if isinstance(guess.evidence, dict) else {}
            signature = evidence.get("candidate_role_signature", {})
            if not isinstance(signature, dict):
                continue
            fit = role_strength(signature, parent_role)
            if fit < 0.90 or fit < broad_fit + 0.20:
                continue
            candidates.append(guess)
        if not candidates:
            return None

        def leaf_role_fit(guess: CategoryGuess) -> float:
            evidence = guess.evidence if isinstance(guess.evidence, dict) else {}
            signature = evidence.get("candidate_role_signature", {})
            return role_strength(signature if isinstance(signature, dict) else {}, parent_role)

        candidates.sort(
            key=lambda guess: (
                -leaf_role_fit(guess),
                guess.rank,
                guess.score,
                guess.label,
            )
        )
        leaf = candidates[0]
        return claim_from_folder_path(
            folder_path=leaf.folder_path,
            source="role_sanity_family_bridge",
            reason=(
                "measured parent role matched a brain leaf while physics supported the same broad family: "
                f"{parent_role}={parent_strength:.2f}, physics_family_rank={physics_family_rank}"
            ),
            shared=[broad_row],
            raw_candidate_score=float(leaf.rank + physics_family_rank),
            brain_rank=leaf.rank,
            physics_rank=physics_family_rank,
            shared_winner=leaf.label,
            can_override=True,
            strength=parent_strength,
            is_real_candidate=True,
        )

    @staticmethod
    def best_family_rank(result: VoterResult, top_family: str) -> int:
        """Return the best rank where a voter supports a broad family."""
        ranks = [guess.rank for guess in result.guesses if guess.top_family == top_family]
        return min(ranks) if ranks else 9999

    def role_sanity_has_raw_support(self, row: dict[str, Any]) -> bool:
        """Return whether a compatible candidate is close enough in raw voter ranks."""
        combined = float(row.get("combined_rank_score", 9999.0))
        brain_rank = int(row.get("brain_rank", 9999))
        physics_rank = int(row.get("physics_rank", 9999))
        if combined > self.policy.role_sanity_max_combined_rank_score:
            return False
        return (
            min(brain_rank, physics_rank) <= self.policy.role_sanity_primary_voter_rank
            and max(brain_rank, physics_rank) <= self.policy.role_sanity_secondary_voter_rank
        )

    def shared_candidates(self, brain_result: VoterResult, physics_result: VoterResult) -> list[dict[str, Any]]:
        """Return shared category candidates sorted by combined rank strength."""
        return self.shared_candidate_builder.build(brain_result, physics_result)

    @staticmethod
    def candidate_row(label: str, brain_guess: CategoryGuess, physics_guess: CategoryGuess) -> dict[str, Any]:
        """Build one comparable shared-candidate row."""
        return SharedCandidateBuilder.row(label, brain_guess, physics_guess)

    @staticmethod
    def review_decision(
        *,
        label: str,
        reason: str,
        status: str,
        shared: list[dict[str, Any]] | None = None,
        winner: dict[str, Any] | None = None,
    ) -> ConsensusClaim:
        """Return a review decision."""
        return review_claim(
            label=label,
            reason=reason,
            source=status,
            shared=shared,
            winner=winner,
        )
