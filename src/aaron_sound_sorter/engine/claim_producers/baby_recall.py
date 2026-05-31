"""Baby-brain recall claim producer.

Baby brains are diagnostic recall lanes.  They can be valuable when the full
brain collapses rare sounds into a broad or wrong FX neighborhood, but they are
not terminal truth.  This producer converts strong baby-lane guesses into broad,
measured-role-compatible claims only.  The FamilyClaimArbiter still owns the
final placement decision.
"""

from __future__ import annotations

from typing import Any

from aaron_sound_sorter.engine.brain_competence_policy import BrainCompetencePolicy
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.decision_helpers import (
    _feature_number_from_facts,
    _has_drum_loop_structure_support,
    _measured_role_from_facts,
    _norm_path,
    _path_has_any,
    _safe_int,
    _shape_confidence_from_facts,
    _shape_vote_from_facts,
)
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_folder_path


class BabyRecallClaimProducer:
    """Produce safe broad claims from optional baby-brain recall lanes.

    The producer is deliberately narrow: every candidate needs both a baby-lane
    path match and measured audio support.  This keeps brain specialization in a
    readable module while avoiding a return to filename-like or first-match
    routing.
    """

    def __init__(self, competence_policy: BrainCompetencePolicy | None = None) -> None:
        self.competence_policy = competence_policy or BrainCompetencePolicy()

    def produce(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return zero or one baby-recall claim for this decision context."""
        claim = self.best_claim(context)
        return [] if claim is None else [claim]

    def best_claim(self, context: DecisionContext) -> ConsensusClaim | None:
        """Return the strongest measured-compatible baby claim, if one exists."""
        if context.facts is None or not isinstance(context.facts.evidence, dict):
            return None

        role_name = str(context.eligibility.role_name or "")
        shape_name = _shape_vote_from_facts(context.facts)
        shape_confidence = _shape_confidence_from_facts(context.facts)
        measured_role = _measured_role_from_facts(context.facts)

        rows = self.baby_guess_rows(context)
        brass_lane_count = self._supporting_lane_count(
            rows,
            fragments=("sax", "saxophone"),
            max_rank=5,
        )
        candidates: list[ConsensusClaim] = []
        for row in rows:
            claim = self.claim_from_row(
                row=row,
                context=context,
                role_name=role_name,
                measured_role=measured_role,
                shape_name=shape_name,
                shape_confidence=shape_confidence,
                brass_lane_count=brass_lane_count,
            )
            if claim is not None:
                candidates.append(claim)

        candidates.sort(key=lambda claim: (-claim.strength, claim.raw_candidate_score or 9999.0, claim.folder_path))
        return candidates[0] if candidates else None

    @staticmethod
    def baby_guess_rows(context: DecisionContext) -> list[dict[str, Any]]:
        """Return top baby-lane guesses from fact diagnostics."""
        if context.facts is None or not isinstance(context.facts.evidence, dict):
            return []
        rows: list[dict[str, Any]] = []
        evidence = context.facts.evidence
        for key in ("core_baby_vote_result", "spread_baby_vote_result", "outlier_baby_vote_result"):
            result = evidence.get(key, {})
            if not isinstance(result, dict):
                continue
            diagnostics = result.get("diagnostics", {}) if isinstance(result.get("diagnostics", {}), dict) else {}
            if diagnostics.get("enabled") is False:
                continue
            lane_name = str(diagnostics.get("lane_name") or key.replace("_vote_result", ""))
            for guess in result.get("top_guesses", [])[:5]:
                if isinstance(guess, dict):
                    rows.append({**guess, "_lane_name": lane_name})
        return rows

    @staticmethod
    def _supporting_lane_count(
        rows: list[dict[str, Any]],
        *,
        fragments: tuple[str, ...],
        max_rank: int,
    ) -> int:
        """Return how many baby lanes support a path family in their top guesses."""
        lanes: set[str] = set()
        for row in rows:
            path = _norm_path(str(row.get("folder_path") or row.get("label") or ""))
            rank = _safe_int(row.get("rank"), 999)
            if rank > max_rank:
                continue
            if _path_has_any(path, fragments):
                lanes.add(str(row.get("_lane_name") or "baby"))
        return len(lanes)

    def claim_from_row(
        self,
        *,
        row: dict[str, Any],
        context: DecisionContext,
        role_name: str,
        measured_role: str,
        shape_name: str,
        shape_confidence: float,
        brass_lane_count: int = 0,
    ) -> ConsensusClaim | None:
        """Convert one baby-lane guess into a safe broad claim, if legal."""
        path = _norm_path(str(row.get("folder_path") or row.get("label") or ""))
        rank = _safe_int(row.get("rank"), 999)
        lane_name = str(row.get("_lane_name") or "baby")
        lane_strength = self.competence_policy.safe_claim_strength(lane_name)
        raw_score = float(context.raw.raw_candidate_score or 9999.0)

        if self.row_supports_clap(
            path, rank, role_name, shape_name, shape_confidence
        ) and self.raw_context_allows_clap_recall(context):
            return self._folder_claim(
                context=context,
                folder_path="Drums/Claps Snaps Slaps/Generic Clap/One Shots",
                source="baby_recall_clap_claim",
                reason=f"{lane_name} recalled a clap/snap candidate while measured evidence supported a short percussive hit",
                rank=rank,
                strength=max(lane_strength, 0.91),
                raw_score=raw_score,
            )
        if self.row_supports_kick(
            path, rank, role_name, shape_name, shape_confidence
        ) and self.raw_context_allows_kick_recall(context):
            return self._folder_claim(
                context=context,
                folder_path="Drums/Kick Drums/Generic Kick/One Shots",
                source="baby_recall_kick_claim",
                reason=f"{lane_name} recalled a kick candidate while measured evidence supported a percussive one-shot",
                rank=rank,
                strength=max(lane_strength, 0.93),
                raw_score=raw_score,
            )
        if self.row_supports_drum_loop(
            path, rank, role_name, measured_role, shape_name, shape_confidence
        ) and self.product_brain_or_shared_candidates_support_drum_loop(context):
            return self._folder_claim(
                context=context,
                folder_path="Drums/Drum Loops/Loops",
                source="baby_recall_drum_loop_claim",
                reason=f"{lane_name} recalled a drum-loop candidate while measured evidence supported drum-loop structure",
                rank=rank,
                strength=max(lane_strength, 0.90),
                raw_score=raw_score,
            )
        if self.row_supports_brass_woodwind(
            path,
            rank,
            context.raw,
            role_name,
            measured_role,
            shape_name,
            shape_confidence,
            brass_lane_count,
        ):
            return self._folder_claim(
                context=context,
                folder_path="Instruments/Brass and Woodwinds/Loops",
                source="baby_recall_brass_woodwind_claim",
                reason=f"{lane_name} recalled a brass/woodwind candidate while measured evidence supported pitched instrument structure",
                rank=rank,
                strength=max(lane_strength, 0.86),
                raw_score=raw_score,
            )
        if self.row_supports_bass(
            path, rank, role_name, shape_name, shape_confidence
        ) and self.product_brain_or_shared_candidates_support_bass(context):
            return self._folder_claim(
                context=context,
                folder_path="Instruments/Bass/Bass Loops",
                source="baby_recall_bass_claim",
                reason=f"{lane_name} recalled a bass candidate while measured evidence supported bass-loop structure",
                rank=rank,
                strength=max(lane_strength, 0.90),
                raw_score=raw_score,
            )
        return None

    @staticmethod
    def product_brain_or_shared_candidates_support_bass(context: DecisionContext) -> bool:
        """Return True when product-voter evidence also has a bass candidate.

        Baby lanes are useful recall witnesses, but a lone baby bass row can
        over-narrow low-heavy mixed musical loops.  Require the product
        BrainVoter or shared raw candidate table to expose a nearby bass path
        before turning recall into a Bass Loops claim.
        """
        fragments = ("bass", "808", "sub bass", "synth bass", "electric bass", "upright bass")
        if context.brain_result is not None:
            for guess in context.brain_result.guesses[:8]:
                path = _norm_path(str(guess.folder_path or guess.label or ""))
                if _path_has_any(path, fragments):
                    return True
        for row in context.raw.shared_candidates[:12]:
            path = _norm_path(str(row.get("folder_path") or row.get("label") or ""))
            if _path_has_any(path, fragments):
                return True
        return False

    @staticmethod
    def raw_context_allows_clap_recall(context: DecisionContext) -> bool:
        """Return True when a baby clap row is not contradicting stronger raw evidence."""
        snare_score = _feature_number_from_facts(context.facts, "drum_snare_source_score")
        clap_score = _feature_number_from_facts(context.facts, "drum_clap_source_score")
        if snare_score >= 0.76 and snare_score >= clap_score + 0.16:
            return False

        raw_path = _norm_path(str(context.raw.folder_path or context.raw.label or ""))
        if _path_has_any(raw_path, ("bass", "808", "sub hit", "sub bass", "kick", "tom", "boom")):
            return False
        return context.raw.final_top != "Instruments"

    @staticmethod
    def raw_context_allows_kick_recall(context: DecisionContext) -> bool:
        """Return True when baby kick recall has low-hit physics support."""
        low_pitched_hit = _feature_number_from_facts(context.facts, "low_pitched_hit_raw")
        low_total = _feature_number_from_facts(context.facts, "low_total")
        high_total = _feature_number_from_facts(context.facts, "high_total")
        event_count = max(
            _feature_number_from_facts(context.facts, "event_count_estimate"),
            _feature_number_from_facts(context.facts, "onset_count"),
        )
        duration = _feature_number_from_facts(context.facts, "duration_sec")
        has_low_hit_physics = bool(
            low_pitched_hit >= 0.72
            and low_total >= 0.60
            and low_total >= high_total
            and (event_count <= 2.0 or event_count == 0.0)
            and (duration <= 1.25 or duration == 0.0)
        )
        if not has_low_hit_physics:
            return False

        raw_path = _norm_path(str(context.raw.folder_path or context.raw.label or ""))
        if _path_has_any(raw_path, ("snare", "clap", "snap", "rimshot", "sidestick", "hat", "cymbal")):
            return False
        return not (context.raw.final_top == "Instruments" and not _path_has_any(raw_path, ("bass", "808", "sub bass")))

    @staticmethod
    def product_brain_or_shared_candidates_support_drum_loop(context: DecisionContext) -> bool:
        """Return True when product/shared evidence exposes a real drum-loop candidate."""
        fragments = ("drum loop", "drum loops", "breakbeat", "breaks")
        raw_path = _norm_path(str(context.raw.folder_path or context.raw.label or ""))
        if context.raw.final_top == "FX" and _path_has_any(
            raw_path,
            ("riser", "build", "downlifter", "drop", "sweep", "whoosh", "reverse", "transition"),
        ):
            return False
        if context.brain_result is not None:
            for guess in context.brain_result.guesses[:8]:
                path = _norm_path(str(guess.folder_path or guess.label or ""))
                if _path_has_any(path, fragments):
                    return True
        raw_score = float(context.raw.raw_candidate_score or 9999.0)
        for row in context.raw.shared_candidates[:12]:
            path = _norm_path(str(row.get("folder_path") or row.get("label") or ""))
            if not _path_has_any(path, fragments):
                continue
            try:
                score = float(row.get("combined_rank_score", 9999.0) or 9999.0)
            except (TypeError, ValueError):
                score = 9999.0
            if score <= raw_score + 12.0:
                return True
        return False

    @staticmethod
    def _folder_claim(
        *,
        context: DecisionContext,
        folder_path: str,
        source: str,
        reason: str,
        rank: int,
        strength: float,
        raw_score: float,
    ) -> ConsensusClaim:
        """Build a broad baby-recall claim while preserving raw-context evidence."""
        return claim_from_folder_path(
            folder_path=folder_path,
            source=source,
            reason=f"{reason}; raw={context.raw.folder_path}",
            shared=context.raw.shared_candidates,
            raw_candidate_score=min(raw_score + 2.0, float(rank) + 3.0),
            brain_rank=rank,
            physics_rank=None,
            shared_winner=folder_path,
            can_override=True,
            strength=strength,
            is_real_candidate=True,
        )

    @staticmethod
    def row_supports_clap(path: str, rank: int, role_name: str, shape_name: str, shape_confidence: float) -> bool:
        """Return True when a baby row may support a broad clap claim."""
        return bool(
            rank <= 3
            and _path_has_any(path, ("clap", "claps", "snap", "snaps", "slap", "claps snaps slaps"))
            and role_name in {"percussive_one_shot", "protected_percussive_one_shot"}
            and shape_name in {"single_hit", "hit_with_tail"}
            and shape_confidence >= 0.70
        )

    @staticmethod
    def row_supports_kick(path: str, rank: int, role_name: str, shape_name: str, shape_confidence: float) -> bool:
        """Return True when a baby row may support a broad kick claim."""
        return bool(
            rank <= 3
            and _path_has_any(path, ("kick", "kick drums", "sub kick", "short kick"))
            and role_name in {"percussive_one_shot", "protected_percussive_one_shot", "low_kick_like_hit"}
            and shape_name in {"hit_with_tail", "single_hit"}
            and shape_confidence >= 0.70
        )

    @staticmethod
    def row_supports_drum_loop(
        path: str,
        rank: int,
        role_name: str,
        measured_role: str,
        shape_name: str,
        shape_confidence: float,
        brass_lane_count: int = 0,
    ) -> bool:
        """Return True when a baby row may support a broad drum-loop claim."""
        return bool(
            rank <= 4
            and _path_has_any(path, ("drum loop", "drum loops", "break", "breaks"))
            and _has_drum_loop_structure_support(role_name, measured_role, shape_name, shape_confidence)
        )

    @staticmethod
    def row_supports_brass_woodwind(
        path: str,
        rank: int,
        raw: object,
        role_name: str,
        measured_role: str,
        shape_name: str,
        shape_confidence: float,
        brass_lane_count: int = 0,
    ) -> bool:
        """Return True when baby reed recall may rescue unsafe FX into broad winds."""
        raw_path = _norm_path(str(getattr(raw, "folder_path", "")))
        final_top = str(getattr(raw, "final_top", ""))
        raw_is_dangerous_fx = final_top == "FX" and _path_has_any(
            raw_path,
            (
                "alarm",
                "siren",
                "human and voice",
                "glitch",
                "stutter",
                "riser",
                "build",
                "animals",
                "creatures",
                "bird",
                "cat",
                "dog",
            ),
        )
        raw_is_generic_instrument = final_top == "Instruments" and _path_has_any(
            raw_path,
            ("instrument loops", "mixed musical loops"),
        )
        if raw_is_generic_instrument:
            return False

        generic_pitched_role = role_name in {
            "pitched_music_loop",
            "pitched_music_phrase",
            "clean_sustained_tonal_instrument_loop",
            "mixed_music_loop",
        } or measured_role in {
            "pitched_music_loop",
            "pitched_music_phrase",
            "clean_sustained_tonal_instrument_loop",
            "mixed_music_loop",
        }
        specific_reed_role = role_name in {
            "pitched_reed_or_instrument_loop",
            "pitched_reed_or_instrument_phrase",
        } or measured_role in {
            "pitched_reed_or_instrument_loop",
            "pitched_reed_or_instrument_phrase",
        }
        exact_sax_top_hit = bool(rank <= 1 and _path_has_any(path, ("sax", "saxophone")))
        enough_lane_support = bool(brass_lane_count >= 2 or exact_sax_top_hit)
        return bool(
            (rank <= 3 if specific_reed_role else rank <= 2)
            and raw_is_dangerous_fx
            and (specific_reed_role or generic_pitched_role)
            and enough_lane_support
            and shape_name in {"pitched_phrase", "vocal_phrase", "sustained_pad"}
            and shape_confidence >= 0.78
            and _path_has_any(path, ("sax", "saxophone"))
        )

    @staticmethod
    def row_supports_bass(path: str, rank: int, role_name: str, shape_name: str, shape_confidence: float) -> bool:
        """Return True when a baby row may support a broad bass-loop claim."""
        return bool(
            rank <= 3
            and role_name == "bass_loop"
            and shape_name == "bass_phrase"
            and shape_confidence >= 0.80
            and _path_has_any(path, ("bass", "808", "sub bass", "synth bass", "electric bass", "upright bass"))
        )
