"""Protection rules for concrete FX consensus against generic family gates."""

from __future__ import annotations

from typing import Any

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.domain.policies import ConsensusPolicy
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_candidate_row
from aaron_sound_sorter.voters.scoring_tools import role_strength
from aaron_sound_sorter.voters.shape_voter import shape_compatible_tops


class ConcreteFxGateProtector:
    """Keep a generic Instrument gate from erasing strong concrete FX agreement."""

    def __init__(self, policy: ConsensusPolicy) -> None:
        self.policy = policy

    def override(
        self,
        *,
        selected_top: str,
        shared: list[dict[str, Any]],
        winner: dict[str, Any],
        facts: SharedAudioFacts | None = None,
    ) -> ConsensusClaim | None:
        """Return a concrete FX claim when the measured top-family gate overreaches."""
        if selected_top != "Instruments":
            return None
        if str(winner.get("top_family", "")) != "FX":
            return None
        if self.has_supported_shared_family(shared, "Instruments"):
            return None
        if self.measured_pitched_music_should_block_override(facts, shared):
            return None
        if self.measured_instrument_shape_should_block_override(facts):
            return None

        concrete_fx_rows = [row for row in shared if self.is_concrete_fx_candidate(row) and self.has_raw_support(row)]
        if not concrete_fx_rows:
            return None
        concrete_fx_rows.sort(
            key=lambda row: (
                float(row.get("combined_rank_score", 9999.0)),
                int(row.get("physics_rank", 9999)),
                int(row.get("brain_rank", 9999)),
                str(row.get("label", "")),
            )
        )
        rescued = concrete_fx_rows[0]
        return claim_from_candidate_row(
            row=rescued,
            source="concrete_fx_gate_override",
            reason=(
                "concrete FX agreement protected from a generic Instrument-loop "
                "top-family gate; no shared Instrument candidate had raw support"
            ),
            shared=shared,
            can_override=True,
        )

    def measured_instrument_shape_should_block_override(
        self,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when measured music structure makes FX rescue unsafe.

        The concrete-FX gate exists to keep real transition/action FX from being
        flattened into a generic Instrument bucket. It must not do the opposite:
        when ShapeVoter and measured roles both say the audio is a music/instrument
        loop and the FX role layer does not explicitly allow FX, candidate names
        such as ``Synth Riser`` are not enough evidence to force an FX folder.
        """
        if facts is None or not isinstance(facts.evidence, dict):
            return False
        shape = facts.evidence.get("shape_vote", {})
        if not isinstance(shape, dict):
            return False
        primary_shape = str(shape.get("primary_shape", ""))
        try:
            shape_confidence = float(shape.get("confidence", 0.0) or 0.0)
        except Exception:
            shape_confidence = 0.0
        if shape_confidence < 0.70:
            return False
        if self._learned_or_measured_mixed_instrument_loop_shape(primary_shape, shape_confidence, facts.evidence):
            return True
        if "FX" in shape_compatible_tops(primary_shape):
            return False
        roles = facts.evidence.get("measured_roles", {})
        if not isinstance(roles, dict):
            return False
        pitched_strength = max(
            role_strength(roles, "pitched_music_loop"),
            role_strength(roles, "pitched_music_phrase"),
            role_strength(roles, "mixed_music_loop"),
            role_strength(roles, "clean_sustained_tonal_instrument_loop"),
        )
        if pitched_strength < 0.60:
            return False
        layer = facts.evidence.get("physics_layer_decision")
        if isinstance(layer, dict):
            try:
                fx_strength = float(layer.get("fx_role_strength", 0.0) or 0.0)
            except Exception:
                fx_strength = 0.0
            try:
                fx_conflict = float(layer.get("fx_role_conflict_strength", 1.0) or 1.0)
            except Exception:
                fx_conflict = 1.0
            if bool(layer.get("fx_role_allows_fx")) and fx_strength >= 0.70 and fx_conflict < 0.56:
                return False
        return True

    @staticmethod
    def _learned_or_measured_mixed_instrument_loop_shape(
        primary_shape: str,
        shape_confidence: float,
        evidence: dict[str, Any],
    ) -> bool:
        """Return True when measured/learned shape says mixed musical loop.

        Concrete FX protection is useful for real transition/action FX, but it
        must stand down when the low-level shape lane has already identified a
        mixed instrumental loop.  Wet, synthetic, or impact-like colors inside a
        mixed loop are not enough to switch top family to FX.
        """
        if primary_shape == "mixed_instrument_loop" and shape_confidence >= 0.86:
            return True
        learned_shape = str(evidence.get("learned_shape_memory_shape") or "")
        learned_matched = bool(evidence.get("learned_shape_memory_matched"))
        try:
            learned_confidence = float(evidence.get("learned_shape_memory_confidence", 0.0) or 0.0)
        except Exception:
            learned_confidence = 0.0
        return bool(learned_matched and learned_shape == "mixed_instrument_loop" and learned_confidence >= 0.72)

    def measured_pitched_music_should_block_override(
        self,
        facts: SharedAudioFacts | None,
        shared: list[dict[str, Any]],
    ) -> bool:
        """Return True when stable pitched music should not be forced to FX."""
        if facts is None or not isinstance(facts.evidence, dict):
            return False
        roles = facts.evidence.get("measured_roles", {})
        if not isinstance(roles, dict):
            return False
        pitched_strength = max(
            role_strength(roles, "pitched_music_loop"),
            role_strength(roles, "pitched_music_phrase"),
        )
        if pitched_strength < 0.84:
            return False
        shape = facts.evidence.get("shape_vote", {})
        primary_shape = str(shape.get("primary_shape", "")) if isinstance(shape, dict) else str(shape or "")
        if primary_shape not in {"pitched_phrase", "sustained_pad"}:
            return False
        fx_rows = [row for row in shared if str(row.get("top_family", "")) == "FX" and self.has_raw_support(row)]
        if not fx_rows:
            return False
        fx_rows.sort(
            key=lambda row: (
                float(row.get("combined_rank_score", 9999.0)),
                int(row.get("physics_rank", 9999)),
                int(row.get("brain_rank", 9999)),
                str(row.get("label", "")),
            )
        )
        best_path = str(fx_rows[0].get("folder_path", fx_rows[0].get("label", ""))).lower()
        if any(fragment in best_path for fragment in CONCRETE_MOTION_FRAGMENTS):
            return False
        return any(fragment in best_path for fragment in ABSTRACT_FX_FRAGMENTS)

    def has_supported_shared_family(self, shared: list[dict[str, Any]], top_family: str) -> bool:
        """Return True when a shared candidate in a family has real voter support."""
        return any(str(row.get("top_family", "")) == top_family and self.has_raw_support(row) for row in shared)

    def has_raw_support(self, row: dict[str, Any]) -> bool:
        """Return whether a candidate is close enough in raw voter ranks."""
        combined = float(row.get("combined_rank_score", 9999.0))
        brain_rank = int(row.get("brain_rank", 9999))
        physics_rank = int(row.get("physics_rank", 9999))
        if combined > self.policy.role_sanity_max_combined_rank_score:
            return False
        return (
            min(brain_rank, physics_rank) <= self.policy.role_sanity_primary_voter_rank
            and max(brain_rank, physics_rank) <= self.policy.role_sanity_secondary_voter_rank
        )

    @staticmethod
    def is_concrete_fx_candidate(row: dict[str, Any]) -> bool:
        """Return True for specific FX identities, not generic catch-all FX."""
        if str(row.get("top_family", "")) != "FX":
            return False
        path = str(row.get("folder_path", row.get("label", ""))).lower()
        generic_fragments = ("hybrid designed fx", "designed noise fx")
        if any(fragment in path for fragment in generic_fragments):
            return any(fragment in path for fragment in GENERIC_CONCRETE_FX_FRAGMENTS)
        return any(fragment in path for fragment in CONCRETE_FX_FRAGMENTS)


CONCRETE_MOTION_FRAGMENTS = (
    "riser",
    "build",
    "sweep",
    "whoosh",
    "swoosh",
    "swish",
    "drop",
    "downlifter",
    "transition",
    "impact",
    "boom",
    "reverse",
    "glitch",
    "stutter",
    "crash",
    "glass",
    "metal",
)

ABSTRACT_FX_FRAGMENTS = (
    "siren",
    "alarm",
    "beep",
    "blip",
    "motor",
    "machine",
    "cat",
    "dog",
    "animal",
)

CONCRETE_FX_FRAGMENTS = CONCRETE_MOTION_FRAGMENTS + (
    "siren",
    "alarm",
    "motor",
    "machine",
    "radio",
    "electrical",
)

GENERIC_CONCRETE_FX_FRAGMENTS = tuple(
    fragment for fragment in CONCRETE_FX_FRAGMENTS if fragment not in {"glass", "metal", "crash"}
)
