# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Measured true-bucket claim producer coordinator.

The section modules keep bass, drum, voice, and FX rescues readable while this
class preserves the original branch order from the legacy decision core.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.claim_contracts import is_rank_one_concrete_non_sax_instrument_consensus
from aaron_sound_sorter.engine.claim_producers.measured_early_adjudication import (
    MeasuredEarlyCandidateAdjudicator,
)
from aaron_sound_sorter.engine.claim_producers.measured_true_bass_fx import MeasuredTrueBassFxMixin
from aaron_sound_sorter.engine.claim_producers.measured_true_drum_voice_fx import MeasuredTrueDrumVoiceFxMixin
from aaron_sound_sorter.engine.claim_producers.measured_true_safety import MeasuredTrueSafetyMixin
from aaron_sound_sorter.engine.claim_support import ClaimSupportMixin
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.decision_helpers import (
    _measured_role_from_facts,
    _norm_path,
    _shape_confidence_from_facts,
    _shape_vote_from_facts,
)
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import ConsensusClaim, claim_from_folder_path


class MeasuredTrueBucketClaimProducer(
    MeasuredTrueSafetyMixin,
    MeasuredTrueBassFxMixin,
    MeasuredTrueDrumVoiceFxMixin,
    ClaimSupportMixin,
):
    """Produce candidate-conflict measured claims using an early adjudicator."""

    def __init__(self, early_adjudicator: MeasuredEarlyCandidateAdjudicator | None = None) -> None:
        self.early_adjudicator = early_adjudicator or MeasuredEarlyCandidateAdjudicator()

    def produce(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return zero or one measured true-bucket conflict claim."""
        claim = self._true_bucket_for_candidate_conflict(
            context.raw,
            context.eligibility,
            context.facts,
        )
        return [] if claim is None else [claim]

    def _true_bucket_for_candidate_conflict(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        facts: SharedAudioFacts | None = None,
    ) -> ConsensusClaim | None:
        """Choose a useful broad/near-true bucket when evidence is strong enough."""
        if raw.final_top == "_TO_REVIEW":
            return None
        raw_path = _norm_path(raw.folder_path)
        if is_rank_one_concrete_non_sax_instrument_consensus(raw):
            return None
        measured_role = _measured_role_from_facts(facts)
        shape = _shape_vote_from_facts(facts)
        shape_conf = _shape_confidence_from_facts(facts)

        for claim in (
            self._maybe_restore_decisive_woodwind_loop_depth(raw, facts, shape, shape_conf),
            self.early_adjudicator.adjudicate(raw, eligibility, measured_role, shape, shape_conf, facts),
            self._maybe_keep_stable_drum_loop(raw, raw_path),
            self._maybe_keep_pitched_instrument_over_voice(raw, eligibility, raw_path, measured_role),
            self._maybe_rescue_bass_loop(raw, eligibility, raw_path, measured_role, shape, shape_conf, facts),
            self._maybe_rescue_concrete_fx_over_generic_instrument(raw, raw_path),
            self._maybe_rescue_tonal_alert_to_instrument(raw, eligibility, facts),
            self._maybe_keep_pitched_percussion_as_instrument(raw, eligibility, raw_path, measured_role),
            self._maybe_rescue_drum_loop(raw, eligibility, raw_path, measured_role, shape, shape_conf),
            self._maybe_keep_measured_human_voice(raw, raw_path, eligibility, measured_role),
            self._maybe_rescue_instrument_hit_from_fx(raw),
            self._maybe_rescue_nonvoice_fx_from_human_voice(raw, raw_path),
            self._maybe_rescue_tonal_transition_fx(raw, eligibility, raw_path, facts),
        ):
            if claim is not None:
                return claim
        return None

    def _maybe_restore_decisive_woodwind_loop_depth(
        self,
        raw: ConsensusClaim,
        facts: SharedAudioFacts | None,
        shape: str,
        shape_conf: float,
    ) -> ConsensusClaim | None:
        """Emit a sax-loop depth claim when physics already selected Woodwinds.

        This keeps dark low-mid sax phrases from being flattened to generic
        instrument loops or stolen by bass-body scores.  It is source-name-blind
        and uses the measured physics layer, not producer filenames.
        """
        if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
            return None
        parent = facts.evidence.get("parent_eligibility_v2", {})
        if isinstance(parent, dict) and str(parent.get("role_name") or "") in {
            "protected_percussive_one_shot",
            "percussive_one_shot",
            "low_kick_like_hit",
        }:
            return None
        layer = facts.evidence.get("physics_layer_decision", {})
        if not isinstance(layer, dict):
            layer = {}
        branch = str(
            layer.get("instrument_branch_selected")
            or layer.get("physics_layer_branch")
            or facts.evidence.get("physics_layer_branch")
            or facts.evidence.get("physics_top_layer_instrument_branch")
            or ""
        )
        if branch not in {"Woodwinds", "ReedWoodwind"}:
            return None
        woodwind_branch = self._number(layer.get("instrument_branch_Woodwinds"), 0.0)
        dark_reed = bool(layer.get("instrument_dark_low_mid_reed_loop_signal"))
        sax_subpanel = str(layer.get("instrument_Woodwinds_subpanel_selected") or "") == "Sax"
        sax_subpanel_confidence = self._number(layer.get("instrument_Woodwinds_subpanel_confidence"), 0.0)
        sax_score = self._number(facts.evidence.get("woodwind_sax_score"), 0.0)
        if not (
            (dark_reed and sax_score >= 0.58)
            or (woodwind_branch >= 0.86 and sax_subpanel and sax_subpanel_confidence >= 0.74 and sax_score >= 0.60)
        ):
            return None
        if shape not in {
            "bass_phrase",
            "pitched_phrase",
            "pitched_phrase_shape",
            "pitched_repetition_phrase",
            "repeated_phrase_loop",
            "solo_phrase",
            "sustained_pad",
        }:
            return None
        if shape_conf < 0.70:
            return None
        return claim_from_folder_path(
            folder_path="Instruments/Woodwinds/Saxophone/Loops",
            source="final_measured_sax_loop_invariant",
            reason=(
                "measured woodwind-loop claim: physics selected a dark/low-mid sax branch before final arbitration"
            ),
            shared=raw.shared_candidates,
            raw_candidate_score=raw.raw_candidate_score,
            brain_rank=raw.brain_rank,
            physics_rank=raw.physics_rank,
            shared_winner=raw.shared_winner or raw.folder_path,
            can_override=True,
            strength=0.95,
            is_real_candidate=False,
        )

    @staticmethod
    def _number(value: object, default: float = 0.0) -> float:
        try:
            number = float(value)
        except Exception:
            return default
        return default if number != number else number
