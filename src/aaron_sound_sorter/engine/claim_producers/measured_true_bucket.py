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
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


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
        measured_role = _measured_role_from_facts(facts)
        shape = _shape_vote_from_facts(facts)
        shape_conf = _shape_confidence_from_facts(facts)

        for claim in (
            self.early_adjudicator.adjudicate(raw, eligibility, measured_role, shape, shape_conf, facts),
            self._maybe_keep_stable_drum_loop(raw, raw_path),
            self._maybe_keep_pitched_instrument_over_voice(raw, eligibility, raw_path, measured_role),
            self._maybe_rescue_bass_loop(raw, eligibility, raw_path, measured_role, shape, shape_conf),
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
