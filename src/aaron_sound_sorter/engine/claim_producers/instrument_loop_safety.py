# SOURCE-NAME BLINDNESS INVARIANT:
# This claim producer may inspect voter output and measured audio facts only.
# It must never inspect producer filenames, source folder names, ZIP member
# names, path tokens, or sample-pack labels as classification evidence.
"""Instrument-loop safety claim producer coordinator."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.claim_producers.instrument_loop_conflicts import InstrumentLoopConflictMixin
from aaron_sound_sorter.engine.claim_producers.instrument_loop_one_shot import InstrumentOneShotLoopMixin
from aaron_sound_sorter.engine.claim_support import ClaimSupportMixin
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.decision_helpers import _norm_path, _path_has_any, _shape_metric_from_facts
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class InstrumentLoopSafetyClaimProducer(
    InstrumentOneShotLoopMixin,
    InstrumentLoopConflictMixin,
    ClaimSupportMixin,
):
    """Produce broad Instrument-loop safety claims."""

    def produce(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return ordered Instrument-loop safety claims for this context."""
        claims: list[ConsensusClaim] = []
        for claim in (
            self.one_shot_leaf_loop_claim(context.raw, context.eligibility, context.facts),
            self.tonal_conflict_loop_claim(context.raw, context.eligibility, context.facts),
            self.ambiguous_fx_music_loop_claim(context.raw, context.eligibility, context.facts),
        ):
            if claim is not None:
                claims.append(claim)
        return claims

    @staticmethod
    def should_skip_decisive_parent_broaden(
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        facts: SharedAudioFacts | None,
    ) -> bool:
        """Return True when generic Instrument Loops is safer than reed broadening."""
        role_name = str(eligibility.role_name or "")
        broad_path = _norm_path(str(eligibility.broad_folder_path or ""))
        raw_path = _norm_path(raw.folder_path)
        risky_reed_broad = role_name in {
            "pitched_reed_or_instrument_loop",
            "pitched_reed_or_instrument_phrase",
        } and _path_has_any(broad_path, ("brass", "woodwind", "sax", "reed"))
        if not risky_reed_broad:
            return False
        if raw.final_top != "Instruments" or not _path_has_any(raw_path, ("instrument loops", "mixed musical loops")):
            return False
        low_ratio = _shape_metric_from_facts(facts, "low_event_ratio")
        high_ratio = _shape_metric_from_facts(facts, "high_event_ratio")
        pitch_conf = _shape_metric_from_facts(facts, "pitch_confidence")
        plausible_reed_body = (
            0.035 <= low_ratio <= 0.45
            and high_ratio <= 0.42
            and pitch_conf >= 0.74
            and not (high_ratio >= 0.24 and low_ratio <= 0.03)
        )
        return not plausible_reed_body
