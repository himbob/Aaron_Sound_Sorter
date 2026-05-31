# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Measured early-candidate adjudication coordinator.

This class keeps the legacy branch order but delegates each policy section to a
small, named mixin.  The split is intentionally behavior-preserving: thresholds,
folder targets, and candidate ordering remain in the section modules where they
can be reviewed independently.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.claim_producers.measured_early_drum_fx import DrumFxRescueMixin
from aaron_sound_sorter.engine.claim_producers.measured_early_preflight import EarlyPreflightMixin
from aaron_sound_sorter.engine.claim_producers.measured_early_reed_voice_conflicts import (
    ReedVoiceConflictMixin,
)
from aaron_sound_sorter.engine.claim_producers.measured_early_short_hit import ShortHitGuardMixin
from aaron_sound_sorter.engine.claim_producers.measured_early_voice import VoiceEvidenceMixin
from aaron_sound_sorter.engine.claim_support import ClaimSupportMixin
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class MeasuredEarlyCandidateAdjudicator(
    EarlyPreflightMixin,
    ShortHitGuardMixin,
    VoiceEvidenceMixin,
    ReedVoiceConflictMixin,
    DrumFxRescueMixin,
    ClaimSupportMixin,
):
    """Adjudicate early measured-role and candidate conflicts."""

    def adjudicate(
        self,
        raw: ConsensusClaim,
        eligibility: EligibilityDecision,
        measured_role: str,
        shape: str,
        shape_conf: float,
        facts: SharedAudioFacts | None,
    ) -> ConsensusClaim | None:
        """Resolve high-confidence candidate contradictions before fallbacks."""
        context = self._build_early_context(
            raw=raw,
            eligibility=eligibility,
            measured_role=measured_role,
            shape=shape,
            shape_conf=shape_conf,
            facts=facts,
        )
        decision = self._adjudicate_preflight_claims(context)
        if decision is not None:
            return decision
        decision = self._adjudicate_short_hit_guard(context)
        if decision is not None:
            return decision

        voice_evidence = self._build_voice_candidate_evidence(context)
        decision = self._adjudicate_voice_claims(context, voice_evidence)
        if decision is not None:
            return decision
        decision = self._adjudicate_reed_and_voice_conflicts(context, voice_evidence)
        if decision is not None:
            return decision
        return self._adjudicate_drum_and_fx_rescues(context, voice_evidence)
