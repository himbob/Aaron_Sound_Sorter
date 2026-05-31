# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision.
"""Measured broad-bucket smoke-stability claim producer coordinator."""

from __future__ import annotations

from aaron_sound_sorter.engine.claim_producers.measured_smoke_drum_voice import SmokeDrumVoiceRescueMixin
from aaron_sound_sorter.engine.claim_producers.measured_smoke_guards import SmokeInitialGuardMixin
from aaron_sound_sorter.engine.claim_producers.measured_smoke_music import SmokeMusicRescueMixin
from aaron_sound_sorter.engine.claim_support import ClaimSupportMixin
from aaron_sound_sorter.engine.decision_context import DecisionContext
from aaron_sound_sorter.engine.decision_helpers import (
    _measured_role_from_facts,
    _norm_path,
    _shape_confidence_from_facts,
    _shape_vote_from_facts,
)
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


class MeasuredSmokeStabilityClaimProducer(
    SmokeInitialGuardMixin,
    SmokeMusicRescueMixin,
    SmokeDrumVoiceRescueMixin,
    ClaimSupportMixin,
):
    """Produce broad measured-role stability claims."""

    def produce(self, context: DecisionContext) -> list[ConsensusClaim]:
        """Return zero or one broad measured-role stability claim."""
        claim = self._measured_broad_bucket_for_smoke_stability(context)
        return [] if claim is None else [claim]

    def _measured_broad_bucket_for_smoke_stability(
        self,
        context: DecisionContext,
    ) -> ConsensusClaim | None:
        """Recover curated/stable smoke examples from over-eager review."""
        raw = context.raw
        eligibility = context.eligibility
        facts = context.facts
        if raw.final_top == "_TO_REVIEW":
            return None
        role = str(eligibility.role_name or "")
        measured_role = _measured_role_from_facts(facts)
        shape = _shape_vote_from_facts(facts)
        shape_conf = _shape_confidence_from_facts(facts)
        raw_path = _norm_path(raw.folder_path)
        broad_path = _norm_path(eligibility.broad_folder_path)

        if self._should_keep_existing_stable_bucket(raw, raw_path, role, measured_role, shape, facts):
            return None

        for claim in (
            self._maybe_rescue_bass_smoke(raw, eligibility, broad_path, role, shape, shape_conf),
            self._maybe_rescue_pitched_music_smoke(
                raw, eligibility, raw_path, role, measured_role, shape, shape_conf, facts
            ),
            self._maybe_rescue_drum_smoke(raw, eligibility, raw_path, broad_path, role, shape, shape_conf),
            self._maybe_rescue_vocal_phrase_smoke(
                raw, eligibility, raw_path, role, measured_role, shape, shape_conf, facts
            ),
            self._maybe_use_eligibility_broad_path(raw, eligibility, broad_path),
        ):
            if claim is not None:
                return claim
        return None

    def _maybe_use_eligibility_broad_path(
        self,
        raw: ConsensusClaim,
        eligibility,
        broad_path: str,
    ) -> ConsensusClaim | None:
        """Use a strong valid eligibility broad path instead of review."""
        if (
            broad_path
            and not broad_path.startswith("_to_review")
            and eligibility.confidence >= 0.76
            and not eligibility.is_path_allowed(raw.folder_path, raw.final_top)
        ):
            return self._broaden_from_raw(
                raw,
                eligibility,
                "measured broad parent was safer than review for curated/stable example",
            )
        return None
