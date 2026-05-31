"""Shared decision context objects for claim-producing policy code.

The decision core should coordinate policy objects, not pass long loose
argument lists through large private methods.  These small dataclasses make
claim producers easier to read, type-check, and unit test without changing the
sorter's external behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

from aaron_sound_sorter.domain.models import SharedAudioFacts, VoterResult
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import ConsensusClaim


@dataclass(frozen=True)
class DecisionContext:
    """Immutable inputs available to one claim-producing pass.

    Claim producers may inspect these values and return evidence claims.  They
    must not finalize placement.  Final placement remains owned by
    :class:`FamilyClaimArbiter`.
    """

    raw: ConsensusClaim
    eligibility: EligibilityDecision
    facts: SharedAudioFacts | None = None
    brain_result: VoterResult | None = None
    physics_result: VoterResult | None = None
