"""Competence policy for optional brain-family recall lanes.

The full brain and the smaller baby brains are not equally reliable for every
kind of sound.  This module keeps lane competence weights in one explicit place
so decision code does not scatter baby-brain exceptions through a giant method.

This is intentionally conservative: the policy only controls how strong a claim
is allowed to be after measured audio evidence has already made the claim
eligible.  It never turns a lane guess into final truth by itself.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BrainLaneCompetence:
    """Documented competence profile for one diagnostic brain lane."""

    lane_name: str
    safe_claim_strength: float
    purpose: str
    caution: str


class BrainCompetencePolicy:
    """Central policy for brain-lane claim strength and reporting.

    New lane specialization should be added here first, then tested with small
    synthetic and real-audio panels.  Claim producers should not hard-code lane
    strength numbers locally.
    """

    _DEFAULT = BrainLaneCompetence(
        lane_name="baby",
        safe_claim_strength=0.88,
        purpose="legacy or unnamed baby recall lane",
        caution="allowed only after measured role and shape evidence agree",
    )

    _LANES = {
        "core_baby": BrainLaneCompetence(
            lane_name="core_baby",
            safe_claim_strength=0.94,
            purpose="clean-center precision recall lane",
            caution="should not override incompatible measured role evidence",
        ),
        "spread_baby": BrainLaneCompetence(
            lane_name="spread_baby",
            safe_claim_strength=0.91,
            purpose="balanced diversity recall lane",
            caution="useful for broad rescue, not terminal truth alone",
        ),
        "outlier_baby": BrainLaneCompetence(
            lane_name="outlier_baby",
            safe_claim_strength=0.84,
            purpose="edge-case recall lane",
            caution="weak global authority; needs strong measured support",
        ),
    }

    def profile_for_lane(self, lane_name: str) -> BrainLaneCompetence:
        """Return the documented competence profile for ``lane_name``."""
        normalized = str(lane_name or "baby")
        return self._LANES.get(normalized, self._DEFAULT)

    def safe_claim_strength(self, lane_name: str) -> float:
        """Return the maximum broad-claim strength this lane may contribute."""
        return self.profile_for_lane(lane_name).safe_claim_strength

    def validation_rows(self) -> list[dict[str, object]]:
        """Return simple rows suitable for a future lane-validation report."""
        rows: list[dict[str, object]] = []
        for profile in sorted(self._LANES.values(), key=lambda item: item.lane_name):
            rows.append(
                {
                    "lane_name": profile.lane_name,
                    "safe_claim_strength": profile.safe_claim_strength,
                    "purpose": profile.purpose,
                    "caution": profile.caution,
                }
            )
        return rows
