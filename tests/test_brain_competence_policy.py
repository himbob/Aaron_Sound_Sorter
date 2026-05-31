"""Tests for isolated brain-lane competence policy."""

from __future__ import annotations

from aaron_sound_sorter.engine.brain_competence_policy import BrainCompetencePolicy


def test_brain_lane_competence_policy_documents_each_baby_lane() -> None:
    """Each baby lane should have explicit strength, purpose, and caution text."""
    policy = BrainCompetencePolicy()
    rows = policy.validation_rows()
    by_lane = {str(row["lane_name"]): row for row in rows}

    assert set(by_lane) == {"core_baby", "spread_baby", "outlier_baby"}
    assert by_lane["core_baby"]["safe_claim_strength"] > by_lane["outlier_baby"]["safe_claim_strength"]
    assert all(str(row["purpose"]).strip() for row in rows)
    assert all(str(row["caution"]).strip() for row in rows)


def test_unknown_baby_lane_gets_legacy_safe_default() -> None:
    """Unrecognized baby lane names should stay conservative, not crash."""
    policy = BrainCompetencePolicy()

    assert policy.safe_claim_strength("legacy_baby") == 0.88
