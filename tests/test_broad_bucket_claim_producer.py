"""Compatibility tests for the broad-bucket claim producer public seam."""

from __future__ import annotations

import inspect
from pathlib import Path

from aaron_sound_sorter.engine import claim_producers
from aaron_sound_sorter.engine.claim_producers import BroadBucketClaimProducer


def test_decision_core_uses_broad_bucket_claim_producer() -> None:
    """DecisionCoreV2 should keep the public compatibility seam visible."""
    source = Path("src/aaron_sound_sorter/engine/decision_core_v2.py").read_text(encoding="utf-8")
    assert "BroadBucketClaimProducer(self)" in source


def test_broad_bucket_claim_producer_is_documented_and_typed() -> None:
    """The public broad-bucket producer should remain documented and typed."""
    assert BroadBucketClaimProducer.__doc__
    assert BroadBucketClaimProducer.produce.__annotations__["return"] == "list[ConsensusClaim]"
    assert inspect.getdoc(BroadBucketClaimProducer.produce)


def test_claim_producer_public_exports_are_import_compatible() -> None:
    """Refactors must not silently drop the producer export surface."""
    expected = {
        "BabyRecallClaimProducer",
        "BroadBucketClaimProducer",
        "CandidateConflictClaimProducer",
        "ClaimProducer",
        "FinalEligibilityClaimProducer",
        "InstrumentLoopSafetyClaimProducer",
        "MeasuredBucketClaimProducer",
        "MeasuredEarlyCandidateAdjudicator",
        "MeasuredSmokeStabilityClaimProducer",
        "MeasuredTrueBucketClaimProducer",
        "ProfileCandidateClaimProducer",
        "VoicePercussiveBucketClaimProducer",
    }

    assert set(claim_producers.__all__) == expected
    for name in expected:
        assert getattr(claim_producers, name) is not None
