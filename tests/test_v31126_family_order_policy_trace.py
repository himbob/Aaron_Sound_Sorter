from __future__ import annotations

from aaron_sound_sorter.engine.family_order_policy import TOP_FAMILY_REVIEW_ORDER, family_order_policy_summary


def test_family_order_policy_keeps_fx_after_drums_and_instruments() -> None:
    summary = family_order_policy_summary()

    assert TOP_FAMILY_REVIEW_ORDER.index("Drums") < TOP_FAMILY_REVIEW_ORDER.index("FX")
    assert TOP_FAMILY_REVIEW_ORDER.index("Instruments") < TOP_FAMILY_REVIEW_ORDER.index("FX")
    assert summary["changes_scores"] is False
    assert "concrete FX" in summary["fx_policy"]
