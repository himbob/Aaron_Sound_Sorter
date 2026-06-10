from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.voter_calibration_panels import flatten_fact_values


def test_flatten_fact_values_caches_per_decision_fact_map() -> None:
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={"shape_vote": {"confidence": 0.75}, "_private_cache": {"bad": 1.0}},
    )

    first = flatten_fact_values(facts)
    second = flatten_fact_values(facts)

    assert first is second
    assert first["confidence"] == 0.75
    assert "_private_cache_bad" not in first
