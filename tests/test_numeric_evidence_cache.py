from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.numeric_evidence import cached_numeric_leaf_values, numeric_evidence_value


def test_numeric_evidence_value_caches_first_seen_leaf_values() -> None:
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "shape_vote": {"voice_score": 0.7},
            "physics_subpanels": {"voice_score": 0.2, "nested": {"reed_sax_identity_score": 0.6}},
            "_private_debug": {"ignored_score": 1.0},
        },
    )

    first = cached_numeric_leaf_values(facts)
    second = cached_numeric_leaf_values(facts)

    assert first is second
    assert numeric_evidence_value(facts, "voice_score", 0.0) == 0.7
    assert numeric_evidence_value(facts, "reed_sax_identity_score", 0.0) == 0.6
    assert numeric_evidence_value(facts, "ignored_score", 0.0) == 0.0
