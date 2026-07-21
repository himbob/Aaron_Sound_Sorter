from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.learned_memory_contracts import (
    has_dominant_non_voice_memory,
    has_exact_dominant_memory_for_path,
    has_non_voice_memory_match,
    is_voice_category_path,
    iter_learned_memory_matches,
    memory_conflicts_with_candidate_top_family,
)


def facts_with_memory(*memory_rows: tuple[str, dict[str, object]]) -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={key: value for key, value in memory_rows},
    )


def test_voice_category_path_detects_internal_voice_labels() -> None:
    assert is_voice_category_path("Instruments/Voice/Vocal Loops/Loops")
    assert is_voice_category_path("FX/Human and Voice FX/Spoken Voice/Long FX")
    assert not is_voice_category_path("Instruments/Synths/Synth Arp/Loops")


def test_iter_learned_memory_matches_normalizes_memory_rows() -> None:
    facts = facts_with_memory(
        (
            "learned_voter_memory",
            {
                "matched": True,
                "label": "Instruments/Synths/Synth Arp/Loops",
                "confidence": "0.94",
                "effective_weight": "1200",
            },
        )
    )

    matches = iter_learned_memory_matches(facts, minimum_confidence=0.90)

    assert len(matches) == 1
    assert matches[0].label == "Instruments/Synths/Synth Arp/Loops"
    assert matches[0].top_family == "Instruments"
    assert matches[0].effective_weight == 1200
    assert matches[0].is_non_voice_source


def test_non_voice_memory_can_dominate_voice_memory() -> None:
    facts = facts_with_memory(
        (
            "learned_voter_memory",
            {
                "matched": True,
                "top_family": "Instruments",
                "label": "Instruments/Synths/Synth Arp/Loops",
                "confidence": 0.94,
            },
        ),
        (
            "learned_physics_memory",
            {
                "matched": True,
                "top_family": "Instruments",
                "branch": "Voice",
                "label": "Instruments/Voice/Vocal Loops/Loops",
                "confidence": 0.86,
            },
        ),
    )

    assert has_non_voice_memory_match(facts)
    assert has_dominant_non_voice_memory(facts)


def test_exact_memory_match_protects_trained_label_depth() -> None:
    facts = facts_with_memory(
        (
            "learned_physics_memory",
            {
                "matched": True,
                "top_family": "Instruments",
                "branch": "Synth",
                "label": "Instruments/Synths/Synth Arp/Loops",
                "confidence": 0.96,
            },
        )
    )

    assert has_exact_dominant_memory_for_path(facts, "Instruments/Synths/Synth Arp/Loops")
    assert not has_exact_dominant_memory_for_path(facts, "Instruments/Instrument Loops/Loops")


def test_memory_conflict_uses_top_family_but_not_exact_label() -> None:
    facts = facts_with_memory(
        (
            "learned_voter_memory",
            {
                "matched": True,
                "top_family": "Instruments",
                "label": "Instruments/Voice/Vocal Loops/Loops",
                "confidence": 0.96,
            },
        )
    )

    assert not memory_conflicts_with_candidate_top_family(facts, "Instruments/Voice/Vocal Loops/Loops")
    assert memory_conflicts_with_candidate_top_family(facts, "FX/Human and Voice FX/Spoken Voice/Long FX")
