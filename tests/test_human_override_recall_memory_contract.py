from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.voters.human_override_recall import human_override_recall_allowed_by_memory


def facts_with_memory(memory: dict[str, object]) -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={"learned_physics_memory": memory},
    )


def test_human_override_recall_allowed_for_exact_learned_memory_label() -> None:
    facts = facts_with_memory(
        {
            "matched": True,
            "top_family": "Instruments",
            "branch": "Voice",
            "label": "Instruments/Voice/Vocal Loops/Loops",
            "confidence": 0.96,
        }
    )

    allowed = human_override_recall_allowed_by_memory("Instruments/Voice/Vocal Loops/Loops", facts)

    assert allowed is True


def test_human_override_recall_suppressed_for_conflicting_learned_memory_top_family() -> None:
    facts = facts_with_memory(
        {
            "matched": True,
            "top_family": "Instruments",
            "branch": "Voice",
            "label": "Instruments/Voice/Vocal Loops/Loops",
            "confidence": 0.96,
        }
    )

    allowed = human_override_recall_allowed_by_memory("FX/Human and Voice FX/Spoken Voice/Long FX", facts)

    assert allowed is False


def test_human_override_recall_ignores_weak_learned_memory_conflict() -> None:
    facts = facts_with_memory(
        {
            "matched": True,
            "top_family": "Instruments",
            "branch": "Voice",
            "label": "Instruments/Voice/Vocal Loops/Loops",
            "confidence": 0.72,
        }
    )

    allowed = human_override_recall_allowed_by_memory("FX/Human and Voice FX/Spoken Voice/Long FX", facts)

    assert allowed is True
