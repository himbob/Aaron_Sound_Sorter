from __future__ import annotations

from pathlib import Path

from aaron_sound_sorter.neural_audio.panns_mapping import (
    PannsEventScore,
    PannsMappingRegistry,
)
from aaron_sound_sorter.taxonomy_registry import TaxonomyRegistry


def _registry() -> tuple[Path, TaxonomyRegistry]:
    root = Path(__file__).resolve().parents[1]
    registry = TaxonomyRegistry.load(
        root / "config" / "canonical_taxonomy.json",
        root / "config" / "taxonomy_aliases.json",
    )
    return root, registry


def test_panns_saxophone_supports_sax_and_contradicts_voice() -> None:
    root, taxonomy = _registry()
    mappings = PannsMappingRegistry.load(root / "config" / "panns_audioset_mapping.json", taxonomy)
    events = (PannsEventScore("Saxophone", 0.88), PannsEventScore("Music", 0.94))

    sax = mappings.evaluate(events, "Instruments/Woodwinds/Saxophone/Alto/One Shots")
    voice = mappings.evaluate(events, "Instruments/Voice/Vocal One Shots/One Shots")

    assert sax.support_score == 0.88
    assert sax.contradiction_score == 0.0
    assert voice.contradiction_score == 0.88
    assert [event.label for event in voice.neutral_events] == ["Music"]


def test_repository_panns_mappings_resolve_to_canonical_prefixes() -> None:
    root, taxonomy = _registry()

    mappings = PannsMappingRegistry.load(root / "config" / "panns_audioset_mapping.json", taxonomy)

    assert mappings.mapping_version
    assert "Snare drum" in mappings.mappings
    assert "Water" in mappings.mappings


def test_panns_speech_supports_voice_and_rejects_nonvoice_fx() -> None:
    root, taxonomy = _registry()
    mappings = PannsMappingRegistry.load(root / "config" / "panns_audioset_mapping.json", taxonomy)
    events = (PannsEventScore("Speech", 0.73),)

    voice = mappings.evaluate(events, "Instruments/Voice/Vocal Loops/Loops")
    glitch = mappings.evaluate(
        events,
        "FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/Long FX",
    )

    assert voice.support_score == 0.73
    assert voice.contradiction_score == 0.0
    assert glitch.support_score == 0.0
    assert glitch.contradiction_score == 0.73


def test_panns_combines_several_quiet_voice_events() -> None:
    root, taxonomy = _registry()
    mappings = PannsMappingRegistry.load(root / "config" / "panns_audioset_mapping.json", taxonomy)
    events = (
        PannsEventScore("Singing", 0.13),
        PannsEventScore("Yodeling", 0.11),
        PannsEventScore("Speech", 0.09),
        PannsEventScore("A capella", 0.09),
    )

    voice = mappings.evaluate(events, "FX/Human and Voice FX/Altered Voice/Long FX")
    sax = mappings.evaluate(events, "Instruments/Woodwinds/Saxophone/Loops")

    assert voice.support_score > 0.30
    assert {event.label for event in voice.supporting_events} == {
        "A capella",
        "Singing",
        "Speech",
        "Yodeling",
    }
    assert sax.contradiction_score == voice.support_score
