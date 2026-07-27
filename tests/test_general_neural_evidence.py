from __future__ import annotations

from pathlib import Path

from aaron_sound_sorter.neural_audio.panns_mapping import PannsEventScore, PannsMappingRegistry
from aaron_sound_sorter.neural_audio.semantic_panel import SEMANTIC_FAMILY_PROMPTS
from aaron_sound_sorter.taxonomy_registry import TaxonomyRegistry


def _mapping_registry() -> PannsMappingRegistry:
    root = Path(__file__).resolve().parents[1]
    taxonomy = TaxonomyRegistry.load(
        root / "config" / "canonical_taxonomy.json",
        root / "config" / "taxonomy_aliases.json",
    )
    return PannsMappingRegistry.load(root / "config" / "panns_audioset_mapping.json", taxonomy)


def test_every_clap_family_has_multiple_source_descriptions() -> None:
    assert len(SEMANTIC_FAMILY_PROMPTS) >= 20
    assert all(len(prompts) >= 3 for prompts in SEMANTIC_FAMILY_PROMPTS.values())
    assert any("processed" in prompt or "pitch-shifted" in prompt for prompt in SEMANTIC_FAMILY_PROMPTS["human_voice"])
    assert any("processed" in prompt for prompt in SEMANTIC_FAMILY_PROMPTS["keys"])
    assert any("processed" in prompt for prompt in SEMANTIC_FAMILY_PROMPTS["guitar_plucked"])
    assert any("electronic" in prompt for prompt in SEMANTIC_FAMILY_PROMPTS["drums"])


def test_panns_groups_related_events_without_needing_the_current_candidate() -> None:
    mappings = _mapping_registry()
    grouped = mappings.aggregate_family_scores(
        (
            PannsEventScore("Speech", 0.24),
            PannsEventScore("Female singing", 0.23),
            PannsEventScore("Humming", 0.17),
            PannsEventScore("Singing", 0.14),
            PannsEventScore("Music", 0.52),
        )
    )

    by_family = {row.family: row for row in grouped}
    assert by_family["human_voice"].score > 0.55
    assert {event.label for event in by_family["human_voice"].events} == {
        "Female singing",
        "Humming",
        "Singing",
        "Speech",
    }
    assert "Music" not in {event.label for event in by_family["human_voice"].events}


def test_panns_mapping_covers_every_major_sorter_family() -> None:
    mappings = _mapping_registry()
    grouped = mappings.aggregate_family_scores(
        (
            PannsEventScore("Hi-hat", 0.8),
            PannsEventScore("Synthesizer", 0.7),
            PannsEventScore("Dog", 0.6),
            PannsEventScore("Engine", 0.5),
            PannsEventScore("Rain", 0.4),
            PannsEventScore("Whoosh, swoosh, swish", 0.3),
        )
    )
    families = {row.family for row in grouped}
    assert {"drum_cymbal", "synth", "animal_creature", "fx_machine", "fx_nature", "fx_transition"} <= families
