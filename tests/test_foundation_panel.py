from __future__ import annotations

from pathlib import Path

from aaron_sound_sorter.neural_audio.foundation_panel import build_foundation_panel
from aaron_sound_sorter.neural_audio.panns_mapping import PannsMappingRegistry
from aaron_sound_sorter.neural_audio.runtime import (
    NeuralEventSuggestion,
    NeuralPromptSuggestion,
    NeuralRuntimePrediction,
)
from aaron_sound_sorter.taxonomy_registry import TaxonomyRegistry


def _prediction() -> NeuralRuntimePrediction:
    return NeuralRuntimePrediction(
        row_id="00001",
        file_sha256="a" * 64,
        predicted_label="Instruments/Voice/Vocal One Shots/One Shots",
        second_label="Instruments/Woodwinds/Saxophone/Alto/One Shots",
        top_similarity=0.82,
        second_similarity=0.62,
        margin=0.20,
        radius_ratio=0.72,
        known_distribution=True,
        label_example_count=5,
        exact_training_match=False,
        ownership_ready=True,
        ownership_block_reason="supported_separated_neighborhood",
        semantic_family="brass",
        semantic_top_score=0.31,
        semantic_margin=0.09,
        semantic_family_scores={"brass": 0.31, "human_voice": 0.18},
        prompt_brain_status="advisory_only",
        prompt_suggestions=(
            NeuralPromptSuggestion(
                "Instruments/Woodwinds/Saxophone/Alto/One Shots",
                0.55,
                0.20,
                0.35,
                "an isolated alto saxophone note",
                0.58,
                "a human voice",
                0.20,
            ),
        ),
        panns_status="advisory_only",
        panns_model_id="panns/test",
        panns_events=(NeuralEventSuggestion("Saxophone", 0.88), NeuralEventSuggestion("Music", 0.74)),
    )


def _mapping() -> tuple[TaxonomyRegistry, PannsMappingRegistry]:
    root = Path(__file__).resolve().parents[1]
    taxonomy = TaxonomyRegistry.load(
        root / "config/canonical_taxonomy.json",
        root / "config/taxonomy_aliases.json",
    )
    return taxonomy, PannsMappingRegistry.load(root / "config/panns_audioset_mapping.json", taxonomy)


def test_foundation_panel_keeps_model_lanes_separate_and_visible() -> None:
    taxonomy, mapping = _mapping()

    panel = build_foundation_panel(
        _prediction(),
        taxonomy_version=taxonomy.taxonomy_version,
        brain_version="prototype/run-test",
        panns_mapping=mapping,
    )

    assert set(panel.lanes) == {
        "aaron_prototype",
        "clap_broad",
        "clap_prompt",
        "panns",
        "human_memory",
    }
    assert panel.lanes["aaron_prototype"].scores[0].label.startswith("Instruments/Voice")
    assert panel.lanes["clap_prompt"].scores[0].label.startswith("Instruments/Woodwinds")
    assert panel.lanes["panns"].scores[0].label == "Saxophone"
    assert any("CLAP broad family contradiction" in conflict for conflict in panel.conflicts)
    assert any("PANNs broad contradiction" in conflict for conflict in panel.conflicts)
    assert panel.review_reason


def test_foundation_panel_serializes_without_source_locator() -> None:
    taxonomy, mapping = _mapping()
    panel = build_foundation_panel(
        _prediction(),
        taxonomy_version=taxonomy.taxonomy_version,
        brain_version="prototype/run-test",
        panns_mapping=mapping,
    )

    payload = panel.to_mapping()

    assert payload["audio_sha256"] == "a" * 64
    assert "source_path" not in payload
    assert payload["lanes"]["clap_prompt"]["scores"][0]["evidence"]["production_ownership_enabled"] is False
