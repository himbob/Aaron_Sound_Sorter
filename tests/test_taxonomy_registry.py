from __future__ import annotations

import json
from pathlib import Path

import pytest

from aaron_sound_sorter.taxonomy_registry import TaxonomyCategory, TaxonomyRegistry


def _category(path: str, *, status: str = "tier_0_manual_only") -> TaxonomyCategory:
    return TaxonomyCategory(
        path=path,
        display_name=path.split("/")[-2],
        manual_selection_enabled=True,
        training_example_count=0,
        prototype_count=0,
        automatic_classification_status=status,
        fallback_parent="",
    )


def test_registry_resolves_deprecated_prefix_without_using_source_names() -> None:
    canonical = "Instruments/Woodwinds/Saxophone/Alto/One Shots"
    registry = TaxonomyRegistry(
        {canonical: _category(canonical)},
        prefix_aliases={"Instruments/Winds": "Instruments/Woodwinds"},
        taxonomy_version="test",
    )

    resolution = registry.resolve("Instruments/Winds/Saxophone/Alto/One Shots")

    assert resolution.canonical_path == canonical
    assert resolution.status == "alias"
    assert resolution.reasons == ("prefix_alias",)


def test_registry_keeps_manual_visibility_independent_from_readiness() -> None:
    manual_only = "FX/Everyday Foley/Glass/One Shots"
    ready = "Drums/Kick Drums/Generic Kick/One Shots"
    registry = TaxonomyRegistry(
        {
            manual_only: _category(manual_only, status="tier_0_manual_only"),
            ready: _category(ready, status="tier_3_unvalidated_suggestion_only"),
        }
    )

    assert registry.manual_labels() == [ready, manual_only]


def test_registry_rejects_duplicate_canonical_paths_on_load(tmp_path: Path) -> None:
    category = {
        "path": "FX/Everyday Foley/Glass/One Shots",
        "manual_selection_enabled": True,
    }
    registry_path = tmp_path / "canonical.json"
    registry_path.write_text(json.dumps({"categories": [category, category]}), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate canonical taxonomy path"):
        TaxonomyRegistry.load(registry_path)


def test_repository_registry_has_unique_paths_and_resolvable_aliases() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = TaxonomyRegistry.load(
        root / "config" / "canonical_taxonomy.json",
        root / "config" / "taxonomy_aliases.json",
    )

    assert len(registry.categories) == len(set(registry.categories))
    assert "FX/Everyday Foley/Glass/One Shots" in registry.manual_labels()
    assert registry.resolve("Instruments/Winds/Saxophone/Alto/One Shots").status == "alias"
