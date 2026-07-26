from __future__ import annotations

import json
from pathlib import Path

from tools.build_canonical_taxonomy import build_canonical_taxonomy_payload


def test_canonical_taxonomy_keeps_empty_catalog_category(tmp_path: Path) -> None:
    brain = tmp_path / "brain.json"
    brain.write_text(
        json.dumps({"counts": {"Drums/Kick Drums/Generic Kick/One Shots": 8}}),
        encoding="utf-8",
    )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps({"labels": ["FX/Everyday Foley/Glass/One Shots"]}),
        encoding="utf-8",
    )
    aliases = tmp_path / "aliases.json"
    aliases.write_text(json.dumps({"exact_aliases": {}, "prefix_aliases": {}}), encoding="utf-8")
    training_root = tmp_path / "training"
    training_root.mkdir()

    payload = build_canonical_taxonomy_payload(
        brain_path=brain,
        gui_catalog_path=catalog,
        training_root=training_root,
        aliases_path=aliases,
        taxonomy_version="test",
    )
    categories = {row["path"]: row for row in payload["categories"]}

    assert categories["FX/Everyday Foley/Glass/One Shots"]["manual_selection_enabled"] is True
    assert categories["FX/Everyday Foley/Glass/One Shots"]["automatic_classification_status"] == "tier_0_manual_only"
