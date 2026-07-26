from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gui_taxonomy_catalog_retains_empty_future_categories_without_duplicates() -> None:
    payload = json.loads((ROOT / "config" / "gui_taxonomy_catalog.json").read_text(encoding="utf-8"))
    labels = payload["labels"]

    assert len(labels) == len(set(labels))
    assert "FX/Everyday Foley/Glass/One Shots" in labels
    assert "Drums/World Percussion/Tabla/Loops" in labels
    assert "Instruments/Plucked Strings/Koto/Loops" in labels
    assert "Instruments/Strings Bowed/Double Bass/One Shots" in labels
