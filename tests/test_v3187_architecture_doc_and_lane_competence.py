"""Current architecture-document regression checks."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_current_architecture_boundary_is_packaged_for_future_ai() -> None:
    must_read_doc = PROJECT_ROOT / "docs" / "AI_MUST_READ_CURRENT_ARCHITECTURE.md"

    assert must_read_doc.exists()
    text = must_read_doc.read_text(encoding="utf-8")
    normalized_text = " ".join(text.split())
    assert "limited GUI proposal authority" in normalized_text
    assert "Unknown cross-family conflicts go to Review" in normalized_text
    assert "not a place to add new category rules" in normalized_text
