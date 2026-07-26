"""Render the small human-facing neural Learning Center page."""

from __future__ import annotations

from pathlib import Path


def render_learning_center_html() -> str:
    """Return the bundled Learning Center HTML shell.

    Returns:
        Static browser application markup. Runtime evidence is loaded through
        local JSON endpoints so no private paths are embedded in the page.

    Raises:
        OSError: If the bundled page asset cannot be read.

    Side Effects:
        Reads one package-local HTML file.
    """
    page_path = Path(__file__).with_name("static") / "learning_center.html"
    return page_path.read_text(encoding="utf-8")
