"""Brain JSON loading and saving."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class BrainRepository:
    """Repository for learned brain JSON files."""

    def load(self, path: Path) -> dict[str, Any]:
        """Load a brain JSON file."""
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Brain JSON not found: {resolved}")
        return json.loads(resolved.read_text(encoding="utf-8"))

    def save(self, path: Path, brain: dict[str, Any]) -> None:
        """Save a brain JSON file atomically enough for local use."""
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        temp = resolved.with_name(f".{resolved.name}.tmp")
        temp.write_text(json.dumps(brain, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(resolved)
