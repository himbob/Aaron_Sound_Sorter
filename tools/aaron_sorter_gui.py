#!/usr/bin/env python3
"""Launch the Aaron Sound Sorter desktop GUI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if __name__ == "__main__":
    if "--tk" in sys.argv:
        from aaron_sound_sorter.gui.app import main as tk_main  # noqa: PLC0415

        mode = "probe" if "--probe-layout" in sys.argv else "run"
        raise SystemExit(tk_main(mode=mode))  # type: ignore[arg-type]

    from aaron_sound_sorter.gui.web_app import main as web_main  # noqa: PLC0415

    raise SystemExit(web_main(probe="--probe-layout" in sys.argv, open_browser="--no-open" not in sys.argv))
