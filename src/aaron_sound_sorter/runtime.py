"""Runtime helpers retained for direct component tests.

The command-line application main lives only in ``Aaron_Sound_Sorter.py``.
This module no longer exposes a CLI main wrapper.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

MODULE_NAMES = [
    "core",
    "io_utils",
    "training_labels",
    "features",
    "brain",
    "committee",
    "reports",
    "preview",
    "selftests",
    "commands",
]


def load_components():
    modules = [import_module(f"aaron_sound_sorter.{name}") for name in MODULE_NAMES]
    shared = {}
    for module in modules:
        for key, value in vars(module).items():
            if key in {"__builtins__", "__doc__", "__loader__", "__name__", "__package__", "__spec__"}:
                continue
            shared[key] = value
    for module in modules:
        module.__dict__.update(shared)
    return {module.__name__.rsplit(".", 1)[-1]: module for module in modules}


def run_self_tests(tmp_root: Path | None = None):
    modules = load_components()
    return modules["selftests"].run_self_tests(tmp_root)
