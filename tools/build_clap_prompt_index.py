#!/usr/bin/env python3
"""Build and activate a versioned read-only CLAP taxonomy prompt index."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.neural_audio.gui_training import configured_clap_trainer  # noqa: E402
from aaron_sound_sorter.neural_audio.prompt_brain import ClapPromptIndex  # noqa: E402

DEFAULT_CATALOG = Path("config/neural_taxonomy_prompts.json")
DEFAULT_INDEX_ROOT = Path("neural_artifacts/prompt_indexes/hf_clap")
DEFAULT_POINTER = Path("config/runtime/neural_clap_prompt_index_path.txt")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--index-root", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build, validate, version, and atomically activate a prompt index."""
    args = build_parser().parse_args(argv)
    root = args.project_root.expanduser().resolve()
    catalog_path = resolve_path(root, args.catalog)
    index_root = resolve_path(root, args.index_root)
    pointer_path = resolve_path(root, args.pointer)
    trainer = configured_clap_trainer(root)
    index = ClapPromptIndex.build(catalog_path, trainer.provider)
    version_dir = index_root / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    index.save(version_dir)
    loaded = ClapPromptIndex.load(version_dir)
    if loaded.metadata != index.metadata:
        raise RuntimeError("serialized prompt index failed metadata validation")
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_value = str(version_dir.relative_to(root)) if version_dir.is_relative_to(root) else str(version_dir)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=pointer_path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
        handle.write(pointer_value + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary_path.replace(pointer_path)
    payload = {
        "schema_version": 1,
        "status": "built",
        "index_path": pointer_value,
        "pointer_path": str(pointer_path),
        "metadata": index.metadata.__dict__,
        "production_ownership_enabled": False,
    }
    if args.output_json is not None:
        output_path = resolve_path(root, args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def resolve_path(project_root: Path, path: Path) -> Path:
    """Resolve a project-relative or absolute path."""
    expanded = Path(path).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root / expanded).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
