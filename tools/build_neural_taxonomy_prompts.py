#!/usr/bin/env python3
"""Build a source-name-blind read-only CLAP prompt catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.taxonomy_registry import TaxonomyRegistry  # noqa: E402

DEFAULT_TAXONOMY = Path("config/canonical_taxonomy.json")
DEFAULT_ALIASES = Path("config/taxonomy_aliases.json")
DEFAULT_OUTPUT = Path("config/neural_taxonomy_prompts.json")

STRUCTURE_DESCRIPTIONS = {
    "One Shots": "short isolated one-shot",
    "Loops": "repeating musical or rhythmic loop",
    "Long FX": "long-running sound effect",
    "Long Running": "long-running audio sample",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate and write a deterministic prompt catalog."""
    args = build_parser().parse_args(argv)
    root = args.project_root.expanduser().resolve()
    taxonomy_path = resolve_path(root, args.taxonomy)
    registry = TaxonomyRegistry.load(taxonomy_path, resolve_path(root, args.aliases))
    payload = build_prompt_catalog_payload(registry)
    output_path = resolve_path(root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Prompt categories: {len(payload['categories'])}")
    print(f"Output: {output_path}")
    return 0


def resolve_path(project_root: Path, path: Path) -> Path:
    """Resolve a project-relative or absolute path."""
    expanded = Path(path).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root / expanded).resolve()


def build_prompt_catalog_payload(registry: TaxonomyRegistry) -> dict[str, Any]:
    """Return generated prompt ensembles for every canonical category.

    Args:
        registry: Frozen canonical category registry.

    Returns:
        JSON-compatible read-only prompt catalog.

    Side Effects:
        None. Prompts derive only from internal taxonomy descriptions.
    """
    categories = []
    paths = registry.manual_labels()
    for path in paths:
        categories.append(prompt_category(path, paths))
    return {
        "schema_version": 1,
        "prompt_catalog_version": f"{registry.taxonomy_version}-generated-v1",
        "taxonomy_version": registry.taxonomy_version,
        "production_ownership_enabled": False,
        "source_name_policy": "taxonomy descriptions only; source filenames and paths are forbidden",
        "categories": categories,
    }


def prompt_category(path: str, all_paths: list[str]) -> dict[str, Any]:
    """Build one conservative positive/negative taxonomy prompt ensemble."""
    parts = path.split("/")
    structure = parts[-1]
    leaf = parts[-2]
    family_words = ", ".join(parts[1:-2]) or parts[0]
    audible_subject = audible_subject_for_path(path)
    structure_description = STRUCTURE_DESCRIPTIONS.get(structure, "audio sample")
    positive_prompts = [
        f"a {structure_description} of {audible_subject}",
        f"a clean producer sample featuring {audible_subject}",
        f"the audible sound of {audible_subject}",
        f"{audible_subject} in the {family_words.lower()} sound family",
    ]
    sibling_subjects = sibling_confusion_subjects(path, all_paths)
    negative_prompts = [
        f"a {STRUCTURE_DESCRIPTIONS.get(structure, 'sample')} of {subject}" for subject in sibling_subjects
    ]
    return {
        "path": path,
        "positive_prompts": positive_prompts,
        "negative_prompts": negative_prompts,
        "broad_family": "/".join(parts[:-2]),
        "structure": structure,
        "review_status": "generated_needs_human_review",
        "production_ownership_enabled": False,
        "display_leaf": leaf,
    }


def audible_subject_for_path(path: str) -> str:
    """Return an audible concept description from a canonical taxonomy path."""
    parts = path.split("/")
    content_parts = parts[1:-1]
    leaf = content_parts[-1]
    parent = content_parts[-2] if len(content_parts) > 1 else parts[0]
    if leaf.casefold().startswith("generic "):
        leaf = leaf[8:]
    if leaf.casefold() in parent.casefold() or parent.casefold() in leaf.casefold():
        return leaf.lower()
    return f"{leaf.lower()} from {parent.lower()}"


def sibling_confusion_subjects(path: str, all_paths: list[str], *, limit: int = 4) -> list[str]:
    """Return deterministic same-structure sibling concepts as negatives."""
    parts = path.split("/")
    structure = parts[-1]
    parent_prefix = "/".join(parts[:-2])
    siblings = [
        candidate
        for candidate in all_paths
        if candidate != path and candidate.endswith(f"/{structure}") and candidate.startswith(f"{parent_prefix}/")
    ]
    subjects: list[str] = []
    for sibling in siblings:
        subject = audible_subject_for_path(sibling)
        if subject not in subjects:
            subjects.append(subject)
        if len(subjects) >= limit:
            break
    if subjects:
        return subjects
    fallback = {
        "Drums": "a tonal instrument",
        "Instruments": "a non-musical sound effect",
        "FX": "a clean musical instrument performance",
    }
    return [fallback.get(parts[0], "an unrelated sound source")]


if __name__ == "__main__":
    raise SystemExit(main())
