#!/usr/bin/env python3
"""Build the frozen canonical GUI taxonomy from reviewed local metadata."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aaron_sound_sorter.gui.preview_service import (  # noqa: E402
    load_json_taxonomy_labels,
    load_training_taxonomy_labels,
)
from aaron_sound_sorter.taxonomy_contracts import canonicalize_taxonomy_label, is_trainable_taxonomy_label  # noqa: E402
from aaron_sound_sorter.taxonomy_registry import resolve_configured_aliases  # noqa: E402

DEFAULT_BRAIN = Path("stage4_folder_brain.json")
DEFAULT_GUI_CATALOG = Path("config/gui_taxonomy_catalog.json")
DEFAULT_TRAINING_ROOT = Path("training/locked_curated_v1")
DEFAULT_ALIASES = Path("config/taxonomy_aliases.json")
DEFAULT_OUTPUT = Path("config/canonical_taxonomy.json")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--brain", type=Path, default=DEFAULT_BRAIN)
    parser.add_argument("--gui-catalog", type=Path, default=DEFAULT_GUI_CATALOG)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--taxonomy-version", default="2026-07-25")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build and write one deterministic canonical taxonomy registry."""
    args = build_parser().parse_args(argv)
    root = args.project_root.expanduser().resolve()
    payload = build_canonical_taxonomy_payload(
        brain_path=resolve_path(root, args.brain),
        gui_catalog_path=resolve_path(root, args.gui_catalog),
        training_root=resolve_path(root, args.training_root),
        aliases_path=resolve_path(root, args.aliases),
        taxonomy_version=str(args.taxonomy_version),
    )
    output_path = resolve_path(root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Canonical categories: {len(payload['categories'])}")
    print(f"Output: {output_path}")
    return 0


def resolve_path(project_root: Path, path: Path) -> Path:
    """Resolve a project-relative or absolute path."""
    expanded = Path(path).expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root / expanded).resolve()


def build_canonical_taxonomy_payload(
    *,
    brain_path: Path,
    gui_catalog_path: Path,
    training_root: Path,
    aliases_path: Path,
    taxonomy_version: str,
) -> dict[str, Any]:
    """Return canonical taxonomy JSON derived from explicit taxonomy sources.

    Args:
        brain_path: Existing detailed brain used only to seed the first frozen
            registry and capture current training counts.
        gui_catalog_path: Manually maintained future-category catalog.
        training_root: Human-curated training slot tree.
        aliases_path: Reviewed deprecated-path mappings.
        taxonomy_version: Version recorded in the output.

    Returns:
        JSON-compatible canonical taxonomy payload.

    Side Effects:
        Reads local metadata and training slot counts. Audio filenames are not
        interpreted as taxonomy evidence.
    """
    brain_payload = json.loads(brain_path.read_text(encoding="utf-8")) if brain_path.is_file() else {}
    brain_counts = _brain_counts(brain_payload)
    catalog_labels = load_json_taxonomy_labels(gui_catalog_path)
    training_labels = load_training_taxonomy_labels(training_root)
    exact_aliases, prefix_aliases = _alias_rules(aliases_path)

    raw_labels = set(brain_counts) | set(catalog_labels) | set(training_labels)
    canonical_labels = {
        resolve_configured_aliases(
            canonicalize_taxonomy_label(label),
            exact_aliases=exact_aliases,
            prefix_aliases=prefix_aliases,
        )[0]
        for label in raw_labels
    }
    canonical_labels = {label for label in canonical_labels if is_trainable_taxonomy_label(label)}
    approved_training_counts = _training_counts(training_root, exact_aliases, prefix_aliases)
    canonical_brain_counts = Counter()
    for label, count in brain_counts.items():
        canonical_label, _reasons = resolve_configured_aliases(
            canonicalize_taxonomy_label(label),
            exact_aliases=exact_aliases,
            prefix_aliases=prefix_aliases,
        )
        canonical_brain_counts[canonical_label] += count

    categories = []
    for label in sorted(canonical_labels):
        training_count = approved_training_counts.get(label, 0)
        categories.append(
            {
                "path": label,
                "display_name": display_name_for_label(label),
                "manual_selection_enabled": True,
                "training_example_count": training_count,
                "prototype_count": 0,
                "automatic_classification_status": readiness_status(training_count),
                "fallback_parent": fallback_parent_for_label(label),
                "legacy_brain_example_count": canonical_brain_counts.get(label, 0),
                "sources": sorted(
                    source
                    for source, present in {
                        "legacy_brain_seed": label in canonical_brain_counts,
                        "future_gui_catalog": label in set(catalog_labels),
                        "curated_training_slot": label in set(training_labels),
                    }.items()
                    if present
                ),
            }
        )
    return {
        "schema_version": 1,
        "taxonomy_version": taxonomy_version,
        "description": (
            "Frozen end-user category registry. Visibility is independent from classifier readiness; "
            "runtime audio evidence never uses source filenames or paths."
        ),
        "categories": categories,
    }


def display_name_for_label(label: str) -> str:
    """Return a compact display name from an internal taxonomy path."""
    parts = label.split("/")
    if len(parts) >= 2 and parts[-1] in {"One Shots", "Loops", "Long FX", "Long Running"}:
        return f"{parts[-2]} — {parts[-1]}"
    return parts[-1]


def fallback_parent_for_label(label: str) -> str:
    """Return a structure-preserving parent fallback for a detailed label."""
    parts = label.split("/")
    if len(parts) >= 4 and parts[-1] in {"One Shots", "Loops", "Long FX", "Long Running"}:
        return "/".join([*parts[:-2], parts[-1]])
    return "/".join(parts[:-1])


def readiness_status(training_example_count: int) -> str:
    """Return a conservative readiness tier without enabling ownership."""
    if training_example_count <= 0:
        return "tier_0_manual_only"
    if training_example_count <= 4:
        return "tier_1_sparse_suggestion_only"
    if training_example_count <= 14:
        return "tier_2_developing_suggestion_only"
    return "tier_3_unvalidated_suggestion_only"


def _brain_counts(payload: Any) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}
    raw_counts = payload.get("counts", {})
    if isinstance(raw_counts, dict):
        return {str(label): max(0, int(count)) for label, count in raw_counts.items()}
    raw_labels = payload.get("labels", [])
    return {str(label): 0 for label in raw_labels} if isinstance(raw_labels, list) else {}


def _alias_rules(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    exact = payload.get("exact_aliases", {}) if isinstance(payload, dict) else {}
    prefixes = payload.get("prefix_aliases", {}) if isinstance(payload, dict) else {}
    if not isinstance(exact, dict) or not isinstance(prefixes, dict):
        raise ValueError("taxonomy alias maps must be objects")
    return (
        {str(key): str(value) for key, value in exact.items()},
        {str(key): str(value) for key, value in prefixes.items()},
    )


def _training_counts(
    training_root: Path,
    exact_aliases: dict[str, str],
    prefix_aliases: dict[str, str],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not training_root.is_dir():
        return counts
    structure_by_slot = {
        "_ONE_SHOTS": "One Shots",
        "_LOOPS": "Loops",
        "_LONG_FX": "Long FX",
        "_LONG_RUNNING": "Long Running",
    }
    for slot_path in training_root.rglob("*"):
        structure = structure_by_slot.get(slot_path.name)
        if structure is None or not slot_path.is_dir():
            continue
        raw_label = "/".join([*slot_path.parent.relative_to(training_root).parts, structure])
        label, _reasons = resolve_configured_aliases(
            canonicalize_taxonomy_label(raw_label),
            exact_aliases=exact_aliases,
            prefix_aliases=prefix_aliases,
        )
        counts[label] += sum(path.is_file() or path.is_symlink() for path in slot_path.iterdir())
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
