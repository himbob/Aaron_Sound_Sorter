"""Canonical end-user taxonomy registry and deprecated-path resolution."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .taxonomy_contracts import (
    canonicalize_taxonomy_label,
    is_trainable_taxonomy_label,
    normalize_taxonomy_path,
    taxonomy_label_contract,
)


@dataclass(frozen=True)
class TaxonomyCategory:
    """One canonical GUI category with readiness separate from visibility.

    Args:
        path: Canonical trainable taxonomy path.
        display_name: Compact human-facing category name.
        manual_selection_enabled: Whether the GUI may offer the category.
        training_example_count: Approved example count known at registry build.
        prototype_count: Prototype count known at registry build.
        automatic_classification_status: Readiness label; visibility never
            depends on this value.
        fallback_parent: Safer parent category when a detailed leaf is not
            ready.

    Side Effects:
        None.
    """

    path: str
    display_name: str
    manual_selection_enabled: bool
    training_example_count: int
    prototype_count: int
    automatic_classification_status: str
    fallback_parent: str


@dataclass(frozen=True)
class TaxonomyResolution:
    """Result of resolving one internal label through canonical aliases.

    Args:
        input_label: Original label supplied by a brain, GUI, or ledger.
        normalized_label: Grammar-canonical label before configured aliases.
        canonical_path: Resolved canonical category path.
        status: ``canonical``, ``alias``, or ``unknown``.
        reasons: Stable resolution diagnostics.

    Side Effects:
        None.
    """

    input_label: str
    normalized_label: str
    canonical_path: str
    status: str
    reasons: tuple[str, ...]


class TaxonomyRegistry:
    """Resolve labels against one versioned canonical category registry.

    Args:
        categories: Canonical categories keyed by path.
        exact_aliases: Deprecated full paths mapped to canonical paths.
        prefix_aliases: Deprecated hierarchy prefixes mapped to canonical
            hierarchy prefixes.
        taxonomy_version: Registry version written into evidence reports.

    Raises:
        ValueError: If categories collide, aliases cycle, or aliases resolve to
            categories absent from the registry.

    Side Effects:
        None.
    """

    def __init__(
        self,
        categories: Mapping[str, TaxonomyCategory],
        *,
        exact_aliases: Mapping[str, str] | None = None,
        prefix_aliases: Mapping[str, str] | None = None,
        taxonomy_version: str = "",
    ) -> None:
        self.categories = dict(categories)
        self.exact_aliases = _normalized_alias_mapping(exact_aliases or {})
        self.prefix_aliases = _normalized_alias_mapping(prefix_aliases or {})
        self.taxonomy_version = str(taxonomy_version)
        _validate_registry(self.categories, self.exact_aliases, self.prefix_aliases)

    @classmethod
    def load(cls, registry_path: Path, aliases_path: Path | None = None) -> TaxonomyRegistry:
        """Load a canonical registry and optional alias file from JSON.

        Args:
            registry_path: JSON file containing canonical category records.
            aliases_path: Optional JSON file containing exact and prefix maps.

        Returns:
            Validated taxonomy registry.

        Raises:
            OSError: If a requested file cannot be read.
            ValueError: If JSON structure or registry contracts are invalid.

        Side Effects:
            Reads the supplied files.
        """
        registry_payload = json.loads(Path(registry_path).read_text(encoding="utf-8"))
        raw_categories = registry_payload.get("categories", [])
        if not isinstance(raw_categories, list):
            raise ValueError("canonical taxonomy categories must be a list")
        categories: dict[str, TaxonomyCategory] = {}
        for raw_category in raw_categories:
            category = _category_from_mapping(raw_category)
            if category.path in categories:
                raise ValueError(f"duplicate canonical taxonomy path: {category.path}")
            categories[category.path] = category
        alias_payload: dict[str, Any] = {}
        if aliases_path is not None and Path(aliases_path).is_file():
            loaded_aliases = json.loads(Path(aliases_path).read_text(encoding="utf-8"))
            if not isinstance(loaded_aliases, dict):
                raise ValueError("taxonomy alias payload must be an object")
            alias_payload = loaded_aliases
        return cls(
            categories,
            exact_aliases=_string_mapping(alias_payload.get("exact_aliases", {})),
            prefix_aliases=_string_mapping(alias_payload.get("prefix_aliases", {})),
            taxonomy_version=str(registry_payload.get("taxonomy_version", "")),
        )

    def resolve(self, label: object) -> TaxonomyResolution:
        """Resolve a label without consulting source filenames or paths.

        Args:
            label: Internal taxonomy label from trusted metadata.

        Returns:
            Canonical, aliased, or unknown resolution.

        Side Effects:
            None.
        """
        input_label = str(label or "")
        contract = taxonomy_label_contract(input_label)
        normalized = contract.canonical_label
        resolved, alias_reasons = resolve_configured_aliases(
            normalized,
            exact_aliases=self.exact_aliases,
            prefix_aliases=self.prefix_aliases,
        )
        grammar_reasons = tuple(reason for reason in contract.reasons if reason.startswith("canonicalized_"))
        all_reasons = (*grammar_reasons, *alias_reasons)
        if resolved in self.categories:
            status = "alias" if all_reasons else "canonical"
            reasons = all_reasons
        else:
            status = "unknown"
            reasons = (*all_reasons, "unknown_canonical_path")
        return TaxonomyResolution(input_label, normalized, resolved, status, reasons)

    def manual_labels(self) -> list[str]:
        """Return sorted GUI-selectable labels regardless of readiness."""
        return sorted(category.path for category in self.categories.values() if category.manual_selection_enabled)


def resolve_configured_aliases(
    label: str,
    *,
    exact_aliases: Mapping[str, str],
    prefix_aliases: Mapping[str, str],
) -> tuple[str, tuple[str, ...]]:
    """Resolve exact and longest-prefix aliases for an internal label.

    Args:
        label: Grammar-canonical internal taxonomy label.
        exact_aliases: Full-path aliases.
        prefix_aliases: Hierarchy-prefix aliases.

    Returns:
        Resolved path and applied-alias diagnostics.

    Raises:
        ValueError: If alias resolution cycles.

    Side Effects:
        None.
    """
    current = canonicalize_taxonomy_label(label)
    reasons: list[str] = []
    seen: set[str] = set()
    for _iteration in range(32):
        if current in seen:
            raise ValueError(f"taxonomy alias cycle includes: {current}")
        seen.add(current)
        exact_target = exact_aliases.get(current)
        if exact_target is not None:
            current = exact_target
            reasons.append("exact_alias")
            continue
        prefix_match = next(
            (
                source_prefix
                for source_prefix in sorted(prefix_aliases, key=lambda value: (-len(value), value))
                if current == source_prefix or current.startswith(f"{source_prefix}/")
            ),
            None,
        )
        if prefix_match is None:
            return current, tuple(reasons)
        suffix = current[len(prefix_match) :]
        current = canonicalize_taxonomy_label(f"{prefix_aliases[prefix_match]}{suffix}")
        reasons.append("prefix_alias")
    raise ValueError("taxonomy alias chain exceeded 32 steps")


def _category_from_mapping(payload: Any) -> TaxonomyCategory:
    if not isinstance(payload, dict):
        raise ValueError("canonical taxonomy category must be an object")
    path = canonicalize_taxonomy_label(payload.get("path", ""))
    if not is_trainable_taxonomy_label(path):
        raise ValueError(f"canonical taxonomy path is not trainable: {path or '<empty>'}")
    return TaxonomyCategory(
        path=path,
        display_name=str(payload.get("display_name", path.split("/")[-2])).strip(),
        manual_selection_enabled=bool(payload.get("manual_selection_enabled", True)),
        training_example_count=max(0, int(payload.get("training_example_count", 0))),
        prototype_count=max(0, int(payload.get("prototype_count", 0))),
        automatic_classification_status=str(payload.get("automatic_classification_status", "manual_only")),
        fallback_parent=canonicalize_taxonomy_label(payload.get("fallback_parent", "")),
    )


def _normalized_alias_mapping(mapping: Mapping[str, str]) -> dict[str, str]:
    return {
        normalize_taxonomy_path(source): canonicalize_taxonomy_label(target)
        for source, target in mapping.items()
        if normalize_taxonomy_path(source) and canonicalize_taxonomy_label(target)
    }


def _string_mapping(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError("taxonomy aliases must be an object")
    return {str(key): str(value) for key, value in payload.items()}


def _validate_registry(
    categories: Mapping[str, TaxonomyCategory],
    exact_aliases: Mapping[str, str],
    prefix_aliases: Mapping[str, str],
) -> None:
    for path, category in categories.items():
        if path != category.path or not is_trainable_taxonomy_label(path):
            raise ValueError(f"invalid canonical taxonomy category: {path}")
    for alias_path in exact_aliases:
        resolved, _reasons = resolve_configured_aliases(
            alias_path,
            exact_aliases=exact_aliases,
            prefix_aliases=prefix_aliases,
        )
        if resolved not in categories:
            raise ValueError(f"taxonomy alias target is absent from registry: {alias_path} -> {resolved}")
    for source_prefix, target_prefix in prefix_aliases.items():
        matching_targets = [
            path for path in categories if path == target_prefix or path.startswith(f"{target_prefix}/")
        ]
        if not matching_targets:
            raise ValueError(
                f"taxonomy prefix alias target is absent from registry: {source_prefix} -> {target_prefix}"
            )
