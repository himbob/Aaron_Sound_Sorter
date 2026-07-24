# SOURCE-NAME BLINDNESS INVARIANT:
# This module handles measured evidence dictionaries only. It must never parse
# producer filenames, source paths, ZIP member names, or sample-pack folders as
# classification evidence.
"""Reusable numeric evidence lookup utilities.

Several voter and arbiter paths need to read previously measured scores from
``SharedAudioFacts.evidence``. Walking the same nested diagnostics repeatedly is
expensive on real smoke panels, so this module centralizes cached numeric lookup
without changing classifier behavior.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from aaron_sound_sorter.domain.models import SharedAudioFacts

NUMERIC_EVIDENCE_CACHE_ATTR = "_numeric_evidence_leaf_values_cache"


def numeric_evidence_value(facts: SharedAudioFacts, key: str, default: float = 0.0) -> float:
    """Return one numeric evidence value by leaf key.

    Args:
        facts: Shared measured audio facts for one file.
        key: Evidence key to find. Only dictionary leaf names are matched.
        default: Value returned when the key is missing or nonnumeric.

    Returns:
        A finite float from the measured evidence cache, or ``default``.

    Side Effects:
        Builds a per-facts-object cache on first use.

    Important Constraints:
        This lookup reads measured evidence only. It does not inspect input file
        names, source folders, ZIP paths, or output folders.
    """
    values = cached_numeric_leaf_values(facts)
    return float(values.get(key, default))


def cached_numeric_leaf_values(facts: SharedAudioFacts) -> dict[str, float]:
    """Return a cached first-seen map of numeric evidence leaf values."""
    cached = getattr(facts, NUMERIC_EVIDENCE_CACHE_ATTR, None)
    if isinstance(cached, dict):
        return cached
    values: dict[str, float] = {}
    evidence = getattr(facts, "evidence", {})
    if isinstance(evidence, dict):
        add_numeric_leaf_values(values, evidence)
    try:
        object.__setattr__(facts, NUMERIC_EVIDENCE_CACHE_ATTR, values)
    except Exception:
        pass
    return values


def add_numeric_leaf_values(output: dict[str, float], value: Any) -> None:
    """Walk evidence once and store first-seen finite numeric leaf values."""
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            child_name = str(child_key)
            if child_name.startswith("_"):
                continue
            if isinstance(child_value, (Mapping, list, tuple)):
                add_numeric_leaf_values(output, child_value)
                continue
            number = finite_number(child_value)
            if number is not None and child_name not in output:
                output[child_name] = number
        return
    if isinstance(value, (list, tuple)):
        for child_value in value:
            add_numeric_leaf_values(output, child_value)


def add_finite_numeric(
    output: dict[str, float],
    key: str,
    value: Any,
    *,
    clamp_to_unit: bool = False,
    replace: bool = True,
) -> None:
    """Add one finite numeric value to ``output``.

    Args:
        output: Mutable numeric evidence map.
        key: Destination evidence key.
        value: Numeric-like value to add.
        clamp_to_unit: Clamp the stored value to ``0..1`` when true.
        replace: Whether a later value may replace an existing key.
    """
    if not replace and key in output:
        return
    number = finite_number(value)
    if number is None:
        return
    if clamp_to_unit:
        number = max(0.0, min(1.0, number))
    output[key] = number


def add_direct_numeric_children(
    output: dict[str, float],
    values: Mapping[str, Any],
    *,
    clamp_to_unit: bool = False,
) -> None:
    """Add direct non-container numeric values from an evidence dictionary."""
    for key, value in values.items():
        key_name = str(key)
        if key_name.startswith("_") or isinstance(value, (Mapping, list, tuple)):
            continue
        add_finite_numeric(output, key_name, value, clamp_to_unit=clamp_to_unit)


def finite_number(value: Any) -> float | None:
    """Return a finite float for numeric-like values, otherwise ``None``."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number
