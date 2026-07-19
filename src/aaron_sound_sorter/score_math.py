from __future__ import annotations

from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    """Return ``value`` as a float, or ``default`` when conversion fails."""
    try:
        return float(value)
    except Exception:
        return float(default)


def clamp01(value: Any) -> float:
    """Clamp a numeric value into the 0..1 evidence-score range."""
    return max(0.0, min(1.0, safe_float(value, 0.0)))


def average_score(*values: Any) -> float:
    """Return a clamped, zero-safe average for evidence score fragments."""
    if not values:
        return 0.0
    usable = [clamp01(value) for value in values]
    return sum(usable) / len(usable)
