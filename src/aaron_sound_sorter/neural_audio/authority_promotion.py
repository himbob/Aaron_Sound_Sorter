"""Data gates for promoting small neural category groups into production."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aaron_sound_sorter.taxonomy_contracts import canonicalize_taxonomy_label

SUPPORTED_AUTHORITY_GROUPS = frozenset({"voice", "drums", "environmental_fx", "designed_fx", "isolated_instruments"})


@dataclass(frozen=True)
class AuthorityPromotionGates:
    """Minimum independent evidence required for one production group."""

    minimum_calibration_reviews: int = 40
    minimum_calibration_accepts: int = 8
    minimum_calibration_rejects: int = 8
    minimum_heldout_examples: int = 12
    minimum_top1_accuracy: float = 0.80
    minimum_parent_family_accuracy: float = 0.90
    maximum_incorrect_auto_placement_rate: float = 0.02


def category_authority_group(label: str) -> str:
    """Map a canonical or historical evaluation label to a rollout group."""
    normalized = canonicalize_taxonomy_label(label)
    lowered = normalized.casefold()
    if lowered.startswith("voice/") or normalized.startswith("Instruments/Voice/"):
        return "voice"
    if normalized.startswith("FX/Human and Voice FX/"):
        return "voice"
    if normalized.startswith("Drums/") or lowered.startswith("nonvoice/drums"):
        return "drums"
    if normalized.startswith("FX/Environmental/") or normalized.startswith("FX/Nature/"):
        return "environmental_fx"
    if normalized.startswith("FX/"):
        return "designed_fx"
    if normalized.startswith("Instruments/") or lowered.startswith("nonvoice/"):
        if "loop" not in lowered and "phrase" not in lowered:
            return "isolated_instruments"
    return ""


def assess_group_promotion(
    *,
    group: str,
    calibration_status: dict[str, Any],
    heldout_metrics: dict[str, Any],
    gates: AuthorityPromotionGates | None = None,
) -> dict[str, Any]:
    """Return a transparent pass/fail record for one limited rollout group."""
    gates = gates or AuthorityPromotionGates()
    if group not in SUPPORTED_AUTHORITY_GROUPS:
        raise ValueError(f"unsupported authority group: {group}")
    checks = {
        "calibration_built": calibration_status.get("status") == "built",
        "calibration_reviews": int(calibration_status.get("reviewed_unique_hash_count", 0))
        >= gates.minimum_calibration_reviews,
        "calibration_accepts": int(calibration_status.get("accepted_count", 0)) >= gates.minimum_calibration_accepts,
        "calibration_rejects": int(calibration_status.get("rejected_count", 0)) >= gates.minimum_calibration_rejects,
        "heldout_examples": int(heldout_metrics.get("example_count", 0)) >= gates.minimum_heldout_examples,
        "heldout_top1": float(heldout_metrics.get("top1_accuracy", 0.0)) >= gates.minimum_top1_accuracy,
        "heldout_parent_family": float(heldout_metrics.get("parent_family_accuracy", 0.0))
        >= gates.minimum_parent_family_accuracy,
        "incorrect_auto_placement": float(heldout_metrics.get("incorrect_auto_placement_rate", 1.0))
        <= gates.maximum_incorrect_auto_placement_rate,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "group": group,
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "gates": asdict(gates),
        "heldout_metrics": heldout_metrics,
        "message": (
            "This category group passed the limited neural authority gates."
            if not failed
            else f"Kept this category group in suggestion/review mode: {', '.join(failed)}."
        ),
    }


def enabled_authority_groups(project_root: Path) -> frozenset[str]:
    """Load only explicitly promoted groups from the calibration status."""
    status_path = Path(project_root) / "neural_artifacts" / "calibration" / "current" / "calibration_status.json"
    if not status_path.is_file():
        return frozenset()
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return frozenset()
    if status.get("status") != "built" or not bool(status.get("production_authority_enabled", False)):
        return frozenset()
    groups = frozenset(str(group) for group in status.get("enabled_authority_groups", []))
    return groups & SUPPORTED_AUTHORITY_GROUPS
