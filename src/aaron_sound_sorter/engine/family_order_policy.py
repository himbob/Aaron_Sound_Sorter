# SOURCE-NAME BLINDNESS INVARIANT:
# This module documents family-order policy only. It must never use source
# paths, folder names, or producer text as classification evidence.
"""Top-family ordering policy for final arbitration and debugging.

FX is the most abstract family in the library. This module exposes a stable
policy summary so debug traces and reports state the intended order: Drums and
Instruments must be considered before broad FX abstractions unless concrete FX
motion, designed-transition evidence, or other measured FX authority exists.
"""

from __future__ import annotations

from typing import Any

TOP_FAMILY_REVIEW_ORDER = ("Drums", "Instruments", "Textures", "FX", "_TO_REVIEW")


def family_order_policy_summary() -> dict[str, Any]:
    """Return a JSON-safe source-blind family-order policy summary.

    Args:
        None.

    Returns:
        Dictionary suitable for debug traces and manifests.

    Side Effects:
        None.

    Raises:
        No intentional exceptions.

    Important Constraints:
        This is a policy trace, not a classifier. It does not choose a folder or
        change voter scores.
    """
    return {
        "policy_name": "drums_and_instruments_before_abstract_fx",
        "review_order": list(TOP_FAMILY_REVIEW_ORDER),
        "fx_policy": (
            "Treat FX as the most abstract family. Prefer valid Drums or "
            "Instruments evidence before broad FX unless measured concrete FX "
            "motion/design authority is present."
        ),
        "behavior_scope": "debug_trace_and_human_policy_guard",
        "changes_scores": False,
    }
