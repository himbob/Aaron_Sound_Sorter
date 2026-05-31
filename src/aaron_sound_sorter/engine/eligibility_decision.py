"""Eligibility decision value object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EligibilityDecision:
    """Broad placement constraints derived from measured audio facts."""

    role_name: str
    confidence: float
    allowed_top_families: tuple[str, ...] = field(default_factory=tuple)
    blocked_path_fragments: tuple[str, ...] = field(default_factory=tuple)
    broad_folder_path: str = "_TO_REVIEW/Measured Role Conflict"
    reason: str = "no decisive parent eligibility"

    @property
    def is_decisive(self) -> bool:
        """Return True when this eligibility should constrain the final result."""
        return self.confidence >= 0.60 and self.role_name != "unknown"

    def is_path_allowed(self, folder_path: str, top_family: str) -> bool:
        """Return True when a final folder path is compatible with eligibility."""
        if not self.is_decisive:
            return True
        if self.allowed_top_families and str(top_family) not in set(self.allowed_top_families):
            return False
        low_path = str(folder_path).replace("\\", "/").lower()
        return all(fragment.lower() not in low_path for fragment in self.blocked_path_fragments)

    def to_report_dict(self) -> dict[str, Any]:
        """Return JSON-serializable report data."""
        return {
            "role_name": self.role_name,
            "confidence": round(float(self.confidence), 6),
            "allowed_top_families": list(self.allowed_top_families),
            "blocked_path_fragments": list(self.blocked_path_fragments),
            "broad_folder_path": self.broad_folder_path,
            "reason": self.reason,
            "decisive": self.is_decisive,
        }
