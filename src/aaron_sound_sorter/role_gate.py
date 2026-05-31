"""Dynamic role-neighborhood gate for the two-voter sorter.

This module deliberately avoids hand-written leaf-category rules.  It derives
broad candidate lanes from the brain's own label physics profiles.  Every label
contributes one equal label-profile vote when parent buckets are built, so large
training folders do not dominate small labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from aaron_sound_sorter.core import FEATURE_NAMES, FP_SIZE
from aaron_sound_sorter.domain.models import AudioPhysics, SharedAudioFacts


@dataclass(frozen=True)
class RoleGateDecision:
    """Candidate-lane decision derived from learned label-profile buckets."""

    enabled: bool
    selected_top_families: tuple[str, ...] = ()
    selected_branch_prefixes: tuple[str, ...] = ()
    allowed_labels: tuple[str, ...] = ()
    top_scores: tuple[dict[str, Any], ...] = ()
    branch_scores: tuple[dict[str, Any], ...] = ()
    reason: str = ""
    final_effects_enabled: bool = False

    def allows(self, label: str) -> bool:
        if not self.enabled or not self.allowed_labels:
            return True
        return str(label) in set(self.allowed_labels)

    def evidence(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "selected_top_families": list(self.selected_top_families),
            "selected_branch_prefixes": list(self.selected_branch_prefixes),
            "allowed_label_count": len(self.allowed_labels),
            "allowed_labels": list(self.allowed_labels),
            "top_scores": list(self.top_scores),
            "branch_scores": list(self.branch_scores),
            "reason": self.reason,
            "final_effects_enabled": self.final_effects_enabled,
            "mode": "final_filter" if self.final_effects_enabled else "diagnostic_only",
        }


TOP_FAMILIES = ("Drums", "Instruments", "Textures", "FX")


def label_parts(label: str) -> list[str]:
    return [p for p in str(label or "").replace("\\", "/").split("/") if p]


def top_family(label: str) -> str:
    parts = label_parts(label)
    return parts[0] if parts else ""


def prefix(label: str, depth: int) -> str:
    parts = label_parts(label)
    return "/".join(parts[:depth])


def safe_stat_median(profile: dict[str, Any], name: str) -> float | None:
    stats = profile.get("feature_stats", {}) if isinstance(profile, dict) else {}
    stat = stats.get(name) if isinstance(stats, dict) else None
    if not isinstance(stat, dict):
        return None
    try:
        value = float(stat.get("median", 0.0) or 0.0)
        if np.isfinite(value):
            return value
    except Exception:
        return None
    return None


def label_profile_matrix(
    labels: Sequence[str], profiles: dict[str, Any], feature_names: Sequence[str]
) -> tuple[np.ndarray, list[str]]:
    """Return equal-label profile medians for labels with complete feature stats."""
    rows: list[list[float]] = []
    kept: list[str] = []
    for label in labels:
        profile = profiles.get(label)
        if not isinstance(profile, dict):
            continue
        values: list[float] = []
        ok = True
        for name in feature_names:
            value = safe_stat_median(profile, str(name))
            if value is None:
                ok = False
                break
            values.append(value)
        if ok:
            rows.append(values)
            kept.append(label)
    if not rows:
        return np.zeros((0, len(feature_names)), dtype=np.float32), []
    return np.asarray(rows, dtype=np.float32), kept


def parent_bucket_from_labels(
    labels: Sequence[str], profiles: dict[str, Any], feature_names: Sequence[str]
) -> dict[str, np.ndarray] | None:
    """Build one parent bucket from child label medians, count-neutral by label."""
    matrix, kept = label_profile_matrix(labels, profiles, feature_names)
    if matrix.size == 0 or len(kept) < 1:
        return None
    p10 = np.percentile(matrix, 10, axis=0)
    p25 = np.percentile(matrix, 25, axis=0)
    median = np.percentile(matrix, 50, axis=0)
    p75 = np.percentile(matrix, 75, axis=0)
    p90 = np.percentile(matrix, 90, axis=0)
    iqr = np.maximum(np.asarray(p75 - p25, dtype=np.float32), 0.035)
    return {
        "median": np.asarray(median, dtype=np.float32),
        "p10": np.asarray(p10, dtype=np.float32),
        "p90": np.asarray(p90, dtype=np.float32),
        "scale": iqr,
        "label_count": np.asarray([len(kept)], dtype=np.float32),
    }


def score_values_against_bucket(values: np.ndarray, bucket: dict[str, np.ndarray]) -> tuple[float, int, int]:
    """Score values against a learned parent bucket. Lower is better."""
    median = bucket["median"]
    p10 = bucket["p10"]
    p90 = bucket["p90"]
    scale = np.maximum(bucket["scale"], 0.035)
    usable = min(values.size, median.size, p10.size, p90.size, scale.size)
    if usable <= 0:
        return float("inf"), 0, 0
    v = values[:usable]
    m = median[:usable]
    lo = np.minimum(p10[:usable], p90[:usable])
    hi = np.maximum(p10[:usable], p90[:usable])
    sc = scale[:usable]
    center = np.abs(v - m) / sc
    below = np.where(v < lo, (lo - v) / sc, 0.0)
    above = np.where(v > hi, (v - hi) / sc, 0.0)
    outside = below + above
    range_term = np.where(outside > 0.0, 1.0 + outside, 0.0)
    residual = 0.35 * center + 0.65 * range_term
    inside = int(np.sum(outside <= 0.0))
    violations = int(usable - inside)
    return float(np.mean(residual)), inside, violations


def feature_values_array(facts: SharedAudioFacts, feature_names: Sequence[str]) -> np.ndarray:
    values = []
    for name in feature_names:
        try:
            values.append(float(facts.feature_values_by_name.get(str(name), 0.0)))
        except Exception:
            values.append(0.0)
    return np.asarray(values, dtype=np.float32)


def score_prefixes(
    *,
    sample_values: np.ndarray,
    labels: Sequence[str],
    profiles: dict[str, Any],
    feature_names: Sequence[str],
    depth: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = {}
    for label in labels:
        key = prefix(label, depth)
        if key:
            groups.setdefault(key, []).append(label)
    rows: list[dict[str, Any]] = []
    for key, group_labels in groups.items():
        bucket = parent_bucket_from_labels(group_labels, profiles, feature_names)
        if bucket is None:
            continue
        score, inside, violations = score_values_against_bucket(sample_values, bucket)
        rows.append(
            {
                "prefix": key,
                "score": round(float(score), 6),
                "inside": int(inside),
                "violations": int(violations),
                "label_count": int(bucket["label_count"][0]),
            }
        )
    rows.sort(key=lambda row: (float(row["score"]), str(row["prefix"])))
    return rows


def dynamic_role_gate(
    physics: AudioPhysics,
    facts: SharedAudioFacts,
    brain: dict[str, Any],
    *,
    decisive_gap: float = 0.85,
    soft_gap: float = 1.25,
    branch_soft_gap: float = 0.75,
    branch_max_count: int = 3,
    final_effects_enabled: bool = False,
) -> RoleGateDecision:
    """Derive diagnostic parent buckets from learned label profiles.

    The gate now defaults to diagnostic-only mode.  It reports the learned
    parent and branch neighborhoods, but it does not narrow final candidates
    unless ``final_effects_enabled`` is explicitly set by a future, proven-safe
    policy.
    """
    labels = [str(label) for label in brain.get("labels", []) if str(label)]
    profiles = (
        brain.get("category_fact_profiles", {}) if isinstance(brain.get("category_fact_profiles", {}), dict) else {}
    )
    if not labels or not profiles or not facts.feature_values_by_name:
        return RoleGateDecision(enabled=False, allowed_labels=tuple(labels), reason="missing_labels_or_profiles")

    feature_names = tuple(str(name) for name in FEATURE_NAMES[:FP_SIZE])
    sample_values = feature_values_array(facts, feature_names)
    top_rows = score_prefixes(
        sample_values=sample_values, labels=labels, profiles=profiles, feature_names=feature_names, depth=1
    )
    if not top_rows:
        return RoleGateDecision(enabled=False, allowed_labels=tuple(labels), reason="no_parent_scores")

    best = float(top_rows[0]["score"])
    second = float(top_rows[1]["score"]) if len(top_rows) > 1 else best + decisive_gap + 1.0
    gap = second - best
    if gap >= decisive_gap:
        selected_tops = [str(top_rows[0]["prefix"])]
        reason = f"decisive_top_family_gap={gap:.3f}"
    else:
        selected_tops = [str(row["prefix"]) for row in top_rows if float(row["score"]) <= best + soft_gap]
        if not selected_tops:
            selected_tops = [str(top_rows[0]["prefix"])]
        reason = f"soft_top_family_gate_gap={gap:.3f}"

    if not final_effects_enabled:
        return RoleGateDecision(
            enabled=False,
            selected_top_families=tuple(selected_tops),
            selected_branch_prefixes=(),
            allowed_labels=tuple(labels),
            top_scores=tuple(top_rows[:8]),
            branch_scores=tuple(
                score_prefixes(
                    sample_values=sample_values,
                    labels=[label for label in labels if top_family(label) in set(selected_tops)],
                    profiles=profiles,
                    feature_names=feature_names,
                    depth=2,
                )[:12]
            ),
            reason=reason + ";final_effects_disabled",
            final_effects_enabled=False,
        )

    # Conservative phase-1 gating: only make a hard top-family candidate cut
    # when the learned parent bucket decisively identifies Instruments.  The
    # current FX and Drums parent buckets are broad enough that narrowing on
    # them caused good drum loops to be swallowed by Long FX branches.  For
    # non-Instruments top-family decisions we keep the role-gate diagnostics
    # but leave all labels eligible so the two voters can still agree on broad
    # Drum Loops or other safer folders.  This is a gate-confidence rule, not a
    # category-specific score penalty.
    if not (gap >= decisive_gap and selected_tops == ["Instruments"]):
        return RoleGateDecision(
            enabled=False,
            selected_top_families=tuple(selected_tops),
            selected_branch_prefixes=(),
            allowed_labels=tuple(labels),
            top_scores=tuple(top_rows[:8]),
            branch_scores=(),
            reason=reason + ";diagnostic_only_non_instrument_or_soft_gate",
            final_effects_enabled=False,
        )

    # Branch-local identity lane.  When the top family is decisive, derive a
    # smaller set of second-level branches from the learned parent buckets.
    # This is still dynamic: the selected branches come from label-profile
    # distances, not from leaf-category rules or sample filenames.  Generic
    # broad loop folders are kept as fallbacks, but unrelated sibling branches
    # are removed before the voters rank leaf labels.
    branch_rows: list[dict[str, Any]] = []
    for top in selected_tops:
        top_labels = [label for label in labels if top_family(label) == top]
        branch_rows.extend(
            score_prefixes(
                sample_values=sample_values, labels=top_labels, profiles=profiles, feature_names=feature_names, depth=2
            )[:8]
        )
    branch_rows.sort(key=lambda row: (float(row["score"]), str(row["prefix"])))

    selected_branches: list[str] = []
    if len(selected_tops) == 1 and branch_rows:
        best_branch_score = float(branch_rows[0]["score"])
        for row in branch_rows:
            if len(selected_branches) >= max(1, int(branch_max_count)):
                break
            if float(row["score"]) <= best_branch_score + float(branch_soft_gap):
                selected_branches.append(str(row["prefix"]))
        if not selected_branches:
            selected_branches = [str(branch_rows[0]["prefix"])]

    generic_branch_names = {
        "Instrument Loops",
        "Drum Loops",
        "Mixed Musical Loops",
    }

    def is_generic_fallback_label(label: str) -> bool:
        parts = label_parts(label)
        return len(parts) >= 2 and parts[1] in generic_branch_names

    if selected_branches:
        selected_branch_set = set(selected_branches)
        allowed = [
            label
            for label in labels
            if top_family(label) in set(selected_tops)
            and (prefix(label, 2) in selected_branch_set or is_generic_fallback_label(label))
        ]
        reason = reason + f";branch_gate={','.join(selected_branches)}"
    else:
        allowed = [label for label in labels if top_family(label) in set(selected_tops)]

    return RoleGateDecision(
        enabled=True,
        selected_top_families=tuple(selected_tops),
        selected_branch_prefixes=tuple(selected_branches),
        allowed_labels=tuple(allowed),
        top_scores=tuple(top_rows[:8]),
        branch_scores=tuple(branch_rows[:12]),
        reason=reason,
        final_effects_enabled=True,
    )
