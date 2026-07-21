#!/usr/bin/env python3
"""Audit whether sort decisions are owned by brains or static contracts.

This report is intentionally source-name blind. File paths in manifests are
used only as row identifiers for audit output. Classification ownership is
derived from internal sorter diagnostics such as final claim source, learned
memory diagnostics, brain agreement columns, and final labels.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path("/Volumes/T9/testbed/Aaron_Sound_Sorter")
DEFAULT_REPORT_ROOT = Path("_reports/decision_ownership_audit")

LEARNED_OWNER_SOURCES = {
    "learned_owner_body_claim",
    "learned_owner_voice_body_claim",
}

STATIC_GUARD_SOURCE_PREFIXES = (
    "final_measured_",
    "raw_contract_",
    "profile_candidate_",
    "baby_recall_",
    "final_clean_",
    "final_false_",
    "final_conflicted_",
    "instrument_sibling_",
    "shape_sanity_",
    "role_sanity_",
    "concrete_fx_",
)

BROAD_FALLBACK_MARKERS = (
    "Instruments/Instrument Loops/Loops",
    "Instruments/Mixed Musical Loops",
    "Drums/Drum Loops/Loops",
    "Drums/Percussion/Generic Percussion",
    "FX/Hybrid Designed FX",
    "_TO_REVIEW",
)


@dataclass(frozen=True)
class DecisionOwnership:
    """Brain/static ownership classification for one manifest row.

    Args:
        owner_type: High-level owner bucket for the final decision.
        owner_detail: More specific source or explanation.
        brain_agrees: Whether any trained brain lane agrees with final output.
        learned_memory_agrees: Whether learned memory owns the final output.
        learned_memory_matched: Whether any learned memory lane matched.
        learned_memory_blocked: Whether memory matched but did not own output.
        static_over_brain: Whether a static contract won over brain agreement.
        broad_fallback: Whether the final output is a broad fallback/review.

    Side Effects:
        None.
    """

    owner_type: str
    owner_detail: str
    brain_agrees: bool
    learned_memory_agrees: bool
    learned_memory_matched: bool
    learned_memory_blocked: bool
    static_over_brain: bool
    broad_fallback: bool


def normalize_path(value: object) -> str:
    """Return a normalized internal taxonomy path string.

    Args:
        value: Loose manifest value.

    Returns:
        Slash-normalized string without leading or trailing slashes.

    Side Effects:
        None.
    """
    return str(value or "").replace("\\", "/").strip("/")


def truthy(value: object) -> bool:
    """Return whether a manifest field represents true."""
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "matched"}


def same_or_child_path(path: str, owner_path: str) -> bool:
    """Return whether ``path`` equals or lives under ``owner_path``."""
    normalized_path = normalize_path(path).lower()
    normalized_owner = normalize_path(owner_path).lower()
    if not normalized_path or not normalized_owner:
        return False
    return normalized_path == normalized_owner or normalized_path.startswith(f"{normalized_owner}/")


def row_final_path(row: Mapping[str, str]) -> str:
    """Return the final folder path from a manifest row."""
    for column in ("folder_path", "final_label", "approved_folder", "proposed_folder", "placed_path"):
        value = normalize_path(row.get(column, ""))
        if value:
            return value
    return ""


def diagnostic_summary_value(row: Mapping[str, str], key: str) -> str:
    """Return a compact GUI diagnostic summary value.

    Args:
        row: Manifest or GUI preview row.
        key: Key such as ``brain`` or ``physics``.

    Returns:
        Parsed value, or an empty string when absent.

    Side Effects:
        None.
    """
    summary = str(row.get("diagnostic_summary") or "")
    marker = f"{key}="
    start = summary.find(marker)
    if start < 0:
        return ""
    value_start = start + len(marker)
    value_end = summary.find(";", value_start)
    if value_end < 0:
        value_end = len(summary)
    return normalize_path(summary[value_start:value_end])


def row_claim_source(row: Mapping[str, str]) -> str:
    """Return the most specific available final claim source."""
    return str(row.get("final_claim_source") or row.get("consensus_status") or "").strip()


def row_brain_agrees(row: Mapping[str, str]) -> bool:
    """Return true when any old trained brain lane agrees with final output."""
    agreement_columns = (
        "final_agrees_with_brain_ensemble",
        "final_agrees_with_full_brain",
        "final_agrees_with_core_baby",
        "final_agrees_with_spread_baby",
        "final_agrees_with_outlier_baby",
    )
    if any(truthy(row.get(column, "")) for column in agreement_columns):
        return True
    final_path = row_final_path(row)
    brain_columns = (
        "brain_ensemble_vote_1",
        "brain_vote_1",
        "full_brain_vote_1",
        "core_baby_vote_1",
        "spread_baby_vote_1",
        "outlier_baby_vote_1",
    )
    if any(same_or_child_path(final_path, str(row.get(column, ""))) for column in brain_columns):
        return True
    return same_or_child_path(final_path, diagnostic_summary_value(row, "brain"))


def row_learned_memory_labels(row: Mapping[str, str]) -> list[str]:
    """Return learned-memory label paths from a manifest row."""
    labels: list[str] = []
    for column in ("learned_voter_memory_label", "learned_physics_memory_label"):
        label = normalize_path(row.get(column, ""))
        if label:
            labels.append(label)
    return labels


def row_learned_memory_matched(row: Mapping[str, str]) -> bool:
    """Return true when any dedicated memory lane matched this audio."""
    return truthy(row.get("learned_voter_memory_matched", "")) or truthy(row.get("learned_physics_memory_matched", ""))


def row_learned_memory_agrees(row: Mapping[str, str]) -> bool:
    """Return true when learned memory owns the final output path."""
    final_path = row_final_path(row)
    return any(
        same_or_child_path(final_path, label) or same_or_child_path(label, final_path)
        for label in row_learned_memory_labels(row)
    )


def row_is_broad_fallback(row: Mapping[str, str]) -> bool:
    """Return true when final output is a broad fallback or review bucket."""
    final_path = row_final_path(row)
    return any(
        same_or_child_path(final_path, marker) or same_or_child_path(marker, final_path)
        for marker in BROAD_FALLBACK_MARKERS
    )


def classify_decision_ownership(row: Mapping[str, str]) -> DecisionOwnership:
    """Classify whether one final decision was owned by brains or static code.

    Args:
        row: One sorter manifest row.

    Returns:
        Ownership classification with audit flags.

    Side Effects:
        None.
    """
    final_path = row_final_path(row)
    source = row_claim_source(row)
    status = str(row.get("consensus_status") or "").strip()
    brain_agrees = row_brain_agrees(row)
    memory_matched = row_learned_memory_matched(row)
    memory_agrees = row_learned_memory_agrees(row)
    broad_fallback = row_is_broad_fallback(row)
    source_key = source or status

    if final_path.lower().startswith("_to_review"):
        owner_type = "review"
        owner_detail = source_key or "review_folder"
    elif source in LEARNED_OWNER_SOURCES or memory_agrees:
        owner_type = "learned_memory"
        owner_detail = source or "learned_memory_label_agreement"
    elif brain_agrees and source_key in {"strong_consensus", "raw_shared_winner", ""}:
        owner_type = "trained_brain_consensus"
        owner_detail = source_key or "brain_lane_agreement"
    elif source_key.startswith(STATIC_GUARD_SOURCE_PREFIXES):
        owner_type = "static_measured_contract"
        owner_detail = source_key
    elif brain_agrees:
        owner_type = "trained_brain_assisted"
        owner_detail = source_key or "brain_agreement_with_other_claim"
    else:
        owner_type = "unowned_or_legacy"
        owner_detail = source_key or "no_claim_source"

    learned_memory_blocked = bool(memory_matched and not memory_agrees and owner_type != "learned_memory")
    static_over_brain = bool(owner_type == "static_measured_contract" and brain_agrees)
    return DecisionOwnership(
        owner_type=owner_type,
        owner_detail=owner_detail,
        brain_agrees=brain_agrees,
        learned_memory_agrees=memory_agrees,
        learned_memory_matched=memory_matched,
        learned_memory_blocked=learned_memory_blocked,
        static_over_brain=static_over_brain,
        broad_fallback=broad_fallback,
    )


def audit_row(row: Mapping[str, str], manifest_path: Path) -> dict[str, str]:
    """Return one CSV-ready ownership audit row."""
    ownership = classify_decision_ownership(row)
    return {
        "manifest_path": str(manifest_path),
        "source_path": str(row.get("source_path", "")),
        "final_label": row_final_path(row),
        "final_top": str(row.get("final_top", "")),
        "consensus_status": str(row.get("consensus_status", "")),
        "final_claim_source": str(row.get("final_claim_source", "")),
        "decision_owner_type": ownership.owner_type,
        "decision_owner_detail": ownership.owner_detail,
        "brain_agrees": str(ownership.brain_agrees),
        "learned_memory_matched": str(ownership.learned_memory_matched),
        "learned_memory_agrees": str(ownership.learned_memory_agrees),
        "learned_memory_blocked": str(ownership.learned_memory_blocked),
        "static_over_brain": str(ownership.static_over_brain),
        "broad_fallback": str(ownership.broad_fallback),
        "brain_ensemble_vote_1": str(row.get("brain_ensemble_vote_1", "")),
        "physics_vote_1": str(row.get("physics_vote_1", "") or diagnostic_summary_value(row, "physics")),
        "shape_vote": str(row.get("shape_vote", "") or diagnostic_summary_value(row, "shape")),
        "learned_voter_memory_label": str(row.get("learned_voter_memory_label", "")),
        "learned_physics_memory_label": str(row.get("learned_physics_memory_label", "")),
    }


def read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    """Read a sorter manifest as a list of dictionaries."""
    with manifest_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Iterable[Mapping[str, str]]) -> None:
    """Write CSV rows, preserving the first row's field order."""
    materialized = list(rows)
    fieldnames = list(materialized[0].keys()) if materialized else ["message"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def summary_lines(rows: list[Mapping[str, str]]) -> list[str]:
    """Return human-readable ownership summary lines."""
    owner_counts = Counter(row["decision_owner_type"] for row in rows)
    broad_count = sum(truthy(row["broad_fallback"]) for row in rows)
    memory_matched = sum(truthy(row["learned_memory_matched"]) for row in rows)
    memory_agrees = sum(truthy(row["learned_memory_agrees"]) for row in rows)
    memory_blocked = sum(truthy(row["learned_memory_blocked"]) for row in rows)
    static_over_brain = sum(truthy(row["static_over_brain"]) for row in rows)
    lines = [
        "Decision Ownership Audit",
        f"Rows audited: {len(rows)}",
        "",
        "Owner counts:",
    ]
    for owner_type, count in sorted(owner_counts.items()):
        lines.append(f"  {owner_type}: {count}")
    lines.extend(
        [
            "",
            f"Learned memory matched: {memory_matched}",
            f"Learned memory owned final label: {memory_agrees}",
            f"Learned memory matched but did not own final label: {memory_blocked}",
            f"Static measured contracts over brain agreement: {static_over_brain}",
            f"Broad fallback or review placements: {broad_count}",
            "",
            "Interpretation:",
            "  learned_memory means the new memory brains owned the destination.",
            "  trained_brain_* means the older full/baby brain lanes agreed with the final destination.",
            "  static_measured_contract means Python guardrails or measured contracts owned the destination.",
            "  learned_memory_blocked=True is the first place to inspect when GUI training seems ignored.",
        ]
    )
    return lines


def default_output_dir(project_root: Path) -> Path:
    """Return a timestamped default report directory."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / DEFAULT_REPORT_ROOT / f"run_{stamp}"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path, help="Sorter manifest CSV files to audit.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help=f"Project root. Default: {PROJECT_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Report directory. Default: _reports/decision_ownership_audit/run_...",
    )
    return parser.parse_args()


def main() -> int:
    """Run the decision ownership audit command."""
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else default_output_dir(project_root)
    audit_rows: list[dict[str, str]] = []
    for manifest in args.manifests:
        manifest_path = manifest.expanduser()
        if not manifest_path.is_absolute():
            manifest_path = project_root / manifest_path
        for row in read_manifest_rows(manifest_path):
            audit_rows.append(audit_row(row, manifest_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "decision_ownership_audit.csv", audit_rows)
    summary = "\n".join(summary_lines(audit_rows)) + "\n"
    (output_dir / "decision_ownership_summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"Report folder: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
