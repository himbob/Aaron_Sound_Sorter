"""Audit taxonomy label contracts across brains, training slots, and GUI catalog."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aaron_sound_sorter.taxonomy_contracts import taxonomy_label_contract

DEFAULT_REPORT_ROOT = Path("_reports/category_contract_audit")
DEFAULT_TRAINING_ROOT = Path("training/locked_curated_v1")
DEFAULT_GUI_CATALOG = Path("config/gui_taxonomy_catalog.json")
TRAINING_STRUCTURE_FOLDER_NAMES = {
    "_ONE_SHOTS": "One Shots",
    "_LOOPS": "Loops",
    "_LONG_FX": "Long FX",
}


@dataclass(frozen=True)
class CategoryContractRow:
    """One audited taxonomy label.

    Args:
        source_kind: Origin type such as ``brain`` or ``training_slot``.
        source_name: File or root path that supplied the label.
        raw_label: Original internal taxonomy label.
        canonical_label: Contract-normalized label.
        valid: Whether the label is valid under the current contract.
        reasons: Contract reasons joined by semicolon for CSV output.

    Side Effects:
        None.
    """

    source_kind: str
    source_name: str
    raw_label: str
    canonical_label: str
    valid: bool
    reasons: str


def main() -> int:
    """Run the category contract audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    report_dir = project_root / args.report_root / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    report_dir.mkdir(parents=True, exist_ok=False)
    rows = collect_category_contract_rows(project_root)
    duplicate_rows = duplicate_canonical_rows(rows)
    write_contract_rows(report_dir / "category_contract_audit.csv", rows)
    write_duplicate_rows(report_dir / "category_contract_duplicates.csv", duplicate_rows)
    write_summary(report_dir / "README_CATEGORY_CONTRACT_AUDIT.txt", rows, duplicate_rows)
    print(f"Audited labels: {len(rows)}")
    print(f"Invalid labels: {sum(1 for row in rows if not row.valid)}")
    print(f"Canonicalized labels: {sum(1 for row in rows if row.raw_label != row.canonical_label)}")
    print(f"Duplicate canonical groups: {len(duplicate_rows)}")
    print(f"Report folder: {report_dir}")
    return 0


def collect_category_contract_rows(project_root: Path) -> list[CategoryContractRow]:
    """Collect labels from active local taxonomy sources."""
    rows: list[CategoryContractRow] = []
    for brain_path in sorted(project_root.glob("stage4*brain*.json")):
        if not is_folder_taxonomy_brain(brain_path):
            continue
        rows.extend(rows_from_brain(brain_path))
    rows.extend(rows_from_training_slots(project_root / DEFAULT_TRAINING_ROOT))
    rows.extend(rows_from_json_catalog(project_root / DEFAULT_GUI_CATALOG))
    return rows


def is_folder_taxonomy_brain(brain_path: Path) -> bool:
    """Return whether a brain stores final folder taxonomy labels.

    Args:
        brain_path: Active brain JSON path.

    Returns:
        True for folder and folder-user-memory brains. False for role, shape,
        and physics memory brains because those labels are intentionally compact
        internal targets such as ``transition_riser`` or ``Drums/Kick``.

    Side Effects:
        None.
    """
    return brain_path.name.startswith("stage4_folder_brain")


def rows_from_brain(brain_path: Path) -> list[CategoryContractRow]:
    """Return contract rows from one brain JSON file."""
    if not brain_path.is_file():
        return []
    try:
        payload = json.loads(brain_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    labels = payload.get("labels", [])
    if not isinstance(labels, list):
        return []
    return [contract_row("brain", brain_path.name, str(label)) for label in labels if str(label)]


def rows_from_training_slots(training_root: Path) -> list[CategoryContractRow]:
    """Return contract rows from curated training slot folders."""
    if not training_root.is_dir():
        return []
    rows: list[CategoryContractRow] = []
    for slot_dir in sorted(training_root.rglob("*")):
        if not slot_dir.is_dir():
            continue
        structure = TRAINING_STRUCTURE_FOLDER_NAMES.get(slot_dir.name)
        if structure is None:
            continue
        label = "/".join([*slot_dir.parent.relative_to(training_root).parts, structure])
        rows.append(contract_row("training_slot", str(slot_dir.relative_to(training_root)), label))
    return rows


def rows_from_json_catalog(catalog_path: Path) -> list[CategoryContractRow]:
    """Return contract rows from the GUI taxonomy catalog."""
    if not catalog_path.is_file():
        return []
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [contract_row("gui_catalog", catalog_path.name, label) for label in discover_label_strings(payload)]


def discover_label_strings(payload: Any) -> list[str]:
    """Return taxonomy-shaped strings from nested JSON data."""
    if isinstance(payload, str):
        return [payload] if "/" in payload else []
    if isinstance(payload, list):
        labels: list[str] = []
        for value in payload:
            labels.extend(discover_label_strings(value))
        return labels
    if isinstance(payload, dict):
        labels: list[str] = []
        for value in payload.values():
            labels.extend(discover_label_strings(value))
        return labels
    return []


def contract_row(source_kind: str, source_name: str, raw_label: str) -> CategoryContractRow:
    """Build one CSV-ready contract row."""
    contract = taxonomy_label_contract(raw_label)
    return CategoryContractRow(
        source_kind=source_kind,
        source_name=source_name,
        raw_label=contract.normalized_label,
        canonical_label=contract.canonical_label,
        valid=contract.valid,
        reasons=";".join(contract.reasons),
    )


def duplicate_canonical_rows(rows: list[CategoryContractRow]) -> list[dict[str, str]]:
    """Return canonical labels with multiple raw spellings."""
    by_canonical: dict[str, set[str]] = {}
    for row in rows:
        if not row.valid:
            continue
        by_canonical.setdefault(row.canonical_label, set()).add(row.raw_label)
    duplicates = []
    for canonical_label, raw_labels in sorted(by_canonical.items()):
        if len(raw_labels) <= 1:
            continue
        duplicates.append(
            {
                "canonical_label": canonical_label,
                "raw_label_count": str(len(raw_labels)),
                "raw_labels": " || ".join(sorted(raw_labels)),
            }
        )
    return duplicates


def write_contract_rows(path: Path, rows: list[CategoryContractRow]) -> None:
    """Write detailed contract rows."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("source_kind", "source_name", "raw_label", "canonical_label", "valid", "reasons"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source_kind": row.source_kind,
                    "source_name": row.source_name,
                    "raw_label": row.raw_label,
                    "canonical_label": row.canonical_label,
                    "valid": str(row.valid),
                    "reasons": row.reasons,
                }
            )


def write_duplicate_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """Write duplicate canonical groups."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("canonical_label", "raw_label_count", "raw_labels"))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[CategoryContractRow], duplicate_rows: list[dict[str, str]]) -> None:
    """Write a compact human-readable audit summary."""
    invalid_rows = [row for row in rows if not row.valid]
    canonicalized_rows = [row for row in rows if row.raw_label != row.canonical_label]
    lines = [
        "Category contract audit",
        f"Audited labels: {len(rows)}",
        f"Invalid labels: {len(invalid_rows)}",
        f"Canonicalized labels: {len(canonicalized_rows)}",
        f"Duplicate canonical groups: {len(duplicate_rows)}",
        "",
        "This audit uses internal taxonomy labels only. It does not inspect source filenames as sorting evidence.",
    ]
    if invalid_rows:
        lines.extend(["", "Invalid examples:"])
        for row in invalid_rows[:20]:
            lines.append(f"- {row.raw_label} [{row.reasons}]")
    if canonicalized_rows:
        lines.extend(["", "Canonicalized examples:"])
        for row in canonicalized_rows[:20]:
            lines.append(f"- {row.raw_label} -> {row.canonical_label} [{row.reasons}]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
