"""Audit FX taxonomy, training, GUI, and memory-brain coverage.

The sorter has enough FX labels to look broad on paper, but broad labels are
not enough if the trusted training tree and trainable voter-memory lanes are
thin. This diagnostic reads local metadata only. It never trains, sorts audio,
or changes classifier behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_ONTOLOGY = Path("config/fx_role_ontology_v1.json")
DEFAULT_FULL_BRAIN = Path("stage4_folder_brain.json")
DEFAULT_GUI_CATALOG = Path("config/gui_taxonomy_catalog.json")
DEFAULT_TRAINING_ROOT = Path("training/locked_curated_v1")
DEFAULT_USER_MEMORY = Path("stage4_folder_brain_user_memory.json")
DEFAULT_VOTER_MEMORY = Path("stage4_voter_memory_brain.json")
DEFAULT_PHYSICS_MEMORY = Path("stage4_physics_memory_brain.json")
DEFAULT_REPORT_ROOT = Path("_reports/fx_taxonomy_audit")

STRUCTURE_TO_SLOT = {
    "Long FX": "_LONG_FX",
    "One Shots": "_ONE_SHOTS",
    "Loops": "_LOOPS",
}


@dataclass(frozen=True)
class FxRoleDefinition:
    """One FX role from the ontology.

    Args:
        role_id: Stable role identifier.
        display_name: Human-friendly role name for reports.
        label_prefixes: FX label prefixes belonging to the role.
        target_memory_roles: Voter-memory role keys expected for this role.
        target_physics_branches: Physics-memory branch keys expected for this
            role.
        training_goal_per_role: Minimum trusted audio target for the role.
    """

    role_id: str
    display_name: str
    label_prefixes: tuple[str, ...]
    target_memory_roles: tuple[str, ...]
    target_physics_branches: tuple[str, ...]
    training_goal_per_role: int


@dataclass(frozen=True)
class FxLabelCoverage:
    """Coverage information for one public FX label."""

    label: str
    role_id: str
    full_brain_count: int
    locked_training_audio_count: int
    visible_in_gui_catalog: bool
    present_in_user_memory: bool
    recommendation: str


@dataclass(frozen=True)
class FxRoleCoverage:
    """Aggregated coverage information for one FX role."""

    role_id: str
    display_name: str
    brain_label_count: int
    full_brain_audio_count: int
    locked_training_audio_count: int
    gui_visible_label_count: int
    user_memory_label_count: int
    voter_memory_target_count: int
    physics_memory_target_count: int
    training_goal: int
    recommendation: str


@dataclass(frozen=True)
class FxAuditSnapshot:
    """Loaded local FX state used by the audit."""

    roles: tuple[FxRoleDefinition, ...]
    brain_counts: dict[str, int]
    gui_labels: set[str]
    locked_training_counts: dict[str, int]
    user_memory_labels: set[str]
    voter_memory_ids: set[str]
    physics_memory_ids: set[str]


def main() -> int:
    """Run the FX taxonomy coverage audit."""
    parser = argparse.ArgumentParser(description="Audit FX taxonomy and memory-brain coverage.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--full-brain", type=Path, default=DEFAULT_FULL_BRAIN)
    parser.add_argument("--gui-catalog", type=Path, default=DEFAULT_GUI_CATALOG)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--user-memory", type=Path, default=DEFAULT_USER_MEMORY)
    parser.add_argument("--voter-memory", type=Path, default=DEFAULT_VOTER_MEMORY)
    parser.add_argument("--physics-memory", type=Path, default=DEFAULT_PHYSICS_MEMORY)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    report_dir = make_report_dir(project_root / args.report_root)
    snapshot = load_audit_snapshot(
        project_root=project_root,
        ontology_path=args.ontology,
        full_brain_path=args.full_brain,
        gui_catalog_path=args.gui_catalog,
        training_root=args.training_root,
        user_memory_path=args.user_memory,
        voter_memory_path=args.voter_memory,
        physics_memory_path=args.physics_memory,
    )
    label_coverage = build_label_coverage(snapshot)
    role_coverage = build_role_coverage(snapshot, label_coverage)
    write_label_coverage(report_dir / "fx_label_coverage.csv", label_coverage)
    write_role_coverage(report_dir / "fx_role_coverage.csv", role_coverage)
    write_summary(report_dir / "README_FX_TAXONOMY_AUDIT.txt", snapshot, label_coverage, role_coverage)
    print(f"FX labels in full brain: {len(snapshot.brain_counts)}")
    print(f"FX labels in GUI catalog: {len(snapshot.gui_labels)}")
    print(
        "FX full-brain labels visible in GUI catalog: "
        f"{sum(1 for label in snapshot.brain_counts if label in snapshot.gui_labels)}"
    )
    print(
        f"FX labels with locked training audio: {sum(1 for count in snapshot.locked_training_counts.values() if count > 0)}"
    )
    print(f"Report folder: {report_dir}")
    return 0


def make_report_dir(report_root: Path) -> Path:
    """Create and return a timestamped report directory."""
    run_dir = report_root / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_audit_snapshot(
    *,
    project_root: Path,
    ontology_path: Path,
    full_brain_path: Path,
    gui_catalog_path: Path,
    training_root: Path,
    user_memory_path: Path,
    voter_memory_path: Path,
    physics_memory_path: Path,
) -> FxAuditSnapshot:
    """Load all local metadata used by the FX coverage audit."""
    resolved_ontology = resolve_project_path(project_root, ontology_path)
    resolved_brain = resolve_project_path(project_root, full_brain_path)
    resolved_gui_catalog = resolve_project_path(project_root, gui_catalog_path)
    resolved_training_root = resolve_project_path(project_root, training_root)
    resolved_user_memory = resolve_project_path(project_root, user_memory_path)
    resolved_voter_memory = resolve_project_path(project_root, voter_memory_path)
    resolved_physics_memory = resolve_project_path(project_root, physics_memory_path)
    brain_counts = read_fx_brain_counts(resolved_brain)
    return FxAuditSnapshot(
        roles=load_role_definitions(resolved_ontology),
        brain_counts=brain_counts,
        gui_labels=read_fx_label_set(resolved_gui_catalog),
        locked_training_counts=count_locked_training_audio(resolved_training_root, brain_counts),
        user_memory_labels=read_fx_label_set(resolved_user_memory),
        voter_memory_ids=read_memory_identifiers(resolved_voter_memory),
        physics_memory_ids=read_memory_identifiers(resolved_physics_memory),
    )


def resolve_project_path(project_root: Path, raw_path: Path) -> Path:
    """Resolve ``raw_path`` against ``project_root`` when relative."""
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else project_root / path


def load_role_definitions(ontology_path: Path) -> tuple[FxRoleDefinition, ...]:
    """Load FX role definitions from the ontology JSON file."""
    payload = json.loads(ontology_path.read_text(encoding="utf-8"))
    raw_roles = payload.get("roles", [])
    roles: list[FxRoleDefinition] = []
    for raw_role in raw_roles:
        if not isinstance(raw_role, dict):
            continue
        roles.append(
            FxRoleDefinition(
                role_id=str(raw_role.get("role_id", "")).strip(),
                display_name=str(raw_role.get("display_name", "")).strip(),
                label_prefixes=tuple(str(value) for value in raw_role.get("label_prefixes", []) if str(value)),
                target_memory_roles=tuple(
                    str(value) for value in raw_role.get("target_memory_roles", []) if str(value)
                ),
                target_physics_branches=tuple(
                    str(value) for value in raw_role.get("target_physics_branches", []) if str(value)
                ),
                training_goal_per_role=safe_int(raw_role.get("training_goal_per_role"), 10),
            )
        )
    return tuple(role for role in roles if role.role_id and role.label_prefixes)


def read_fx_brain_counts(brain_path: Path) -> dict[str, int]:
    """Read public FX labels and support counts from a brain JSON file."""
    if not brain_path.exists():
        return {}
    payload = json.loads(brain_path.read_text(encoding="utf-8"))
    labels = [str(label) for label in payload.get("labels", []) if str(label).startswith("FX/")]
    counts = payload.get("counts", {})
    if not isinstance(counts, dict):
        counts = {}
    return {label: safe_int(counts.get(label), 0) for label in sorted(labels)}


def read_fx_label_set(json_path: Path) -> set[str]:
    """Read public FX labels from a generic JSON brain/catalog file."""
    if not json_path.exists():
        return set()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    labels: set[str] = set()
    collect_label_values(payload, labels)
    return {label for label in labels if label.startswith("FX/")}


def collect_label_values(payload: Any, labels: set[str]) -> None:
    """Collect likely public labels from nested JSON data."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"label", "training_label", "approved_label"} and isinstance(value, str):
                labels.add(value)
            elif key in {"labels", "label_names"} and isinstance(value, list):
                labels.update(str(label) for label in value if isinstance(label, str))
            else:
                collect_label_values(value, labels)
    elif isinstance(payload, list):
        for value in payload:
            collect_label_values(value, labels)


def read_memory_identifiers(memory_path: Path) -> set[str]:
    """Read role, target, branch, and label identifiers from a memory brain."""
    if not memory_path.exists():
        return set()
    payload = json.loads(memory_path.read_text(encoding="utf-8"))
    identifiers: set[str] = set()
    collect_memory_identifiers(payload, identifiers)
    return {identifier for identifier in identifiers if identifier.startswith("FX/") or identifier.startswith("fx_")}


def collect_memory_identifiers(payload: Any, identifiers: set[str]) -> None:
    """Collect memory identifiers from nested JSON data."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"role", "target_key", "branch", "label"} and isinstance(value, str):
                identifiers.add(value)
            elif key in {"labels", "examples_by_role", "examples_by_target", "target_metadata"}:
                add_key_or_list_identifiers(value, identifiers)
            else:
                collect_memory_identifiers(value, identifiers)
    elif isinstance(payload, list):
        for value in payload:
            collect_memory_identifiers(value, identifiers)


def add_key_or_list_identifiers(payload: Any, identifiers: set[str]) -> None:
    """Collect identifiers from known memory maps and lists."""
    if isinstance(payload, dict):
        identifiers.update(str(key) for key in payload)
        for value in payload.values():
            collect_memory_identifiers(value, identifiers)
    elif isinstance(payload, list):
        for value in payload:
            if isinstance(value, str):
                identifiers.add(value)
            else:
                collect_memory_identifiers(value, identifiers)


def count_locked_training_audio(training_root: Path, brain_counts: dict[str, int]) -> dict[str, int]:
    """Count trusted training audio references for each public FX brain label."""
    counts: dict[str, int] = {}
    for label in brain_counts:
        slot_path = training_slot_for_label(training_root, label)
        counts[label] = count_audio_references(slot_path)
    return counts


def training_slot_for_label(training_root: Path, label: str) -> Path:
    """Return the expected locked-training slot for a public label."""
    parts = [part for part in label.split("/") if part]
    if not parts:
        return training_root
    structure = parts[-1]
    slot_name = STRUCTURE_TO_SLOT.get(structure, f"_{structure.upper().replace(' ', '_')}")
    return training_root.joinpath(*parts[:-1], slot_name)


def count_audio_references(slot_path: Path) -> int:
    """Count audio files or symlinks in a locked-training slot."""
    if not slot_path.exists() or not slot_path.is_dir():
        return 0
    return sum(1 for path in slot_path.iterdir() if path.is_file() or path.is_symlink())


def build_label_coverage(snapshot: FxAuditSnapshot) -> list[FxLabelCoverage]:
    """Build one coverage row per public FX brain label."""
    rows: list[FxLabelCoverage] = []
    for label, brain_count in sorted(snapshot.brain_counts.items()):
        locked_count = snapshot.locked_training_counts.get(label, 0)
        visible = label in snapshot.gui_labels
        in_user_memory = label in snapshot.user_memory_labels
        rows.append(
            FxLabelCoverage(
                label=label,
                role_id=role_for_label(label, snapshot.roles),
                full_brain_count=brain_count,
                locked_training_audio_count=locked_count,
                visible_in_gui_catalog=visible,
                present_in_user_memory=in_user_memory,
                recommendation=label_recommendation(locked_count, visible, in_user_memory),
            )
        )
    return rows


def role_for_label(label: str, roles: tuple[FxRoleDefinition, ...]) -> str:
    """Return the ontology role id for an FX label."""
    for role in roles:
        if any(label.startswith(prefix) for prefix in role.label_prefixes):
            return role.role_id
    return "fx_unmapped"


def label_recommendation(locked_count: int, visible: bool, in_user_memory: bool) -> str:
    """Return the highest-priority label-level recommendation."""
    if locked_count == 0:
        return "seed_locked_training"
    if not visible:
        return "add_to_gui_catalog"
    if not in_user_memory:
        return "optional_memory_seed"
    return "covered"


def build_role_coverage(
    snapshot: FxAuditSnapshot,
    label_coverage: list[FxLabelCoverage],
) -> list[FxRoleCoverage]:
    """Aggregate label rows into FX role coverage rows."""
    role_rows: list[FxRoleCoverage] = []
    for role in snapshot.roles:
        labels = [row for row in label_coverage if row.role_id == role.role_id]
        voter_hits = sum(1 for target in role.target_memory_roles if target in snapshot.voter_memory_ids)
        physics_hits = sum(1 for target in role.target_physics_branches if target in snapshot.physics_memory_ids)
        locked_training_total = sum(row.locked_training_audio_count for row in labels)
        role_rows.append(
            FxRoleCoverage(
                role_id=role.role_id,
                display_name=role.display_name,
                brain_label_count=len(labels),
                full_brain_audio_count=sum(row.full_brain_count for row in labels),
                locked_training_audio_count=locked_training_total,
                gui_visible_label_count=sum(1 for row in labels if row.visible_in_gui_catalog),
                user_memory_label_count=sum(1 for row in labels if row.present_in_user_memory),
                voter_memory_target_count=voter_hits,
                physics_memory_target_count=physics_hits,
                training_goal=role.training_goal_per_role,
                recommendation=role_recommendation(
                    labels,
                    locked_training_total,
                    role.training_goal_per_role,
                    voter_hits,
                    physics_hits,
                ),
            )
        )
    unmapped_labels = [row for row in label_coverage if row.role_id == "fx_unmapped"]
    if unmapped_labels:
        role_rows.append(unmapped_role_row(unmapped_labels))
    return role_rows


def role_recommendation(
    labels: list[FxLabelCoverage],
    locked_training_total: int,
    training_goal: int,
    voter_hits: int,
    physics_hits: int,
) -> str:
    """Return the highest-priority role-level recommendation."""
    if not labels:
        return "missing_from_brain"
    if locked_training_total < training_goal:
        return "seed_trusted_fx_role_panel"
    if voter_hits == 0 or physics_hits == 0:
        return "seed_memory_brains"
    if any(not row.visible_in_gui_catalog for row in labels):
        return "refresh_gui_taxonomy_catalog"
    return "covered"


def unmapped_role_row(unmapped_labels: list[FxLabelCoverage]) -> FxRoleCoverage:
    """Return an aggregate row for labels not covered by the ontology."""
    return FxRoleCoverage(
        role_id="fx_unmapped",
        display_name="Unmapped FX Labels",
        brain_label_count=len(unmapped_labels),
        full_brain_audio_count=sum(row.full_brain_count for row in unmapped_labels),
        locked_training_audio_count=sum(row.locked_training_audio_count for row in unmapped_labels),
        gui_visible_label_count=sum(1 for row in unmapped_labels if row.visible_in_gui_catalog),
        user_memory_label_count=sum(1 for row in unmapped_labels if row.present_in_user_memory),
        voter_memory_target_count=0,
        physics_memory_target_count=0,
        training_goal=0,
        recommendation="extend_fx_role_ontology",
    )


def write_label_coverage(path: Path, rows: list[FxLabelCoverage]) -> None:
    """Write label-level coverage rows."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "role_id",
                "full_brain_count",
                "locked_training_audio_count",
                "visible_in_gui_catalog",
                "present_in_user_memory",
                "recommendation",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.label,
                    row.role_id,
                    row.full_brain_count,
                    row.locked_training_audio_count,
                    row.visible_in_gui_catalog,
                    row.present_in_user_memory,
                    row.recommendation,
                ]
            )


def write_role_coverage(path: Path, rows: list[FxRoleCoverage]) -> None:
    """Write role-level coverage rows."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "role_id",
                "display_name",
                "brain_label_count",
                "full_brain_audio_count",
                "locked_training_audio_count",
                "gui_visible_label_count",
                "user_memory_label_count",
                "voter_memory_target_count",
                "physics_memory_target_count",
                "training_goal",
                "recommendation",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.role_id,
                    row.display_name,
                    row.brain_label_count,
                    row.full_brain_audio_count,
                    row.locked_training_audio_count,
                    row.gui_visible_label_count,
                    row.user_memory_label_count,
                    row.voter_memory_target_count,
                    row.physics_memory_target_count,
                    row.training_goal,
                    row.recommendation,
                ]
            )


def write_summary(
    path: Path,
    snapshot: FxAuditSnapshot,
    label_rows: list[FxLabelCoverage],
    role_rows: list[FxRoleCoverage],
) -> None:
    """Write a concise human-readable audit summary."""
    labels_with_training = sum(1 for row in label_rows if row.locked_training_audio_count > 0)
    labels_visible_in_gui = sum(1 for row in label_rows if row.visible_in_gui_catalog)
    empty_training = len(label_rows) - labels_with_training
    gui_catalog_only = sorted(label for label in snapshot.gui_labels if label not in snapshot.brain_counts)
    role_lines = [
        f"  {row.role_id}: {row.recommendation} "
        f"(training={row.locked_training_audio_count}/{row.training_goal}, "
        f"gui={row.gui_visible_label_count}, voter_memory={row.voter_memory_target_count}, "
        f"physics_memory={row.physics_memory_target_count})"
        for row in role_rows
    ]
    lines = [
        "FX Taxonomy And Brain Coverage Audit",
        "",
        "This is diagnostic only. It does not sort audio, train brains, or change runtime behavior.",
        "",
        f"FX ontology roles: {len(snapshot.roles)}",
        f"FX labels in full brain: {len(label_rows)}",
        f"FX labels with locked curated training audio: {labels_with_training}",
        f"FX labels without locked curated training audio: {empty_training}",
        f"FX labels in GUI taxonomy catalog: {len(snapshot.gui_labels)}",
        f"FX full-brain labels visible in GUI taxonomy catalog: {labels_visible_in_gui}",
        f"FX GUI-only labels absent from full brain: {len(gui_catalog_only)}",
        f"Voter-memory FX identifiers: {len(snapshot.voter_memory_ids)}",
        f"Physics-memory FX identifiers: {len(snapshot.physics_memory_ids)}",
        "",
        "Role recommendations:",
        *role_lines,
        "",
        "Interpretation:",
        "  Use this report to seed trusted FX memory panels before retuning sorter logic.",
        "  Prioritize roles with seed_trusted_fx_role_panel or seed_memory_brains.",
        "  Refreshing the GUI catalog should expose safe destination choices; it should not narrow runtime taxonomy.",
    ]
    if gui_catalog_only:
        lines.extend(["", "FX GUI-only labels absent from full brain:"])
        lines.extend(f"  {label}" for label in gui_catalog_only[:30])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_int(value: Any, default: int) -> int:
    """Convert ``value`` to an int with a fallback."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())
