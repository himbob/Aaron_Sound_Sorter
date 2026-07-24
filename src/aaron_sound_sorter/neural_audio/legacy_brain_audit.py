"""Reversible contamination inventory for legacy folder and memory brains."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from .hashing import sha256_file


@dataclass(frozen=True)
class LegacyBrainTrainerRow:
    """One source trainer referenced by a legacy brain lane."""

    brain_path: Path
    lane: str
    section: str
    owner_target: str
    approved_label: str
    source_path: Path | None
    source_exists: bool
    file_sha256: str
    fingerprint_sha256: str
    identity_kind: str
    identity_sha256: str
    assigned_label_count: int = 0
    owner_target_count: int = 0
    status: str = "pending"
    reasons: tuple[str, ...] = ()


def _fingerprint_sha256(raw_fingerprint: object) -> str:
    if not isinstance(raw_fingerprint, list) or not raw_fingerprint:
        return ""
    canonical = json.dumps(raw_fingerprint, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source_path(raw_path: object, brain_path: Path) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = brain_path.parent / candidate
    return candidate


def _make_row(
    *,
    brain_path: Path,
    lane: str,
    section: str,
    owner_target: str,
    approved_label: str,
    raw_path: object,
    raw_fingerprint: object,
) -> LegacyBrainTrainerRow:
    source_path = _source_path(raw_path, brain_path)
    source_exists = bool(source_path and source_path.is_file())
    file_digest = sha256_file(source_path) if source_exists and source_path else ""
    fingerprint_digest = _fingerprint_sha256(raw_fingerprint)
    if file_digest:
        identity_kind, identity_digest = "file_sha256", file_digest
    elif fingerprint_digest:
        identity_kind, identity_digest = "fingerprint_sha256", fingerprint_digest
    else:
        identity_kind, identity_digest = "unresolved", ""
    return LegacyBrainTrainerRow(
        brain_path=brain_path,
        lane=lane,
        section=section,
        owner_target=owner_target,
        approved_label=approved_label,
        source_path=source_path,
        source_exists=source_exists,
        file_sha256=file_digest,
        fingerprint_sha256=fingerprint_digest,
        identity_kind=identity_kind,
        identity_sha256=identity_digest,
    )


def _folder_rows(brain_path: Path, payload: dict[str, object], lane: str) -> list[LegacyBrainTrainerRow]:
    rows: list[LegacyBrainTrainerRow] = []
    detailed = payload.get("training_examples_detailed_by_label")
    if isinstance(detailed, dict):
        for label, examples in sorted(detailed.items()):
            if not isinstance(label, str) or not isinstance(examples, list):
                continue
            for example in examples:
                if not isinstance(example, dict):
                    continue
                rows.append(
                    _make_row(
                        brain_path=brain_path,
                        lane=lane,
                        section="folder.training_examples_detailed_by_label",
                        owner_target=label,
                        approved_label=label,
                        raw_path=example.get("source_path"),
                        raw_fingerprint=example.get("fingerprint"),
                    )
                )
        return rows

    examples_by_label = payload.get("examples_by_label")
    if not isinstance(examples_by_label, dict):
        return rows
    for label, examples in sorted(examples_by_label.items()):
        if not isinstance(label, str) or not isinstance(examples, list):
            continue
        for raw_path in examples:
            rows.append(
                _make_row(
                    brain_path=brain_path,
                    lane=lane,
                    section="folder.examples_by_label",
                    owner_target=label,
                    approved_label=label,
                    raw_path=raw_path,
                    raw_fingerprint=None,
                )
            )
    return rows


def _memory_rows(
    *,
    brain_path: Path,
    lane: str,
    payload: dict[str, object],
    memory_key: str,
    examples_key: str,
    target_field: str,
) -> list[LegacyBrainTrainerRow]:
    memory = payload.get(memory_key)
    if not isinstance(memory, dict):
        return []
    examples_by_target = memory.get(examples_key)
    if not isinstance(examples_by_target, dict):
        return []
    rows: list[LegacyBrainTrainerRow] = []
    for dictionary_target, examples in sorted(examples_by_target.items()):
        if not isinstance(dictionary_target, str) or not isinstance(examples, list):
            continue
        for example in examples:
            if not isinstance(example, dict):
                continue
            owner_target = str(example.get(target_field) or dictionary_target)
            approved_label = str(example.get("approved_label") or "")
            rows.append(
                _make_row(
                    brain_path=brain_path,
                    lane=lane,
                    section=f"{memory_key}.{examples_key}",
                    owner_target=owner_target,
                    approved_label=approved_label,
                    raw_path=example.get("source_path"),
                    raw_fingerprint=example.get("fingerprint"),
                )
            )
    return rows


def audit_legacy_brain_trainers(brain_paths: Sequence[Path]) -> tuple[LegacyBrainTrainerRow, ...]:
    """Inventory trainer identity conflicts without modifying any brain.

    Args:
        brain_paths: Legacy folder or dedicated memory brain JSON files.

    Returns:
        Trainer occurrences annotated with missing-source, label-conflict, and
        same-owner-lane conflict evidence.

    Raises:
        ValueError: If no brain paths are supplied or a brain root is invalid.
        OSError: If a brain or referenced source cannot be read.
        json.JSONDecodeError: If a brain is malformed.
    """
    if not brain_paths:
        raise ValueError("at least one legacy brain path is required")

    raw_rows: list[LegacyBrainTrainerRow] = []
    for raw_brain_path in sorted(Path(path) for path in brain_paths):
        brain_path = raw_brain_path.resolve()
        payload = json.loads(brain_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"legacy brain root must be an object: {brain_path}")
        lane = str(payload.get("brain_family_role") or payload.get("brain_type") or brain_path.stem)
        raw_rows.extend(_folder_rows(brain_path, payload, lane))
        raw_rows.extend(
            _memory_rows(
                brain_path=brain_path,
                lane=lane,
                payload=payload,
                memory_key="voter_memory",
                examples_key="examples_by_role",
                target_field="target_role",
            )
        )
        raw_rows.extend(
            _memory_rows(
                brain_path=brain_path,
                lane=lane,
                payload=payload,
                memory_key="physics_memory",
                examples_key="examples_by_target",
                target_field="target_key",
            )
        )
        raw_rows.extend(
            _memory_rows(
                brain_path=brain_path,
                lane=lane,
                payload=payload,
                memory_key="shape_memory",
                examples_key="shape_examples_by_shape",
                target_field="target_shape",
            )
        )

    labels_by_identity: dict[str, set[str]] = defaultdict(set)
    targets_by_section_identity: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in raw_rows:
        if not row.identity_sha256:
            continue
        if row.approved_label:
            labels_by_identity[row.identity_sha256].add(row.approved_label)
        targets_by_section_identity[(row.section, row.identity_sha256)].add(row.owner_target)

    audited: list[LegacyBrainTrainerRow] = []
    for row in raw_rows:
        labels = labels_by_identity.get(row.identity_sha256, set())
        targets = targets_by_section_identity.get((row.section, row.identity_sha256), set())
        reasons: list[str] = []
        if row.source_path is None:
            reasons.append("source path is absent")
        elif not row.source_exists:
            reasons.append("source path no longer exists; fingerprint identity only")
        if not row.identity_sha256:
            reasons.append("no file hash or stored fingerprint is available")
        if len(labels) > 1:
            reasons.append("same trainer identity has multiple approved labels")
        if len(targets) > 1:
            reasons.append("same trainer identity has multiple targets in one owner lane")

        if len(labels) > 1:
            status = "approved_label_conflict"
        elif len(targets) > 1:
            status = "owner_target_conflict"
        elif not row.identity_sha256:
            status = "unresolved_identity"
        elif not row.source_exists:
            status = "missing_source"
        else:
            status = "clean"
        audited.append(
            replace(
                row,
                assigned_label_count=len(labels),
                owner_target_count=len(targets),
                status=status,
                reasons=tuple(reasons),
            )
        )
    return tuple(
        sorted(
            audited,
            key=lambda row: (
                row.status == "clean",
                row.status,
                row.identity_sha256,
                str(row.brain_path),
                row.section,
                row.owner_target,
            ),
        )
    )


def write_legacy_brain_audit(
    rows: Sequence[LegacyBrainTrainerRow],
    csv_path: Path,
    summary_path: Path,
) -> None:
    """Write the reversible legacy-brain inventory and aggregate counts."""
    csv_path = Path(csv_path)
    summary_path = Path(summary_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "brain_path",
                "lane",
                "section",
                "owner_target",
                "approved_label",
                "source_path",
                "source_exists",
                "file_sha256",
                "fingerprint_sha256",
                "identity_kind",
                "identity_sha256",
                "assigned_label_count",
                "owner_target_count",
                "status",
                "reasons",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.brain_path,
                    row.lane,
                    row.section,
                    row.owner_target,
                    row.approved_label,
                    row.source_path or "",
                    int(row.source_exists),
                    row.file_sha256,
                    row.fingerprint_sha256,
                    row.identity_kind,
                    row.identity_sha256,
                    row.assigned_label_count,
                    row.owner_target_count,
                    row.status,
                    "; ".join(row.reasons),
                ]
            )

    unique_identities = {row.identity_sha256 for row in rows if row.identity_sha256}
    conflict_identities = {
        row.identity_sha256
        for row in rows
        if row.identity_sha256 and row.status in {"approved_label_conflict", "owner_target_conflict"}
    }
    summary = {
        "schema_version": 1,
        "brain_files": sorted({str(row.brain_path) for row in rows}),
        "occurrence_count": len(rows),
        "unique_trainer_identity_count": len(unique_identities),
        "conflicting_identity_count": len(conflict_identities),
        "status_counts": dict(sorted(Counter(row.status for row in rows).items())),
        "section_counts": dict(sorted(Counter(row.section for row in rows).items())),
        "warning": "This is a reversible inventory. It does not mutate trainer data or brain JSON files.",
        "cleanup_plan": [
            "Review every approved_label_conflict and owner_target_conflict against the source audio.",
            "Copy human-approved trainers into a new versioned curated root; do not edit the locked source in place.",
            "Quarantine rejected or unresolved identities in a separate review manifest.",
            "Rebuild full/core/spread/outlier and dedicated memory brains from the reviewed root.",
            "Keep the current brains as rollback artifacts until held-out and shadow gates pass.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
