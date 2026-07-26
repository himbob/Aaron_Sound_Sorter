"""Explicit, reversible safe-core approval imports for neural training."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aaron_sound_sorter.taxonomy_contracts import is_trainable_taxonomy_label

from .gui_training import NeuralIntakeSummary, NeuralTrainingInbox
from .hashing import sha256_file


@dataclass(frozen=True)
class ClusterApprovalImportSummary:
    """Result of one explicit safe-core import without a prototype rebuild."""

    cluster_id: str
    approved_category: str
    imported_count: int
    intake_summary: NeuralIntakeSummary
    import_manifest_path: Path
    rollback_record_path: Path
    copied_audio_paths: tuple[Path, ...]


def import_cluster_approval(
    project_root: Path,
    approval_manifest_path: Path,
    output_dir: Path,
    *,
    inbox_root: Path | None = None,
) -> ClusterApprovalImportSummary:
    """Verify and import an explicitly approved safe core into the inbox.

    Args:
        project_root: Repository root.
        approval_manifest_path: Human-authored approval JSON.
        output_dir: New report directory for provenance and rollback metadata.
        inbox_root: Optional isolated inbox for tests.

    Returns:
        Import counts, durable inbox result, and rollback record.

    Raises:
        FileExistsError: If the report directory already exists.
        ValueError: If approval, hashes, taxonomy, or duplicate labels conflict.
        FileNotFoundError: If a manifest or approved member is missing.

    Side Effects:
        Copies approved hash-named audio, appends inbox events, and writes a
        reversible import record. It does not rebuild any prototype index.
    """
    root = Path(project_root).expanduser().resolve()
    approval_path = Path(approval_manifest_path).expanduser().resolve()
    report_dir = Path(output_dir).expanduser().resolve()
    if report_dir.exists():
        raise FileExistsError(f"cluster approval report already exists: {report_dir}")
    approval = _load_object(approval_path)
    cluster_id = str(approval.get("cluster_id", "")).strip()
    approved_category = str(approval.get("approved_category", "")).strip()
    approved_hashes = _string_tuple(approval.get("approved_safe_core_hashes", []))
    reviewed_by = str(approval.get("reviewed_by", "")).strip()
    if not cluster_id or not reviewed_by or not approved_hashes:
        raise ValueError("cluster approval requires cluster_id, reviewed_by, and approved safe-core hashes")
    if not is_trainable_taxonomy_label(approved_category):
        raise ValueError(f"invalid approved cluster category: {approved_category or '<empty>'}")
    cluster_manifest_path = _resolve_from_manifest(approval_path, approval.get("cluster_manifest_path"))
    cluster_manifest = _load_object(cluster_manifest_path)
    cluster = _find_cluster(cluster_manifest, cluster_id)
    safe_core_hashes = set(_string_tuple(cluster.get("safe_core_hashes", [])))
    approved_set = set(approved_hashes)
    approval_scope = str(approval.get("approval_scope", "safe_core"))
    explicit_all_members = bool(approval.get("allow_non_core", False)) and approval_scope == "all_members"
    if explicit_all_members:
        member_hashes = set(_string_tuple(cluster.get("member_hashes", [])))
        if not member_hashes or approved_set != member_hashes:
            raise ValueError("explicit all-member approval must include every cluster member exactly once")
    elif not approved_set.issubset(safe_core_hashes):
        raise ValueError("approval includes hashes outside the cluster safe core")
    member_paths = cluster.get("member_paths_by_hash", {})
    if not isinstance(member_paths, dict):
        raise ValueError("cluster manifest member_paths_by_hash must be an object")

    inbox = NeuralTrainingInbox(root, inbox_root=inbox_root)
    _reject_existing_label_conflicts(inbox.current_manifest_path, approved_hashes, approved_category)
    report_dir.mkdir(parents=True)
    audio_root = inbox.inbox_root / "approved_cluster_audio" / cluster_id
    audio_root.mkdir(parents=True, exist_ok=True)
    before_manifest = report_dir / "inbox_current_before.csv"
    prior_manifest_exists = inbox.current_manifest_path.is_file()
    if prior_manifest_exists:
        shutil.copy2(inbox.current_manifest_path, before_manifest)
    previous_event_log_size = inbox.event_log_path.stat().st_size if inbox.event_log_path.is_file() else 0
    copied_paths: list[Path] = []
    import_rows: list[dict[str, str]] = []
    pack_root = cluster_manifest_path.parent.resolve()
    try:
        for file_sha256 in approved_hashes:
            relative_member_path = str(member_paths.get(file_sha256, "")).strip()
            source_path = (pack_root / relative_member_path).resolve()
            if not source_path.is_relative_to(pack_root) or not source_path.is_file():
                raise FileNotFoundError(f"approved cluster member is missing: {file_sha256}")
            if sha256_file(source_path) != file_sha256:
                raise ValueError(f"approved cluster member hash mismatch: {file_sha256}")
            copied_path = audio_root / f"audio_{file_sha256}{source_path.suffix.casefold()}"
            if copied_path.exists() and sha256_file(copied_path) != file_sha256:
                raise ValueError(f"approved cluster destination conflict: {copied_path}")
            if not copied_path.exists():
                shutil.copy2(source_path, copied_path)
                copied_paths.append(copied_path)
            import_rows.append(
                {
                    "approved_folder": approved_category,
                    "staged_path": str(copied_path),
                    "status": "staged",
                }
            )
        import_manifest_path = report_dir / "cluster_training_import.csv"
        _write_import_rows(import_manifest_path, import_rows)
        intake_summary = inbox.queue_import_manifest(import_manifest_path)
        if intake_summary.errors:
            raise ValueError("cluster neural intake rejected rows: " + "; ".join(intake_summary.errors))
    except Exception:
        _restore_inbox_state(
            inbox,
            prior_manifest_exists=prior_manifest_exists,
            before_manifest=before_manifest,
            previous_event_log_size=previous_event_log_size,
            copied_paths=copied_paths,
        )
        raise

    rollback_record_path = report_dir / "rollback_record.json"
    rollback_payload = {
        "schema_version": 1,
        "cluster_id": cluster_id,
        "approved_category": approved_category,
        "approval_scope": approval_scope,
        "approval_manifest": str(approval_path),
        "inbox_root": str(inbox.inbox_root),
        "current_manifest_path": str(inbox.current_manifest_path),
        "prior_manifest_exists": prior_manifest_exists,
        "before_manifest_path": str(before_manifest) if prior_manifest_exists else "",
        "event_log_path": str(inbox.event_log_path),
        "previous_event_log_size": previous_event_log_size,
        "copied_audio_paths": [str(path) for path in copied_paths],
        "imported_hashes": list(approved_hashes),
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "prototype_rebuild_performed": False,
    }
    rollback_record_path.write_text(json.dumps(rollback_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (report_dir / "import_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cluster_id": cluster_id,
                "approved_category": approved_category,
                "imported_count": len(approved_hashes),
                "intake": {
                    **asdict(intake_summary),
                    "current_manifest_path": str(intake_summary.current_manifest_path),
                    "event_log_path": str(intake_summary.event_log_path),
                },
                "prototype_rebuild_performed": False,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return ClusterApprovalImportSummary(
        cluster_id,
        approved_category,
        len(approved_hashes),
        intake_summary,
        import_manifest_path,
        rollback_record_path,
        tuple(copied_paths),
    )


def rollback_cluster_import(rollback_record_path: Path) -> None:
    """Restore the inbox state recorded before one cluster import."""
    record_path = Path(rollback_record_path).expanduser().resolve()
    record = _load_object(record_path)
    current_manifest = Path(str(record["current_manifest_path"])).resolve()
    prior_manifest_exists = bool(record.get("prior_manifest_exists", False))
    if prior_manifest_exists:
        before_manifest = Path(str(record["before_manifest_path"])).resolve()
        if not before_manifest.is_file():
            raise FileNotFoundError(f"rollback inbox backup is missing: {before_manifest}")
        shutil.copy2(before_manifest, current_manifest)
    elif current_manifest.exists():
        current_manifest.unlink()
    event_log_path = Path(str(record["event_log_path"])).resolve()
    previous_size = int(record.get("previous_event_log_size", 0))
    if event_log_path.is_file():
        with event_log_path.open("r+b") as handle:
            handle.truncate(previous_size)
        if previous_size == 0:
            event_log_path.unlink()
    for raw_path in record.get("copied_audio_paths", []):
        copied_path = Path(str(raw_path)).resolve()
        if copied_path.is_file():
            copied_path.unlink()


def _reject_existing_label_conflicts(path: Path, approved_hashes: tuple[str, ...], approved_category: str) -> None:
    if not path.is_file():
        return
    approved_set = set(approved_hashes)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("file_sha256", "")) not in approved_set:
                continue
            existing_label = str(row.get("approved_folder", "")).strip()
            if existing_label and existing_label != approved_category:
                raise ValueError(
                    f"approved cluster hash already has another label: {existing_label} vs {approved_category}"
                )


def _restore_inbox_state(
    inbox: NeuralTrainingInbox,
    *,
    prior_manifest_exists: bool,
    before_manifest: Path,
    previous_event_log_size: int,
    copied_paths: list[Path],
) -> None:
    if prior_manifest_exists and before_manifest.is_file():
        shutil.copy2(before_manifest, inbox.current_manifest_path)
    elif inbox.current_manifest_path.is_file():
        inbox.current_manifest_path.unlink()
    if inbox.event_log_path.is_file():
        with inbox.event_log_path.open("r+b") as handle:
            handle.truncate(previous_event_log_size)
        if previous_event_log_size == 0:
            inbox.event_log_path.unlink()
    for copied_path in copied_paths:
        if copied_path.is_file():
            copied_path.unlink()


def _write_import_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["approved_folder", "staged_path", "status"])
        writer.writeheader()
        writer.writerows(rows)


def _find_cluster(manifest: dict[str, Any], cluster_id: str) -> dict[str, Any]:
    clusters = manifest.get("clusters", [])
    if not isinstance(clusters, list):
        raise ValueError("cluster manifest clusters must be a list")
    for cluster in clusters:
        if isinstance(cluster, dict) and str(cluster.get("cluster_id", "")) == cluster_id:
            return cluster
    raise ValueError(f"cluster is absent from review manifest: {cluster_id}")


def _resolve_from_manifest(manifest_path: Path, raw_path: object) -> Path:
    path = Path(str(raw_path or "")).expanduser()
    if not str(raw_path or "").strip():
        raise ValueError("cluster_manifest_path is required")
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _string_tuple(payload: object) -> tuple[str, ...]:
    if not isinstance(payload, list):
        raise ValueError("approved hashes must be a list")
    return tuple(dict.fromkeys(str(value).strip() for value in payload if str(value).strip()))
