"""Read legacy placement manifests for report-only shadow comparisons."""

from __future__ import annotations

import csv
from pathlib import Path

from .hashing import sha256_file


def read_legacy_folder_map_by_hash(path: Path, *, compute_missing_hashes: bool = True) -> dict[str, str]:
    """Read file-hash to legacy-folder mappings for shadow reports.

    New manifests should carry ``file_sha256`` directly.  During migration, a
    missing hash may be computed from an existing source path.  This happens only
    after the legacy decision and is never fed back into category scoring.
    """
    path = Path(path)
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            digest = (row.get("file_sha256") or row.get("sha256") or "").strip()
            if len(digest) != 64 and compute_missing_hashes:
                source_text = (
                    row.get("original_path") or row.get("source_path") or row.get("input_path") or ""
                ).strip()
                source_path = Path(source_text).expanduser() if source_text else None
                if source_path and source_path.is_file():
                    digest = sha256_file(source_path)
            output = (
                row.get("new_relative_path")
                or row.get("final_folder")
                or row.get("folder_path")
                or row.get("folder")
                or ""
            ).strip()
            if len(digest) == 64 and output:
                output_path = Path(output)
                folder = str(output_path.parent if output_path.suffix else output_path).replace("\\", "/")
                result[digest] = folder
    return result
