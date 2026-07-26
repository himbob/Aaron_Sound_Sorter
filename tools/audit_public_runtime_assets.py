#!/usr/bin/env python3
"""Remove or reject private source names in publishable runtime brains."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

PRIVATE_PATH_PATTERN = re.compile(r"(?:/Volumes/|/Users/|[A-Za-z]:[\\/])")
AUDIO_NAME_PATTERN = re.compile(r"(?i)\.(?:wav|aif|aiff|flac|ogg|mp3|m4a|aac)(?:$|[?#])")
SOURCE_KEY_RENAMES = {
    "source_path": "source_id",
    "new_source_path": "new_source_id",
    "last_incremental_source_path": "last_incremental_source_id",
    "source_path_for_audit_only": "source_id_for_audit_only",
    "source_pack": "source_group_id",
    "source_pack_counts_top10": "source_group_counts_top10",
}


def opaque_identifier(value: str) -> str:
    """Return a stable, non-reversible identifier for private metadata."""
    digest = hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()
    return f"private-sha256:{digest}"


def is_private_string(value: str) -> bool:
    """Return whether a string exposes a local path or audio filename."""
    return bool(PRIVATE_PATH_PATTERN.search(value) or AUDIO_NAME_PATTERN.search(value))


def sanitize_value(value: Any, *, parent_key: str = "") -> Any:
    """Return JSON-compatible data with private source metadata anonymized."""
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            renamed_key = SOURCE_KEY_RENAMES.get(key, key)
            if key == "source_pack_counts_top10" and isinstance(raw_value, dict):
                sanitized[renamed_key] = {
                    opaque_identifier(str(pack_name)): sanitize_value(count) for pack_name, count in raw_value.items()
                }
            elif key in {"source_path", "new_source_path", "last_incremental_source_path", "source_pack"}:
                sanitized[renamed_key] = opaque_identifier(str(raw_value)) if str(raw_value) else ""
            else:
                sanitized[renamed_key] = sanitize_value(raw_value, parent_key=renamed_key)
        return sanitized
    if isinstance(value, list):
        return [sanitize_value(element, parent_key=parent_key) for element in value]
    if isinstance(value, str) and is_private_string(value):
        return opaque_identifier(value)
    return value


def private_value_locations(value: Any, *, location: str = "$") -> list[str]:
    """Return JSON locations that still expose private source metadata."""
    violations: list[str] = []
    if isinstance(value, dict):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_location = f"{location}.{key}"
            if key in SOURCE_KEY_RENAMES:
                violations.append(child_location)
            violations.extend(private_value_locations(child, location=child_location))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            violations.extend(private_value_locations(child, location=f"{location}[{index}]"))
    elif isinstance(value, str) and is_private_string(value):
        violations.append(location)
    return violations


def runtime_brain_paths(project_root: Path) -> list[Path]:
    """Return active root brain JSON paths in stable order."""
    return sorted(path for path in project_root.glob("stage4_*brain*.json") if path.is_file())


def write_sanitized_brain(path: Path) -> None:
    """Atomically rewrite one runtime brain without private source metadata."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    sanitized = sanitize_value(payload)
    temporary_path = path.with_suffix(path.suffix + ".sanitizing")
    temporary_path.write_text(json.dumps(sanitized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)


def audit_brains(project_root: Path) -> list[str]:
    """Return publishability violations without printing private values."""
    violations: list[str] = []
    for path in runtime_brain_paths(project_root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for location in private_value_locations(payload):
            violations.append(f"{path.name}:{location}")
    return violations


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--apply", action="store_true", help="Anonymize private metadata before auditing.")
    return parser.parse_args()


def main() -> int:
    """Sanitize when requested, then fail if publishable brains still leak names."""
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    paths = runtime_brain_paths(project_root)
    if args.apply:
        for path in paths:
            write_sanitized_brain(path)
    violations = audit_brains(project_root)
    if violations:
        print("FAIL: publishable runtime brains contain private source metadata:")
        for violation in violations[:100]:
            print(f"- {violation}")
        if len(violations) > 100:
            print(f"- ... {len(violations) - 100} more")
        return 1
    print(f"PASS: {len(paths)} runtime brain file(s) contain no local paths or audio filenames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
