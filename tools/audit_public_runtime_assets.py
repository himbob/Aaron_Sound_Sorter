#!/usr/bin/env python3
"""Reject private/generated assets and local paths tracked by Git."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Iterable
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
FORBIDDEN_TRACKED_PREFIXES = ("_models/", "_reports/", "neural_artifacts/", "training/")
FORBIDDEN_TRACKED_PATTERNS = (
    "stage4_*brain*.json",
    "tests/acceptance/**/trusted_training_seed*.json",
)
FORBIDDEN_TRACKED_SUFFIXES = (
    ".aac",
    ".aif",
    ".aiff",
    ".au",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".rar",
    ".tar",
    ".tgz",
    ".wav",
    ".zip",
    ".7z",
)
PRIVATE_MACHINE_MARKERS = (
    "".join(("/Users", "/aaron")),
    "".join(("/Volumes", "/T9")),
)


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


def tracked_repository_paths(project_root: Path) -> tuple[str, ...]:
    """Return normalized Git-tracked paths without inspecting ignored local data."""
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=project_root,
        check=True,
        capture_output=True,
    )
    return tuple(sorted(path for path in completed.stdout.decode("utf-8").split("\0") if path))


def forbidden_tracked_paths(paths: Iterable[str]) -> list[str]:
    """Return tracked paths forbidden by the public repository contract."""
    violations: list[str] = []
    for raw_path in paths:
        path = str(raw_path).replace("\\", "/")
        basename = Path(path).name
        if path.startswith(FORBIDDEN_TRACKED_PREFIXES):
            violations.append(path)
            continue
        if any(
            fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(basename, pattern)
            for pattern in FORBIDDEN_TRACKED_PATTERNS
        ):
            violations.append(path)
            continue
        if path.casefold().endswith(FORBIDDEN_TRACKED_SUFFIXES):
            violations.append(path)
    return sorted(set(violations))


def tracked_private_path_locations(project_root: Path, paths: Iterable[str]) -> list[str]:
    """Return tracked text locations containing this machine's private roots."""
    violations: list[str] = []
    for relative_path in paths:
        path = project_root / relative_path
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(marker in line for marker in PRIVATE_MACHINE_MARKERS):
                violations.append(f"{relative_path}:{line_number}")
    return violations


def runtime_brain_paths(project_root: Path, tracked_paths: Iterable[str] | None = None) -> list[Path]:
    """Return tracked root brain JSON paths in stable order."""
    candidates = tracked_paths if tracked_paths is not None else tracked_repository_paths(project_root)
    return sorted(
        project_root / path for path in candidates if "/" not in path and fnmatch.fnmatch(path, "stage4_*brain*.json")
    )


def write_sanitized_brain(path: Path) -> None:
    """Atomically rewrite one runtime brain without private source metadata."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    sanitized = sanitize_value(payload)
    temporary_path = path.with_suffix(path.suffix + ".sanitizing")
    temporary_path.write_text(json.dumps(sanitized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)


def audit_brains(project_root: Path, tracked_paths: Iterable[str] | None = None) -> list[str]:
    """Return publishability violations without printing private values."""
    violations: list[str] = []
    for path in runtime_brain_paths(project_root, tracked_paths):
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
    """Sanitize when requested, then enforce the public repository contract."""
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    tracked_paths = tracked_repository_paths(project_root)
    paths = runtime_brain_paths(project_root, tracked_paths)
    if args.apply:
        for path in paths:
            write_sanitized_brain(path)
    brain_violations = audit_brains(project_root, tracked_paths)
    asset_violations = forbidden_tracked_paths(tracked_paths)
    path_violations = tracked_private_path_locations(project_root, tracked_paths)
    violations = [
        *(f"private brain metadata: {value}" for value in brain_violations),
        *(f"forbidden tracked asset: {value}" for value in asset_violations),
        *(f"private machine path: {value}" for value in path_violations),
    ]
    if violations:
        print("FAIL: public repository safety violations:")
        for violation in violations[:100]:
            print(f"- {violation}")
        if len(violations) > 100:
            print(f"- ... {len(violations) - 100} more")
        return 1
    print(f"PASS: {len(tracked_paths)} tracked path(s) contain no private/generated runtime assets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
