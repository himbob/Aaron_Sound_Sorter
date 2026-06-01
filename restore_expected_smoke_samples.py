#!/usr/bin/env python3
"""
restore_expected_smoke_samples.py

Restore WAV files listed in an expected_results.json manifest by searching one
or more sample-library roots and copying matching files into the acceptance
sample folder.

Default project assumptions:
  Project root: /Volumes/T9/testbed/Aaron_Sound_Sorter
  Manifest:     ./expected_results.json
  Destination:  ./tests/acceptance/locked_smoke_v1/samples
  Search root:  /Volumes/T9/music_production/samples

The script only matches by manifest filename/basename. It does not inspect audio
content or use category labels.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_PROJECT_ROOT = Path("/Volumes/T9/testbed/Aaron_Sound_Sorter")
DEFAULT_SEARCH_ROOTS = [Path("/Volumes/T9/music_production/samples")]
DEFAULT_DEST = DEFAULT_PROJECT_ROOT / "tests/acceptance/locked_smoke_v1/samples"
DEFAULT_MANIFEST = DEFAULT_PROJECT_ROOT / "expected_results.json"

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    ".venv_phase4",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "outputs",
    "reports",
    "_reports",
    "dist",
    "build",
}


@dataclass(frozen=True)
class RestoreResult:
    """Result for one expected sample restore attempt."""

    filename: str
    status: str
    source: Path | None
    destination: Path
    note: str


def normalize_name(value: str) -> str:
    """Normalize a filename for case-insensitive and Unicode-safe lookup."""
    return unicodedata.normalize("NFC", value).casefold()


def iter_files(root: Path) -> Iterable[Path]:
    """Yield files under root while skipping known junk/generated directories."""
    if not root.exists():
        return

    stack = [root]
    while stack:
        current = stack.pop()
        try:
            children = list(current.iterdir())
        except OSError:
            continue

        for child in children:
            if child.is_dir():
                if child.name in SKIP_DIR_NAMES:
                    continue
                stack.append(child)
            elif child.is_file():
                yield child


def load_expected_filenames(manifest_path: Path) -> list[str]:
    """Load unique expected filenames from an expected_results.json file."""
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    cases = data.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Manifest has no list field named 'cases': {manifest_path}")

    filenames: list[str] = []
    seen: set[str] = set()

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Manifest case #{index} is not an object")

        filename = case.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"Manifest case #{index} has no valid filename")

        base = Path(filename).name
        key = normalize_name(base)

        if key not in seen:
            seen.add(key)
            filenames.append(base)

    return filenames


def build_source_index(search_roots: list[Path]) -> dict[str, list[Path]]:
    """Build a basename lookup index for all candidate files under search roots."""
    index: dict[str, list[Path]] = {}

    for root in search_roots:
        for path in iter_files(root):
            key = normalize_name(path.name)
            index.setdefault(key, []).append(path)

    for paths in index.values():
        paths.sort(key=lambda item: str(item))

    return index


def pick_source(filename: str, candidates: list[Path]) -> Path | None:
    """Pick the best source candidate for one expected filename."""
    if not candidates:
        return None

    exact = [path for path in candidates if path.name == filename]
    if exact:
        return exact[0]

    return candidates[0]


def restore_samples(
    filenames: list[str],
    source_index: dict[str, list[Path]],
    destination_dir: Path,
    dry_run: bool,
    overwrite: bool,
) -> list[RestoreResult]:
    """Copy found samples into the destination folder."""
    results: list[RestoreResult] = []
    destination_dir.mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        destination = destination_dir / filename

        if destination.exists() and not overwrite:
            results.append(
                RestoreResult(
                    filename=filename,
                    status="exists",
                    source=None,
                    destination=destination,
                    note="already present; use --overwrite to replace",
                )
            )
            continue

        candidates = source_index.get(normalize_name(filename), [])
        source = pick_source(filename, candidates)

        if source is None:
            results.append(
                RestoreResult(
                    filename=filename,
                    status="missing",
                    source=None,
                    destination=destination,
                    note="no matching basename found in search roots",
                )
            )
            continue

        if source.resolve() == destination.resolve():
            results.append(
                RestoreResult(
                    filename=filename,
                    status="same-file",
                    source=source,
                    destination=destination,
                    note="source and destination are the same file",
                )
            )
            continue

        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        status = "would-copy" if dry_run else "copied"
        note = f"{len(candidates)} candidate(s)" if len(candidates) > 1 else "exact basename match"

        results.append(
            RestoreResult(
                filename=filename,
                status=status,
                source=source,
                destination=destination,
                note=note,
            )
        )

    return results


def print_report(results: list[RestoreResult]) -> int:
    """Print a restore report and return a process exit code."""
    counts: dict[str, int] = {}

    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
        source = str(result.source) if result.source else "-"
        print(f"{result.status:10} | {result.filename}")
        print(f"           source: {source}")
        print(f"             dest: {result.destination}")
        print(f"             note: {result.note}")

    print()
    print("Summary:")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")

    return 1 if counts.get("missing", 0) else 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Restore expected acceptance smoke WAV files from sample library roots."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Path to expected_results.json. Default: {DEFAULT_MANIFEST}",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination sample folder. Default: {DEFAULT_DEST}",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Root folder to search recursively. Can be used more than once. "
            f"Default: {DEFAULT_SEARCH_ROOTS[0]}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be copied without copying files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace destination files that already exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """Run the restore tool."""
    args = parse_args(argv)

    manifest_path = args.manifest.expanduser()
    destination_dir = args.dest.expanduser()
    search_roots = args.search_root or DEFAULT_SEARCH_ROOTS
    search_roots = [root.expanduser() for root in search_roots]

    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    print(f"Manifest: {manifest_path}")
    print(f"Destination: {destination_dir}")
    print("Search roots:")
    for root in search_roots:
        print(f"  {root}")
    print()

    try:
        filenames = load_expected_filenames(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: failed to read manifest: {exc}", file=sys.stderr)
        return 2

    print(f"Expected filenames: {len(filenames)}")
    print("Indexing source files. This can take a minute on a large sample drive...")
    source_index = build_source_index(search_roots)

    results = restore_samples(
        filenames=filenames,
        source_index=source_index,
        destination_dir=destination_dir,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )

    return print_report(results)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
