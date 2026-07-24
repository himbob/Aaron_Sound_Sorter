#!/usr/bin/env python3
"""
recover_regression_audio_from_samples.py

Standalone recovery tool.

Purpose:
  Find audio files referenced by tests, search user-supplied library roots, and
  copy matches into <project>/tests/regression_audio.

Run:

    python3 tools/recovery/recover_regression_audio_from_samples.py \
      --search-root /path/to/samples

Stronger search:

    python3 tools/recovery/recover_regression_audio_from_samples.py \
      --search-root /path/to/larger/library

No third-party packages required.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
import sys
import unicodedata
import zipfile
from pathlib import Path

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEARCH_ROOTS = [DEFAULT_PROJECT_ROOT / "tests" / "acceptance" / "locked_smoke_v1" / "samples"]
DEFAULT_DEST_RELATIVE = Path("tests/regression_audio")

AUDIO_EXTS = {
    ".wav",
    ".aif",
    ".aiff",
    ".flac",
    ".ogg",
    ".mp3",
    ".m4a",
    ".aac",
    ".au",
}

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
    "node_modules",
}

# These were visible in the latest failing pytest output.
ALWAYS_FIND_THESE = {
    "33157.wav",
    "50728.wav",
    "AA_JBL_78bpm_Cm_Sax_Loop_1.wav",
    "EWS_Keys_resampled_HipHop_RnB_G_Major_88BPM.wav",
    "Money_vocals_female_rap_110bpm.wav",
    "ABOUTME_94_DRUMLOOP.wav",
    "04_Dmn_176bpm_bass.wav",
    "Vocal Phrase We Up 140bpm.wav",
}


def normalize_name(value: str) -> str:
    """Normalize a filename for case-insensitive matching."""
    return unicodedata.normalize("NFC", value).casefold()


def is_audio_name(value: str) -> bool:
    """Return True when value ends in a supported audio extension."""
    return Path(value).suffix.lower() in AUDIO_EXTS


def filename_only(value: str) -> str:
    """Return just the filename component."""
    return Path(value.strip()).name


def read_text_best_effort(path: Path) -> str:
    """Read text files without crashing on odd encodings."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def collect_audio_names_from_tests(project_root: Path) -> set[str]:
    """Find audio filenames referenced anywhere under tests/*.py or manifests."""
    names: set[str] = set()
    tests_root = project_root / "tests"

    if not tests_root.exists():
        print(f"WARNING: tests folder missing: {tests_root}")
        return names

    quoted_audio = re.compile(
        r"""["']([^"'\n\r]+?\.(?:wav|aif|aiff|flac|ogg|mp3|m4a|aac|au))["']""",
        re.IGNORECASE,
    )

    bracket_audio = re.compile(
        r"""\[([^\]\n\r]+?\.(?:wav|aif|aiff|flac|ogg|mp3|m4a|aac|au))\]""",
        re.IGNORECASE,
    )

    for path in tests_root.rglob("*"):
        if path.is_dir():
            continue

        if path.suffix.lower() not in {".py", ".json", ".txt", ".md", ".csv"}:
            continue

        text = read_text_best_effort(path)

        for match in quoted_audio.finditer(text):
            names.add(filename_only(match.group(1)))

        for match in bracket_audio.finditer(text):
            names.add(filename_only(match.group(1)))

        if path.suffix.lower() == ".py":
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    value = node.value.strip()
                    if is_audio_name(value):
                        names.add(filename_only(value))

    names.update(ALWAYS_FIND_THESE)
    return names


def walk_sample_files(search_roots: list[Path]) -> tuple[dict[str, list[Path]], list[Path]]:
    """Index loose audio files and ZIP files under the sample roots."""
    loose_index: dict[str, list[Path]] = {}
    zip_paths: list[Path] = []

    for root in search_roots:
        if not root.exists():
            print(f"WARNING: search root does not exist: {root}")
            continue

        print(f"Searching loose files under: {root}")

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in SKIP_DIR_NAMES]
            current_dir = Path(dirpath)

            for filename in filenames:
                path = current_dir / filename
                suffix = path.suffix.lower()

                if suffix == ".zip":
                    zip_paths.append(path)

                if suffix in AUDIO_EXTS:
                    key = normalize_name(filename)
                    loose_index.setdefault(key, []).append(path)

    for paths in loose_index.values():
        paths.sort(key=lambda item: str(item))

    return loose_index, zip_paths


def copy_loose_file(filename: str, source: Path, dest: Path, overwrite: bool) -> str:
    """Copy one loose file into the destination folder."""
    if dest.exists() and not overwrite:
        return "exists"

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    print(f"COPIED  | {filename}")
    print(f"          from: {source}")
    print(f"          to:   {dest}")
    return "copied"


def extract_from_zip(filename: str, zip_paths: list[Path], dest: Path, overwrite: bool) -> str | None:
    """Extract one expected filename from the first ZIP containing it."""
    if dest.exists() and not overwrite:
        return "exists"

    wanted_key = normalize_name(filename)

    for zip_path in zip_paths:
        try:
            with zipfile.ZipFile(zip_path) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue

                    member_name = Path(member.filename).name
                    if normalize_name(member_name) != wanted_key:
                        continue

                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as src, dest.open("wb") as out:
                        shutil.copyfileobj(src, out)

                    print(f"EXTRACT | {filename}")
                    print(f"          zip:    {zip_path}")
                    print(f"          member: {member.filename}")
                    print(f"          to:     {dest}")
                    return "extracted"

        except (OSError, zipfile.BadZipFile):
            continue

    return None


def restore_files(
    expected_names: set[str],
    loose_index: dict[str, list[Path]],
    zip_paths: list[Path],
    dest_dir: Path,
    overwrite: bool,
) -> dict[str, int]:
    """Restore every expected file into tests/regression_audio."""
    counts = {
        "exists": 0,
        "copied": 0,
        "extracted": 0,
        "missing": 0,
    }

    dest_dir.mkdir(parents=True, exist_ok=True)

    for filename in sorted(expected_names):
        dest = dest_dir / filename

        if dest.exists() and not overwrite:
            counts["exists"] += 1
            print(f"EXISTS  | {filename}")
            continue

        candidates = loose_index.get(normalize_name(filename), [])
        if candidates:
            status = copy_loose_file(filename, candidates[0], dest, overwrite)
            counts[status] += 1
            continue

        zip_status = extract_from_zip(filename, zip_paths, dest, overwrite)
        if zip_status:
            counts[zip_status] += 1
            continue

        counts["missing"] += 1
        print(f"MISSING | {filename}")

    return counts


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command line args."""
    parser = argparse.ArgumentParser(
        description="Recover test audio files from the sample folder into tests/regression_audio."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help=f"Project root. Default: {DEFAULT_PROJECT_ROOT}",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        default=None,
        help=(f"Folder to search recursively. Can be repeated. Default: {DEFAULT_SEARCH_ROOTS[0]}"),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination folder. Default: <project-root>/tests/regression_audio",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files already present in tests/regression_audio.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print expected filenames, do not search or copy.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    """Run the recovery tool."""
    args = parse_args(argv)

    project_root = args.project_root.expanduser()
    search_roots = args.search_root or DEFAULT_SEARCH_ROOTS
    search_roots = [path.expanduser() for path in search_roots]
    dest_dir = args.dest.expanduser() if args.dest else project_root / DEFAULT_DEST_RELATIVE

    expected_names = collect_audio_names_from_tests(project_root)

    print(f"Project root: {project_root}")
    print(f"Destination:  {dest_dir}")
    print(f"Expected audio filenames found in tests/log list: {len(expected_names)}")
    print()

    if args.list_only:
        for name in sorted(expected_names):
            print(name)
        return 0

    loose_index, zip_paths = walk_sample_files(search_roots)

    print()
    print(f"Loose audio filenames indexed: {len(loose_index)}")
    print(f"ZIP files found:               {len(zip_paths)}")
    print()

    counts = restore_files(
        expected_names=expected_names,
        loose_index=loose_index,
        zip_paths=zip_paths,
        dest_dir=dest_dir,
        overwrite=args.overwrite,
    )

    print()
    print("Summary")
    print("-------")
    for key in ["exists", "copied", "extracted", "missing"]:
        print(f"{key}: {counts[key]}")

    print()
    print("Final destination audio count")
    print("-----------------------------")
    final_count = sum(1 for path in dest_dir.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTS)
    print(f"{final_count} audio files in {dest_dir}")

    if counts["missing"]:
        print()
        print("Some files were not found in the default sample folder.")
        print("Try a wider search:")
        print("  python3 tools/recovery/recover_regression_audio_from_samples.py --search-root /path/to/samples")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
