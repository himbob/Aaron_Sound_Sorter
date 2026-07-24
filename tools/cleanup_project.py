#!/usr/bin/env python3
"""Clean disposable Aaron Sound Sorter project junk.

This tool is intentionally boring and conservative. It removes caches, test
outputs, preview runs, and root clutter. It preserves GUI training evidence,
rollback backups, classifier code, active brains, source, tests, docs, training
audio, models, and neural artifacts by default.
"""

from __future__ import annotations

import argparse
import fnmatch
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

PROTECTED_ROOT_NAMES = {
    "Aaron_Sound_Sorter.py",
    "AGENTS.md",
    "AI_READ_THIS_FIRST.md",
    "CURRENT_STATUS.md",
    "LICENSE",
    "NOTICE.md",
    "README.md",
    "MASTER_PROJECT_GOALS.md",
    "pyproject.toml",
    "requirements-dev.txt",
    "requirements-quality.txt",
    "requirements.txt",
    "Makefile",
    ".gitignore",
}

PROTECTED_ROOT_DIRS = {
    "src",
    "tests",
    "tools",
    "commands",
    "config",
    "docs",
    ".github",
    "training",  # huge, but user-controlled source/training data; never delete by default.
    "_models",  # pinned local foundation-model snapshots; expensive but reproducible critical inputs.
    "neural_artifacts",  # compact provenance, registry, quarantine, and held-out evidence.
}

ACTIVE_BRAIN_PATTERNS = [
    "stage4_folder_brain*.json",
    "stage4_*brain*.json",
    "v31_brain*.json",
    "brain*.json",
    "brains*.json",
]

STANDARD_DIRS = [
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "htmlcov",
    "reports",
    "_real_sort_tests",
    "outputs",
    "output",
    "sorted_output",
    "test_output",
    "Aaron_Sorted_Sounds",
    "Aaron_Test_Cases",
    "source_cache",
    "_pytest_outputs",
    "_pytest_regression_outputs",
    "_qa_parent_eligibility_v2",
    "_backups",
    "_patch_backups",
]

STANDARD_REPORT_DIRS = [
    "quality",
    "python_pycache",
    "pytest_outputs",
    "pytest_regression_outputs",
    "gui_preview",
    "generated_private_regression_audio",
]

STANDARD_REPORT_DIR_GLOBS = [
    "pytest_*",
    "generated_*",
]

STANDARD_DIR_GLOBS = [
    "_backup*",
]

STANDARD_ROOT_FILE_GLOBS = [
    "tree.out",
    "validation.log",
    "installed_files.txt",
    "REAL_SAMPLE_RETEST_RESULTS_*.csv",
    "LOW_REPRESENTATION_LABELS_REPORT_*.csv",
    "Aaron_Brain_*.csv",
    "Aaron_Sorted_Sounds_manifest.csv",
    "Aaron_Sorted_Sounds_summary.txt",
    "*.log",
    "*.tmp",
    "*.bak",
    "Aaron_Sound_Sorter_*.zip",
    "Archive.zip",
]

CACHE_FILE_GLOBS_RECURSIVE = [
    "*.pyc",
    ".DS_Store",
    "._*",
]

CACHE_DIR_NAMES_RECURSIVE = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

HEAVY_LOCAL_DIRS = [
    "_balanced_baby_training",
    "stage4_brain_family_training",
    "stage4_folder_brain_training",
]


@dataclass(frozen=True)
class RemovalCandidate:
    path: Path
    reason: str


def is_active_brain_file(path: Path) -> bool:
    """Return True for active root brain JSON files that cleanup must preserve."""
    if path.parent != path.parent.parent / path.parent.name:
        # This helper is only called on project-root children, but keep it simple.
        return False
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in ACTIVE_BRAIN_PATTERNS)


def is_protected_root_child(project_root: Path, path: Path) -> bool:
    """Return True when a root child should never be removed by this cleanup tool."""
    try:
        rel = path.relative_to(project_root)
    except ValueError:
        return True
    if len(rel.parts) != 1:
        return False
    name = rel.parts[0]
    if name in PROTECTED_ROOT_NAMES or name in PROTECTED_ROOT_DIRS:
        return True
    return bool(path.is_file() and any(fnmatch.fnmatch(name, pattern) for pattern in ACTIVE_BRAIN_PATTERNS))


def add_if_exists(candidates: list[RemovalCandidate], path: Path, reason: str) -> None:
    """Append a removal candidate if it exists and is not already present."""
    if not path.exists() and not path.is_symlink():
        return
    resolved = str(path)
    if any(str(candidate.path) == resolved for candidate in candidates):
        return
    candidates.append(RemovalCandidate(path=path, reason=reason))


def collect_root_dir_candidates(project_root: Path, mode: str) -> list[RemovalCandidate]:
    """Collect generated root directories for cleanup."""
    candidates: list[RemovalCandidate] = []
    for name in STANDARD_DIRS:
        add_if_exists(candidates, project_root / name, "generated directory")
    for pattern in STANDARD_DIR_GLOBS:
        for path in project_root.glob(pattern):
            add_if_exists(candidates, path, f"generated directory pattern {pattern}")
    reports_root = project_root / "_reports"
    for name in STANDARD_REPORT_DIRS:
        add_if_exists(candidates, reports_root / name, "disposable generated report directory")
    for pattern in STANDARD_REPORT_DIR_GLOBS:
        for path in reports_root.glob(pattern):
            add_if_exists(candidates, path, f"disposable report directory pattern {pattern}")
    if mode == "heavy-local":
        for name in HEAVY_LOCAL_DIRS:
            add_if_exists(candidates, project_root / name, "heavy local generated training selection folder")
    return candidates


def collect_root_file_candidates(project_root: Path) -> list[RemovalCandidate]:
    """Collect generated root files for cleanup."""
    candidates: list[RemovalCandidate] = []
    for pattern in STANDARD_ROOT_FILE_GLOBS:
        for path in project_root.glob(pattern):
            if path.is_dir():
                continue
            add_if_exists(candidates, path, f"generated root file pattern {pattern}")
    return candidates


def collect_recursive_cache_candidates(project_root: Path) -> list[RemovalCandidate]:
    """Collect cache files and cache directories recursively."""
    candidates: list[RemovalCandidate] = []
    skip_roots = {
        project_root / ".git",
        project_root / "training",
        project_root / "_training_data",
    }
    for path in project_root.rglob("*"):
        if any(path == root or root in path.parents for root in skip_roots):
            continue
        if path.is_dir() and path.name in CACHE_DIR_NAMES_RECURSIVE:
            add_if_exists(candidates, path, f"recursive cache directory {path.name}")
        elif path.is_file() and any(fnmatch.fnmatch(path.name, pattern) for pattern in CACHE_FILE_GLOBS_RECURSIVE):
            add_if_exists(candidates, path, "recursive cache/metadata file")
    return candidates


def prune_children_of_removed_dirs(candidates: Sequence[RemovalCandidate]) -> list[RemovalCandidate]:
    """Drop child candidates when an ancestor directory is already removed.

    The cleanup collector intentionally finds both cache directories and files inside
    those directories.  Removing both is noisy and can race on macOS AppleDouble
    files such as ``._AI_READ_THIS_FIRST.md``.  Keeping the ancestor directory is
    enough, so child candidates are pruned before removal.
    """
    sorted_candidates = sorted(candidates, key=lambda item: (len(item.path.parts), str(item.path)))
    kept: list[RemovalCandidate] = []
    removed_dirs: list[Path] = []
    for candidate in sorted_candidates:
        if any(parent == candidate.path or parent in candidate.path.parents for parent in removed_dirs):
            continue
        kept.append(candidate)
        if candidate.path.is_dir() and not candidate.path.is_symlink():
            removed_dirs.append(candidate.path)
    return sorted(kept, key=lambda item: str(item.path))


def collect_candidates(project_root: Path, mode: str) -> list[RemovalCandidate]:
    """Collect all cleanup candidates for a mode."""
    candidates: list[RemovalCandidate] = []
    candidates.extend(collect_root_dir_candidates(project_root, mode))
    candidates.extend(collect_root_file_candidates(project_root))
    candidates.extend(collect_recursive_cache_candidates(project_root))
    clean_candidates: list[RemovalCandidate] = []
    for candidate in candidates:
        if is_protected_root_child(project_root, candidate.path):
            continue
        clean_candidates.append(candidate)
    return prune_children_of_removed_dirs(clean_candidates)


def _ignore_missing_rmtree_error(function, path, exc_info) -> None:
    """Ignore files that disappear while deleting a generated directory.

    Finder, unzip, rsync, and AppleDouble cleanup can make ``._*`` files appear or
    disappear while ``shutil.rmtree`` is walking a directory.  Missing files should
    never make ``make clean`` fail.  Permission errors and other real problems are
    still raised.
    """
    error = exc_info[1]
    if isinstance(error, FileNotFoundError):
        return
    raise error


def remove_candidate(candidate: RemovalCandidate) -> None:
    """Remove a file, symlink, or directory candidate safely and idempotently."""
    path = candidate.path
    if not path.exists() and not path.is_symlink():
        return
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, onerror=_ignore_missing_rmtree_error)
        else:
            path.unlink(missing_ok=True)
    except FileNotFoundError:
        return


def write_cleanup_report(project_root: Path, candidates: Sequence[RemovalCandidate], apply: bool) -> None:
    """Print a human-readable cleanup report."""
    action = "REMOVED" if apply else "WOULD_REMOVE"
    print(f"Project root: {project_root}")
    print(f"Mode: {'apply' if apply else 'dry-run'}")
    print(f"Candidates: {len(candidates)}")
    for candidate in candidates:
        try:
            rel = candidate.path.relative_to(project_root)
        except ValueError:
            rel = candidate.path
        print(f"{action}: {rel}  # {candidate.reason}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean generated Aaron Sound Sorter project junk.")
    parser.add_argument("--project-root", default=".", help="Project root. Defaults to current directory.")
    parser.add_argument(
        "--mode",
        choices=["standard", "bundle", "heavy-local"],
        default="standard",
        help="Cleanup mode. bundle currently matches standard; heavy-local also removes generated training selection folders.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually remove files. Without this flag, dry-run only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    if not (project_root / "Aaron_Sound_Sorter.py").exists():
        print(f"ERROR: not an Aaron_Sound_Sorter project root: {project_root}")
        return 2
    candidates = collect_candidates(project_root, args.mode)
    write_cleanup_report(project_root, candidates, args.apply)
    if args.apply:
        for candidate in candidates:
            remove_candidate(candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
