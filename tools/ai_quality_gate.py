#!/usr/bin/env python3
"""Run AI changed-files quality gates for Aaron Sound Sorter.

This tool exists because the current repository has legacy lint, format, type,
and docstring debt.  A whole-repo strict gate is useful as a baseline report,
but it is too noisy to use as a blocker for every small AI patch.  This command
implements a ratchet policy: AI-touched files must be clean, generated junk must
not be present, and source-name sorting audits still run globally.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

JUNK_DIRECTORY_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", ".venv_py313"}
JUNK_FILE_NAMES = {".DS_Store", ".coverage"}
BUNDLE_FORBIDDEN_TOP_LEVEL = {"_reports"}
CHECKABLE_PYTHON_ROOTS = {"src", "tests", "tools"}
PROJECT_PYTHON_CHECK_ROOTS = ("src", "tests", "tools")
SOURCE_PACKAGE_PREFIX = Path("src/aaron_sound_sorter")


@dataclass(frozen=True)
class CommandStatus:
    """Process result for a quality command.

    Args:
        name: Human-readable command name.
        command: Executed command arguments.
        exit_code: Process exit code.

    Attributes:
        name: Human-readable command name.
        command: Executed command arguments.
        exit_code: Process exit code.
    """

    name: str
    command: tuple[str, ...]
    exit_code: int


@dataclass(frozen=True)
class ChangedFiles:
    """Changed file groups used by the ratchet gate.

    Args:
        all_paths: All changed, added, or untracked paths that still exist.
        python_paths: Changed Python files eligible for syntax/lint/format checks.
        source_python_paths: Changed package files under ``src/aaron_sound_sorter``.

    Attributes:
        all_paths: Changed paths relative to the project root.
        python_paths: Changed Python paths relative to the project root.
        source_python_paths: Changed source package paths relative to the project root.
    """

    all_paths: tuple[Path, ...]
    python_paths: tuple[Path, ...]
    source_python_paths: tuple[Path, ...]


def normalize_relative_path(project_root: Path, raw_path: str) -> Path | None:
    """Normalize one changed-file path while preventing path escape.

    Args:
        project_root: Project checkout root.
        raw_path: Raw relative path from git or environment input.

    Returns:
        Normalized relative path, or ``None`` when the input is empty or escapes
        the project root.

    Side Effects:
        None.

    Raises:
        No intentional exceptions.

    Important Constraints:
        This function only accepts paths inside the project.  It must not allow
        ``..`` escapes from a shell-provided changed-file list.
    """
    stripped_path = raw_path.strip()
    if not stripped_path:
        return None
    candidate_path = Path(stripped_path)
    if candidate_path.is_absolute():
        try:
            return candidate_path.resolve().relative_to(project_root)
        except ValueError:
            return None
    parts = candidate_path.parts
    if ".." in parts:
        return None
    return Path(*parts) if parts else None


def run_command(command: Sequence[str], project_root: Path, *, check: bool = False) -> CommandStatus:
    """Run one command with visible output.

    Args:
        command: Command arguments to run.
        project_root: Working directory for the process.
        check: Whether to raise ``SystemExit`` when the command fails.

    Returns:
        Command status containing the exit code.

    Side Effects:
        Spawns a subprocess and streams command output to the terminal.

    Raises:
        SystemExit: Raised when ``check`` is true and the command exits non-zero.

    Important Constraints:
        Commands must be source-blind quality checks.  They must not train the
        brain or sort user libraries.
    """
    printable = " ".join(command)
    print(f"\n$ {printable}")
    completed = subprocess.run(list(command), cwd=str(project_root), check=False)
    status = CommandStatus(name=command[0], command=tuple(command), exit_code=int(completed.returncode))
    if check and completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return status


def git_command(project_root: Path, args: Sequence[str]) -> list[str]:
    """Run a git command and return output lines.

    Args:
        project_root: Project checkout root.
        args: Arguments following ``git -C <root>``.

    Returns:
        Output lines, or an empty list when git is unavailable or the directory
        is not a repository.

    Side Effects:
        Spawns a git subprocess.

    Raises:
        No intentional exceptions.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *args],
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def environment_changed_files(project_root: Path) -> list[Path]:
    """Read explicitly supplied changed files from ``AI_CHANGED_FILES``.

    Args:
        project_root: Project checkout root.

    Returns:
        Existing changed paths relative to the project root.

    Side Effects:
        Reads the process environment.

    Raises:
        No intentional exceptions.
    """
    raw_value = os.environ.get("AI_CHANGED_FILES", "")
    if not raw_value.strip():
        return []
    normalized_paths: list[Path] = []
    for raw_path in raw_value.replace(",", "\n").splitlines():
        relative_path = normalize_relative_path(project_root, raw_path)
        if relative_path is None:
            continue
        if (project_root / relative_path).exists():
            normalized_paths.append(relative_path)
    return sorted(set(normalized_paths))


def discover_changed_files(project_root: Path, base_ref: str) -> ChangedFiles:
    """Discover changed files for the AI ratchet gate.

    Args:
        project_root: Project checkout root.
        base_ref: Git reference used as the diff base.

    Returns:
        Grouped changed-file information.

    Side Effects:
        May run git subprocesses and reads ``AI_CHANGED_FILES``.

    Raises:
        No intentional exceptions.

    Important Constraints:
        Deleted files are excluded because lint/type/doc checks need existing
        filesystem paths.
    """
    explicit_paths = environment_changed_files(project_root)
    raw_paths: list[Path] = list(explicit_paths)
    if not raw_paths:
        diff_lines = git_command(project_root, ["diff", "--name-only", "--diff-filter=ACMR", base_ref, "--"])
        untracked_lines = git_command(project_root, ["ls-files", "--others", "--exclude-standard"])
        for raw_path in [*diff_lines, *untracked_lines]:
            relative_path = normalize_relative_path(project_root, raw_path)
            if relative_path is None:
                continue
            if (project_root / relative_path).exists():
                raw_paths.append(relative_path)

    unique_paths = tuple(sorted(set(raw_paths), key=lambda path: path.as_posix()))
    python_paths = tuple(path for path in unique_paths if is_checkable_python_file(path))
    source_paths = tuple(
        path for path in python_paths if path.as_posix().startswith(SOURCE_PACKAGE_PREFIX.as_posix() + "/")
    )
    return ChangedFiles(all_paths=unique_paths, python_paths=python_paths, source_python_paths=source_paths)


def is_checkable_python_file(relative_path: Path) -> bool:
    """Return whether a changed file should receive Python quality checks.

    Args:
        relative_path: Path relative to the project root.

    Returns:
        ``True`` for project Python files that are not generated junk.

    Side Effects:
        None.

    Raises:
        No intentional exceptions.
    """
    if relative_path.suffix != ".py":
        return False
    if is_junk_relative_path(relative_path):
        return False
    if relative_path.name.startswith("._"):
        return False
    first_part = relative_path.parts[0] if relative_path.parts else ""
    return first_part in CHECKABLE_PYTHON_ROOTS or relative_path.name == "Aaron_Sound_Sorter.py"


def is_junk_relative_path(relative_path: Path, *, forbid_reports: bool = False) -> bool:
    """Return whether a relative path is generated junk.

    Args:
        relative_path: Path relative to a project or bundle root.
        forbid_reports: Whether ``_reports`` is considered invalid.

    Returns:
        ``True`` when the path is a cache, compiled file, macOS metadata file,
        coverage output, or optionally a report folder.

    Side Effects:
        None.

    Raises:
        No intentional exceptions.
    """
    parts = relative_path.parts
    if any(part in JUNK_DIRECTORY_NAMES for part in parts):
        return True
    if forbid_reports and parts and parts[0] in BUNDLE_FORBIDDEN_TOP_LEVEL:
        return True
    if relative_path.name in JUNK_FILE_NAMES:
        return True
    if relative_path.name.startswith("._"):
        return True
    return relative_path.suffix == ".pyc"


def scan_junk(root: Path, *, forbid_reports: bool = False) -> list[Path]:
    """Find generated files and directories that should not be committed or bundled.

    Args:
        root: Root directory to scan.
        forbid_reports: Whether report folders are invalid for this scan.

    Returns:
        Sorted relative paths that match junk rules.

    Side Effects:
        Reads the filesystem.

    Raises:
        No intentional exceptions.

    Important Constraints:
        The scan ignores ``.git`` and ``.venv_phase4`` so local tooling does not
        make normal development fail.
    """
    matches: set[Path] = set()
    if not root.exists():
        return []
    for current_path in root.rglob("*"):
        try:
            relative_path = current_path.relative_to(root)
        except ValueError:
            continue
        if not relative_path.parts:
            continue
        if relative_path.parts[0] in {".git", ".venv_phase4"}:
            continue
        if relative_path.parts[0] == "_reports" and not forbid_reports:
            continue
        if is_junk_relative_path(relative_path, forbid_reports=forbid_reports):
            matches.add(relative_path)
    return sorted(matches, key=lambda path: path.as_posix())


def print_changed_files(changed_files: ChangedFiles) -> None:
    """Print changed files in a human-readable form.

    Args:
        changed_files: Grouped changed-file data.

    Returns:
        None.

    Side Effects:
        Writes to stdout.
    """
    if not changed_files.all_paths:
        print("No changed files detected. Set AI_CHANGED_FILES to force a file list when working outside git.")
        return
    print("Changed files:")
    for relative_path in changed_files.all_paths:
        print(f"  {relative_path.as_posix()}")
    if changed_files.python_paths:
        print("Changed Python files:")
        for relative_path in changed_files.python_paths:
            print(f"  {relative_path.as_posix()}")


def fail_if_junk_exists(root: Path, *, forbid_reports: bool = False) -> None:
    """Fail when generated junk exists under a root.

    Args:
        root: Directory to scan.
        forbid_reports: Whether report folders are invalid for this scan.

    Returns:
        None.

    Side Effects:
        Writes diagnostics to stdout.

    Raises:
        SystemExit: Raised with exit code 1 when junk is found.
    """
    junk_paths = scan_junk(root, forbid_reports=forbid_reports)
    if not junk_paths:
        print("Generated-junk scan: PASS")
        return
    print("Generated-junk scan: FAIL")
    for relative_path in junk_paths[:200]:
        print(f"  {relative_path.as_posix()}")
    if len(junk_paths) > 200:
        print(f"  ... {len(junk_paths) - 200} more")
    print("Run: make clean-generated")
    raise SystemExit(1)


def run_preflight(project_root: Path, base_ref: str) -> int:
    """Run the AI preflight checks.

    Args:
        project_root: Project checkout root.
        base_ref: Git diff base.

    Returns:
        Process exit code.

    Side Effects:
        Prints environment, changed-file, and generated-junk status.

    Raises:
        No intentional exceptions.
    """
    print(f"Project root: {project_root}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]} at {sys.executable}")
    changed_files = discover_changed_files(project_root, base_ref)
    print_changed_files(changed_files)
    fail_if_junk_exists(project_root, forbid_reports=False)
    return 0


def run_fix(project_root: Path, base_ref: str) -> int:
    """Apply safe Ruff fixes to changed Python files only.

    Args:
        project_root: Project checkout root.
        base_ref: Git diff base.

    Returns:
        Process exit code.

    Side Effects:
        May edit changed Python files using Ruff safe fixes and formatting.
    """
    changed_files = discover_changed_files(project_root, base_ref)
    if not changed_files.python_paths:
        print("No changed Python files to fix.")
        return 0
    python_args = [path.as_posix() for path in changed_files.python_paths]
    run_command([sys.executable, "-m", "ruff", "check", "--fix", *python_args], project_root, check=True)
    run_command([sys.executable, "-m", "ruff", "format", *python_args], project_root, check=True)
    return 0


def run_check(project_root: Path, base_ref: str) -> int:
    """Run the strict AI project quality gate.

    Args:
        project_root: Project checkout root.
        base_ref: Git diff base.  It is still printed for context, but this
            command now checks the maintained project Python tree, not only
            changed files.

    Returns:
        Process exit code.

    Side Effects:
        Runs whole-project py_compile, Ruff lint, Ruff format check, generated
        junk scan, and the no-source-name sorting audit.

    Raises:
        SystemExit: Raised when any required check fails.
    """
    changed_files = discover_changed_files(project_root, base_ref)
    print_changed_files(changed_files)
    fail_if_junk_exists(project_root, forbid_reports=False)

    existing_roots = [root for root in PROJECT_PYTHON_CHECK_ROOTS if (project_root / root).exists()]
    compile_roots = [*existing_roots]
    if (project_root / "Aaron_Sound_Sorter.py").exists():
        compile_roots.append("Aaron_Sound_Sorter.py")

    run_command([sys.executable, "-m", "compileall", "-q", *compile_roots], project_root, check=True)
    run_command([sys.executable, "-m", "ruff", "check", *existing_roots], project_root, check=True)
    run_command([sys.executable, "-m", "ruff", "format", "--check", *existing_roots], project_root, check=True)

    audit_script = project_root / "tools" / "audit_no_source_name_sorting.py"
    if audit_script.exists():
        run_command([sys.executable, str(audit_script), "--project-root", str(project_root)], project_root, check=True)
    else:
        print("Skipping source-name audit because tools/audit_no_source_name_sorting.py is missing.")
    return 0


def run_bundle_check(bundle_root: Path) -> int:
    """Validate an AI handoff bundle root.

    Args:
        bundle_root: Extracted or staged bundle root to scan.

    Returns:
        Process exit code.

    Side Effects:
        Prints bundle hygiene status.

    Raises:
        SystemExit: Raised when bundle junk is found or the directory is missing.
    """
    if not bundle_root.exists() or not bundle_root.is_dir():
        print(f"Bundle directory does not exist: {bundle_root}")
        return 2
    fail_if_junk_exists(bundle_root, forbid_reports=True)
    print("Bundle hygiene: PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Args:
        None.

    Returns:
        Configured argument parser.

    Side Effects:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Project checkout root")
    parser.add_argument("--base", default="HEAD", help="Git diff base for changed-file discovery")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", help="Show changed files and fail on generated junk")
    subparsers.add_parser("list", help="List changed files")
    subparsers.add_parser("fix", help="Apply safe Ruff fixes to changed Python files")
    subparsers.add_parser("check", help="Run the strict changed-files AI gate")
    bundle_parser = subparsers.add_parser("bundle-check", help="Validate staged bundle hygiene")
    bundle_parser.add_argument("--bundle-dir", required=True, help="Bundle root directory to scan")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Run the command-line entry point.

    Args:
        argv: Optional CLI argument iterable. ``None`` uses ``sys.argv``.

    Returns:
        Process exit code.

    Side Effects:
        Runs requested quality commands and prints status.
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    project_root = Path(args.project_root).expanduser().resolve()

    if args.command == "preflight":
        return run_preflight(project_root, str(args.base))
    if args.command == "list":
        print_changed_files(discover_changed_files(project_root, str(args.base)))
        return 0
    if args.command == "fix":
        return run_fix(project_root, str(args.base))
    if args.command == "check":
        return run_check(project_root, str(args.base))
    if args.command == "bundle-check":
        return run_bundle_check(Path(args.bundle_dir).expanduser().resolve())
    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
