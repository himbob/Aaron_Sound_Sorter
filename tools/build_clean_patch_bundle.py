#!/usr/bin/env python3
"""Build a clean Aaron Sound Sorter patch bundle.

The bundle builder is intentionally conservative. It packages source files that
were changed for an AI handoff, writes a no-backup installer, rejects generated
junk, and writes the finished ZIP under a reports subfolder instead of the
project root.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PATCH_ALLOWED_TOP_LEVEL_DIRECTORIES = {
    "src",
    "tests",
    "tools",
    "docs",
    "commands",
    "neural_artifacts",
}
PATCH_PERSISTENT_EVIDENCE_DIRECTORIES = {"neural_artifacts"}
PATCH_ALLOWED_ROOT_FILES = {
    ".gitignore",
    "AGENTS.md",
    "AI_READ_THIS_FIRST.md",
    "CURRENT_STATUS.md",
    "Makefile",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-quality.txt",
    "requirements-dev.txt",
}
PATCH_ALLOWED_ROOT_PREFIXES = ("AI_STATUS_",)
PATCH_BLOCKED_TOP_LEVEL_DIRECTORIES = {".git", ".venv", ".venv_phase4", "_reports", "build", "dist", "node_modules"}
PATCH_BLOCKED_ANYWHERE_DIRECTORIES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
PATCH_BLOCKED_FILE_NAMES = {".DS_Store", ".coverage"}
PATCH_BLOCKED_SUFFIXES = {
    ".aif",
    ".aiff",
    ".au",
    ".dylib",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".pyo",
    ".pyc",
    ".so",
    ".wav",
    ".zip",
}
DEFAULT_MAX_FILE_BYTES = 5_000_000


@dataclass(frozen=True)
class BundleRequest:
    """User request for one clean patch bundle.

    Args:
        project_root: Root of the project checkout to package from.
        base_ref: Git reference used to discover changed files when
            ``AI_CHANGED_FILES`` is not supplied.
        bundle_name: Human-readable bundle folder and ZIP stem.
        output_dir: Directory that receives the final ZIP.
        dry_run: Whether to print the plan without writing the ZIP.
        max_file_bytes: Maximum size for one packaged file.

    Attributes:
        project_root: Root of the project checkout to package from.
        base_ref: Git reference used to discover changed files.
        bundle_name: Bundle folder and ZIP stem.
        output_dir: Directory that receives the final ZIP.
        dry_run: Whether to print the plan only.
        max_file_bytes: Maximum size for one packaged file.
    """

    project_root: Path
    base_ref: str
    bundle_name: str
    output_dir: Path
    dry_run: bool
    max_file_bytes: int


@dataclass(frozen=True)
class BundleResult:
    """Result from building a clean patch bundle.

    Args:
        zip_path: Final ZIP path, or the path that would be written in dry-run
            mode.
        staged_file_count: Number of source files selected for packaging.
        selected_paths: Relative project paths staged into ``files/``.

    Attributes:
        zip_path: Final ZIP path, or dry-run destination path.
        staged_file_count: Number of selected source files.
        selected_paths: Relative project paths staged into the bundle.
    """

    zip_path: Path
    staged_file_count: int
    selected_paths: tuple[Path, ...]


def normalize_bundle_name(raw_name: str) -> str:
    """Return a filesystem-safe bundle name.

    Args:
        raw_name: Requested bundle name.

    Returns:
        Safe bundle name using only letters, digits, dashes, underscores, and
        periods.

    Side Effects:
        None.

    Raises:
        ValueError: Raised when the resulting name is empty.
    """
    cleaned = "".join(char if char.isalnum() or char in "-_." else "_" for char in raw_name.strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("Bundle name cannot be empty")
    return cleaned


def run_git_lines(project_root: Path, arguments: Sequence[str]) -> list[str]:
    """Run a Git command and return stdout lines.

    Args:
        project_root: Project checkout root.
        arguments: Arguments to pass after ``git -C <project_root>``.

    Returns:
        Non-empty output lines. An empty list is returned when Git is not
        available or the directory is not a repository.

    Side Effects:
        Spawns one Git subprocess.

    Raises:
        No intentional exceptions.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def normalize_relative_path(project_root: Path, raw_path: str) -> Path | None:
    """Normalize a user or Git supplied path.

    Args:
        project_root: Project checkout root.
        raw_path: Raw relative or absolute path.

    Returns:
        Relative project path, or ``None`` when the input is empty or escapes
        the project root.

    Side Effects:
        None.

    Raises:
        No intentional exceptions.
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
    if ".." in candidate_path.parts:
        return None
    return Path(*candidate_path.parts)


def read_environment_changed_files(project_root: Path) -> list[Path]:
    """Read explicit changed-file paths from ``AI_CHANGED_FILES``.

    Args:
        project_root: Project checkout root.

    Returns:
        Existing relative file paths supplied through the environment.

    Side Effects:
        Reads the process environment.

    Raises:
        No intentional exceptions.
    """
    raw_value = os.environ.get("AI_CHANGED_FILES", "")
    if not raw_value.strip():
        return []
    paths: list[Path] = []
    for raw_path in raw_value.replace(",", "\n").splitlines():
        relative_path = normalize_relative_path(project_root, raw_path)
        if relative_path is None:
            continue
        if (project_root / relative_path).is_file():
            paths.append(relative_path)
    return sorted(set(paths), key=lambda path: path.as_posix())


def discover_persistent_evidence_files(project_root: Path) -> list[Path]:
    """Return local evidence explicitly included even when Git ignores it."""
    evidence_paths: list[Path] = []
    for directory_name in PATCH_PERSISTENT_EVIDENCE_DIRECTORIES:
        evidence_root = project_root / directory_name
        if not evidence_root.is_dir():
            continue
        evidence_paths.extend(path.relative_to(project_root) for path in evidence_root.rglob("*") if path.is_file())
    return sorted(set(evidence_paths), key=lambda path: path.as_posix())


def discover_changed_files(project_root: Path, base_ref: str) -> tuple[Path, ...]:
    """Discover changed files for bundle packaging.

    Args:
        project_root: Project checkout root.
        base_ref: Git reference used when ``AI_CHANGED_FILES`` is not supplied.

    Returns:
        Existing relative file paths selected from explicit input or Git.

    Side Effects:
        May run Git subprocesses and reads ``AI_CHANGED_FILES``.

    Raises:
        No intentional exceptions.
    """
    explicit_paths = read_environment_changed_files(project_root)
    evidence_paths = discover_persistent_evidence_files(project_root)
    if explicit_paths:
        return tuple(sorted(set([*explicit_paths, *evidence_paths]), key=lambda path: path.as_posix()))

    raw_paths: list[Path] = []
    changed_lines = run_git_lines(project_root, ["diff", "--name-only", "--diff-filter=ACMR", base_ref, "--"])
    untracked_lines = run_git_lines(project_root, ["ls-files", "--others", "--exclude-standard"])
    for raw_path in [*changed_lines, *untracked_lines]:
        relative_path = normalize_relative_path(project_root, raw_path)
        if relative_path is None:
            continue
        if (project_root / relative_path).is_file():
            raw_paths.append(relative_path)
    return tuple(sorted(set([*raw_paths, *evidence_paths]), key=lambda path: path.as_posix()))


def is_generated_or_blocked_path(relative_path: Path) -> bool:
    """Return whether a path is generated junk or unsafe for a patch bundle.

    Args:
        relative_path: Project-relative path.

    Returns:
        ``True`` when the path is cache output, macOS metadata, compiled code,
        report output, a virtualenv, or other unsafe bundle content.

    Side Effects:
        None.

    Raises:
        No intentional exceptions.
    """
    if relative_path.parts and relative_path.parts[0] in PATCH_BLOCKED_TOP_LEVEL_DIRECTORIES:
        return True
    if any(part in PATCH_BLOCKED_ANYWHERE_DIRECTORIES for part in relative_path.parts):
        return True
    if any(part.endswith(".egg-info") for part in relative_path.parts):
        return True
    if relative_path.name in PATCH_BLOCKED_FILE_NAMES:
        return True
    if relative_path.name.startswith("._"):
        return True
    return relative_path.suffix.lower() in PATCH_BLOCKED_SUFFIXES


def is_allowed_patch_file(relative_path: Path) -> bool:
    """Return whether a file is allowed in a clean patch bundle.

    Args:
        relative_path: Project-relative path.

    Returns:
        ``True`` when the path lives in an allowed source, test, tool, command,
        documentation, or persistent neural-evidence location.

    Side Effects:
        None.

    Raises:
        No intentional exceptions.
    """
    if is_generated_or_blocked_path(relative_path):
        return False
    if len(relative_path.parts) == 1:
        name = relative_path.name
        return name in PATCH_ALLOWED_ROOT_FILES or name.startswith(PATCH_ALLOWED_ROOT_PREFIXES)
    return relative_path.parts[0] in PATCH_ALLOWED_TOP_LEVEL_DIRECTORIES


def select_bundle_files(request: BundleRequest) -> tuple[Path, ...]:
    """Select clean changed files for the patch bundle.

    Args:
        request: Bundle request configuration.

    Returns:
        Relative paths that should be staged into the bundle.

    Side Effects:
        Reads file metadata.

    Raises:
        SystemExit: Raised when no files are selected or an unsafe changed file
        is detected.
    """
    changed_files = discover_changed_files(request.project_root, request.base_ref)
    if not changed_files:
        raise SystemExit(
            "No changed files detected. Commit changes, set AI_CHANGED_FILES, or pass a different AI_BASE before make bundle."
        )

    selected_paths: list[Path] = []
    rejected_paths: list[str] = []
    oversized_paths: list[str] = []
    for relative_path in changed_files:
        absolute_path = request.project_root / relative_path
        if not absolute_path.is_file():
            continue
        if not is_allowed_patch_file(relative_path):
            rejected_paths.append(relative_path.as_posix())
            continue
        file_size = absolute_path.stat().st_size
        if file_size > request.max_file_bytes:
            oversized_paths.append(f"{relative_path.as_posix()} ({file_size} bytes)")
            continue
        selected_paths.append(relative_path)

    if rejected_paths:
        print("Refusing to bundle unsafe or generated paths:")
        for relative_path in rejected_paths:
            print(f"  {relative_path}")
        raise SystemExit(1)
    if oversized_paths:
        print("Refusing to bundle oversized files:")
        for relative_path in oversized_paths:
            print(f"  {relative_path}")
        raise SystemExit(1)
    if not selected_paths:
        raise SystemExit("No allowed source files were selected for the patch bundle.")
    return tuple(sorted(selected_paths, key=lambda path: path.as_posix()))


def scan_for_staged_junk(staged_root: Path) -> list[Path]:
    """Find forbidden files inside a staged bundle root.

    Args:
        staged_root: Root folder that will be zipped.

    Returns:
        Relative paths that should not be present in the staged bundle.

    Side Effects:
        Reads the staged filesystem.

    Raises:
        No intentional exceptions.
    """
    junk_paths: list[Path] = []
    for current_path in staged_root.rglob("*"):
        try:
            relative_path = current_path.relative_to(staged_root)
        except ValueError:
            continue
        if not relative_path.parts:
            continue
        if is_generated_or_blocked_path(relative_path):
            junk_paths.append(relative_path)
    return sorted(junk_paths, key=lambda path: path.as_posix())


def copy_selected_files(project_root: Path, staged_files_root: Path, selected_paths: Sequence[Path]) -> None:
    """Copy selected project files into the staged ``files`` folder.

    Args:
        project_root: Project checkout root.
        staged_files_root: Destination ``files`` directory inside the staged bundle.
        selected_paths: Relative paths to copy.

    Returns:
        None.

    Side Effects:
        Creates directories and copies files.

    Raises:
        OSError: Raised by the filesystem when a copy fails.
    """
    for relative_path in selected_paths:
        source_path = project_root / relative_path
        destination_path = staged_files_root / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def write_installer(staged_root: Path, bundle_name: str) -> None:
    """Write the no-backup installer script for the bundle.

    Args:
        staged_root: Root directory of the staged bundle.
        bundle_name: Bundle name used in status output.

    Returns:
        None.

    Side Effects:
        Writes ``INSTALL_NO_BACKUP.command`` and marks it executable.
    """
    installer_path = staged_root / "INSTALL_NO_BACKUP.command"
    installer_text = f"""#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${{PROJECT_ROOT:-/Volumes/T9/testbed/Aaron_Sound_Sorter}}"
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
FILES_DIR="$SCRIPT_DIR/files"

if [ ! -d "$PROJECT_ROOT" ]; then
  echo "Project root does not exist: $PROJECT_ROOT" >&2
  exit 2
fi
if [ ! -d "$FILES_DIR" ]; then
  echo "Bundle files directory missing: $FILES_DIR" >&2
  exit 2
fi

find "$FILES_DIR" -name '._*' -type f -delete
find "$FILES_DIR" -name '.DS_Store' -type f -delete
find "$FILES_DIR" -name '__pycache__' -type d -prune -exec rm -rf {{}} +
find "$FILES_DIR" -name '.pytest_cache' -type d -prune -exec rm -rf {{}} +
find "$FILES_DIR" -name '.mypy_cache' -type d -prune -exec rm -rf {{}} +
find "$FILES_DIR" -name '.ruff_cache' -type d -prune -exec rm -rf {{}} +
find "$FILES_DIR" -name '*.pyc' -type f -delete

cd "$FILES_DIR"
while IFS= read -r -d '' source_file; do
  relative_path="${{source_file#./}}"
  target_file="$PROJECT_ROOT/$relative_path"
  mkdir -p "$(dirname "$target_file")"
  cp "$source_file" "$target_file"
  if [[ "$target_file" == *.command ]]; then
    chmod +x "$target_file"
  fi
  echo "installed $relative_path"
done < <(find . -type f -print0 | LC_ALL=C sort -z)

echo "Installed {bundle_name} into $PROJECT_ROOT"
"""
    installer_path.write_text(installer_text, encoding="utf-8")
    installer_path.chmod(0o755)


def write_manifest(staged_root: Path, request: BundleRequest, selected_paths: Sequence[Path]) -> None:
    """Write a human-readable manifest into the staged bundle.

    Args:
        staged_root: Root directory of the staged bundle.
        request: Bundle request configuration.
        selected_paths: Relative project files included in the bundle.

    Returns:
        None.

    Side Effects:
        Writes ``AI_BUNDLE_MANIFEST.md``.
    """
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {request.bundle_name}",
        "",
        f"Generated: {generated_at}",
        f"Platform: {platform.system()} {platform.release()}",
        f"Project root: `{request.project_root}`",
        f"Base ref: `{request.base_ref}`",
        "",
        "## Included files",
        "",
    ]
    for relative_path in selected_paths:
        lines.append(f"- `{relative_path.as_posix()}`")
    lines.extend(
        [
            "",
            "## Install",
            "",
            "```bash",
            "chmod +x INSTALL_NO_BACKUP.command",
            'PROJECT_ROOT="/Volumes/T9/testbed/Aaron_Sound_Sorter" ./INSTALL_NO_BACKUP.command',
            "```",
            "",
            "## Hygiene policy",
            "",
            "This bundle was built after `make clean-for-bundle` and rejects caches, compiled Python, reports,",
            "virtual environments, macOS metadata, audio files, and ZIP archives from the staged payload.",
        ]
    )
    (staged_root / "AI_BUNDLE_MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def zip_directory(staged_root: Path, zip_path: Path) -> None:
    """Zip one staged bundle directory.

    Args:
        staged_root: Directory to zip.
        zip_path: Destination ZIP path.

    Returns:
        None.

    Side Effects:
        Writes the ZIP file.

    Raises:
        OSError: Raised when the destination cannot be written.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for current_path in sorted(staged_root.rglob("*"), key=lambda path: path.relative_to(staged_root).as_posix()):
            archive.write(current_path, staged_root.name / current_path.relative_to(staged_root))


def build_bundle(request: BundleRequest) -> BundleResult:
    """Build or preview one clean patch bundle.

    Args:
        request: Bundle build request.

    Returns:
        Bundle build result.

    Side Effects:
        Copies selected files into a temporary staging directory and writes a ZIP
        unless ``request.dry_run`` is true.

    Raises:
        SystemExit: Raised when selected files are unsafe or staging contains junk.
    """
    selected_paths = select_bundle_files(request)
    safe_bundle_name = normalize_bundle_name(request.bundle_name)
    zip_path = request.output_dir / f"{safe_bundle_name}.zip"

    print("Selected bundle files:")
    for relative_path in selected_paths:
        print(f"  {relative_path.as_posix()}")

    if request.dry_run:
        print(f"Dry run: would write {zip_path}")
        return BundleResult(zip_path=zip_path, staged_file_count=len(selected_paths), selected_paths=selected_paths)

    request.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aaron_bundle_stage_") as temporary_directory:
        staged_parent = Path(temporary_directory)
        staged_root = staged_parent / safe_bundle_name
        staged_files_root = staged_root / "files"
        staged_files_root.mkdir(parents=True, exist_ok=True)
        copy_selected_files(request.project_root, staged_files_root, selected_paths)
        write_installer(staged_root, safe_bundle_name)
        write_manifest(staged_root, request, selected_paths)
        junk_paths = scan_for_staged_junk(staged_root)
        if junk_paths:
            print("Refusing to zip staged bundle because junk was found:")
            for relative_path in junk_paths:
                print(f"  {relative_path.as_posix()}")
            raise SystemExit(1)
        zip_directory(staged_root, zip_path)

    print(f"Wrote bundle: {zip_path}")
    return BundleResult(zip_path=zip_path, staged_file_count=len(selected_paths), selected_paths=selected_paths)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser.

    Args:
        None.

    Returns:
        Configured argument parser.

    Side Effects:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", "."), help="Project checkout root")
    parser.add_argument("--base", default=os.environ.get("AI_BASE", "HEAD"), help="Git diff base")
    parser.add_argument(
        "--bundle-name",
        default=os.environ.get("BUNDLE_NAME", "Aaron_Sound_Sorter_AI_patch"),
        help="Bundle folder and ZIP stem",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("BUNDLE_OUTPUT_DIR", "_reports/bundles"),
        help="Output directory for the final ZIP",
    )
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES, help="Maximum staged file size")
    parser.add_argument("--dry-run", action="store_true", help="Print selected files without writing a ZIP")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Run the bundle builder entry point.

    Args:
        argv: Optional command-line arguments. ``None`` uses ``sys.argv``.

    Returns:
        Process exit code.

    Side Effects:
        Writes a bundle ZIP unless ``--dry-run`` is supplied.
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    project_root = Path(args.project_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    request = BundleRequest(
        project_root=project_root,
        base_ref=str(args.base),
        bundle_name=normalize_bundle_name(str(args.bundle_name)),
        output_dir=output_dir.resolve(),
        dry_run=bool(args.dry_run),
        max_file_bytes=int(args.max_file_bytes),
    )
    build_bundle(request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
