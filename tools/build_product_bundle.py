#!/usr/bin/env python3
"""Build clean Aaron Sound Sorter code and runtime-asset bundles.

The builder supports three payload modes:

``code``
    Maintained application source, commands, tools, configuration, current
    documentation, and Python test code without test audio.
``ai-runtime``
    Active trained brains, pinned neural model snapshots, and compact neural
    prototype indexes without application source.
``product``
    The combined code and AI-runtime payload retained for compatibility.

Generated reports, virtual environments, Git history, training audio, test
audio, caches, backups, and archived runtime artifacts are excluded from every
mode.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PRODUCT_SOURCE_DIRECTORIES = (
    "src",
    "commands",
    "config",
    "docs",
    "tools",
    "tests",
)

PRODUCT_ROOT_FILES = (
    ".gitattributes",
    ".gitignore",
    "AGENTS.md",
    "AI_READ_THIS_FIRST.md",
    "Aaron_Sound_Sorter.py",
    "CURRENT_STATUS.md",
    "LICENSE",
    "Makefile",
    "NOTICE.md",
    "README.md",
    "pyproject.toml",
    "requirements-dev.txt",
    "requirements-quality.txt",
    "requirements.txt",
)

ASSET_MANIFEST_PATH = Path("config/product_bundle_assets.txt")

EXCLUDED_TOP_LEVEL_NAMES = {
    ".git",
    ".github",
    ".venv",
    ".venv_phase4",
    ".venv_neural",
    ".vscode",
    "_models",
    "_reports",
    "_real_sort_tests",
    "_pytest_outputs",
    "_pytest_regression_outputs",
    "build",
    "coverage_ledgers",
    "dist",
    "neural_artifacts",
    "outputs",
    "reports",
    "stage4_brain_family_training",
    "stage4_folder_brain_training",
    "training",
}

EXCLUDED_ANYWHERE_DIRECTORY_NAMES = {
    ".cache",
    ".git",
    ".github",
    ".locks",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "archive",
    "archives",
    "backup",
    "backups",
    "cache",
    "logs",
    "reports",
    "run",
    "runs",
    "temp",
    "tmp",
}

EXCLUDED_RELATIVE_PREFIXES = (
    Path("commands/legacy_root_commands"),
    Path("docs/archive"),
    Path("docs/archived_bundle_handoffs"),
    Path("docs/legacy_docs"),
)

EXCLUDED_FILE_NAMES = {
    ".DS_Store",
    ".coverage",
}

EXCLUDED_FILE_SUFFIXES = {
    ".aif",
    ".aiff",
    ".au",
    ".flac",
    ".log",
    ".m4a",
    ".mp3",
    ".ogg",
    ".pyc",
    ".pyo",
    ".tmp",
    ".wav",
    ".zip",
}

BACKUP_NAME_MARKERS = (
    ".bak",
    ".backup",
    "_backup",
    "-backup",
    "_archived",
    "-archived",
    "_old",
    "-old",
    "_previous",
    "-previous",
)


@dataclass(frozen=True)
class RuntimeAsset:
    """One explicitly declared runtime asset.

    Attributes:
        kind: Manifest kind: required-file, optional-file, model-dir, or
            artifact-dir.
        relative_path: Project-relative source path.
    """

    kind: str
    relative_path: Path


@dataclass(frozen=True)
class ProductBundleRequest:
    """Configuration for one bundle build."""

    project_root: Path
    bundle_name: str
    output_dir: Path
    dry_run: bool
    include_neural_models: bool
    bundle_mode: str = "product"
    compression_level: int = 1
    payload_root_name: str | None = None


@dataclass(frozen=True)
class SelectedFile:
    """One file selected for the product bundle."""

    relative_path: Path
    category: str
    size_bytes: int


@dataclass(frozen=True)
class ProductBundleResult:
    """Result from building or previewing one product bundle."""

    zip_path: Path
    selected_files: tuple[SelectedFile, ...]


def normalize_bundle_name(raw_name: str) -> str:
    """Return a safe bundle directory and ZIP stem."""
    cleaned = "".join(character if character.isalnum() or character in "-_." else "_" for character in raw_name)
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError("Bundle name cannot be empty")
    return cleaned


def normalize_bundle_mode(raw_mode: str) -> str:
    """Return a validated bundle payload mode."""
    normalized = str(raw_mode).strip().lower().replace("_", "-")
    aliases = {
        "ai": "ai-runtime",
        "runtime": "ai-runtime",
        "code-only": "code",
        "full": "product",
        "combined": "product",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"code", "ai-runtime", "product"}:
        raise ValueError(f"Unknown bundle mode: {raw_mode!r}")
    return normalized


def normalize_manifest_path(raw_path: str) -> Path:
    """Validate and normalize one project-relative manifest path."""
    candidate = Path(raw_path.strip())
    if not raw_path.strip() or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"invalid product asset path: {raw_path!r}")
    return Path(*candidate.parts)


def read_asset_manifest(project_root: Path) -> tuple[RuntimeAsset, ...]:
    """Read the versioned runtime asset manifest."""
    manifest_path = project_root / ASSET_MANIFEST_PATH
    if not manifest_path.is_file():
        raise SystemExit(f"Product asset manifest is missing: {manifest_path}")

    allowed_kinds = {"required-file", "optional-file", "model-dir", "artifact-dir"}
    assets: list[RuntimeAsset] = []
    for line_number, raw_line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            kind, raw_path = stripped.split(maxsplit=1)
        except ValueError as error:
            raise SystemExit(f"Bad product asset manifest line {line_number}: {raw_line}") from error
        if kind not in allowed_kinds:
            raise SystemExit(f"Unknown product asset kind on line {line_number}: {kind}")
        assets.append(RuntimeAsset(kind=kind, relative_path=normalize_manifest_path(raw_path)))
    return tuple(assets)


def path_has_excluded_prefix(relative_path: Path) -> bool:
    """Return whether a project-relative path is under an excluded prefix."""
    return any(relative_path == prefix or prefix in relative_path.parents for prefix in EXCLUDED_RELATIVE_PREFIXES)


def is_backup_or_archive_file(path: Path) -> bool:
    """Return whether a filename clearly identifies a backup or archive."""
    lowered_name = path.name.lower()
    return lowered_name.startswith("._") or any(marker in lowered_name for marker in BACKUP_NAME_MARKERS)


def is_clean_product_file(relative_path: Path, *, allow_binary_runtime_asset: bool = False) -> bool:
    """Return whether a path is safe for the end-user product payload."""
    if not relative_path.parts:
        return False
    if relative_path.parts[0] in EXCLUDED_TOP_LEVEL_NAMES and not allow_binary_runtime_asset:
        return False
    if any(part.lower() in EXCLUDED_ANYWHERE_DIRECTORY_NAMES for part in relative_path.parts[:-1]):
        return False
    if path_has_excluded_prefix(relative_path):
        return False
    if relative_path.name in EXCLUDED_FILE_NAMES:
        return False
    if is_backup_or_archive_file(relative_path):
        return False
    if relative_path.suffix.lower() in EXCLUDED_FILE_SUFFIXES:
        return False
    return True


def iter_directory_files(
    project_root: Path,
    relative_directory: Path,
    *,
    allow_binary_runtime_asset: bool = False,
) -> Iterator[Path]:
    """Yield clean files beneath one selected project directory."""
    directory = project_root / relative_directory
    if not directory.is_dir():
        return
    for source_path in sorted(directory.rglob("*"), key=lambda path: path.as_posix()):
        if not source_path.is_file():
            continue
        relative_path = source_path.relative_to(project_root)
        if is_clean_product_file(relative_path, allow_binary_runtime_asset=allow_binary_runtime_asset):
            yield relative_path


def select_product_files(request: ProductBundleRequest) -> tuple[SelectedFile, ...]:
    """Select maintained files for the requested payload mode."""
    selected_categories: dict[Path, str] = {}
    bundle_mode = normalize_bundle_mode(request.bundle_mode)
    include_application = bundle_mode in {"code", "product"}
    include_runtime_assets = bundle_mode in {"ai-runtime", "product"}

    if include_application:
        for root_filename in PRODUCT_ROOT_FILES:
            relative_path = Path(root_filename)
            source_path = request.project_root / relative_path
            if source_path.is_file() and is_clean_product_file(relative_path):
                selected_categories[relative_path] = "application"

        for directory_name in PRODUCT_SOURCE_DIRECTORIES:
            relative_directory = Path(directory_name)
            for relative_path in iter_directory_files(request.project_root, relative_directory):
                selected_categories[relative_path] = "application"

    missing_required: list[Path] = []
    missing_models: list[Path] = []
    skipped_optional: list[Path] = []
    if include_runtime_assets:
        for asset in read_asset_manifest(request.project_root):
            source_path = request.project_root / asset.relative_path
            if asset.kind in {"required-file", "optional-file"}:
                if source_path.is_file():
                    selected_categories[asset.relative_path] = "brain"
                elif asset.kind == "required-file":
                    missing_required.append(asset.relative_path)
                else:
                    skipped_optional.append(asset.relative_path)
                continue

            if asset.kind == "model-dir":
                if not request.include_neural_models:
                    continue
                if not source_path.is_dir():
                    missing_models.append(asset.relative_path)
                    continue
                for relative_path in iter_directory_files(
                    request.project_root,
                    asset.relative_path,
                    allow_binary_runtime_asset=True,
                ):
                    selected_categories[relative_path] = "neural-model"
                continue

            if asset.kind == "artifact-dir":
                if not source_path.is_dir():
                    skipped_optional.append(asset.relative_path)
                    continue
                for relative_path in iter_directory_files(
                    request.project_root,
                    asset.relative_path,
                    allow_binary_runtime_asset=True,
                ):
                    selected_categories[relative_path] = "neural-runtime-artifact"

    if missing_required:
        print("Cannot build an end-user product bundle. Required runtime brains are missing:")
        for relative_path in missing_required:
            print(f"  {relative_path.as_posix()}")
        raise SystemExit(1)

    if missing_models:
        print("Cannot build the requested full product bundle. Neural model snapshots are missing:")
        for relative_path in missing_models:
            print(f"  {relative_path.as_posix()}")
        print("Use BUNDLE_INCLUDE_NEURAL_MODELS=0 only when intentionally building a brain-only product bundle.")
        raise SystemExit(1)

    if skipped_optional:
        print("Optional runtime assets not present:")
        for relative_path in skipped_optional:
            print(f"  {relative_path.as_posix()}")

    selected_files = [
        SelectedFile(
            relative_path=relative_path,
            category=category,
            size_bytes=(request.project_root / relative_path).stat().st_size,
        )
        for relative_path, category in selected_categories.items()
    ]
    return tuple(sorted(selected_files, key=lambda selected: selected.relative_path.as_posix()))


def human_size(byte_count: int) -> str:
    """Return a compact binary file-size string."""
    value = float(byte_count)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or suffix == "TiB":
            return f"{value:.1f} {suffix}"
        value /= 1024.0
    return f"{byte_count} B"


def copy_selected_files(project_root: Path, staged_root: Path, selected_files: Sequence[SelectedFile]) -> None:
    """Copy selected files into a standalone product directory."""
    for selected_file in selected_files:
        source_path = project_root / selected_file.relative_path
        destination_path = staged_root / selected_file.relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        if source_path.stat().st_mode & stat.S_IXUSR:
            destination_path.chmod(destination_path.stat().st_mode | stat.S_IXUSR)


def write_product_manifest(
    staged_root: Path,
    request: ProductBundleRequest,
    selected_files: Sequence[SelectedFile],
) -> None:
    """Write an inspectable mode-specific payload manifest."""
    bundle_mode = normalize_bundle_mode(request.bundle_mode)
    totals: dict[str, int] = {}
    for selected_file in selected_files:
        totals[selected_file.category] = totals.get(selected_file.category, 0) + selected_file.size_bytes
    total_bytes = sum(totals.values())
    lines = [
        f"# {request.bundle_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Platform: {platform.system()} {platform.release()}",
        f"Files: {len(selected_files)}",
        f"Uncompressed payload: {human_size(total_bytes)}",
        f"Bundle mode: {bundle_mode}",
        f"ZIP compression level: {request.compression_level}",
        f"Neural model snapshots included: {'yes' if request.include_neural_models else 'no'}",
        "",
        "## Payload groups",
        "",
    ]
    for category, byte_count in sorted(totals.items()):
        lines.append(f"- {category}: {human_size(byte_count)}")
    lines.extend(
        [
            "",
        ]
    )
    if bundle_mode == "code":
        lines.extend(
            [
                "## Add the AI runtime",
                "",
                "Extract the matching `Aaron_Sound_Sorter_AI_Runtime_*.zip` into the same parent folder.",
                "Both archives use the same top-level `Aaron_Sound_Sorter/` directory, so the runtime files merge into place.",
                "",
                "## First run after both archives are merged",
                "",
                "```bash",
                "python3 -m venv .venv",
                "source .venv/bin/activate",
                "python -m pip install --upgrade pip",
                "python -m pip install -e .",
                "python Aaron_Sound_Sorter.py self-test",
                "./commands/gui/RUN_SORTER_GUI.command",
                "```",
                "",
                "This archive intentionally excludes trained brains, model snapshots, prototype indexes, reports, audio, and caches.",
            ]
        )
    elif bundle_mode == "ai-runtime":
        lines.extend(
            [
                "## Installation",
                "",
                "Extract this archive into the same parent folder as the matching code archive.",
                "Allow the shared top-level `Aaron_Sound_Sorter/` folder to merge.",
                "",
                "This archive contains active runtime brains, selected neural model snapshots, and compact prototype indexes only.",
                "It intentionally excludes application source, reports, virtual environments, training audio, test audio, caches, backups, and archives.",
                "CLAP and MERT remain shadow-only and do not own production placement.",
            ]
        )
    else:
        lines.extend(
            [
                "## First run",
                "",
                "```bash",
                "python3 -m venv .venv",
                "source .venv/bin/activate",
                "python -m pip install --upgrade pip",
                "python -m pip install -e .",
                "python Aaron_Sound_Sorter.py self-test",
                "./commands/gui/RUN_SORTER_GUI.command",
                "```",
                "",
                "The trained runtime brains are included. CLAP and MERT remain shadow-only and do not own production placement.",
                "Training audio, reports, virtual environments, test audio, caches, backups, and archived artifacts are excluded. Python test code is included.",
            ]
        )
    lines.extend(["", "## Included files", ""])
    for selected_file in selected_files:
        lines.append(
            f"- `{selected_file.relative_path.as_posix()}` ({selected_file.category}, {human_size(selected_file.size_bytes)})"
        )
    manifest_filename = {
        "code": "CODE_BUNDLE_MANIFEST.md",
        "ai-runtime": "AI_RUNTIME_BUNDLE_MANIFEST.md",
        "product": "PRODUCT_BUNDLE_MANIFEST.md",
    }[bundle_mode]
    (staged_root / manifest_filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def scan_staged_payload(staged_root: Path) -> list[Path]:
    """Return forbidden generated or backup paths found after staging."""
    junk_paths: list[Path] = []
    for current_path in staged_root.rglob("*"):
        relative_path = current_path.relative_to(staged_root)
        if not relative_path.parts:
            continue
        if current_path.is_dir() and current_path.name.lower() in EXCLUDED_ANYWHERE_DIRECTORY_NAMES:
            junk_paths.append(relative_path)
            continue
        if current_path.is_file() and (
            current_path.name in EXCLUDED_FILE_NAMES
            or current_path.name.startswith("._")
            or current_path.suffix.lower() in {".pyc", ".pyo", ".tmp", ".log"}
            or is_backup_or_archive_file(current_path)
        ):
            junk_paths.append(relative_path)
    return sorted(junk_paths, key=lambda path: path.as_posix())


def zip_directory(staged_root: Path, zip_path: Path, *, compression_level: int) -> None:
    """Write one standalone ZIP using the requested DEFLATE level."""
    if compression_level < 0 or compression_level > 9:
        raise ValueError("compression_level must be between 0 and 9")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=compression_level,
        allowZip64=True,
    ) as archive:
        for current_path in sorted(staged_root.rglob("*"), key=lambda path: path.relative_to(staged_root).as_posix()):
            archive.write(current_path, staged_root.name / current_path.relative_to(staged_root))


def print_selection_summary(selected_files: Sequence[SelectedFile]) -> None:
    """Print selected file counts and sizes by payload category."""
    category_sizes: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for selected_file in selected_files:
        category_sizes[selected_file.category] = category_sizes.get(selected_file.category, 0) + selected_file.size_bytes
        category_counts[selected_file.category] = category_counts.get(selected_file.category, 0) + 1
    print("Bundle selection:")
    for category in sorted(category_sizes):
        print(f"  {category}: {category_counts[category]} files, {human_size(category_sizes[category])}")
    total_bytes = sum(category_sizes.values())
    print(f"  total: {len(selected_files)} files, {human_size(total_bytes)}")
    if total_bytes >= 1_000_000_000:
        print("Note: the bundle is over 1 GB because the selected neural model snapshots themselves exceed 1 GB.")


def build_product_bundle(request: ProductBundleRequest) -> ProductBundleResult:
    """Build or preview a clean standalone product ZIP."""
    selected_files = select_product_files(request)
    if not selected_files:
        raise SystemExit("No product files were selected.")
    print_selection_summary(selected_files)

    safe_bundle_name = normalize_bundle_name(request.bundle_name)
    payload_root_name = normalize_bundle_name(request.payload_root_name or safe_bundle_name)
    zip_path = request.output_dir / f"{safe_bundle_name}.zip"
    if request.dry_run:
        print(f"Dry run: would write {zip_path}")
        return ProductBundleResult(zip_path=zip_path, selected_files=selected_files)

    request.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aaron_product_bundle_") as temporary_directory:
        staged_root = Path(temporary_directory) / payload_root_name
        staged_root.mkdir(parents=True)
        copy_selected_files(request.project_root, staged_root, selected_files)
        write_product_manifest(staged_root, request, selected_files)
        junk_paths = scan_staged_payload(staged_root)
        if junk_paths:
            print("Refusing to build the product bundle because staged junk was found:")
            for junk_path in junk_paths:
                print(f"  {junk_path.as_posix()}")
            raise SystemExit(1)
        zip_directory(staged_root, zip_path, compression_level=request.compression_level)

    print(f"Wrote product bundle: {zip_path}")
    return ProductBundleResult(zip_path=zip_path, selected_files=selected_files)


def build_parser() -> argparse.ArgumentParser:
    """Create the product bundle command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT", "."), help="Project checkout root")
    parser.add_argument(
        "--bundle-name",
        default=os.environ.get("BUNDLE_NAME", "Aaron_Sound_Sorter_Product"),
        help="Product folder and ZIP stem",
    )
    parser.add_argument(
        "--mode",
        choices=("code", "ai-runtime", "product"),
        default=os.environ.get("BUNDLE_MODE", "product"),
        help="Payload mode: source code only, AI runtime assets only, or combined product",
    )
    parser.add_argument(
        "--payload-root-name",
        default=os.environ.get("BUNDLE_PAYLOAD_ROOT_NAME"),
        help="Top-level folder stored inside the ZIP; defaults to the bundle name",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        choices=range(0, 10),
        default=int(os.environ.get("BUNDLE_COMPRESSION_LEVEL", "1")),
        metavar="0-9",
        help="ZIP DEFLATE compression level; 9 is maximum",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("BUNDLE_OUTPUT_DIR", "_reports/bundles"),
        help="Output directory for the finished ZIP",
    )
    parser.add_argument("--without-neural-models", action="store_true", help="Build a smaller brain-only product ZIP")
    parser.add_argument("--dry-run", action="store_true", help="Print selection and size without writing a ZIP")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Run the end-user product bundle builder."""
    parser = build_parser()
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    project_root = Path(arguments.project_root).expanduser().resolve()
    output_dir = Path(arguments.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    request = ProductBundleRequest(
        project_root=project_root,
        bundle_name=normalize_bundle_name(str(arguments.bundle_name)),
        output_dir=output_dir.resolve(),
        dry_run=bool(arguments.dry_run),
        include_neural_models=not bool(arguments.without_neural_models),
        bundle_mode=normalize_bundle_mode(str(arguments.mode)),
        compression_level=int(arguments.compression_level),
        payload_root_name=(
            normalize_bundle_name(str(arguments.payload_root_name)) if arguments.payload_root_name else None
        ),
    )
    build_product_bundle(request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())