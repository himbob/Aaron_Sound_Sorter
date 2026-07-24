from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_TOOL = ROOT / "tools" / "build_clean_patch_bundle.py"
MAKEFILE = ROOT / "Makefile"


def run_bundle_tool(project_root: Path, changed_files: str, *extra_args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["AI_CHANGED_FILES"] = changed_files
    command = [
        sys.executable,
        str(BUNDLE_TOOL),
        "--project-root",
        str(project_root),
        "--bundle-name",
        "Aaron_Sound_Sorter_test_patch",
        "--output-dir",
        "_reports/bundles",
        *extra_args,
    ]
    return subprocess.run(command, cwd=str(project_root), env=environment, text=True, capture_output=True, check=False)


def make_tiny_project(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    (project_root / "src" / "aaron_sound_sorter").mkdir(parents=True)
    (project_root / "tests").mkdir()
    (project_root / "tools").mkdir()
    (project_root / "docs").mkdir()
    (project_root / "commands" / "build").mkdir(parents=True)
    (project_root / "neural_artifacts" / "verified_run").mkdir(parents=True)
    (project_root / "src" / "aaron_sound_sorter" / "example.py").write_text('"""Example module."""\nVALUE = 1\n')
    (project_root / "tests" / "test_example.py").write_text("def test_example():\n    assert True\n")
    (project_root / "tools" / "example_tool.py").write_text('"""Example tool."""\n')
    (project_root / "docs" / "NOTE.md").write_text("# Note\n")
    (project_root / "commands" / "build" / "EXAMPLE.command").write_text("#!/bin/bash\necho ok\n")
    (project_root / "neural_artifacts" / "verified_run" / "summary.json").write_text("{}\n")
    return project_root


def test_makefile_bundle_target_cleans_before_building() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")

    assert "bundle: clean-for-bundle ai-check" in text
    assert "./commands/build/BUILD_CLEAN_PATCH_BUNDLE.command" in text
    assert "bundle-dry-run: clean-for-bundle" in text


def test_bundle_tool_dry_run_selects_safe_changed_files(tmp_path: Path) -> None:
    project_root = make_tiny_project(tmp_path)

    completed = run_bundle_tool(
        project_root,
        (
            "src/aaron_sound_sorter/example.py\n"
            "tests/test_example.py\n"
            "docs/NOTE.md\n"
            "neural_artifacts/verified_run/summary.json"
        ),
        "--dry-run",
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "Selected bundle files:" in completed.stdout
    assert "src/aaron_sound_sorter/example.py" in completed.stdout
    assert "tests/test_example.py" in completed.stdout
    assert "docs/NOTE.md" in completed.stdout
    assert "neural_artifacts/verified_run/summary.json" in completed.stdout
    assert "Dry run:" in completed.stdout


def test_bundle_tool_creates_clean_zip_with_installer(tmp_path: Path) -> None:
    project_root = make_tiny_project(tmp_path)
    (project_root / "src" / "aaron_sound_sorter" / "__pycache__").mkdir()
    (project_root / "src" / "aaron_sound_sorter" / "__pycache__" / "bad.pyc").write_bytes(b"junk")

    completed = run_bundle_tool(project_root, "src/aaron_sound_sorter/example.py\ncommands/build/EXAMPLE.command")

    assert completed.returncode == 0, completed.stderr + completed.stdout
    zip_path = project_root / "_reports" / "bundles" / "Aaron_Sound_Sorter_test_patch.zip"
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path, "r") as archive:
        names = set(archive.namelist())

    assert "Aaron_Sound_Sorter_test_patch/files/src/aaron_sound_sorter/example.py" in names
    assert "Aaron_Sound_Sorter_test_patch/files/commands/build/EXAMPLE.command" in names
    assert "Aaron_Sound_Sorter_test_patch/files/neural_artifacts/verified_run/summary.json" in names
    assert "Aaron_Sound_Sorter_test_patch/INSTALL_NO_BACKUP.command" in names
    assert "Aaron_Sound_Sorter_test_patch/AI_BUNDLE_MANIFEST.md" in names
    assert not any("__pycache__" in name for name in names)
    assert not any(name.endswith(".pyc") for name in names)
    assert not any(".pytest_cache" in name for name in names)


def test_bundle_tool_rejects_reports_and_audio_files(tmp_path: Path) -> None:
    project_root = make_tiny_project(tmp_path)
    (project_root / "_reports").mkdir()
    (project_root / "_reports" / "bad.txt").write_text("bad")
    (project_root / "sound.wav").write_bytes(b"RIFF")

    completed = run_bundle_tool(project_root, "_reports/bad.txt\nsound.wav")

    assert completed.returncode != 0
    assert "Refusing to bundle unsafe or generated paths" in completed.stdout
