"""Regression tests for v31128 AGENTS.md and AI Makefile ratchet gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_ai_quality_gate_module():
    """Load the ai_quality_gate module directly from the tools folder."""
    module_path = PROJECT_ROOT / "tools" / "ai_quality_gate.py"
    spec = importlib.util.spec_from_file_location("ai_quality_gate", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ai_quality_gate"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_agents_md_tells_ai_to_use_makefile_quality_gate() -> None:
    """The root agent instruction file must force AI workers through Makefile gates."""
    agents_text = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "make ai-preflight" in agents_text
    assert "make ai-fix" in agents_text
    assert "make ai-check" in agents_text
    assert "make ai-bundle-check" in agents_text
    assert "Do not use filenames" in agents_text
    assert "AI-touched files must be clean" in agents_text


def test_makefile_contains_ai_ratchet_targets() -> None:
    """The Makefile must expose the targets named in AGENTS.md."""
    makefile_text = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")

    for target_name in [
        "ai-preflight:",
        "ai-changed-files:",
        "ai-fix:",
        "ai-check:",
        "ai-test:",
        "ai-bundle-check:",
        "clean-generated:",
    ]:
        assert target_name in makefile_text

    assert "tools/ai_quality_gate.py" in makefile_text
    assert "quality-strict" in makefile_text
    assert "quality-baseline" in makefile_text


def test_ai_quality_gate_rejects_bundle_junk_paths() -> None:
    """Bundle hygiene must reject cache, pyc, report, and macOS metadata paths."""
    gate = load_ai_quality_gate_module()

    assert gate.is_junk_relative_path(Path("files/src/__pycache__/x.pyc"), forbid_reports=True)
    assert gate.is_junk_relative_path(Path("files/tests/.pytest_cache/CACHEDIR.TAG"), forbid_reports=True)
    assert gate.is_junk_relative_path(Path("files/.DS_Store"), forbid_reports=True)
    assert gate.is_junk_relative_path(Path("files/._features.py"), forbid_reports=True)
    assert gate.is_junk_relative_path(Path("_reports/quality/report.md"), forbid_reports=True)
    assert not gate.is_junk_relative_path(Path("files/src/aaron_sound_sorter/features.py"), forbid_reports=True)


def test_ai_quality_gate_filters_checkable_python_files() -> None:
    """The changed-files gate should check real Python files but skip generated junk."""
    gate = load_ai_quality_gate_module()

    assert gate.is_checkable_python_file(Path("src/aaron_sound_sorter/features.py"))
    assert gate.is_checkable_python_file(Path("tests/test_example.py"))
    assert gate.is_checkable_python_file(Path("tools/example.py"))
    assert gate.is_checkable_python_file(Path("Aaron_Sound_Sorter.py"))
    assert not gate.is_checkable_python_file(Path("docs/example.py"))
    assert not gate.is_checkable_python_file(Path("src/aaron_sound_sorter/._features.py"))
    assert not gate.is_checkable_python_file(Path("src/aaron_sound_sorter/__pycache__/features.pyc"))


def test_pyproject_excludes_appledouble_from_docstyle() -> None:
    """pydocstyle config must not scan macOS AppleDouble files."""
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "\\\\._" in pyproject_text or "\\._" in pyproject_text
    assert ".pytest_cache" in pyproject_text
    assert ".ruff_cache" in pyproject_text
