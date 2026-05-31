"""Regression tests for the developer Makefile bootstrap workflow."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_makefile_has_company_style_bootstrap_and_quality_targets() -> None:
    """Makefile should expose explicit install, test, quality, and report targets."""
    makefile_text = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    required_targets = [
        "bootstrap:",
        "init:",
        "venv:",
        "upgrade-pip:",
        "install-runtime:",
        "install-quality:",
        "install-dev:",
        "install-all:",
        "doctor:",
        "audit-source-names:",
        "pycompile:",
        "test:",
        "test-one:",
        "test-coverage:",
        "test-coverage-html:",
        "lint:",
        "format:",
        "format-check:",
        "type-check:",
        "docstyle:",
        "quality:",
        "qa:",
        "quality-report:",
        "quality-gate-report:",
        "ci:",
    ]
    missing_targets = [target for target in required_targets if target not in makefile_text]
    assert not missing_targets


def test_makefile_installs_dependencies_before_quality_targets() -> None:
    """Quality and test targets should install the declared tool stack first."""
    makefile_text = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    for target_name in ["test", "test-coverage", "lint", "format-check", "type-check", "docstyle"]:
        assert f"{target_name}: install-quality" in makefile_text
    assert "bootstrap: install-dev doctor" in makefile_text
    assert "install-all: install-dev" in makefile_text


def test_requirements_install_runtime_and_quality_tool_stack() -> None:
    """Requirements files should declare runtime and developer quality dependencies."""
    runtime_requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    quality_requirements = (PROJECT_ROOT / "requirements-quality.txt").read_text(encoding="utf-8")
    dev_requirements = (PROJECT_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")

    for package_name in ["numpy", "soundfile"]:
        assert package_name in runtime_requirements
    for package_name in ["pytest", "pytest-cov", "coverage", "ruff", "mypy", "pydocstyle"]:
        assert package_name in quality_requirements
    assert "-r requirements.txt" in dev_requirements
    assert "-r requirements-quality.txt" in dev_requirements
    assert "build" in dev_requirements


def test_pyproject_declares_optional_and_dependency_group_tooling() -> None:
    """pyproject should mirror the developer tooling for modern Python workflows."""
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[project.optional-dependencies]" in pyproject_text
    assert "[dependency-groups]" in pyproject_text
    for package_name in ["pytest", "pytest-cov", "coverage", "ruff", "mypy", "pydocstyle"]:
        assert package_name in pyproject_text


def test_quality_report_no_longer_silently_skips_missing_tools() -> None:
    """Quality report should require tools unless explicitly allowed to skip."""
    report_script = (PROJECT_ROOT / "tools" / "pro_quality_report.py").read_text(encoding="utf-8")
    assert "allow_missing_tools" in report_script
    assert "--allow-missing-tools" in report_script
    assert "--report-only" in report_script
    assert "Required developer tools are missing" in report_script


def test_check_dev_environment_lists_required_quality_modules() -> None:
    """Environment checker should verify every required quality module."""
    checker_text = (PROJECT_ROOT / "tools" / "check_dev_environment.py").read_text(encoding="utf-8")
    for module_name in ["pytest", "pytest_cov", "coverage", "ruff", "mypy", "pydocstyle"]:
        assert module_name in checker_text
