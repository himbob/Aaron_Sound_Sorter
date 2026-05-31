#!/usr/bin/env python3
"""Write a professional quality status report for the local project checkout.

The report is intentionally stricter than the older version.  Missing quality
packages are treated as a setup problem unless ``--allow-missing-tools`` is used.
The Makefile installs the required tools before calling this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ToolCheck:
    """Required developer tool used by the quality report.

    Args:
        module_name: Import name used to verify the tool is installed.
        description: Human-readable tool description.
        version_command: Command used to capture the installed version.

    Attributes:
        module_name: Importable module checked with ``find_spec``.
        description: Short tool description for Markdown output.
        version_command: Python arguments following ``sys.executable``.
    """

    module_name: str
    description: str
    version_command: tuple[str, ...]


@dataclass(frozen=True)
class CommandResult:
    """Captured result from one quality command.

    Args:
        name: Human-readable check name.
        command: Full command argument list.
        exit_code: Process exit code.
        output: Combined stdout and stderr.

    Attributes:
        name: Human-readable check name.
        command: Full command argument list.
        exit_code: Process exit code.
        output: Trimmed combined stdout and stderr.
    """

    name: str
    command: tuple[str, ...]
    exit_code: int
    output: str


REQUIRED_TOOLS: tuple[ToolCheck, ...] = (
    ToolCheck("pytest", "pytest test runner", ("-m", "pytest", "--version")),
    ToolCheck("pytest_cov", "pytest coverage plugin", ("-m", "pytest", "--version")),
    ToolCheck("coverage", "coverage.py", ("-m", "coverage", "--version")),
    ToolCheck("ruff", "Ruff lint and format checker", ("-m", "ruff", "--version")),
    ToolCheck("mypy", "Mypy static type checker", ("-m", "mypy", "--version")),
    ToolCheck("pydocstyle", "Google-style docstring checker", ("-m", "pydocstyle", "--version")),
)


def run_command(command: list[str], cwd: Path, timeout_seconds: int = 300) -> CommandResult:
    """Run one command and capture status plus combined output.

    Args:
        command: Full command argument list.
        cwd: Project root.
        timeout_seconds: Maximum number of seconds to allow the command to run.

    Returns:
        Captured command result.

    Side Effects:
        Spawns a subprocess.

    Raises:
        No intentional exceptions.  Subprocess launch errors are converted to a
        non-zero result.

    Important Constraints:
        This helper must not modify classifier behavior.  It only runs explicit
        quality commands supplied by ``write_report``.
    """
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        return CommandResult(
            name=" ".join(command),
            command=tuple(command),
            exit_code=completed.returncode,
            output=completed.stdout.strip(),
        )
    except Exception as exc:  # pragma: no cover - defensive reporting only
        return CommandResult(
            name=" ".join(command),
            command=tuple(command),
            exit_code=1,
            output=f"could not run {' '.join(command)}: {exc}",
        )


def module_available(module_name: str) -> bool:
    """Return whether a Python module is importable.

    Args:
        module_name: Import name to check.

    Returns:
        ``True`` when Python can find the module, otherwise ``False``.

    Side Effects:
        None.

    Raises:
        No intentional exceptions.

    Important Constraints:
        This is only an availability check.  It must not import or execute the
        target package.
    """
    return importlib.util.find_spec(module_name) is not None


def capture_tool_version(tool_check: ToolCheck, project_root: Path) -> str:
    """Return one short version line for an installed tool.

    Args:
        tool_check: Required tool definition.
        project_root: Project checkout root.

    Returns:
        First output line from the version command, or a failure description.

    Side Effects:
        Spawns a subprocess.

    Raises:
        No intentional exceptions.

    Important Constraints:
        Version commands must be read-only.
    """
    command_result = run_command([sys.executable, *tool_check.version_command], project_root, timeout_seconds=60)
    first_line = command_result.output.splitlines()[0] if command_result.output else "no version output"
    return first_line


def build_tool_status(project_root: Path) -> tuple[list[str], list[str]]:
    """Build Markdown tool status lines and return missing tool names.

    Args:
        project_root: Project checkout root.

    Returns:
        Tuple containing Markdown status lines and missing tool display names.

    Side Effects:
        Runs version commands for installed tools.

    Raises:
        No intentional exceptions.

    Important Constraints:
        Tool status must be honest.  Missing tools are returned to the caller,
        not silently skipped.
    """
    status_lines: list[str] = []
    missing_tools: list[str] = []
    for tool_check in REQUIRED_TOOLS:
        if module_available(tool_check.module_name):
            version_text = capture_tool_version(tool_check, project_root)
            status_lines.append(f"- `{tool_check.module_name}` ({tool_check.description}): available, `{version_text}`")
        else:
            status_lines.append(f"- `{tool_check.module_name}` ({tool_check.description}): **missing**")
            missing_tools.append(tool_check.module_name)
    return status_lines, missing_tools


def make_quality_commands(
    project_root: Path, *, run_pytest: bool, run_coverage: bool
) -> list[tuple[str, list[str], int]]:
    """Create the quality command list.

    Args:
        project_root: Project checkout root.
        run_pytest: Whether to include the normal pytest pass.
        run_coverage: Whether to include the slower pytest coverage pass.

    Returns:
        List of tuples containing name, command, and timeout seconds.

    Side Effects:
        None.

    Raises:
        No intentional exceptions.

    Important Constraints:
        These commands are source-blind quality checks.  They must not train the
        brain, rewrite ledgers, or sort user sample libraries.
    """
    commands: list[tuple[str, list[str], int]] = [
        (
            "No source-name sorting audit",
            [sys.executable, "tools/audit_no_source_name_sorting.py", "--project-root", str(project_root)],
            300,
        ),
        ("Ruff lint", [sys.executable, "-m", "ruff", "check", "src", "tests", "tools"], 300),
        ("Ruff format check", [sys.executable, "-m", "ruff", "format", "--check", "src", "tests", "tools"], 300),
        ("Mypy", [sys.executable, "-m", "mypy", "src/aaron_sound_sorter"], 300),
        ("Pydocstyle", [sys.executable, "-m", "pydocstyle", "src/aaron_sound_sorter"], 300),
    ]
    if run_pytest:
        commands.append(("Pytest", [sys.executable, "-m", "pytest", "-q"], 900))
    if run_coverage:
        commands.append(
            (
                "Coverage summary",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "--cov=src/aaron_sound_sorter",
                    "--cov-report=term-missing",
                    "--cov-report=xml:_reports/quality/coverage.xml",
                    "-q",
                ],
                900,
            )
        )
    return commands


def append_command_result(lines: list[str], name: str, command_result: CommandResult) -> None:
    """Append one command result section to Markdown lines.

    Args:
        lines: Markdown line buffer to mutate.
        name: Human-readable check name.
        command_result: Captured command result.

    Returns:
        None.

    Side Effects:
        Mutates ``lines``.

    Raises:
        No intentional exceptions.

    Important Constraints:
        Output is trimmed to keep reports readable in large failing projects.
    """
    lines.append(f"### {name}")
    lines.append("")
    lines.append(f"Command: `{' '.join(command_result.command)}`")
    lines.append(f"Exit code: `{command_result.exit_code}`")
    lines.append("")
    if command_result.output:
        lines.append("```text")
        lines.append(command_result.output[-8000:])
        lines.append("```")
    else:
        lines.append("No output captured.")
    lines.append("")


def write_report(
    project_root: Path,
    *,
    run_pytest: bool,
    run_coverage: bool,
    allow_missing_tools: bool,
    report_only: bool,
) -> tuple[Path, int]:
    """Write a quality report under ``_reports/quality``.

    Args:
        project_root: Project checkout root.
        run_pytest: Whether to run the pytest suite as part of the report.
        run_coverage: Whether to run the slower pytest-cov coverage pass.
        allow_missing_tools: Whether missing quality tools should be reported but
            not treated as a failure.
        report_only: Whether command failures should be recorded without
            becoming the script exit code.

    Returns:
        Tuple containing the report path and recommended process exit code.

    Side Effects:
        Creates a report file under ``_reports/quality`` and may create coverage
        output under the same folder.

    Raises:
        Filesystem errors if the report cannot be written.

    Important Constraints:
        Generated reports stay under ``_reports``.  This tool must not edit
        source, tests, training data, brains, or ledgers.
    """
    report_dir = project_root / "_reports" / "quality"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"pro_quality_report_{stamp}.md"

    tool_status_lines, missing_tools = build_tool_status(project_root)
    lines = ["# Aaron Sound Sorter professional quality report", "", f"Generated: {stamp}", ""]
    lines.append("## Required tool availability")
    lines.extend(tool_status_lines)
    lines.append("")

    if missing_tools and not allow_missing_tools:
        lines.append("## Setup failure")
        lines.append("")
        lines.append("Required developer tools are missing. Run:")
        lines.append("")
        lines.append("```bash")
        lines.append("make install-dev")
        lines.append("```")
        lines.append("")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path, 2

    lines.append("## Checks")
    exit_code = 0
    for name, command, timeout_seconds in make_quality_commands(
        project_root,
        run_pytest=run_pytest,
        run_coverage=run_coverage,
    ):
        command_result = run_command(command, project_root, timeout_seconds=timeout_seconds)
        append_command_result(lines, name, command_result)
        if command_result.exit_code != 0:
            exit_code = 1

    report_path.write_text("\n".join(lines), encoding="utf-8")
    if report_only:
        return report_path, 0
    return report_path, exit_code


def main(argv: Iterable[str] | None = None) -> int:
    """Run the quality report command-line entry point.

    Args:
        argv: Optional command-line arguments for tests.  ``None`` uses
            ``sys.argv``.

    Returns:
        Process exit code.  ``0`` means all included checks passed, ``1`` means
        at least one quality command failed, and ``2`` means required tools were
        missing.

    Side Effects:
        Writes a Markdown report path to stdout and creates the report file.

    Raises:
        No intentional exceptions.

    Important Constraints:
        The command is reporting-only and must not alter classifier behavior.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Project checkout root")
    parser.add_argument("--with-pytest", action="store_true", help="Include pytest in the report")
    parser.add_argument("--with-coverage", action="store_true", help="Include slower pytest-cov coverage pass")
    parser.add_argument(
        "--allow-missing-tools",
        action="store_true",
        help="Write the report even when quality tools are missing",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Always return zero after writing the report unless setup is invalid",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    report_path, exit_code = write_report(
        Path(args.project_root).expanduser().resolve(),
        run_pytest=bool(args.with_pytest),
        run_coverage=bool(args.with_coverage),
        allow_missing_tools=bool(args.allow_missing_tools),
        report_only=bool(args.report_only),
    )
    print(report_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
