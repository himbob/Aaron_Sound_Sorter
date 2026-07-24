#!/usr/bin/env python3
"""Verify that the local developer environment has required project tools.

The Makefile calls this after installing the developer requirements.  It gives a
clear failure when a tool is missing instead of letting a later report silently
skip important checks.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RequiredTool:
    """Definition for one required importable developer tool.

    Args:
        module_name: Import name used to verify the package is installed.
        display_name: Human-readable tool name for command output.
        version_command: Command used to print the installed version.

    Attributes:
        module_name: Import name checked with ``importlib.util.find_spec``.
        display_name: Human-readable tool name.
        version_command: Version command executed with the current Python.
    """

    module_name: str
    display_name: str
    version_command: tuple[str, ...]


REQUIRED_TOOLS: tuple[RequiredTool, ...] = (
    RequiredTool("numpy", "NumPy runtime", ("-c", "import numpy; print(numpy.__version__)")),
    RequiredTool("soundfile", "SoundFile runtime", ("-c", "import soundfile; print(soundfile.__version__)")),
    RequiredTool("librosa", "librosa runtime", ("-c", "import librosa; print(librosa.__version__)")),
    RequiredTool("pytest", "pytest", ("-m", "pytest", "--version")),
    RequiredTool("pytest_cov", "pytest-cov", ("-m", "pytest", "--version")),
    RequiredTool("coverage", "coverage", ("-m", "coverage", "--version")),
    RequiredTool("ruff", "ruff", ("-m", "ruff", "--version")),
    RequiredTool("mypy", "mypy", ("-m", "mypy", "--version")),
    RequiredTool("pydocstyle", "pydocstyle", ("-m", "pydocstyle", "--version")),
)


def module_is_available(module_name: str) -> bool:
    """Return whether a Python module can be imported.

    Args:
        module_name: Import name to check.

    Returns:
        ``True`` when Python can find the module, otherwise ``False``.

    Side Effects:
        None.

    Raises:
        No intentional exceptions.

    Important Constraints:
        This check only verifies import availability.  It does not install or
        modify anything.
    """
    return importlib.util.find_spec(module_name) is not None


def run_version_command(version_command: tuple[str, ...], project_root: Path) -> str:
    """Run a tool version command and return short output.

    Args:
        version_command: Python arguments following the current executable.
        project_root: Project root used as the subprocess working directory.

    Returns:
        Trimmed output from stdout and stderr, or an error description.

    Side Effects:
        Spawns a subprocess.

    Raises:
        No intentional exceptions.  Subprocess errors are converted to text.

    Important Constraints:
        Version commands must not edit project files.
    """
    command = [sys.executable, *version_command]
    try:
        completed = subprocess.run(
            command,
            cwd=str(project_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - defensive command wrapper
        return f"could not run {' '.join(command)}: {exc}"
    output = completed.stdout.strip().splitlines()
    return output[0] if output else f"exit code {completed.returncode}"


def check_required_tools(project_root: Path, *, quick: bool = False, quiet: bool = False) -> int:
    """Print required tool status and return a process exit code.

    Args:
        project_root: Project checkout root.
        quick: When true, check only import availability and skip slower
            subprocess version commands.
        quiet: When true, print only missing-tool guidance.

    Returns:
        ``0`` when every required tool is installed, otherwise ``1``.

    Side Effects:
        Writes status lines to stdout.

    Raises:
        No intentional exceptions.

    Important Constraints:
        This function reports only.  Installation belongs to the Makefile.
    """
    missing_tools: list[str] = []
    for required_tool in REQUIRED_TOOLS:
        if module_is_available(required_tool.module_name):
            if not quiet:
                if quick:
                    print(f"PASS {required_tool.display_name}: installed")
                else:
                    version_text = run_version_command(required_tool.version_command, project_root)
                    print(f"PASS {required_tool.display_name}: {version_text}")
        else:
            missing_tools.append(required_tool.display_name)
            if not quiet:
                print(f"MISSING {required_tool.display_name}: module {required_tool.module_name!r} is not importable")

    if missing_tools:
        if quiet:
            print("Missing developer tools: " + ", ".join(missing_tools))
        print("")
        print("Run this to install missing developer tools:")
        print("  make install-dev")
        return 1
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    """Run the command-line entry point.

    Args:
        argv: Optional argument iterable for tests.  ``None`` uses ``sys.argv``.

    Returns:
        Process exit code.

    Side Effects:
        Prints environment status to stdout.

    Raises:
        No intentional exceptions.

    Important Constraints:
        The command must not create, modify, or delete project source files.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Project checkout root")
    parser.add_argument("--quick", action="store_true", help="Skip slower tool version subprocess checks")
    parser.add_argument("--quiet", action="store_true", help="Print only missing-tool guidance")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return check_required_tools(Path(args.project_root).expanduser().resolve(), quick=args.quick, quiet=args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
