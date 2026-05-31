#!/usr/bin/env python3
"""Run pytest one collected test node at a time.

This is for Aaron Sound Sorter, where running a whole file can hide which
real-audio fixture hangs or fails.  It also prevents one long test file from
masking earlier passing node-level results.

Rules:
  - Collect node IDs first.
  - Run exactly one test node per subprocess.
  - Write a separate log per node.
  - Never use Bash-only features, so it works on macOS.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:220] or "test"


def collect_nodes(project_root: Path, pytest_args: list[str]) -> list[str]:
    """Collect concrete pytest node IDs, preserving nested test folders."""
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-vv", *pytest_args]
    result = subprocess.run(
        cmd,
        cwd=project_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
    output = result.stdout
    if result.returncode not in {0, 5}:
        raise SystemExit(f"pytest collection failed:\n{output}")

    nodes: list[str] = []
    dir_stack: list[tuple[int, str]] = []
    current_module = ""
    current_class = ""

    for line in output.splitlines():
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)

        dir_match = re.search(r"<Dir ([^>]+)>", stripped)
        package_match = re.search(r"<Package ([^>]+)>", stripped)
        if dir_match or package_match:
            while dir_stack and dir_stack[-1][0] >= indent:
                dir_stack.pop()
            dirname = (dir_match or package_match).group(1)
            # Pytest shows the project root as the first <Dir>. It is not part
            # of the node path. Nested packages below it, such as tests, must
            # be preserved so node IDs can be re-run directly.
            if not (not dir_stack and dirname == project_root.name):
                dir_stack.append((indent, dirname))
            current_class = ""
            continue

        module_match = re.search(r"<Module ([^>]+)>", stripped)
        if module_match:
            # A root-level module can appear after modules from a nested Dir.
            # Pop stale nested dirs before building the node id.
            while dir_stack and dir_stack[-1][0] >= indent:
                dir_stack.pop()
            module_name = module_match.group(1)
            path_parts = [name for _, name in dir_stack]
            if path_parts and path_parts[0] == project_root.name:
                path_parts = path_parts[1:]
            current_module = "/".join([*path_parts, module_name])
            current_class = ""
            continue

        class_match = re.search(r"<Class ([^>]+)>", stripped)
        if class_match:
            current_class = class_match.group(1)
            continue

        func_match = re.search(r"<Function ([^>]+)>", stripped)
        if func_match and current_module:
            func_name = func_match.group(1)
            if current_class:
                nodes.append(f"{current_module}::{current_class}::{func_name}")
            else:
                nodes.append(f"{current_module}::{func_name}")

    if nodes:
        return nodes

    # Fallback: if parser cannot find nodes, run each selected file as one unit.
    fallback = [arg for arg in pytest_args if arg.endswith(".py") or "/test_" in arg]
    return fallback


def kill_process_tree(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except Exception:
        with contextlib.suppress(Exception):
            process.terminate()
    try:
        process.wait(timeout=5)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            with contextlib.suppress(Exception):
                process.kill()


def run_node(project_root: Path, node: str, log_path: Path, timeout_sec: float) -> tuple[str, int]:
    """Run one pytest node and capture its log.

    subprocess.run() is used intentionally here. Some environments leave pytest
    child processes alive after printing PASS when Popen/start_new_session is
    used. For this project tool, reliable one-node completion is more important
    than preserving a process group for exotic plugin workers.
    """
    cmd = [sys.executable, "-m", "pytest", "-q", node]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root / 'src'}:{project_root}:{env.get('PYTHONPATH', '')}"
    # Third-party pytest plugins can leave background workers alive after a
    # one-node run has printed PASS. Disable plugin autoload so node-by-node
    # timing reflects the project tests, not environment-specific plugins.
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=None if timeout_sec <= 0 else timeout_sec,
        )
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        log_path.write_text(str(output) + f"\nTIMEOUT after {timeout_sec:g} seconds\n", encoding="utf-8")
        return "TIMEOUT", 124
    log_path.write_text(result.stdout, encoding="utf-8", errors="replace")
    if result.returncode == 0:
        return "PASS", 0
    return "FAIL", int(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pytest_args", nargs="*", help="Optional pytest paths/node selectors. Defaults to tests.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--timeout-sec", type=float, default=180.0, help="0 disables timeout")
    parser.add_argument("--out-root", type=Path, default=Path("reports/pytest_node_by_node"))
    parser.add_argument("--stop-after", type=int, default=0, help="Stop after N failing nodes. 0 means run all.")
    parser.add_argument("--start-index", type=int, default=1, help="One-based collected-node index to start from.")
    parser.add_argument(
        "--max-nodes", type=int, default=0, help="Maximum collected nodes to run. 0 means run through the end."
    )
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    pytest_args = args.pytest_args or ["tests"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = (project_root / args.out_root / f"run_{stamp}").resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = report_dir / "pytest_node_by_node_summary.txt"
    node_file = report_dir / "collected_nodes.txt"
    failed_file = report_dir / "failed_nodes.txt"
    timeout_file = report_dir / "timeout_nodes.txt"

    collected_nodes = collect_nodes(project_root, pytest_args)
    start_index = max(1, int(args.start_index))
    if int(args.max_nodes) > 0:
        nodes = collected_nodes[start_index - 1 : start_index - 1 + int(args.max_nodes)]
    else:
        nodes = collected_nodes[start_index - 1 :]
    node_file.write_text("\n".join(collected_nodes) + "\n", encoding="utf-8")

    passed = failed = timed_out = 0
    lines = [
        "Aaron Sound Sorter node-by-node pytest run",
        f"Project: {project_root}",
        f"Selected args: {' '.join(pytest_args)}",
        f"Collected nodes: {len(collected_nodes)}",
        f"Selected run nodes: {len(nodes)}",
        f"Start index: {start_index}",
        f"Max nodes: {'all' if int(args.max_nodes) <= 0 else int(args.max_nodes)}",
        f"Timeout sec: {'none' if args.timeout_sec <= 0 else args.timeout_sec}",
        f"Report dir: {report_dir}",
        "",
    ]
    summary.write_text("\n".join(lines), encoding="utf-8")
    failed_nodes: list[str] = []
    timeout_nodes: list[str] = []

    for offset, node in enumerate(nodes, start=0):
        index = start_index + offset
        log_path = report_dir / f"{index:04d}_{safe_name(node)}.log"
        print(f"[{offset + 1}/{len(nodes)} | collected {index}/{len(collected_nodes)}] {node}", flush=True)
        status, code = run_node(project_root, node, log_path, args.timeout_sec)
        if status == "PASS":
            passed += 1
        elif status == "TIMEOUT":
            timed_out += 1
            timeout_nodes.append(node)
            failed_nodes.append(node)
        else:
            failed += 1
            failed_nodes.append(node)
        with summary.open("a", encoding="utf-8") as handle:
            handle.write(f"{status:7s} {node}  log={log_path.name}\n")
        print(f"  {status}", flush=True)
        if args.stop_after and len(failed_nodes) >= args.stop_after:
            break

    failed_file.write_text("\n".join(failed_nodes) + ("\n" if failed_nodes else ""), encoding="utf-8")
    timeout_file.write_text("\n".join(timeout_nodes) + ("\n" if timeout_nodes else ""), encoding="utf-8")
    with summary.open("a", encoding="utf-8") as handle:
        handle.write("\nFinal result\n")
        handle.write(f"Passed:   {passed}\n")
        handle.write(f"Failed:   {failed}\n")
        handle.write(f"Timeouts: {timed_out}\n")
        handle.write(f"Report:   {report_dir}\n")

    print(f"Report dir: {report_dir}")
    return 1 if failed_nodes else 0


if __name__ == "__main__":
    raise SystemExit(main())
