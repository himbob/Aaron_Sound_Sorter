#!/usr/bin/env python3
"""Locked real-audio smoke acceptance harness.

This tool is intentionally QA-only. It runs the existing sorter as a black box,
reads the produced manifest, and compares placements to a locked expectation
panel. It does not import classifier internals, train brains, or alter policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_PROJECT_ROOT = Path("/Volumes/T9/testbed/Aaron_Sound_Sorter")
DEFAULT_PANEL_REL = Path("tests/acceptance/locked_smoke_v1")
DEFAULT_TIMEOUT_SECONDS = 90

PASS = "PASS"
PASS_REVIEW = "PASS_REVIEW"
FAIL_MISSING_SAMPLE = "FAIL_MISSING_SAMPLE"
FAIL_MISSING_MANIFEST = "FAIL_MISSING_MANIFEST"
FAIL_TIMEOUT = "FAIL_TIMEOUT"
FAIL_SORTER_ERROR = "FAIL_SORTER_ERROR"
FAIL_WRONG_TOP = "FAIL_WRONG_TOP"
FAIL_WRONG_FOLDER = "FAIL_WRONG_FOLDER"

RESULT_COLUMNS = [
    "id",
    "filename",
    "protected",
    "status",
    "actual_path",
    "actual_top",
    "expected_top",
    "matched_prefix",
    "sorter_returncode",
    "sample_path",
    "manifest_path",
    "output_dir",
    "notes",
    "error",
]

COMPARISON_COLUMNS = [
    "id",
    "filename",
    "protected",
    "baseline_status",
    "current_status",
    "comparison",
    "baseline_actual_path",
    "current_actual_path",
    "notes",
]


@dataclass(frozen=True)
class AcceptanceRules:
    fail_on_missing_sample: bool = True
    fail_on_missing_manifest_row: bool = True
    fail_on_wrong_top_family: bool = True
    allow_review_when_listed: bool = True


@dataclass(frozen=True)
class AcceptanceCase:
    id: str
    filename: str
    expected_top: str
    accepted_folder_prefixes: tuple[str, ...]
    accepted_review_prefixes: tuple[str, ...]
    protected: bool
    notes: str = ""


@dataclass(frozen=True)
class AcceptanceConfig:
    panel_name: str
    description: str
    rules: AcceptanceRules
    cases: tuple[AcceptanceCase, ...]


@dataclass(frozen=True)
class SorterResult:
    returncode: int | None
    timed_out: bool
    error: str = ""


@dataclass(frozen=True)
class CaseResult:
    id: str
    filename: str
    protected: bool
    status: str
    actual_path: str
    actual_top: str
    expected_top: str
    matched_prefix: str
    sorter_returncode: int | None
    sample_path: str
    manifest_path: str
    output_dir: str
    notes: str
    error: str = ""


@dataclass(frozen=True)
class CompareResult:
    id: str
    filename: str
    protected: bool
    baseline_status: str
    current_status: str
    comparison: str
    baseline_actual_path: str
    current_actual_path: str
    notes: str = ""


def configure_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = limit // 10


def normalize_manifest_path(value: str) -> str:
    return value.replace("\\", "/").strip()


def normalize_prefix(value: str) -> str:
    return normalize_manifest_path(value).rstrip("/")


def path_starts_with(value: str, prefix: str) -> bool:
    normalized_value = normalize_manifest_path(value)
    normalized_prefix = normalize_prefix(prefix)
    return normalized_value == normalized_prefix or normalized_value.startswith(normalized_prefix + "/")


def top_folder(value: str) -> str:
    normalized = normalize_manifest_path(value)
    if not normalized:
        return ""
    return normalized.split("/", 1)[0]


def parse_rules(data: Mapping[str, Any]) -> AcceptanceRules:
    return AcceptanceRules(
        fail_on_missing_sample=bool(data.get("fail_on_missing_sample", True)),
        fail_on_missing_manifest_row=bool(data.get("fail_on_missing_manifest_row", True)),
        fail_on_wrong_top_family=bool(data.get("fail_on_wrong_top_family", True)),
        allow_review_when_listed=bool(data.get("allow_review_when_listed", True)),
    )


def parse_case(data: Mapping[str, Any]) -> AcceptanceCase:
    return AcceptanceCase(
        id=str(data["id"]),
        filename=str(data["filename"]),
        expected_top=str(data.get("expected_top", "")),
        accepted_folder_prefixes=tuple(str(item) for item in data.get("accepted_folder_prefixes", [])),
        accepted_review_prefixes=tuple(str(item) for item in data.get("accepted_review_prefixes", [])),
        protected=bool(data.get("protected", True)),
        notes=str(data.get("notes", "")),
    )


def load_acceptance_config(expected_path: Path) -> AcceptanceConfig:
    with expected_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return AcceptanceConfig(
        panel_name=str(data.get("panel_name", "locked_smoke")),
        description=str(data.get("description", "")),
        rules=parse_rules(data.get("rules", {})),
        cases=tuple(parse_case(item) for item in data.get("cases", [])),
    )


def default_panel_dir(project_root: Path) -> Path:
    return project_root / DEFAULT_PANEL_REL


def default_reports_root(project_root: Path) -> Path:
    return project_root / "_reports" / "locked_smoke_acceptance"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(reports_root: Path) -> Path:
    run_dir = reports_root / f"run_{timestamp()}"
    counter = 2
    while run_dir.exists():
        run_dir = reports_root / f"run_{timestamp()}_{counter}"
        counter += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def latest_run_file(reports_root: Path) -> Path:
    return reports_root / "latest_run_path.txt"


def latest_baseline_file(reports_root: Path) -> Path:
    return reports_root / "latest_baseline_path.txt"


def write_latest_path(path_file: Path, run_dir: Path) -> None:
    path_file.parent.mkdir(parents=True, exist_ok=True)
    path_file.write_text(str(run_dir) + "\n", encoding="utf-8")


def sample_path_for_case(samples_dir: Path, case: AcceptanceCase) -> Path:
    return samples_dir / case.filename


def build_sort_command(project_root: Path, sample_path: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(project_root / "Aaron_Sound_Sorter.py"),
        "sort",
        str(sample_path),
        str(output_dir),
        "--brain",
        str(project_root / "stage4_folder_brain.json"),
        "--no-zip",
    ]


def append_run_log(log_path: Path, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if text and not text.endswith("\n"):
            handle.write("\n")


def timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def terminate_process(process: subprocess.Popen[str]) -> None:
    """Terminate a sorter subprocess and its child process group when possible."""
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    except Exception:
        try:
            process.kill()
        except Exception:
            return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        return
    except Exception:
        try:
            process.kill()
        except Exception:
            return


def run_sorter(
    project_root: Path, sample_path: Path, output_dir: Path, timeout_seconds: int, log_path: Path
) -> SorterResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_sort_command(project_root, sample_path, output_dir)
    append_run_log(log_path, "\n=== SORT START ===")
    append_run_log(log_path, "sample: " + str(sample_path))
    append_run_log(log_path, "output: " + str(output_dir))
    append_run_log(log_path, "command: " + " ".join(command))

    popen_kwargs: dict[str, Any] = {
        "cwd": str(project_root),
        "text": True,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(command, **popen_kwargs)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        terminate_process(process)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        append_run_log(log_path, "TIMEOUT after " + str(timeout_seconds) + " seconds")
        append_run_log(log_path, timeout_output(stdout))
        append_run_log(log_path, timeout_output(stderr))
        return SorterResult(returncode=None, timed_out=True, error=f"Timed out after {timeout_seconds} seconds")

    append_run_log(log_path, "returncode: " + str(process.returncode))
    append_run_log(log_path, "--- stdout ---")
    append_run_log(log_path, stdout)
    append_run_log(log_path, "--- stderr ---")
    append_run_log(log_path, stderr)
    if process.returncode != 0:
        return SorterResult(returncode=process.returncode, timed_out=False, error="Sorter exited nonzero")
    return SorterResult(returncode=process.returncode, timed_out=False)


def find_manifest_path(output_dir: Path) -> Path | None:
    direct = output_dir / "Aaron_Sorted_Sounds_manifest.csv"
    if direct.is_file():
        return direct
    matches = sorted(output_dir.rglob("Aaron_Sorted_Sounds_manifest.csv"))
    if matches:
        return matches[0]
    return None


def read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    configure_csv_limit()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def manifest_row_filename(row: Mapping[str, str]) -> str:
    columns = [
        "source_path",
        "input_path",
        "file_path",
        "original_path",
        "source_name",
        "filename",
        "file_name",
        "name",
        "placed_path",
    ]
    for column in columns:
        value = str(row.get(column, "")).strip()
        if value:
            return Path(value).name
    return ""


def select_manifest_row(rows: Sequence[Mapping[str, str]], filename: str) -> Mapping[str, str] | None:
    for row in rows:
        if manifest_row_filename(row) == filename:
            return row
    if len(rows) == 1:
        return rows[0]
    return None


def path_after_sorted_root(value: str) -> str:
    normalized = normalize_manifest_path(value)
    marker = "/Aaron_Sorted_Sounds/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    return normalized


def manifest_actual_path(row: Mapping[str, str]) -> str:
    columns = [
        "new_relative_path",
        "final_relative_path",
        "placed_relative_path",
        "relative_path",
        "folder_path",
        "final_label",
        "label",
        "destination_folder",
        "placed_path",
    ]
    for column in columns:
        value = normalize_manifest_path(str(row.get(column, "")))
        if value:
            if column == "placed_path":
                return path_after_sorted_root(value)
            return value
    return ""


def manifest_actual_top(row: Mapping[str, str], actual_path: str) -> str:
    explicit = str(row.get("final_top", "")).strip()
    if explicit:
        return explicit
    return top_folder(actual_path)


def accepted_folder_prefix(actual_path: str, prefixes: Sequence[str]) -> str:
    for prefix in prefixes:
        if path_starts_with(actual_path, prefix):
            return prefix
    return ""


def accepted_review_prefix(actual_path: str, prefixes: Sequence[str]) -> str:
    for prefix in prefixes:
        if path_starts_with(actual_path, prefix):
            return prefix
    return ""


def result_from_status(
    case: AcceptanceCase,
    status: str,
    sample_path: Path,
    output_dir: Path,
    manifest_path: Path | None,
    sorter: SorterResult,
    error: str,
) -> CaseResult:
    return CaseResult(
        id=case.id,
        filename=case.filename,
        protected=case.protected,
        status=status,
        actual_path="",
        actual_top="",
        expected_top=case.expected_top,
        matched_prefix="",
        sorter_returncode=sorter.returncode,
        sample_path=str(sample_path),
        manifest_path=str(manifest_path or ""),
        output_dir=str(output_dir),
        notes=case.notes,
        error=error,
    )


def evaluate_manifest_case(
    case: AcceptanceCase,
    row: Mapping[str, str],
    sample_path: Path,
    output_dir: Path,
    manifest_path: Path,
    sorter: SorterResult,
    rules: AcceptanceRules,
) -> CaseResult:
    actual_path = manifest_actual_path(row)
    actual_top = manifest_actual_top(row, actual_path)
    review_prefix = ""
    if rules.allow_review_when_listed:
        review_prefix = accepted_review_prefix(actual_path, case.accepted_review_prefixes)
    if review_prefix:
        status = PASS_REVIEW
        matched = review_prefix
    else:
        folder_prefix = accepted_folder_prefix(actual_path, case.accepted_folder_prefixes)
        if folder_prefix:
            status = PASS
            matched = folder_prefix
        elif actual_top != case.expected_top and rules.fail_on_wrong_top_family:
            status = FAIL_WRONG_TOP
            matched = ""
        else:
            status = FAIL_WRONG_FOLDER
            matched = ""
    return CaseResult(
        id=case.id,
        filename=case.filename,
        protected=case.protected,
        status=status,
        actual_path=actual_path,
        actual_top=actual_top,
        expected_top=case.expected_top,
        matched_prefix=matched,
        sorter_returncode=sorter.returncode,
        sample_path=str(sample_path),
        manifest_path=str(manifest_path),
        output_dir=str(output_dir),
        notes=case.notes,
        error=sorter.error,
    )


def evaluate_case(
    case: AcceptanceCase,
    sample_path: Path,
    output_dir: Path,
    manifest_path: Path | None,
    sorter: SorterResult,
    rules: AcceptanceRules,
) -> CaseResult:
    if not sample_path.is_file():
        return result_from_status(
            case, FAIL_MISSING_SAMPLE, sample_path, output_dir, manifest_path, sorter, "Sample file is missing"
        )
    if sorter.timed_out:
        return result_from_status(case, FAIL_TIMEOUT, sample_path, output_dir, manifest_path, sorter, sorter.error)
    if sorter.returncode not in (0, None):
        return result_from_status(case, FAIL_SORTER_ERROR, sample_path, output_dir, manifest_path, sorter, sorter.error)
    if manifest_path is None:
        return result_from_status(
            case, FAIL_MISSING_MANIFEST, sample_path, output_dir, manifest_path, sorter, "Manifest file is missing"
        )
    rows = read_manifest_rows(manifest_path)
    row = select_manifest_row(rows, case.filename)
    if row is None:
        return result_from_status(
            case, FAIL_MISSING_MANIFEST, sample_path, output_dir, manifest_path, sorter, "Manifest row is missing"
        )
    return evaluate_manifest_case(case, row, sample_path, output_dir, manifest_path, sorter, rules)


def result_to_row(result: CaseResult) -> dict[str, str]:
    return {
        "id": result.id,
        "filename": result.filename,
        "protected": "true" if result.protected else "false",
        "status": result.status,
        "actual_path": result.actual_path,
        "actual_top": result.actual_top,
        "expected_top": result.expected_top,
        "matched_prefix": result.matched_prefix,
        "sorter_returncode": "" if result.sorter_returncode is None else str(result.sorter_returncode),
        "sample_path": result.sample_path,
        "manifest_path": result.manifest_path,
        "output_dir": result.output_dir,
        "notes": result.notes,
        "error": result.error,
    }


def comparison_to_row(result: CompareResult) -> dict[str, str]:
    return {
        "id": result.id,
        "filename": result.filename,
        "protected": "true" if result.protected else "false",
        "baseline_status": result.baseline_status,
        "current_status": result.current_status,
        "comparison": result.comparison,
        "baseline_actual_path": result.baseline_actual_path,
        "current_actual_path": result.current_actual_path,
        "notes": result.notes,
    }


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, str]], preferred_columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(preferred_columns)
    seen = set(columns)
    for row in rows:
        for key in row:
            if key not in seen:
                columns.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def protected_case_passed(result: CaseResult) -> bool:
    return result.status in {PASS, PASS_REVIEW}


def result_blocks_exit(result: CaseResult, rules: AcceptanceRules | None = None) -> bool:
    active_rules = rules or AcceptanceRules()
    if not result.protected or protected_case_passed(result):
        return False
    if result.status == FAIL_MISSING_SAMPLE:
        return active_rules.fail_on_missing_sample
    if result.status == FAIL_MISSING_MANIFEST:
        return active_rules.fail_on_missing_manifest_row
    if result.status == FAIL_WRONG_TOP:
        return active_rules.fail_on_wrong_top_family
    return True


def exit_code_for_results(results: Sequence[CaseResult], rules: AcceptanceRules | None = None) -> int:
    return 1 if any(result_blocks_exit(result, rules) for result in results) else 0


def failing_results(results: Sequence[CaseResult]) -> list[CaseResult]:
    return [result for result in results if result.status not in {PASS, PASS_REVIEW}]


def combined_manifest_rows(
    manifest_paths: Sequence[Path], case_by_manifest: Mapping[str, AcceptanceCase]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for manifest_path in manifest_paths:
        case = case_by_manifest.get(str(manifest_path))
        if case is None:
            continue
        for row in read_manifest_rows(manifest_path):
            merged = {"case_id": case.id, "case_filename": case.filename}
            merged.update(dict(row))
            rows.append(merged)
    return rows


def write_summary(path: Path, config: AcceptanceConfig, results: Sequence[CaseResult], timeout_seconds: int) -> None:
    protected_total = sum(1 for result in results if result.protected)
    protected_failed = sum(1 for result in results if result_blocks_exit(result, config.rules))
    lines = [
        f"Panel: {config.panel_name}",
        f"Description: {config.description}",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Timeout seconds per sample: {timeout_seconds}",
        f"Cases: {len(results)}",
        f"Protected cases: {protected_total}",
        f"Protected failures: {protected_failed}",
        "",
    ]
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
    lines.extend(f"{status}: {count}" for status, count in sorted(status_counts.items()))
    lines.append("")
    for result in results:
        protected = "protected" if result.protected else "unprotected"
        lines.append(f"{result.status}: {result.id} ({protected}) -> {result.actual_path or result.error}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_upload_zip(run_dir: Path) -> Path:
    zip_path = run_dir / "locked_smoke_acceptance_upload_back.zip"
    names = [
        "actual_results.csv",
        "failures.csv",
        "summary.txt",
        "full_manifest_combined.csv",
        "run.log",
        "expected_results.json",
        "baseline_comparison.csv",
        "baseline_comparison_summary.txt",
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            path = run_dir / name
            if path.is_file():
                archive.write(path, arcname=name)
    return zip_path


def copy_expected_to_run(expected_path: Path, run_dir: Path) -> None:
    shutil.copy2(expected_path, run_dir / "expected_results.json")


def filter_cases_for_run(
    cases: Sequence[AcceptanceCase],
    case_ids: Sequence[str],
    start_index: int,
    max_cases: int,
) -> tuple[AcceptanceCase, ...]:
    """Select a stable subset of acceptance cases for AI-safe chunked runs.

    Indexes are one-based for humans.  This helper is QA-only and does not
    change sorter behavior.
    """
    selected = list(cases)
    if case_ids:
        wanted = {str(case_id) for case_id in case_ids}
        selected = [case for case in selected if case.id in wanted]
    if start_index > 1:
        selected = selected[start_index - 1 :]
    if max_cases > 0:
        selected = selected[:max_cases]
    return tuple(selected)


def print_case_progress(message: str, enabled: bool) -> None:
    """Print progress immediately so AI wrappers do not look hung."""
    if enabled:
        print(message, flush=True)


def run_acceptance(
    project_root: Path,
    panel_dir: Path,
    expected_path: Path,
    samples_dir: Path,
    reports_root: Path,
    timeout_seconds: int,
    case_ids: Sequence[str] = (),
    start_index: int = 1,
    max_cases: int = 0,
    progress: bool = True,
) -> tuple[int, Path, list[CaseResult]]:
    config = load_acceptance_config(expected_path)
    selected_cases = filter_cases_for_run(config.cases, case_ids, start_index, max_cases)
    if not selected_cases:
        raise SystemExit("No locked smoke cases selected. Check --case-id, --start-index, or --max-cases.")
    run_dir = make_run_dir(reports_root)
    log_path = run_dir / "run.log"
    copy_expected_to_run(expected_path, run_dir)
    write_latest_path(latest_run_file(reports_root), run_dir)
    append_run_log(log_path, "project_root: " + str(project_root))
    append_run_log(log_path, "panel_dir: " + str(panel_dir))
    append_run_log(log_path, "expected_path: " + str(expected_path))
    append_run_log(log_path, "samples_dir: " + str(samples_dir))
    append_run_log(log_path, "selected_cases: " + ",".join(case.id for case in selected_cases))
    print_case_progress("RUN_DIR=" + str(run_dir), progress)
    print_case_progress(
        "Selected locked smoke cases: " + str(len(selected_cases)) + " of " + str(len(config.cases)), progress
    )
    results: list[CaseResult] = []
    manifest_paths: list[Path] = []
    case_by_manifest: dict[str, AcceptanceCase] = {}
    for index, case in enumerate(selected_cases, start=1):
        print_case_progress(f"[{index}/{len(selected_cases)}] START {case.id} :: {case.filename}", progress)
        sample_path = sample_path_for_case(samples_dir, case)
        output_dir = run_dir / "case_outputs" / case.id
        if sample_path.is_file():
            sorter = run_sorter(project_root, sample_path, output_dir, timeout_seconds, log_path)
            manifest_path = find_manifest_path(output_dir)
        else:
            sorter = SorterResult(returncode=None, timed_out=False, error="Sample file is missing")
            manifest_path = None
            append_run_log(log_path, "\n=== MISSING SAMPLE ===")
            append_run_log(log_path, str(sample_path))
        result = evaluate_case(case, sample_path, output_dir, manifest_path, sorter, config.rules)
        results.append(result)
        print_case_progress(
            f"[{index}/{len(selected_cases)}] {result.status} {case.id} -> {result.actual_path or result.error}",
            progress,
        )
        if manifest_path is not None and manifest_path.is_file():
            manifest_paths.append(manifest_path)
            case_by_manifest[str(manifest_path)] = case
    result_rows = [result_to_row(result) for result in results]
    failure_rows = [result_to_row(result) for result in failing_results(results)]
    write_csv_rows(run_dir / "actual_results.csv", result_rows, RESULT_COLUMNS)
    write_csv_rows(run_dir / "failures.csv", failure_rows, RESULT_COLUMNS)
    combined_rows = combined_manifest_rows(manifest_paths, case_by_manifest)
    write_csv_rows(run_dir / "full_manifest_combined.csv", combined_rows, ["case_id", "case_filename"])
    write_summary(run_dir / "summary.txt", config, results, timeout_seconds)
    create_upload_zip(run_dir)
    return exit_code_for_results(results, config.rules), run_dir, results


def copy_baseline(run_dir: Path, baseline_dir: Path, reports_root: Path) -> None:
    if baseline_dir.exists():
        shutil.rmtree(baseline_dir)
    baseline_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir, baseline_dir)
    write_latest_path(latest_baseline_file(reports_root), baseline_dir)


def read_results_csv(path: Path) -> dict[str, dict[str, str]]:
    configure_csv_limit()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["id"]: dict(row) for row in csv.DictReader(handle)}


def case_by_id(config: AcceptanceConfig) -> dict[str, AcceptanceCase]:
    return {case.id: case for case in config.cases}


def status_is_pass(status: str) -> bool:
    return status == PASS


def status_is_review(status: str) -> bool:
    return status == PASS_REVIEW


def status_is_failure(status: str) -> bool:
    return status.startswith("FAIL") or status == ""


def case_allows_review(case: AcceptanceCase, rules: AcceptanceRules) -> bool:
    return rules.allow_review_when_listed and bool(case.accepted_review_prefixes)


def compare_status(
    case: AcceptanceCase, rules: AcceptanceRules, baseline: Mapping[str, str], current: Mapping[str, str]
) -> str:
    baseline_status = str(baseline.get("status", ""))
    current_status = str(current.get("status", ""))
    if status_is_pass(baseline_status) and status_is_failure(current_status):
        return "REGRESSION_PASS_TO_FAIL"
    if status_is_review(baseline_status) and status_is_failure(current_status):
        return "REGRESSION_REVIEW_TO_FAIL"
    if status_is_pass(baseline_status) and status_is_review(current_status):
        if case_allows_review(case, rules):
            return "OK_PASS_TO_ACCEPTED_REVIEW"
        return "WARN_PASS_TO_REVIEW"
    if status_is_failure(baseline_status) and (status_is_pass(current_status) or status_is_review(current_status)):
        return "IMPROVEMENT"
    if status_is_failure(baseline_status) and status_is_failure(current_status):
        return "FAIL_STILL_FAIL"
    if baseline_status == current_status:
        return "UNCHANGED"
    return "CHANGED"


def compare_rows_to_result(
    case: AcceptanceCase,
    rules: AcceptanceRules,
    baseline: Mapping[str, str] | None,
    current: Mapping[str, str] | None,
) -> CompareResult:
    if baseline is None:
        baseline = {}
    if current is None:
        current = {}
    comparison = compare_status(case, rules, baseline, current)
    if baseline == {}:
        comparison = "FAIL_MISSING_BASELINE_ROW"
    if current == {}:
        comparison = (
            "REGRESSION_MISSING_CURRENT_ROW"
            if str(baseline.get("status", "")) in {PASS, PASS_REVIEW}
            else "FAIL_MISSING_CURRENT_ROW"
        )
    return CompareResult(
        id=case.id,
        filename=case.filename,
        protected=case.protected,
        baseline_status=str(baseline.get("status", "")),
        current_status=str(current.get("status", "")),
        comparison=comparison,
        baseline_actual_path=str(baseline.get("actual_path", "")),
        current_actual_path=str(current.get("actual_path", "")),
        notes=case.notes,
    )


def comparison_blocks_exit(result: CompareResult) -> bool:
    if not result.protected:
        return False
    if result.comparison.startswith("REGRESSION"):
        return True
    return result.comparison in {"FAIL_STILL_FAIL", "FAIL_MISSING_BASELINE_ROW", "FAIL_MISSING_CURRENT_ROW"}


def write_comparison_summary(
    path: Path, comparisons: Sequence[CompareResult], baseline_dir: Path, current_dir: Path
) -> None:
    protected_blocks = sum(1 for result in comparisons if comparison_blocks_exit(result))
    counts: dict[str, int] = {}
    for result in comparisons:
        counts[result.comparison] = counts.get(result.comparison, 0) + 1
    lines = [
        f"Baseline: {baseline_dir}",
        f"Current: {current_dir}",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Protected blocking comparisons: {protected_blocks}",
        "",
    ]
    lines.extend(f"{name}: {count}" for name, count in sorted(counts.items()))
    lines.append("")
    for result in comparisons:
        protected = "protected" if result.protected else "unprotected"
        lines.append(
            f"{result.comparison}: {result.id} ({protected}) {result.baseline_status} -> {result.current_status}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_acceptance_runs(
    expected_path: Path, baseline_dir: Path, current_dir: Path
) -> tuple[int, list[CompareResult]]:
    config = load_acceptance_config(expected_path)
    baseline_rows = read_results_csv(baseline_dir / "actual_results.csv")
    current_rows = read_results_csv(current_dir / "actual_results.csv")
    comparisons = [
        compare_rows_to_result(case, config.rules, baseline_rows.get(case.id), current_rows.get(case.id))
        for case in config.cases
    ]
    rows = [comparison_to_row(result) for result in comparisons]
    write_csv_rows(current_dir / "baseline_comparison.csv", rows, COMPARISON_COLUMNS)
    write_comparison_summary(current_dir / "baseline_comparison_summary.txt", comparisons, baseline_dir, current_dir)
    create_upload_zip(current_dir)
    return (1 if any(comparison_blocks_exit(result) for result in comparisons) else 0), comparisons


def resolve_latest_run(reports_root: Path, explicit: str) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    marker = latest_run_file(reports_root)
    if marker.is_file():
        return Path(marker.read_text(encoding="utf-8").strip())
    runs = sorted(path for path in reports_root.glob("run_*") if path.is_dir())
    if not runs:
        raise FileNotFoundError("No locked smoke acceptance run found")
    return runs[-1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Locked smoke acceptance harness for Aaron Sound Sorter.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_run_arguments(subparsers.add_parser("run", help="Run locked smoke acceptance."))
    add_run_arguments(subparsers.add_parser("baseline", help="Run acceptance and copy it to latest_baseline."))
    compare = subparsers.add_parser("compare", help="Compare latest acceptance run to latest baseline.")
    compare.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    compare.add_argument("--expected", default="")
    compare.add_argument("--reports-root", default="")
    compare.add_argument("--baseline-dir", default="")
    compare.add_argument("--latest-run-dir", default="")
    return parser


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--panel-dir", default="")
    parser.add_argument("--expected", default="")
    parser.add_argument("--samples-dir", default="")
    parser.add_argument("--reports-root", default="")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--case-id", action="append", default=[], help="Run only this case id. Repeat for multiple cases."
    )
    parser.add_argument("--start-index", type=int, default=1, help="One-based first case index for chunked AI runs.")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Maximum number of cases to run in this chunk. 0 means all selected cases.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Suppress per-case progress output.")


def resolved_run_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    project_root = Path(args.project_root).expanduser().resolve()
    panel_dir = Path(args.panel_dir).expanduser().resolve() if args.panel_dir else default_panel_dir(project_root)
    expected_path = Path(args.expected).expanduser().resolve() if args.expected else panel_dir / "expected_results.json"
    samples_dir = Path(args.samples_dir).expanduser().resolve() if args.samples_dir else panel_dir / "samples"
    reports_root = (
        Path(args.reports_root).expanduser().resolve() if args.reports_root else default_reports_root(project_root)
    )
    return project_root, panel_dir, expected_path, samples_dir, reports_root


def command_run(args: argparse.Namespace) -> int:
    project_root, panel_dir, expected_path, samples_dir, reports_root = resolved_run_paths(args)
    exit_code, run_dir, _results = run_acceptance(
        project_root=project_root,
        panel_dir=panel_dir,
        expected_path=expected_path,
        samples_dir=samples_dir,
        reports_root=reports_root,
        timeout_seconds=int(args.timeout_seconds),
        case_ids=tuple(args.case_id),
        start_index=int(args.start_index),
        max_cases=int(args.max_cases),
        progress=not bool(args.no_progress),
    )
    print("RUN_DIR=" + str(run_dir))
    return exit_code


def command_baseline(args: argparse.Namespace) -> int:
    project_root, panel_dir, expected_path, samples_dir, reports_root = resolved_run_paths(args)
    exit_code, run_dir, _results = run_acceptance(
        project_root=project_root,
        panel_dir=panel_dir,
        expected_path=expected_path,
        samples_dir=samples_dir,
        reports_root=reports_root,
        timeout_seconds=int(args.timeout_seconds),
        case_ids=tuple(args.case_id),
        start_index=int(args.start_index),
        max_cases=int(args.max_cases),
        progress=not bool(args.no_progress),
    )
    baseline_dir = reports_root / "baselines" / "latest_baseline"
    copy_baseline(run_dir, baseline_dir, reports_root)
    print("RUN_DIR=" + str(run_dir))
    print("BASELINE_DIR=" + str(baseline_dir))
    return exit_code


def command_compare(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).expanduser().resolve()
    reports_root = (
        Path(args.reports_root).expanduser().resolve() if args.reports_root else default_reports_root(project_root)
    )
    expected_path = (
        Path(args.expected).expanduser().resolve()
        if args.expected
        else default_panel_dir(project_root) / "expected_results.json"
    )
    baseline_dir = (
        Path(args.baseline_dir).expanduser().resolve()
        if args.baseline_dir
        else reports_root / "baselines" / "latest_baseline"
    )
    current_dir = resolve_latest_run(reports_root, str(args.latest_run_dir))
    exit_code, _comparisons = compare_acceptance_runs(expected_path, baseline_dir, current_dir)
    print("BASELINE_DIR=" + str(baseline_dir))
    print("CURRENT_RUN_DIR=" + str(current_dir))
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return command_run(args)
    if args.command == "baseline":
        return command_baseline(args)
    if args.command == "compare":
        return command_compare(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
