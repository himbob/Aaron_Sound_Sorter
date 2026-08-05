#!/usr/bin/env python3
"""Category stability gate for Aaron Sound Sorter.

This is a QA-only harness. It runs the sorter as a black box against fixed
real-audio anchor packs and checks category firewalls, neighbor traps, and
catastrophic-steal rules. It does not import classifier internals, alter brains,
train, or change sorter behavior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import locked_smoke_acceptance as smoke

DEFAULT_PROJECT_ROOT = Path("/path/to/Aaron_Sound_Sorter")
DEFAULT_STABILITY_REL = Path("tests/stability")
DEFAULT_SAMPLES_REL = Path("tests/acceptance/locked_smoke_v1/samples")
DEFAULT_TIMEOUT_SECONDS = 90

FAIL_CATASTROPHIC_STEAL = "FAIL_CATASTROPHIC_STEAL"
REGRESSION_OUTPUT_DRIFT = "REGRESSION_UNAPPROVED_OUTPUT_DRIFT"

STABILITY_COLUMNS = smoke.RESULT_COLUMNS + [
    "anchor_pack",
    "target_category",
    "scenario",
    "change_scope",
    "failure_type",
    "forbidden_match",
    "severity",
]

DRIFT_COLUMNS = [
    "id",
    "filename",
    "protected",
    "anchor_pack",
    "target_category",
    "scenario",
    "baseline_status",
    "current_status",
    "baseline_actual_path",
    "current_actual_path",
    "comparison",
    "approved_by_change_budget",
]


@dataclass(frozen=True)
class StabilityCase:
    id: str
    filename: str
    anchor_pack: str
    target_category: str
    scenario: str
    expected_top: str
    accepted_folder_prefixes: tuple[str, ...]
    accepted_review_prefixes: tuple[str, ...]
    forbidden_folder_prefixes: tuple[str, ...]
    protected: bool
    change_scope: str = ""
    failure_type: str = ""
    notes: str = ""


@dataclass(frozen=True)
class StabilityPack:
    anchor_pack: str
    target_category: str
    description: str
    cases: tuple[StabilityCase, ...]


@dataclass(frozen=True)
class SortArtifact:
    filename: str
    sample_path: Path
    output_dir: Path
    manifest_path: Path | None
    sorter: smoke.SorterResult


@dataclass(frozen=True)
class ChangeBudget:
    description: str
    allowed_case_ids: tuple[str, ...]
    allowed_anchor_packs: tuple[str, ...]
    allowed_target_categories: tuple[str, ...]
    allowed_scenarios: tuple[str, ...]


@dataclass(frozen=True)
class DriftResult:
    id: str
    filename: str
    protected: bool
    anchor_pack: str
    target_category: str
    scenario: str
    baseline_status: str
    current_status: str
    baseline_actual_path: str
    current_actual_path: str
    comparison: str
    approved_by_change_budget: bool


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_stability_dir(project_root: Path) -> Path:
    return project_root / DEFAULT_STABILITY_REL


def default_samples_dir(project_root: Path) -> Path:
    return project_root / DEFAULT_SAMPLES_REL


def default_reports_root(project_root: Path) -> Path:
    return project_root / "_reports" / "category_stability_gate"


def latest_run_file(reports_root: Path) -> Path:
    return reports_root / "latest_run_path.txt"


def latest_baseline_file(reports_root: Path) -> Path:
    return reports_root / "latest_baseline_path.txt"


def write_latest_path(path_file: Path, run_dir: Path) -> None:
    path_file.parent.mkdir(parents=True, exist_ok=True)
    path_file.write_text(str(run_dir) + "\n", encoding="utf-8")


def make_run_dir(reports_root: Path) -> Path:
    run_dir = reports_root / f"run_{timestamp()}"
    counter = 2
    while run_dir.exists():
        run_dir = reports_root / f"run_{timestamp()}_{counter}"
        counter += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def stable_output_name(filename: str) -> str:
    stem = Path(filename).stem
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem).strip("_")
    digest = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:10]
    return f"{safe[:60]}_{digest}"


def load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return data


def parse_string_tuple(data: Mapping[str, Any], key: str) -> tuple[str, ...]:
    return tuple(str(item) for item in data.get(key, []))


def parse_stability_case(pack_name: str, target_category: str, data: Mapping[str, Any]) -> StabilityCase:
    return StabilityCase(
        id=str(data["id"]),
        filename=str(data["filename"]),
        anchor_pack=pack_name,
        target_category=str(data.get("target_category", target_category)),
        scenario=str(data["scenario"]),
        expected_top=str(data.get("expected_top", "")),
        accepted_folder_prefixes=parse_string_tuple(data, "accepted_folder_prefixes"),
        accepted_review_prefixes=parse_string_tuple(data, "accepted_review_prefixes"),
        forbidden_folder_prefixes=parse_string_tuple(data, "forbidden_folder_prefixes"),
        protected=bool(data.get("protected", True)),
        change_scope=str(data.get("change_scope", "")),
        failure_type=str(data.get("failure_type", "")),
        notes=str(data.get("notes", "")),
    )


def load_stability_pack(path: Path) -> StabilityPack:
    data = load_json(path)
    pack_name = str(data.get("anchor_pack", path.stem))
    target_category = str(data.get("target_category", pack_name))
    cases = tuple(
        parse_stability_case(pack_name, target_category, item)
        for item in data.get("cases", [])
        if isinstance(item, dict)
    )
    return StabilityPack(
        anchor_pack=pack_name,
        target_category=target_category,
        description=str(data.get("description", "")),
        cases=cases,
    )


def load_stability_packs(stability_dir: Path) -> tuple[StabilityPack, ...]:
    paths = sorted(stability_dir.glob("anchors_*.json"))
    if not paths:
        raise FileNotFoundError(f"No stability anchor files found in {stability_dir}")
    return tuple(load_stability_pack(path) for path in paths)


def flatten_cases(packs: Sequence[StabilityPack]) -> tuple[StabilityCase, ...]:
    cases: list[StabilityCase] = []
    seen: set[str] = set()
    for pack in packs:
        for case in pack.cases:
            if case.id in seen:
                raise ValueError(f"Duplicate stability case id: {case.id}")
            seen.add(case.id)
            cases.append(case)
    return tuple(cases)


def to_acceptance_case(case: StabilityCase) -> smoke.AcceptanceCase:
    return smoke.AcceptanceCase(
        id=case.id,
        filename=case.filename,
        expected_top=case.expected_top,
        accepted_folder_prefixes=case.accepted_folder_prefixes,
        accepted_review_prefixes=case.accepted_review_prefixes,
        protected=case.protected,
        notes=case.notes,
    )


def forbidden_prefix_match(actual_path: str, prefixes: Sequence[str]) -> str:
    for prefix in prefixes:
        if smoke.path_starts_with(actual_path, prefix):
            return prefix
    return ""


def apply_firewall(case: StabilityCase, result: smoke.CaseResult) -> smoke.CaseResult:
    forbidden = forbidden_prefix_match(result.actual_path, case.forbidden_folder_prefixes)
    if not forbidden:
        return result
    return smoke.CaseResult(
        id=result.id,
        filename=result.filename,
        protected=result.protected,
        status=FAIL_CATASTROPHIC_STEAL,
        actual_path=result.actual_path,
        actual_top=result.actual_top,
        expected_top=result.expected_top,
        matched_prefix=forbidden,
        sorter_returncode=result.sorter_returncode,
        sample_path=result.sample_path,
        manifest_path=result.manifest_path,
        output_dir=result.output_dir,
        notes=result.notes,
        error=f"Actual path matched forbidden category firewall: {forbidden}",
    )


def stability_result_to_row(case: StabilityCase, result: smoke.CaseResult) -> dict[str, str]:
    row = smoke.result_to_row(result)
    row.update(
        {
            "anchor_pack": case.anchor_pack,
            "target_category": case.target_category,
            "scenario": case.scenario,
            "change_scope": case.change_scope,
            "failure_type": case.failure_type,
            "forbidden_match": result.matched_prefix if result.status == FAIL_CATASTROPHIC_STEAL else "",
            "severity": "catastrophic"
            if result.status == FAIL_CATASTROPHIC_STEAL
            else ("blocking" if smoke.result_blocks_exit(result) else "ok"),
        }
    )
    return row


def run_unique_sorts(
    project_root: Path,
    samples_dir: Path,
    run_dir: Path,
    cases: Sequence[StabilityCase],
    timeout_seconds: int,
    progress: bool,
) -> dict[str, SortArtifact]:
    artifacts: dict[str, SortArtifact] = {}
    log_path = run_dir / "run.log"
    filenames = sorted({case.filename for case in cases})
    for index, filename in enumerate(filenames, start=1):
        sample_path = samples_dir / filename
        output_dir = run_dir / "sample_outputs" / stable_output_name(filename)
        print_progress(f"[sort {index}/{len(filenames)}] {filename}", progress)
        if sample_path.is_file():
            sorter = smoke.run_sorter(project_root, sample_path, output_dir, timeout_seconds, log_path)
            manifest_path = smoke.find_manifest_path(output_dir)
        else:
            sorter = smoke.SorterResult(returncode=None, timed_out=False, error="Sample file is missing")
            manifest_path = None
            smoke.append_run_log(log_path, "\n=== MISSING SAMPLE ===")
            smoke.append_run_log(log_path, str(sample_path))
        artifacts[filename] = SortArtifact(
            filename=filename,
            sample_path=sample_path,
            output_dir=output_dir,
            manifest_path=manifest_path,
            sorter=sorter,
        )
    return artifacts


def evaluate_stability_case(
    case: StabilityCase,
    artifact: SortArtifact,
    rules: smoke.AcceptanceRules,
) -> smoke.CaseResult:
    acceptance_case = to_acceptance_case(case)
    result = smoke.evaluate_case(
        acceptance_case,
        artifact.sample_path,
        artifact.output_dir,
        artifact.manifest_path,
        artifact.sorter,
        rules,
    )
    return apply_firewall(case, result)


def combined_manifest_rows(
    artifacts: Mapping[str, SortArtifact], case_by_filename: Mapping[str, Sequence[StabilityCase]]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for filename, artifact in artifacts.items():
        if artifact.manifest_path is None or not artifact.manifest_path.is_file():
            continue
        for manifest_row in smoke.read_manifest_rows(artifact.manifest_path):
            for case in case_by_filename.get(filename, ()):
                merged = {
                    "case_id": case.id,
                    "case_filename": case.filename,
                    "anchor_pack": case.anchor_pack,
                    "target_category": case.target_category,
                    "scenario": case.scenario,
                }
                merged.update(dict(manifest_row))
                rows.append(merged)
    return rows


def write_pack_snapshot(stability_dir: Path, run_dir: Path) -> None:
    snapshot = run_dir / "anchor_packs"
    snapshot.mkdir(parents=True, exist_ok=True)
    for path in sorted(stability_dir.glob("anchors_*.json")):
        shutil.copy2(path, snapshot / path.name)
    budget = stability_dir / "change_budget.json"
    if budget.is_file():
        shutil.copy2(budget, run_dir / "change_budget.json")


def status_counts(results: Sequence[smoke.CaseResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def metric_ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{numerator / denominator:.3f}"


def changed_outputs_count(results: Sequence[smoke.CaseResult], baseline_dir: Path | None) -> int | None:
    if baseline_dir is None or not (baseline_dir / "actual_results.csv").is_file():
        return None
    baseline = read_results_by_id(baseline_dir / "actual_results.csv")
    count = 0
    for result in results:
        old = baseline.get(result.id)
        if old and old.get("actual_path", "") != result.actual_path:
            count += 1
    return count


def write_summary(
    path: Path,
    packs: Sequence[StabilityPack],
    results: Sequence[smoke.CaseResult],
    timeout_seconds: int,
    baseline_dir: Path | None = None,
) -> None:
    protected = [result for result in results if result.protected]
    protected_total = len(protected)
    protected_passish = sum(1 for result in protected if result.status in {smoke.PASS, smoke.PASS_REVIEW})
    protected_leaf_pass = sum(1 for result in protected if result.status == smoke.PASS)
    review_count = sum(1 for result in protected if result.status == smoke.PASS_REVIEW)
    catastrophic = sum(1 for result in protected if result.status == FAIL_CATASTROPHIC_STEAL)
    drift_count = changed_outputs_count(results, baseline_dir)
    lines = [
        "Category Stability Gate",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Anchor packs: {len(packs)}",
        f"Cases: {len(results)}",
        f"Protected cases: {protected_total}",
        f"Timeout seconds per unique sample: {timeout_seconds}",
        "",
        f"family_accuracy: {metric_ratio(protected_passish, protected_total)}",
        f"leaf_accuracy: {metric_ratio(protected_leaf_pass, protected_total)}",
        f"review_rate: {metric_ratio(review_count, protected_total)}",
        f"catastrophic_steal_count: {catastrophic}",
        f"changed_outputs_count: {drift_count if drift_count is not None else 'not compared'}",
        "",
    ]
    lines.extend(f"{status}: {count}" for status, count in sorted(status_counts(results).items()))
    lines.append("")
    for result in results:
        protected_label = "protected" if result.protected else "unprotected"
        lines.append(f"{result.status}: {result.id} ({protected_label}) -> {result.actual_path or result.error}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_upload_zip(run_dir: Path) -> Path:
    zip_path = run_dir / "category_stability_upload_back.zip"
    names = [
        "actual_results.csv",
        "failures.csv",
        "summary.txt",
        "full_manifest_combined.csv",
        "run.log",
        "drift_comparison.csv",
        "drift_comparison_summary.txt",
        "change_budget.json",
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            path = run_dir / name
            if path.is_file():
                archive.write(path, arcname=name)
        packs = run_dir / "anchor_packs"
        if packs.is_dir():
            for path in sorted(packs.glob("*.json")):
                archive.write(path, arcname=f"anchor_packs/{path.name}")
    return zip_path


def print_progress(message: str, enabled: bool) -> None:
    if enabled:
        print(message, flush=True)


def cases_by_filename(cases: Sequence[StabilityCase]) -> dict[str, list[StabilityCase]]:
    grouped: dict[str, list[StabilityCase]] = {}
    for case in cases:
        grouped.setdefault(case.filename, []).append(case)
    return grouped


def run_stability_gate(
    project_root: Path,
    stability_dir: Path,
    samples_dir: Path,
    reports_root: Path,
    timeout_seconds: int,
    progress: bool = True,
    baseline_dir: Path | None = None,
) -> tuple[int, Path, list[smoke.CaseResult]]:
    packs = load_stability_packs(stability_dir)
    cases = flatten_cases(packs)
    run_dir = make_run_dir(reports_root)
    write_latest_path(latest_run_file(reports_root), run_dir)
    write_pack_snapshot(stability_dir, run_dir)
    log_path = run_dir / "run.log"
    smoke.append_run_log(log_path, "project_root: " + str(project_root))
    smoke.append_run_log(log_path, "stability_dir: " + str(stability_dir))
    smoke.append_run_log(log_path, "samples_dir: " + str(samples_dir))
    smoke.append_run_log(log_path, "cases: " + str(len(cases)))
    print_progress("RUN_DIR=" + str(run_dir), progress)
    print_progress(
        f"Stability cases: {len(cases)} across {len({case.filename for case in cases})} unique samples", progress
    )
    artifacts = run_unique_sorts(project_root, samples_dir, run_dir, cases, timeout_seconds, progress)
    rules = smoke.AcceptanceRules()
    results = [evaluate_stability_case(case, artifacts[case.filename], rules) for case in cases]
    result_rows = [stability_result_to_row(case, result) for case, result in zip(cases, results)]
    failure_rows = [
        row for row, result in zip(result_rows, results) if result.status not in {smoke.PASS, smoke.PASS_REVIEW}
    ]
    smoke.write_csv_rows(run_dir / "actual_results.csv", result_rows, STABILITY_COLUMNS)
    smoke.write_csv_rows(run_dir / "failures.csv", failure_rows, STABILITY_COLUMNS)
    combined = combined_manifest_rows(artifacts, cases_by_filename(cases))
    smoke.write_csv_rows(
        run_dir / "full_manifest_combined.csv",
        combined,
        ["case_id", "case_filename", "anchor_pack", "target_category", "scenario"],
    )
    write_summary(run_dir / "summary.txt", packs, results, timeout_seconds, baseline_dir)
    create_upload_zip(run_dir)
    return exit_code_for_results(results), run_dir, results


def exit_code_for_results(results: Sequence[smoke.CaseResult]) -> int:
    return 1 if any(smoke.result_blocks_exit(result) for result in results) else 0


def copy_baseline(run_dir: Path, baseline_dir: Path, reports_root: Path) -> None:
    if baseline_dir.exists():
        shutil.rmtree(baseline_dir)
    baseline_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir, baseline_dir)
    write_latest_path(latest_baseline_file(reports_root), baseline_dir)


def read_results_by_id(path: Path) -> dict[str, dict[str, str]]:
    smoke.configure_csv_limit()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["id"]: dict(row) for row in csv.DictReader(handle)}


def load_change_budget(path: Path) -> ChangeBudget:
    if not path.is_file():
        return ChangeBudget(
            description="No change budget file found; protected output drift is unapproved.",
            allowed_case_ids=(),
            allowed_anchor_packs=(),
            allowed_target_categories=(),
            allowed_scenarios=(),
        )
    data = load_json(path)
    return ChangeBudget(
        description=str(data.get("description", "")),
        allowed_case_ids=parse_string_tuple(data, "allowed_case_ids"),
        allowed_anchor_packs=parse_string_tuple(data, "allowed_anchor_packs"),
        allowed_target_categories=parse_string_tuple(data, "allowed_target_categories"),
        allowed_scenarios=parse_string_tuple(data, "allowed_scenarios"),
    )


def change_budget_approves(row: Mapping[str, str], budget: ChangeBudget) -> bool:
    return (
        str(row.get("id", "")) in budget.allowed_case_ids
        or str(row.get("anchor_pack", "")) in budget.allowed_anchor_packs
        or str(row.get("target_category", "")) in budget.allowed_target_categories
        or str(row.get("scenario", "")) in budget.allowed_scenarios
    )


def compare_output_drift(
    baseline_dir: Path,
    current_dir: Path,
    budget: ChangeBudget,
) -> tuple[int, list[DriftResult]]:
    baseline = read_results_by_id(baseline_dir / "actual_results.csv")
    current = read_results_by_id(current_dir / "actual_results.csv")
    results: list[DriftResult] = []
    for case_id in sorted(set(baseline) | set(current)):
        old = baseline.get(case_id, {})
        new = current.get(case_id, {})
        protected = str(new.get("protected") or old.get("protected", "")).lower() == "true"
        approved = change_budget_approves(new or old, budget)
        old_path = str(old.get("actual_path", ""))
        new_path = str(new.get("actual_path", ""))
        old_status = str(old.get("status", ""))
        new_status = str(new.get("status", ""))
        if not old:
            comparison = "NEW_CASE"
        elif not new:
            comparison = "MISSING_CURRENT_CASE"
        elif old_path != new_path and protected and not approved:
            comparison = REGRESSION_OUTPUT_DRIFT
        elif old_status in {smoke.PASS, smoke.PASS_REVIEW} and new_status not in {smoke.PASS, smoke.PASS_REVIEW}:
            comparison = "REGRESSION_PASS_TO_FAIL"
        elif old_path != new_path:
            comparison = "APPROVED_OUTPUT_DRIFT" if approved else "OUTPUT_DRIFT_UNPROTECTED"
        else:
            comparison = "UNCHANGED"
        row = new or old
        results.append(
            DriftResult(
                id=case_id,
                filename=str(row.get("filename", "")),
                protected=protected,
                anchor_pack=str(row.get("anchor_pack", "")),
                target_category=str(row.get("target_category", "")),
                scenario=str(row.get("scenario", "")),
                baseline_status=old_status,
                current_status=new_status,
                baseline_actual_path=old_path,
                current_actual_path=new_path,
                comparison=comparison,
                approved_by_change_budget=approved,
            )
        )
    return (1 if any(drift_blocks_exit(item) for item in results) else 0), results


def drift_blocks_exit(result: DriftResult) -> bool:
    if not result.protected:
        return False
    return result.comparison in {
        REGRESSION_OUTPUT_DRIFT,
        "REGRESSION_PASS_TO_FAIL",
        "MISSING_CURRENT_CASE",
    }


def drift_to_row(result: DriftResult) -> dict[str, str]:
    return {
        "id": result.id,
        "filename": result.filename,
        "protected": "true" if result.protected else "false",
        "anchor_pack": result.anchor_pack,
        "target_category": result.target_category,
        "scenario": result.scenario,
        "baseline_status": result.baseline_status,
        "current_status": result.current_status,
        "baseline_actual_path": result.baseline_actual_path,
        "current_actual_path": result.current_actual_path,
        "comparison": result.comparison,
        "approved_by_change_budget": "true" if result.approved_by_change_budget else "false",
    }


def write_drift_summary(path: Path, results: Sequence[DriftResult], budget: ChangeBudget) -> None:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.comparison] = counts.get(result.comparison, 0) + 1
    blocking = sum(1 for result in results if drift_blocks_exit(result))
    lines = [
        "Category Stability Drift Comparison",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Blocking regressions: {blocking}",
        f"Change budget: {budget.description}",
        "",
    ]
    lines.extend(f"{name}: {count}" for name, count in sorted(counts.items()))
    lines.append("")
    for result in results:
        lines.append(f"{result.comparison}: {result.id} {result.baseline_actual_path} -> {result.current_actual_path}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_to_baseline(stability_dir: Path, reports_root: Path, baseline_dir: Path, current_dir: Path) -> int:
    budget = load_change_budget(stability_dir / "change_budget.json")
    exit_code, results = compare_output_drift(baseline_dir, current_dir, budget)
    rows = [drift_to_row(result) for result in results]
    smoke.write_csv_rows(current_dir / "drift_comparison.csv", rows, DRIFT_COLUMNS)
    write_drift_summary(current_dir / "drift_comparison_summary.txt", results, budget)
    create_upload_zip(current_dir)
    return exit_code


def resolve_latest_run(reports_root: Path, explicit: str) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    marker = latest_run_file(reports_root)
    if marker.is_file():
        return Path(marker.read_text(encoding="utf-8").strip())
    runs = sorted(path for path in reports_root.glob("run_*") if path.is_dir())
    if not runs:
        raise FileNotFoundError("No category stability run found")
    return runs[-1]


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--stability-dir", default="")
    parser.add_argument("--samples-dir", default="")
    parser.add_argument("--reports-root", default="")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--no-progress", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run category stability firewalls and neighbor traps.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_common_run_args(subparsers.add_parser("run", help="Run category stability gate."))
    add_common_run_args(subparsers.add_parser("baseline", help="Run gate and copy to latest baseline."))
    compare = subparsers.add_parser("compare", help="Compare latest stability run to baseline.")
    compare.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    compare.add_argument("--stability-dir", default="")
    compare.add_argument("--reports-root", default="")
    compare.add_argument("--baseline-dir", default="")
    compare.add_argument("--latest-run-dir", default="")
    return parser


def resolve_run_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    project_root = Path(args.project_root).expanduser().resolve()
    stability_dir = (
        Path(args.stability_dir).expanduser().resolve() if args.stability_dir else default_stability_dir(project_root)
    )
    samples_dir = (
        Path(args.samples_dir).expanduser().resolve() if args.samples_dir else default_samples_dir(project_root)
    )
    reports_root = (
        Path(args.reports_root).expanduser().resolve() if args.reports_root else default_reports_root(project_root)
    )
    return project_root, stability_dir, samples_dir, reports_root


def command_run(args: argparse.Namespace) -> int:
    project_root, stability_dir, samples_dir, reports_root = resolve_run_paths(args)
    exit_code, run_dir, _results = run_stability_gate(
        project_root=project_root,
        stability_dir=stability_dir,
        samples_dir=samples_dir,
        reports_root=reports_root,
        timeout_seconds=int(args.timeout_seconds),
        progress=not bool(args.no_progress),
    )
    print("RUN_DIR=" + str(run_dir))
    return exit_code


def command_baseline(args: argparse.Namespace) -> int:
    project_root, stability_dir, samples_dir, reports_root = resolve_run_paths(args)
    exit_code, run_dir, _results = run_stability_gate(
        project_root=project_root,
        stability_dir=stability_dir,
        samples_dir=samples_dir,
        reports_root=reports_root,
        timeout_seconds=int(args.timeout_seconds),
        progress=not bool(args.no_progress),
    )
    baseline_dir = reports_root / "baselines" / "latest_baseline"
    copy_baseline(run_dir, baseline_dir, reports_root)
    print("RUN_DIR=" + str(run_dir))
    print("BASELINE_DIR=" + str(baseline_dir))
    return exit_code


def command_compare(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).expanduser().resolve()
    stability_dir = (
        Path(args.stability_dir).expanduser().resolve() if args.stability_dir else default_stability_dir(project_root)
    )
    reports_root = (
        Path(args.reports_root).expanduser().resolve() if args.reports_root else default_reports_root(project_root)
    )
    baseline_dir = (
        Path(args.baseline_dir).expanduser().resolve()
        if args.baseline_dir
        else reports_root / "baselines" / "latest_baseline"
    )
    current_dir = resolve_latest_run(reports_root, str(args.latest_run_dir))
    exit_code = compare_to_baseline(stability_dir, reports_root, baseline_dir, current_dir)
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
