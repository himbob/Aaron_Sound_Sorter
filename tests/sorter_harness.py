from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

csv.field_size_limit(sys.maxsize)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SORTER = PROJECT_ROOT / "Aaron_Sound_Sorter.py"
DEFAULT_BRAIN = PROJECT_ROOT / "stage4_folder_brain.json"
REGRESSION_AUDIO_DIR = PROJECT_ROOT / "tests" / "regression_audio"
PYTEST_OUTPUT_ROOT = PROJECT_ROOT / "_reports" / "pytest_outputs"


def safe_case_name(name: str) -> str:
    """Return a filesystem-safe, readable case name for pytest sort outputs."""
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name).strip("_") or "case"


def run_sort(
    source_path: Path,
    *,
    output_key: str,
    brain_path: Path | None = None,
    neutralize_input_name: bool = False,
    timeout: int = 75,
    workers: int | None = 1,
) -> list[dict[str, str]]:
    """Run the command-line sorter for one pytest fixture and return manifest rows."""
    assert SORTER.exists(), f"Missing sorter: {SORTER}"
    if brain_path is not None:
        assert brain_path.exists(), f"Missing brain: {brain_path}"
    if not source_path.exists():
        pytest.skip(f"missing regression audio: {source_path}")

    case_root = PYTEST_OUTPUT_ROOT / output_key / safe_case_name(source_path.stem)
    if case_root.exists():
        shutil.rmtree(case_root, ignore_errors=True)
    input_path = source_path
    output_path = case_root / "output" if neutralize_input_name else case_root
    if neutralize_input_name:
        input_dir = case_root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        neutral = input_dir / f"sample_0001{source_path.suffix}"
        shutil.copy2(source_path, neutral)
        input_path = neutral
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(SORTER), "sort", str(input_path), str(output_path), "--no-zip"]
    if brain_path is not None:
        cmd.extend(["--brain", str(brain_path)])
    if workers is not None:
        cmd.extend(["--workers", str(int(workers))])

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    assert result.returncode == 0, result.stdout
    manifest = output_path / "Aaron_Sorted_Sounds_manifest.csv"
    assert manifest.exists(), result.stdout
    with manifest.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, f"Manifest had no rows: {manifest}"
    return rows


def run_regression_sample(
    sample_name: str,
    *,
    output_key: str,
    audio_dir: Path = REGRESSION_AUDIO_DIR,
    brain_path: Path | None = None,
    neutralize_input_name: bool = False,
    timeout: int = 75,
    workers: int | None = 1,
) -> dict[str, str]:
    """Sort one named regression fixture and return its single manifest row."""
    rows = run_sort(
        audio_dir / sample_name,
        output_key=output_key,
        brain_path=brain_path,
        neutralize_input_name=neutralize_input_name,
        timeout=timeout,
        workers=workers,
    )
    assert len(rows) == 1
    return rows[0]


def run_folder_once(
    audio_dir: Path,
    *,
    output_key: str,
    brain_path: Path | None = None,
    timeout: int = 240,
) -> dict[str, dict[str, str]]:
    """Sort a fixture folder once and return manifest rows keyed by source name."""
    rows = run_sort(audio_dir, output_key=output_key, brain_path=brain_path, timeout=timeout, workers=None)
    return {Path(row.get("source_path", "")).name: row for row in rows}


def row_label(row: dict[str, str]) -> str:
    """Return the normalized final folder path from current or older manifest fields."""
    return (
        row.get("final_label")
        or row.get("new_relative_path")
        or row.get("folder_path")
        or row.get("placed_path")
        or row.get("category")
        or ""
    ).replace("\\", "/")
