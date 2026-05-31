# SOURCE-NAME BLINDNESS INVARIANT:
# This module records elapsed time for diagnostics only. It must never parse
# producer-provided path text as classification evidence.
"""Lightweight timing profiler for sorter speed work.

The profiler is intentionally small and in-memory. It records where runtime is
spent during a single sort run, then writes report data only to the chosen output
folder. It does not affect voter scores, consensus results, or placement.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any


@dataclass
class TimingBucket:
    """Aggregate timing information for one named stage.

    Args:
        total_seconds: Sum of elapsed seconds recorded for this stage.
        call_count: Number of timed calls included in the bucket.

    Attributes:
        total_seconds: Total measured wall-clock seconds.
        call_count: Number of completed timing spans.

    Side Effects:
        None. The bucket is mutated by ``add``.

    Raises:
        No intentional exceptions.

    Important Constraints:
        This class is diagnostics-only and must never influence sorting.
    """

    total_seconds: float = 0.0
    call_count: int = 0

    def add(self, elapsed_seconds: float) -> None:
        """Add one elapsed duration to the bucket.

        Args:
            elapsed_seconds: Non-negative elapsed wall-clock seconds.

        Returns:
            None.

        Side Effects:
            Updates ``total_seconds`` and ``call_count``.

        Raises:
            No intentional exceptions.

        Important Constraints:
            Negative values are clamped to zero because clock jitter should not
            poison the diagnostic report.
        """
        self.total_seconds += max(0.0, float(elapsed_seconds))
        self.call_count += 1

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-safe representation of this bucket."""
        average = self.total_seconds / self.call_count if self.call_count else 0.0
        return {
            "total_seconds": round(float(self.total_seconds), 6),
            "call_count": int(self.call_count),
            "average_seconds": round(float(average), 6),
        }


@dataclass
class FileTimingRecord:
    """Timing buckets for one analyzed file inside one sorter run.

    Args:
        stage_buckets: Stage-name to aggregate timing bucket.

    Attributes:
        stage_buckets: Per-file timing buckets.

    Side Effects:
        None at construction time. ``add`` mutates timing buckets.

    Raises:
        No intentional exceptions.

    Important Constraints:
        The record does not store final labels or voter decisions.
    """

    stage_buckets: dict[str, TimingBucket] = field(default_factory=dict)

    def add(self, stage_name: str, elapsed_seconds: float) -> None:
        """Add elapsed time for a per-file stage."""
        bucket = self.stage_buckets.setdefault(stage_name, TimingBucket())
        bucket.add(elapsed_seconds)

    def as_dict(self) -> dict[str, dict[str, float | int]]:
        """Return JSON-safe per-stage timing details."""
        return {name: bucket.as_dict() for name, bucket in sorted(self.stage_buckets.items())}

    def total_seconds(self) -> float:
        """Return total measured seconds across all per-file stages."""
        return float(sum(bucket.total_seconds for bucket in self.stage_buckets.values()))


class SortTimingProfiler:
    """In-memory profiler for one sort run.

    The profiler is safe for the existing threaded sorting path. It records
    aggregate run stages plus per-file stages. The source path is used only as a
    diagnostics key and never returned to voters or policy code as evidence.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._run_buckets: dict[str, TimingBucket] = {}
        self._file_records: dict[str, FileTimingRecord] = {}
        self.enabled = True

    def reset(self) -> None:
        """Clear all timing data for a new user-facing run."""
        with self._lock:
            self._run_buckets.clear()
            self._file_records.clear()

    @contextmanager
    def stage(self, stage_name: str) -> Iterator[None]:
        """Measure one run-level stage.

        Args:
            stage_name: Human-readable stage identifier.

        Yields:
            None.

        Side Effects:
            Adds elapsed time to the profiler when the context exits.

        Raises:
            Re-raises any exception from the wrapped code.

        Important Constraints:
            The timer must not catch or hide sorter failures.
        """
        if not self.enabled:
            yield
            return
        started_at = perf_counter()
        try:
            yield
        finally:
            self.add_stage_time(stage_name, perf_counter() - started_at)

    @contextmanager
    def file_stage(self, source_path: Path, stage_name: str) -> Iterator[None]:
        """Measure one per-file stage.

        Args:
            source_path: Staged file path used only as a diagnostics key.
            stage_name: Human-readable per-file stage identifier.

        Yields:
            None.

        Side Effects:
            Adds elapsed time to the per-file record and the aggregate stage.

        Raises:
            Re-raises any exception from the wrapped code.

        Important Constraints:
            The key is never parsed for source words and never used in voting.
        """
        if not self.enabled:
            yield
            return
        started_at = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - started_at
            self.add_file_stage_time(source_path, stage_name, elapsed)

    def add_stage_time(self, stage_name: str, elapsed_seconds: float) -> None:
        """Record elapsed time for a run-level stage."""
        with self._lock:
            bucket = self._run_buckets.setdefault(stage_name, TimingBucket())
            bucket.add(elapsed_seconds)

    def add_file_stage_time(self, source_path: Path, stage_name: str, elapsed_seconds: float) -> None:
        """Record elapsed time for one file-stage pair."""
        source_key = str(Path(source_path))
        with self._lock:
            record = self._file_records.setdefault(source_key, FileTimingRecord())
            record.add(stage_name, elapsed_seconds)
            aggregate_bucket = self._run_buckets.setdefault(f"file:{stage_name}", TimingBucket())
            aggregate_bucket.add(elapsed_seconds)

    def file_summary(self, source_path: Path) -> dict[str, Any]:
        """Return JSON-safe timing data for one file."""
        source_key = str(Path(source_path))
        with self._lock:
            record = self._file_records.get(source_key)
            if record is None:
                return {"total_seconds": 0.0, "stages": {}}
            return {
                "total_seconds": round(record.total_seconds(), 6),
                "stages": record.as_dict(),
            }

    def snapshot(self) -> dict[str, Any]:
        """Return a full JSON-safe profiler snapshot."""
        with self._lock:
            slowest_files = sorted(
                (
                    {
                        "source_path": source_key,
                        "total_seconds": round(record.total_seconds(), 6),
                        "stages": record.as_dict(),
                    }
                    for source_key, record in self._file_records.items()
                ),
                key=lambda row: (-float(row["total_seconds"]), str(row["source_path"])),
            )[:20]
            return {
                "policy": "diagnostic_only_does_not_affect_sorting",
                "run_stages": {name: bucket.as_dict() for name, bucket in sorted(self._run_buckets.items())},
                "file_count_with_timings": len(self._file_records),
                "slowest_files": slowest_files,
            }


def write_sort_timing_reports(output_dir: Path, snapshot: dict[str, Any]) -> tuple[Path, Path]:
    """Write JSON and text timing reports to the sort output folder.

    Args:
        output_dir: User-selected output folder for the current run.
        snapshot: JSON-safe snapshot from ``SortTimingProfiler``.

    Returns:
        Tuple of ``(json_path, text_path)``.

    Side Effects:
        Creates ``Aaron_Sort_Timing_Profile.json`` and
        ``Aaron_Sort_Timing_Profile.txt`` in ``output_dir``.

    Raises:
        Any filesystem exception raised by ``Path.write_text``.

    Important Constraints:
        Reports are output artifacts only. They are never written to ``src/``,
        ``tests/``, or the project root by this helper.
    """
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "Aaron_Sort_Timing_Profile.json"
    text_path = output / "Aaron_Sort_Timing_Profile.txt"
    json_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["Aaron Sound Sorter Timing Profile", "", "Run stages:"]
    run_stages = snapshot.get("run_stages", {}) if isinstance(snapshot, dict) else {}
    if isinstance(run_stages, dict):
        for stage_name, bucket in sorted(run_stages.items()):
            if isinstance(bucket, dict):
                lines.append(
                    f"  {stage_name}: total={bucket.get('total_seconds', 0)}s "
                    f"calls={bucket.get('call_count', 0)} avg={bucket.get('average_seconds', 0)}s"
                )
    lines.append("")
    lines.append("Slowest files:")
    for row in snapshot.get("slowest_files", []) if isinstance(snapshot, dict) else []:
        if isinstance(row, dict):
            lines.append(f"  {row.get('total_seconds', 0)}s  {row.get('source_path', '')}")
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, text_path
