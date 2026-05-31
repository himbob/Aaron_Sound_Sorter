from __future__ import annotations

import json
from pathlib import Path

from aaron_sound_sorter.engine.sort_timing import SortTimingProfiler, write_sort_timing_reports


def test_sort_timing_profiler_records_run_and_file_stages(tmp_path: Path) -> None:
    profiler = SortTimingProfiler()
    source_path = tmp_path / "sample.wav"
    source_path.write_bytes(b"fake")

    with profiler.stage("prepare_input"):
        pass
    with profiler.file_stage(source_path, "analysis_packet"):
        pass

    snapshot = profiler.snapshot()

    assert snapshot["policy"] == "diagnostic_only_does_not_affect_sorting"
    assert "prepare_input" in snapshot["run_stages"]
    assert "file:analysis_packet" in snapshot["run_stages"]
    assert snapshot["file_count_with_timings"] == 1
    assert profiler.file_summary(source_path)["stages"]["analysis_packet"]["call_count"] == 1


def test_sort_timing_reports_write_only_to_requested_output_folder(tmp_path: Path) -> None:
    snapshot = {
        "policy": "diagnostic_only_does_not_affect_sorting",
        "run_stages": {"classify_files": {"total_seconds": 0.1, "call_count": 1, "average_seconds": 0.1}},
        "slowest_files": [],
    }

    json_path, text_path = write_sort_timing_reports(tmp_path, snapshot)

    assert json_path.parent == tmp_path.resolve()
    assert text_path.parent == tmp_path.resolve()
    assert json.loads(json_path.read_text())["policy"] == "diagnostic_only_does_not_affect_sorting"
    assert "classify_files" in text_path.read_text()
