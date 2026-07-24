from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from cleanup_project import collect_candidates  # noqa: E402


def test_standard_cleanup_preserves_critical_training_neural_and_brain_assets(tmp_path: Path) -> None:
    (tmp_path / "Aaron_Sound_Sorter.py").write_text("# runner\n", encoding="utf-8")
    critical_paths = [
        tmp_path / "training" / "locked_curated_v1" / "sample.wav",
        tmp_path / "_models" / "pinned_encoder" / "weights.bin",
        tmp_path / "neural_artifacts" / "run" / "neural_model_registry.json",
        tmp_path / "stage4_folder_brain.json",
    ]
    for path in critical_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"critical")
    generated_report = tmp_path / "_reports" / "temporary" / "report.csv"
    generated_report.parent.mkdir(parents=True)
    generated_report.write_text("generated\n", encoding="utf-8")

    candidates = {candidate.path for candidate in collect_candidates(tmp_path, "standard")}

    assert tmp_path / "_reports" in candidates
    assert all(path not in candidates for path in critical_paths)
    assert not any(
        critical_path == candidate or candidate in critical_path.parents
        for critical_path in critical_paths
        for candidate in candidates
    )
