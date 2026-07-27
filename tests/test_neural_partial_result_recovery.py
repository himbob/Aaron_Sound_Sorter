from __future__ import annotations

import json
from pathlib import Path

from aaron_sound_sorter.gui.models import PreviewRow
from aaron_sound_sorter.gui.neural_explanations import broad_clap_summary, panns_summary
from aaron_sound_sorter.neural_audio.runtime import _batch_from_checkpoint


def test_runtime_recovers_completed_rows_from_timeout_checkpoint(tmp_path: Path) -> None:
    response = tmp_path / "response.json"
    log = tmp_path / "runtime.log"
    response.write_text(
        json.dumps(
            {
                "status": "processing",
                "message": "Processed 1/3 audio waveform(s).",
                "index_path": "/models/index.npz",
                "predictions": [
                    {
                        "row_id": "00001",
                        "file_sha256": "a" * 64,
                        "predicted_label": "FX/Impacts/Generic Impact/One Shots",
                        "second_label": "FX/Designed FX/Generic/One Shots",
                        "top_similarity": 0.61,
                        "second_similarity": 0.52,
                        "margin": 0.09,
                        "radius_ratio": 1.4,
                        "known_distribution": False,
                        "label_example_count": 3,
                        "semantic_status": "advisory_only",
                        "semantic_family": "fx_impact",
                        "semantic_top_score": 0.08,
                        "panns_status": "advisory_only",
                        "panns_events": [{"label": "Explosion", "score": 0.04}],
                    }
                ],
                "row_errors": {"00002": "RuntimeError: unreadable audio"},
            }
        ),
        encoding="utf-8",
    )

    batch = _batch_from_checkpoint(
        response,
        log,
        status="partial_timeout",
        fallback_message="timeout",
    )

    assert batch.status == "partial_timeout"
    assert len(batch.predictions) == 1
    assert batch.predictions[0].semantic_family == "fx_impact"
    assert batch.predictions[0].panns_events[0].label == "Explosion"
    assert batch.row_errors == {"00002": "RuntimeError: unreadable audio"}


def _row() -> PreviewRow:
    return PreviewRow(
        row_id="00001",
        source_path=Path("/audio/test.wav"),
        display_name="test.wav",
        proposed_folder="FX/Review",
        approved_folder="FX/Review",
        final_top="FX",
        consensus_status="test",
        confidence=0.0,
        duration_sec=1.0,
        read_status="ok",
        decision_reason="test",
        diagnostic_summary="test",
    )


def test_gui_says_runtime_error_instead_of_no_result_for_failed_lanes() -> None:
    row = _row()
    row.neural_runtime_status = "row_error"
    row.neural_row_error = "RuntimeError: model failed"
    row.neural_semantic_status = "runtime_error"
    row.panns_status = "unavailable"

    assert "Runtime Error" in broad_clap_summary(row)
    assert "Unavailable" in panns_summary(row)
    assert "No Result" not in broad_clap_summary(row)


def test_gui_keeps_true_empty_output_as_no_result() -> None:
    row = _row()
    row.neural_runtime_status = "predicted"
    row.neural_semantic_status = "advisory_only"
    row.panns_status = "advisory_only"

    assert broad_clap_summary(row).endswith("No Result.")
    assert panns_summary(row).endswith("No Result.")
