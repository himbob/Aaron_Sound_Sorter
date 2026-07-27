from __future__ import annotations

import json
from pathlib import Path

from aaron_sound_sorter.neural_audio.runtime import start_configured_neural_prediction_process


def test_streaming_neural_process_is_immediately_done_when_disabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "neural_training.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"enabled": False}), encoding="utf-8")

    process = start_configured_neural_prediction_process(
        tmp_path,
        [("00001", tmp_path / "sample.wav")],
        tmp_path / "report",
    )

    assert process.done
    assert process.poll().status == "disabled"
    process.close()


def test_streaming_neural_process_skips_empty_input(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "neural_training.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps({"enabled": True, "production_ownership_enabled": True}),
        encoding="utf-8",
    )

    process = start_configured_neural_prediction_process(
        tmp_path,
        [],
        tmp_path / "report",
    )

    assert process.done
    assert process.poll().status == "skipped"
    process.close()
