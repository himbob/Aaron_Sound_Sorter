from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from trusted_memory_seed import (  # noqa: E402
    cases_from_records,
    is_trainable_label,
    load_trusted_seed_cases,
)


def test_loader_accepts_only_explicit_training_labels(tmp_path: Path) -> None:
    project_root = tmp_path
    samples_dir = project_root / "samples"
    samples_dir.mkdir()
    panel_path = project_root / "panel.json"
    panel_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "explicit",
                        "filename": "riser.wav",
                        "training_label": "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX",
                    },
                    {
                        "id": "prefix_only",
                        "filename": "voice.wav",
                        "accepted_folder_prefixes": ["Instruments/Voice"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded = load_trusted_seed_cases(panel_path, project_root=project_root, samples_dir=samples_dir)

    assert [case.case_id for case in loaded.trainable_cases] == ["explicit"]
    assert loaded.trainable_cases[0].audio_path == samples_dir / "riser.wav"
    assert loaded.skipped_rows == [
        {"case_id": "prefix_only", "status": "skipped", "reason": "missing_explicit_training_label"}
    ]


def test_absolute_audio_path_is_preserved_for_location_only(tmp_path: Path) -> None:
    absolute_audio = tmp_path / "known.wav"
    loaded = cases_from_records(
        [
            {
                "id": "known",
                "audio_path": str(absolute_audio),
                "approved_label": "Instruments/Keys/Piano/One Shots",
            }
        ],
        panel_path=tmp_path / "panel.json",
        project_root=tmp_path,
        samples_dir=None,
    )

    assert loaded.skipped_rows == []
    assert loaded.trainable_cases[0].audio_path == absolute_audio
    assert loaded.trainable_cases[0].approved_label == "Instruments/Keys/Piano/One Shots"


def test_gui_training_manifest_fields_are_explicit_trusted_inputs(tmp_path: Path) -> None:
    audio_path = tmp_path / "known.wav"
    loaded = cases_from_records(
        [
            {
                "row_id": "00042",
                "source_path": str(audio_path),
                "display_name": "known.wav",
                "approved_folder": "FX/Human and Voice FX/Spoken Voice/One Shots",
            }
        ],
        panel_path=tmp_path / "Aaron_GUI_Training_Import.csv",
        project_root=tmp_path,
        samples_dir=None,
    )

    assert loaded.skipped_rows == []
    assert loaded.trainable_cases[0].audio_path == audio_path
    assert loaded.trainable_cases[0].approved_label == "FX/Human and Voice FX/Spoken Voice/One Shots"


def test_review_and_missing_labels_are_not_trainable() -> None:
    assert is_trainable_label("FX/Impacts and Hits/Generic Impact/Long FX")
    assert not is_trainable_label("")
    assert not is_trainable_label("_TO_REVIEW/Measured Role Conflict")
    assert not is_trainable_label("FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Loops")
