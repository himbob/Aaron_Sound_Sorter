from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from aaron_sound_sorter.infrastructure.audio_repository import AudioInputRepository, InputPreparationCancelled


def test_folder_input_preparation_can_be_cancelled(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "sound.wav").write_bytes(b"fake wav")

    with pytest.raises(InputPreparationCancelled):
        AudioInputRepository().prepare(
            source_root,
            tmp_path / "out",
            cancel_requested=lambda: True,
        )


def test_valid_zip_without_audio_explains_detected_formats(tmp_path: Path) -> None:
    source_zip = tmp_path / "presets.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("library/song.mid", b"midi")
        archive.writestr("library/preset.x3p", b"preset")

    with pytest.raises(ValueError) as captured:
        AudioInputRepository().prepare(source_zip, tmp_path / "out")

    message = str(captured.value)
    assert "archive is readable" in message
    assert ".mid" in message
    assert ".x3p" in message
