from __future__ import annotations

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
