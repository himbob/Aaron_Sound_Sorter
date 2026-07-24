from pathlib import Path

import numpy as np

from aaron_sound_sorter.neural_audio.contracts import EmbeddingRecord
from aaron_sound_sorter.neural_audio.prototype_index import PrototypeIndexBuilder
from aaron_sound_sorter.neural_audio.shadow import build_shadow_rows


def record(number: int, vector):
    return EmbeddingRecord("p", "m", f"{number:064x}", np.asarray(vector, dtype=np.float32))


def test_shadow_report_surfaces_disagreement() -> None:
    index = PrototypeIndexBuilder().build(
        {
            "Instruments/Voice/Vocal Loops": [record(1, [1.0, 0.0]), record(2, [0.9, 0.1])],
            "Textures/Atmospheres": [record(3, [0.0, 1.0]), record(4, [0.1, 0.9])],
        }
    )
    query = record(10, [0.95, 0.05])
    rows = build_shadow_rows(
        index,
        [(Path("PDHV_demo.wav"), query)],
        legacy_folders_by_hash={query.file_sha256: "_TO_REVIEW/Measured Role Conflict"},
    )
    assert len(rows) == 1
    assert rows[0].neural_label == "Instruments/Voice/Vocal Loops"
    assert rows[0].neural_second_label == "Textures/Atmospheres"
    assert rows[0].neural_prototype_support_count >= 1
    assert rows[0].agreement is False
