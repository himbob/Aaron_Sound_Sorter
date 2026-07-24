from pathlib import Path

import numpy as np

from aaron_sound_sorter.neural_audio.contracts import EmbeddingRecord
from aaron_sound_sorter.neural_audio.prototype_index import PrototypeIndexBuilder


def test_prediction_is_independent_of_display_filename() -> None:
    def record(number: int, vector):
        return EmbeddingRecord("p", "m", f"{number:064x}", np.asarray(vector, dtype=np.float32))

    index = PrototypeIndexBuilder().build(
        {
            "Instruments/Voice": [record(1, [1.0, 0.0]), record(2, [0.9, 0.1])],
            "Textures": [record(3, [0.0, 1.0]), record(4, [0.1, 0.9])],
        }
    )
    same_bytes_embedding = record(99, [0.95, 0.05])
    names = [Path("obvious_vocal.wav"), Path("texture_bed_wrong_name.wav")]
    predictions = [index.predict(same_bytes_embedding).predicted_label for _ in names]
    assert predictions == ["Instruments/Voice", "Instruments/Voice"]
