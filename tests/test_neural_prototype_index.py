from pathlib import Path

import numpy as np
import pytest

from aaron_sound_sorter.neural_audio import prototype_index
from aaron_sound_sorter.neural_audio.contracts import EmbeddingRecord
from aaron_sound_sorter.neural_audio.prototype_index import PrototypeIndex, PrototypeIndexBuilder


def rec(provider: str, model: str, digest_number: int, vector) -> EmbeddingRecord:
    return EmbeddingRecord(provider, model, f"{digest_number:064x}", np.asarray(vector, dtype=np.float32))


def test_multimodal_label_builds_multiple_prototypes() -> None:
    records = {
        "Instruments/Woodwinds/Saxophone": [
            rec("p", "m", 1, [1.0, 0.0, 0.0]),
            rec("p", "m", 2, [0.98, 0.05, 0.0]),
            rec("p", "m", 3, [0.96, 0.10, 0.0]),
            rec("p", "m", 4, [0.0, 1.0, 0.0]),
            rec("p", "m", 5, [0.05, 0.98, 0.0]),
            rec("p", "m", 6, [0.10, 0.96, 0.0]),
        ],
        "Instruments/Voice/Vocal Loops": [
            rec("p", "m", 20, [0.0, 0.0, 1.0]),
            rec("p", "m", 21, [0.0, 0.05, 0.98]),
        ],
    }
    index = PrototypeIndexBuilder(max_prototypes_per_label=4).build(records)
    assert index.metadata.label_prototype_counts["Instruments/Woodwinds/Saxophone"] == 2

    prediction = index.predict(rec("p", "m", 100, [0.02, 0.99, 0.0]))
    assert prediction.predicted_label == "Instruments/Woodwinds/Saxophone"
    assert prediction.margin > 0.5


def test_index_round_trip_without_pickle(tmp_path) -> None:
    index = PrototypeIndexBuilder().build(
        {
            "A": [rec("p", "m", 1, [1.0, 0.0]), rec("p", "m", 2, [0.9, 0.1])],
            "B": [rec("p", "m", 3, [0.0, 1.0]), rec("p", "m", 4, [0.1, 0.9])],
        }
    )
    output = tmp_path / "index"
    index.save(output)
    loaded = PrototypeIndex.load(output)
    prediction = loaded.predict(rec("p", "m", 5, [0.95, 0.05]))
    assert prediction.predicted_label == "A"
    assert loaded.metadata.training_hashes == index.metadata.training_hashes


def test_prediction_uses_finite_reduction_without_matmul_warnings() -> None:
    index = PrototypeIndexBuilder().build(
        {
            "A": [rec("p", "m", 1, [1.0, 0.0])],
            "B": [rec("p", "m", 2, [0.0, 1.0])],
        }
    )

    with np.errstate(all="raise"):
        prediction = index.predict(rec("p", "m", 3, [0.8, 0.2]))

    assert prediction.predicted_label == "A"
    assert np.isfinite(prediction.top_similarity)


def test_label_similarities_exposes_content_scores_without_source_names() -> None:
    index = PrototypeIndexBuilder().build(
        {
            "Instruments/Voice/Vocal Loops": [rec("p", "m", 1, [1.0, 0.0])],
            "Instruments/Woodwinds/Saxophone": [rec("p", "m", 2, [0.0, 1.0])],
        }
    )

    scores = index.label_similarities(rec("p", "m", 3, [0.2, 0.8]))

    assert list(scores) == [
        "Instruments/Woodwinds/Saxophone",
        "Instruments/Voice/Vocal Loops",
    ]
    assert scores["Instruments/Woodwinds/Saxophone"] > scores["Instruments/Voice/Vocal Loops"]


def test_exact_human_label_precedes_prototype_generalization() -> None:
    corrected = rec("p", "m", 3, [0.0, 1.0])
    index = PrototypeIndexBuilder(max_prototypes_per_label=1).build(
        {
            "Approved/A": [
                rec("p", "m", 1, [1.0, 0.0]),
                rec("p", "m", 2, [1.0, 0.0]),
                corrected,
            ],
            "Prototype/B": [rec("p", "m", 4, [0.0, 1.0])],
        }
    )

    assert index.predict(corrected).predicted_label == "Prototype/B"
    exact = index.predict(corrected, exact_label="Approved/A")

    assert exact.predicted_label == "Approved/A"
    assert exact.second_label == "Prototype/B"
    assert exact.known_distribution is True
    assert exact.evidence["prediction_mode"] == "exact_human_training_label"
    assert exact.evidence["prototype_winner_label"] == "Prototype/B"


def test_failed_index_replacement_restores_previous_index(tmp_path, monkeypatch) -> None:
    original = PrototypeIndexBuilder().build(
        {
            "Original/A": [rec("p", "m", 1, [1.0, 0.0])],
            "Original/B": [rec("p", "m", 2, [0.0, 1.0])],
        }
    )
    replacement = PrototypeIndexBuilder().build(
        {
            "Replacement/A": [rec("p", "m", 3, [1.0, 0.0])],
            "Replacement/B": [rec("p", "m", 4, [0.0, 1.0])],
        }
    )
    output = tmp_path / "index"
    original.save(output)

    real_replace = prototype_index.os.replace
    replacement_attempts = 0

    def fail_new_index_install(source: Path, destination: Path) -> None:
        nonlocal replacement_attempts
        if destination == output and Path(source).name.startswith(".index.tmp-"):
            replacement_attempts += 1
            raise OSError("simulated index install failure")
        real_replace(source, destination)

    monkeypatch.setattr(prototype_index.os, "replace", fail_new_index_install)
    with pytest.raises(OSError, match="simulated index install failure"):
        replacement.save(output)

    assert replacement_attempts == 1
    loaded = PrototypeIndex.load(output)
    assert loaded.metadata.training_hashes == original.metadata.training_hashes
