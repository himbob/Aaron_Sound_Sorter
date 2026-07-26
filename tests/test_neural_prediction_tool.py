from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from aaron_sound_sorter.neural_audio.contracts import EmbeddingRecord
from aaron_sound_sorter.neural_audio.prototype_index import PrototypeIndexBuilder
from tools.predict_neural_audio import (
    _training_labels_by_hash,
    load_optional_prompt_index,
    select_neural_prediction,
)


def _record(digest_number: int, vector: list[float]) -> EmbeddingRecord:
    return EmbeddingRecord(
        provider_id="test_provider",
        model_id="test_model",
        file_sha256=f"{digest_number:064x}",
        vector=np.asarray(vector, dtype=np.float32),
    )


def test_runtime_selects_exact_human_label_before_prototype_winner() -> None:
    corrected = _record(3, [0.0, 1.0])
    index = PrototypeIndexBuilder(max_prototypes_per_label=1).build(
        {
            "Approved/A": [
                _record(1, [1.0, 0.0]),
                _record(2, [1.0, 0.0]),
                corrected,
            ],
            "Prototype/B": [_record(4, [0.0, 1.0])],
        }
    )

    prediction, exact = select_neural_prediction(
        index,
        corrected,
        {corrected.file_sha256: "Approved/A"},
    )

    assert prediction.predicted_label == "Approved/A"
    assert prediction.evidence["prototype_winner_label"] == "Prototype/B"
    assert exact is True


def test_training_manifest_omits_conflicting_exact_labels(tmp_path: Path) -> None:
    manifest = tmp_path / "training_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_sha256", "label"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            [
                {"file_sha256": "a" * 64, "label": "Approved/A"},
                {"file_sha256": "a" * 64, "label": "Approved/A"},
                {"file_sha256": "b" * 64, "label": "Approved/A"},
                {"file_sha256": "b" * 64, "label": "Conflicting/B"},
            ]
        )

    labels = _training_labels_by_hash(manifest)

    assert labels == {"a" * 64: "Approved/A"}


def test_missing_prompt_index_remains_optional(tmp_path: Path) -> None:
    prompt_index, status = load_optional_prompt_index(tmp_path, "test-model")

    assert prompt_index is None
    assert status == "unavailable"
