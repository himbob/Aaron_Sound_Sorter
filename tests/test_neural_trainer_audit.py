from __future__ import annotations

import numpy as np

from aaron_sound_sorter.neural_audio.contracts import EmbeddingRecord
from aaron_sound_sorter.neural_audio.trainer_audit import audit_labeled_embeddings


def record(digest_char: str, vector) -> EmbeddingRecord:
    return EmbeddingRecord(
        provider_id="test",
        model_id="test-model",
        file_sha256=digest_char * 64,
        vector=np.asarray(vector, dtype=np.float32),
    )


def test_trainer_audit_flags_cross_label_contamination() -> None:
    rows = audit_labeled_embeddings(
        {
            "Voice": [record("a", [1.0, 0.0]), record("b", [0.98, 0.02]), record("c", [0.0, 1.0])],
            "FX": [record("d", [0.02, 0.98]), record("e", [0.0, 1.0])],
        }
    )
    by_hash = {row.file_sha256: row for row in rows}
    assert by_hash["c" * 64].status == "cross_label_conflict"
    assert by_hash["c" * 64].nearest_other_label == "FX"


def test_trainer_audit_marks_singletons_without_faking_confidence() -> None:
    rows = audit_labeled_embeddings(
        {
            "Solo": [record("a", [1.0, 0.0])],
            "Other": [record("b", [0.0, 1.0]), record("c", [0.1, 0.9])],
        }
    )
    solo = next(row for row in rows if row.label == "Solo")
    assert solo.status == "prototype_only_singleton"
    assert solo.nearest_same_similarity is None


def test_trainer_audit_flags_statistical_outlier_without_category_rules() -> None:
    rows = audit_labeled_embeddings(
        {
            "A": [
                record("a", [1.0, 0.0, 0.0]),
                record("b", [0.999, 0.03, 0.0]),
                record("c", [0.998, -0.04, 0.0]),
                record("d", [0.997, 0.05, 0.0]),
                record("e", [0.75, 0.0, 0.66]),
            ],
            "B": [record("f", [0.0, 1.0, 0.0]), record("1", [0.0, 0.98, 0.2])],
        }
    )
    outlier = next(row for row in rows if row.file_sha256 == "e" * 64)
    assert outlier.status == "possible_label_outlier"


def test_trainer_audit_rejects_duplicate_bytes_across_labels() -> None:
    duplicate = record("a", [1.0, 0.0])
    rows = audit_labeled_embeddings({"A": [duplicate], "B": [duplicate]})
    assert {row.status for row in rows} == {"duplicate_hash_label_conflict"}
