import builtins
from pathlib import Path

from aaron_sound_sorter.neural_audio.calibration import (
    ConfidenceCalibrator,
    ReviewFeedback,
    ReviewFeedbackStore,
)


def feedback(index: int, accepted: bool) -> ReviewFeedback:
    return ReviewFeedback(
        file_sha256=f"{index:064x}",
        provider_id="clap",
        predicted_label="Vocal",
        accepted=accepted,
        top_similarity=0.9 if accepted else 0.4,
        margin=0.3 if accepted else 0.01,
        radius_ratio=0.4 if accepted else 2.0,
        label_example_count=20,
        label_prototype_count=3,
        structure_agreement=0.9 if accepted else 0.1,
        created_utc="2026-07-23T00:00:00+00:00",
    )


def test_feedback_store_and_logistic_calibration(tmp_path: Path) -> None:
    store = ReviewFeedbackStore(tmp_path / "feedback.jsonl")
    rows = [feedback(index, accepted=index % 2 == 0) for index in range(30)]
    for row in rows:
        store.append(row)
    assert len(store.read_all()) == 30

    calibrator = ConfidenceCalibrator()
    calibrator.fit(store.read_all())
    accepted_probability = calibrator.predict_probability(feedback(100, True))
    rejected_probability = calibrator.predict_probability(feedback(101, False))
    assert accepted_probability is not None
    assert rejected_probability is not None
    assert accepted_probability > rejected_probability


def test_calibration_has_numpy_fallback_without_scikit_learn(monkeypatch) -> None:
    original_import = builtins.__import__

    def import_without_sklearn(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("scikit-learn intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_sklearn)
    calibrator = ConfidenceCalibrator()
    calibrator.fit([feedback(index, accepted=index % 2 == 0) for index in range(30)])

    accepted_probability = calibrator.predict_probability(feedback(100, True))
    rejected_probability = calibrator.predict_probability(feedback(101, False))
    assert calibrator.method == "numpy_logistic"
    assert accepted_probability is not None
    assert rejected_probability is not None
    assert accepted_probability > rejected_probability


def test_calibrator_round_trip_is_dependency_free(tmp_path: Path) -> None:
    calibrator = ConfidenceCalibrator()
    calibrator.fit([feedback(index, accepted=index % 2 == 0) for index in range(30)])
    artifact = tmp_path / "calibrator.json"

    calibrator.save(artifact, metadata={"purpose": "test"})
    loaded = ConfidenceCalibrator.load(artifact)

    expected = calibrator.predict_probability(feedback(100, True))
    actual = loaded.predict_probability(feedback(100, True))
    assert expected is not None
    assert actual is not None
    assert abs(expected - actual) < 1e-9
