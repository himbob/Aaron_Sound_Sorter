import numpy as np
import pytest

from aaron_sound_sorter.neural_audio.contracts import EmbeddingRecord, LaneEvaluation
from aaron_sound_sorter.neural_audio.evaluation import choose_per_label_lanes, evaluate_heldout
from aaron_sound_sorter.neural_audio.prototype_index import PrototypeIndexBuilder


def record(provider: str, digest: int, vector):
    return EmbeddingRecord(provider, "model", f"{digest:064x}", np.asarray(vector, dtype=np.float32))


def test_heldout_evaluation_rejects_reused_training_audio() -> None:
    training = {"A": [record("p", 1, [1.0, 0.0]), record("p", 2, [0.9, 0.1])]}
    index = PrototypeIndexBuilder().build(training)
    with pytest.raises(ValueError, match="reused"):
        evaluate_heldout(index, {"A": [record("p", 1, [1.0, 0.0])]})


def test_lane_choice_is_per_label_and_does_not_blindly_fuse() -> None:
    evaluations = [
        LaneEvaluation("clap", "Vocal", 10, 9, 0.9, 0.8, 0.2, 0.9),
        LaneEvaluation("mert", "Vocal", 10, 7, 0.7, 0.9, 0.3, 1.0),
        LaneEvaluation("clap", "Snare", 10, 6, 0.6, 0.8, 0.2, 0.8),
        LaneEvaluation("mert", "Snare", 10, 9, 0.9, 0.8, 0.2, 0.9),
    ]
    choices = {choice.label: choice.provider_id for choice in choose_per_label_lanes(evaluations)}
    assert choices == {"Snare": "mert", "Vocal": "clap"}
