"""Typed contracts for the neural-audio migration lane."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REVIEW_PREVIEW_SPLIT = "review_preview"
HELDOUT_EVAL_SPLIT = "heldout_eval"
VALID_SPLITS = frozenset({REVIEW_PREVIEW_SPLIT, HELDOUT_EVAL_SPLIT})


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return a finite float32 unit vector without mutating the input."""
    arr = np.nan_to_num(np.asarray(vector, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim != 1:
        raise ValueError(f"expected one-dimensional embedding, got shape={arr.shape}")
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError("embedding vector has zero length")
    return (arr / norm).astype(np.float32, copy=False)


@dataclass(frozen=True)
class EmbeddingRecord:
    """One source-blind neural representation of an audio file."""

    provider_id: str
    model_id: str
    file_sha256: str
    vector: np.ndarray = field(repr=False)
    segment_count: int = 1
    sample_rate: int = 0

    def __post_init__(self) -> None:
        if not self.provider_id.strip():
            raise ValueError("provider_id is required")
        if not self.model_id.strip():
            raise ValueError("model_id is required")
        if len(self.file_sha256) != 64:
            raise ValueError("file_sha256 must be a 64-character SHA-256 digest")
        if self.segment_count < 1:
            raise ValueError("segment_count must be positive")
        object.__setattr__(self, "vector", l2_normalize(self.vector))

    @property
    def dimension(self) -> int:
        return int(self.vector.shape[0])


@dataclass(frozen=True)
class LabeledAudioExample:
    """A human-curated training or held-out example.

    ``label`` may come from a deliberately curated folder path.  ``path`` is
    operational metadata only and must never be embedded into the model input.
    """

    label: str
    path: Path
    file_sha256: str
    split: str
    decoded_audio_sha256: str = ""
    normalized_audio_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("label is required")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"unsupported split: {self.split}")
        if len(self.file_sha256) != 64:
            raise ValueError("file_sha256 must be a SHA-256 digest")
        if self.decoded_audio_sha256 and len(self.decoded_audio_sha256) != 64:
            raise ValueError("decoded_audio_sha256 must be empty or a SHA-256 digest")
        if self.normalized_audio_sha256 and len(self.normalized_audio_sha256) != 64:
            raise ValueError("normalized_audio_sha256 must be empty or a SHA-256 digest")


@dataclass(frozen=True)
class EvaluationSplit:
    """Leakage-safe preview and held-out partitions."""

    review_preview: tuple[LabeledAudioExample, ...]
    heldout_eval: tuple[LabeledAudioExample, ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        train_hashes = {example.file_sha256 for example in self.review_preview}
        heldout_hashes = {example.file_sha256 for example in self.heldout_eval}
        overlap = train_hashes & heldout_hashes
        if overlap:
            raise ValueError(f"training/evaluation leakage detected for {len(overlap)} hash(es)")
        train_audio_hashes = {
            example.decoded_audio_sha256 for example in self.review_preview if example.decoded_audio_sha256
        }
        heldout_audio_hashes = {
            example.decoded_audio_sha256 for example in self.heldout_eval if example.decoded_audio_sha256
        }
        decoded_overlap = train_audio_hashes & heldout_audio_hashes
        if decoded_overlap:
            raise ValueError(
                f"training/evaluation decoded-audio leakage detected for {len(decoded_overlap)} content fingerprint(s)"
            )
        train_normalized_hashes = {
            example.normalized_audio_sha256 for example in self.review_preview if example.normalized_audio_sha256
        }
        heldout_normalized_hashes = {
            example.normalized_audio_sha256 for example in self.heldout_eval if example.normalized_audio_sha256
        }
        normalized_overlap = train_normalized_hashes & heldout_normalized_hashes
        if normalized_overlap:
            raise ValueError(
                "training/evaluation normalized-audio leakage detected for "
                f"{len(normalized_overlap)} derived-copy fingerprint(s)"
            )


@dataclass(frozen=True)
class Prototype:
    """One learned neighborhood inside a label."""

    label: str
    prototype_id: str
    vector: np.ndarray = field(repr=False)
    member_hashes: tuple[str, ...] = ()
    radius_p95: float = 0.0
    mean_distance: float = 0.0

    def __post_init__(self) -> None:
        if not self.label.strip() or not self.prototype_id.strip():
            raise ValueError("prototype label and id are required")
        object.__setattr__(self, "vector", l2_normalize(self.vector))
        if self.radius_p95 < 0.0 or self.mean_distance < 0.0:
            raise ValueError("prototype distances cannot be negative")


@dataclass(frozen=True)
class LabelPrediction:
    """Neural evidence for one file; this is not a final folder decision."""

    provider_id: str
    model_id: str
    predicted_label: str
    top_similarity: float
    second_label: str
    second_similarity: float
    margin: float
    prototype_id: str
    prototype_radius_p95: float
    distance_to_prototype: float
    radius_ratio: float
    known_distribution: bool
    confidence_probability: float | None = None
    evidence: Mapping[str, float | int | str] = field(default_factory=dict)


@dataclass(frozen=True)
class PrototypeIndexMetadata:
    """Serializable index metadata."""

    schema_version: int
    provider_id: str
    model_id: str
    dimension: int
    created_utc: str
    training_hashes: tuple[str, ...]
    label_example_counts: Mapping[str, int]
    label_prototype_counts: Mapping[str, int]
    build_settings: Mapping[str, float | int | str]
    label_spread_p95: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LaneEvaluation:
    """Held-out metrics for one provider and label."""

    provider_id: str
    label: str
    heldout_count: int
    correct_count: int
    top1_accuracy: float
    mean_correct_similarity: float
    mean_margin: float
    known_distribution_rate: float


@dataclass(frozen=True)
class LabelLaneChoice:
    """Per-label representation winner selected only from held-out evidence."""

    label: str
    provider_id: str
    score: float
    heldout_count: int
    reason: str


@dataclass(frozen=True)
class FusionGateDecision:
    """Held-out decision about whether an encoder fusion is allowed."""

    label: str
    best_single_provider_id: str
    fused_provider_id: str
    heldout_count: int
    best_single_accuracy: float
    fused_accuracy: float
    accuracy_gain: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class TrainerAuditRow:
    """Leave-one-out evidence about one human-curated trainer.

    This is a contamination diagnostic, not held-out accuracy.  It asks whether
    one trainer is supported by other examples assigned to the same label and
    whether another label supports it more strongly.
    """

    provider_id: str
    model_id: str
    label: str
    file_sha256: str
    label_example_count: int
    nearest_same_similarity: float | None
    nearest_other_label: str
    nearest_other_similarity: float | None
    support_margin: float | None
    robust_same_label_floor: float | None
    status: str
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShadowDiffRow:
    """One disagreement or agreement between legacy and neural evidence."""

    file_sha256: str
    display_name: str
    legacy_folder: str
    neural_label: str
    neural_similarity: float
    neural_second_label: str
    neural_second_similarity: float
    neural_margin: float
    neural_prototype_id: str
    neural_prototype_support_count: int
    neural_prototype_radius_p95: float
    neural_category_spread_p95: float
    neural_distance_to_prototype: float
    neural_radius_ratio: float
    neural_known_distribution: bool
    agreement: bool
    provider_id: str
    objective_structure_evidence: str = ""
    trainer_warnings: str = ""
    human_verdict: str = ""
    notes: str = ""
