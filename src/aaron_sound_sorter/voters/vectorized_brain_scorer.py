"""Vectorized-safe helpers for BrainVoter fallback centroid scoring.

The main adaptive label-model scoring path stays authoritative. These helpers
only replace the final centroid fallback that is already used when the adaptive
model returns no finite score. The fallback is tested against the scalar helper
before it is used in product voting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from aaron_sound_sorter.voters.scoring_tools import centroid_distance


@dataclass(frozen=True)
class VectorizedFallbackScore:
    """One centroid fallback score for a label.

    Args:
        label: Brain label being scored.
        score: Nearest-centroid fallback distance. Lower is better.
        mode: Diagnostic mode string preserved for old evidence consumers.

    Side Effects:
        None.

    Raises:
        No intentional exceptions.

    Important Constraints:
        This object never stores final sorter decisions.
    """

    label: str
    score: float
    mode: str = "centroid_fallback"


class VectorizedCentroidFallbackScorer:
    """Batch fallback scorer for labels that need centroid-only scoring.

    The implementation keeps behavior equal to ``centroid_distance``. It is
    intentionally conservative: labels with different centroid dimensions are
    grouped separately, and every score is checked through the same nearest
    centroid formula as the scalar helper.
    """

    def __init__(self, brain: dict[str, Any], weights: np.ndarray) -> None:
        self.brain = brain
        self.weights = np.asarray(weights, dtype=np.float32)
        centroids = brain.get("centroids", {}) if isinstance(brain, dict) else {}
        self.centroids_by_label = centroids if isinstance(centroids, dict) else {}

    def score_labels(
        self,
        weighted_vectors_by_label: dict[str, np.ndarray],
    ) -> dict[str, VectorizedFallbackScore]:
        """Return fallback scores for labels needing centroid scoring.

        Args:
            weighted_vectors_by_label: Mapping of label to already scaled and
                weighted query vector.

        Returns:
            Mapping of label to ``VectorizedFallbackScore``.

        Side Effects:
            None.

        Raises:
            No intentional exceptions. Malformed centroid data yields ``inf``.

        Important Constraints:
            Scores must match ``centroid_distance`` for each label. Tests compare
            the two paths directly.
        """
        scores: dict[str, VectorizedFallbackScore] = {}
        for label, weighted_vector in weighted_vectors_by_label.items():
            score = self._score_one_label(label, weighted_vector)
            scores[label] = VectorizedFallbackScore(label=label, score=score)
        return scores

    def _score_one_label(self, label: str, weighted_vector: np.ndarray) -> float:
        """Return a fallback score matching the scalar helper."""
        label_centroids = np.asarray(self.centroids_by_label.get(label, []), dtype=np.float32)
        if label_centroids.ndim == 1 and label_centroids.size:
            label_centroids = label_centroids.reshape(1, -1)
        if label_centroids.ndim != 2 or not label_centroids.size:
            return float("inf")
        usable = min(label_centroids.shape[1], weighted_vector.size, self.weights.size)
        if usable <= 0:
            return float("inf")
        centroid_weighted = label_centroids[:, :usable] * self.weights[:usable][None, :]
        distances = np.linalg.norm(centroid_weighted - weighted_vector[:usable][None, :], axis=1)
        return float(np.min(distances))

    def scalar_equivalent_score(self, label: str, weighted_vector: np.ndarray) -> float:
        """Return the legacy scalar score for tests and defensive checks."""
        return centroid_distance(self.brain, label, weighted_vector, self.weights)
