"""BrainVoter implementation."""

from __future__ import annotations

from typing import Any

import numpy as np

from aaron_sound_sorter.brain import label_model_distance_score
from aaron_sound_sorter.domain.facts import fingerprint_array
from aaron_sound_sorter.domain.models import AudioPhysics, SharedAudioFacts, VoterResult
from aaron_sound_sorter.domain.policies import BrainVoterPolicy, ConsensusPolicy
from aaron_sound_sorter.voters.base import Voter
from aaron_sound_sorter.voters.human_override_recall import (
    human_override_recall_allowed_by_memory,
    human_override_recall_match,
)
from aaron_sound_sorter.voters.scoring_tools import (
    canonical_candidate_labels,
    centroid_distance,
    feature_weights,
    label_maps,
    role_compatibility_adjustment,
    scaler_for_label,
)
from aaron_sound_sorter.voters.vectorized_brain_scorer import VectorizedCentroidFallbackScorer


class BrainVoter(Voter):
    """Rank categories by learned brain centroid/exemplar distance."""

    voter_name = "brain"

    def __init__(self, policy: BrainVoterPolicy | None = None) -> None:
        self.policy = policy or BrainVoterPolicy()
        super().__init__(self.policy.top_n)

    def vote(self, physics: AudioPhysics, facts: SharedAudioFacts, brain: dict[str, Any]) -> VoterResult:
        if facts.is_broken_or_tiny:
            return self.review_result(ConsensusPolicy().broken_or_tiny_label, "broken_or_tiny")
        raw_labels = [str(label) for label in brain.get("labels", []) if str(label)]
        labels = canonical_candidate_labels(raw_labels)
        role_gate = facts.evidence.get("dynamic_role_gate", {}) if isinstance(facts.evidence, dict) else {}
        gate_can_filter = bool(isinstance(role_gate, dict) and role_gate.get("final_effects_enabled"))
        allowed = set(role_gate.get("allowed_labels", []) or []) if gate_can_filter else set()
        if gate_can_filter and allowed:
            labels = [label for label in labels if label in allowed]
        if not labels:
            return self.review_result(ConsensusPolicy().no_consensus_label, "brain_has_no_labels")
        rows = self.score_labels(physics, facts, brain, labels)
        folder_map, top_map = label_maps(brain)
        guesses = self.ranked_guesses(
            rows,
            label_to_folder=folder_map,
            label_to_top=top_map,
            lower_score_is_better=True,
            default_reason="brain_distance_match",
        )
        return VoterResult(
            voter_name=self.voter_name,
            guesses=guesses,
            diagnostics={
                "candidate_count": len(labels),
                "discarded_noncanonical_label_count": max(0, len(raw_labels) - len(labels)),
                "returned_count": len(guesses),
                "role_gate_applied": bool(gate_can_filter and allowed),
                "role_gate_mode": str(role_gate.get("mode", "missing")) if isinstance(role_gate, dict) else "missing",
            },
        )

    def score_labels(
        self,
        physics: AudioPhysics,
        facts: SharedAudioFacts,
        brain: dict[str, Any],
        labels: list[str],
        *,
        use_vectorized_fallback: bool = True,
    ) -> list[dict[str, Any]]:
        """Score every label using the learned brain model.

        Args:
            physics: Measured audio physics for the current file.
            facts: Shared facts used by role-compatibility adjustment.
            brain: Loaded brain dictionary.
            labels: Candidate labels to score.
            use_vectorized_fallback: When true, labels whose adaptive model does
                not return a finite score use the batch centroid fallback scorer.

        Returns:
            Score rows consumed by ``ranked_guesses``.

        Side Effects:
            None. Diagnostic counters are stored in row evidence only.

        Raises:
            No intentional exceptions. Bad per-label data becomes an infinite
            fallback score, matching the legacy scalar behavior.

        Important Constraints:
            The vectorized path is only a centroid fallback. It must not change
            adaptive label-model scores or final routing policy.
        """
        vector = fingerprint_array(physics.fingerprint)
        weights = feature_weights(brain)
        folder_map, _top_map = label_maps(brain)
        pending_rows: list[dict[str, Any]] = []
        fallback_vectors: dict[str, np.ndarray] = {}
        for label in labels:
            mean, std = scaler_for_label(brain, label)
            weighted_vector = ((vector - mean) / std) * weights
            score, raw_distance, mode = self.score_one_label(
                brain,
                label,
                weighted_vector,
                weights,
                use_scalar_fallback=not use_vectorized_fallback,
            )
            human_override = human_override_recall_match(
                brain,
                label,
                raw_query_vector=vector,
                weighted_query_vector=weighted_vector,
                scaler_mean=mean,
                scaler_std=std,
                feature_weight_vector=weights,
            )
            human_override_evidence = human_override.evidence()
            folder_path = str(folder_map.get(label, label))
            human_override_allowed = human_override_recall_allowed_by_memory(folder_path, facts)
            if human_override.matched and human_override_allowed:
                score = min(float(score), float(human_override.ranking_score))
                raw_distance = min(float(raw_distance), float(human_override.nearest_distance))
                if human_override.exact_match:
                    mode = "human_override_fingerprint_teacher_match"
                else:
                    mode = f"human_override_{human_override.match_kind}_match"
            elif human_override.matched:
                human_override_evidence["human_override_suppressed_by_learned_memory_conflict"] = True
            if use_vectorized_fallback and not np.isfinite(float(score)):
                fallback_vectors[label] = weighted_vector
            pending_rows.append(
                {
                    "label": label,
                    "score": float(score),
                    "raw_distance": float(raw_distance),
                    "mode": str(mode),
                    "used_vectorized_fallback": False,
                    "human_override_evidence": human_override_evidence,
                }
            )

        if use_vectorized_fallback and fallback_vectors:
            fallback_scorer = VectorizedCentroidFallbackScorer(brain, weights)
            fallback_scores = fallback_scorer.score_labels(fallback_vectors)
            for row in pending_rows:
                label = str(row["label"])
                if label in fallback_scores:
                    fallback = fallback_scores[label]
                    row["score"] = float(fallback.score)
                    row["raw_distance"] = float(fallback.score)
                    row["mode"] = fallback.mode
                    row["used_vectorized_fallback"] = True

        rows: list[dict[str, Any]] = []
        vectorized_count = sum(1 for row in pending_rows if row.get("used_vectorized_fallback"))
        for row in pending_rows:
            label = str(row["label"])
            role_delta, role_evidence = role_compatibility_adjustment(label, brain, facts, voter_name=self.voter_name)
            raw_score = float(row["score"])
            final_score = raw_score + float(role_delta)
            confidence_score = 1.0 / (1.0 + max(0.0, final_score))
            rows.append(
                {
                    "label": label,
                    "score": final_score,
                    "confidence": confidence_score,
                    "reason": "brain_distance_match",
                    "evidence": {
                        "distance_mode": str(row["mode"]),
                        "raw_distance": round(float(row["raw_distance"]), 6),
                        "raw_brain_score": round(raw_score, 6),
                        "brain_score_after_role_adjustment": round(float(final_score), 6),
                        "role_adjustment_applied": abs(float(role_delta)) >= 1e-9,
                        "score_is_raw_voter_score": abs(float(role_delta)) < 1e-9,
                        "used_vectorized_centroid_fallback": bool(row.get("used_vectorized_fallback")),
                        "vectorized_centroid_fallback_label_count": int(vectorized_count),
                        **dict(row.get("human_override_evidence", {})),
                        **role_evidence,
                        "structure_gate": "candidate_pre_filtered",
                    },
                }
            )
        return rows

    def score_one_label(
        self,
        brain: dict[str, Any],
        label: str,
        weighted_vector: np.ndarray,
        weights: np.ndarray,
        *,
        use_scalar_fallback: bool = True,
    ) -> tuple[float, float, str]:
        """Score one label with adaptive model and optional scalar fallback.

        Args:
            brain: Loaded brain dictionary.
            label: Candidate label to score.
            weighted_vector: Already scaled and weighted query vector.
            weights: Feature-weight vector used by centroid fallback.
            use_scalar_fallback: When false, return ``inf`` for fallback-needed
                labels so ``score_labels`` can batch those fallbacks.

        Returns:
            Tuple of ranking score, raw distance, and diagnostic mode.

        Side Effects:
            None.

        Raises:
            No intentional exceptions. Bad model data falls through to fallback.

        Important Constraints:
            Lower scores are better. This method must preserve legacy behavior
            when ``use_scalar_fallback`` is true.
        """
        try:
            ranking_distance, similarity_distance, mode = label_model_distance_score(brain, label, weighted_vector)
            score = float(ranking_distance)
            raw_distance = float(similarity_distance)
            if np.isfinite(score):
                return score, raw_distance, str(mode)
        except Exception:
            pass
        if not use_scalar_fallback:
            return float("inf"), float("inf"), "needs_centroid_fallback"
        fallback = centroid_distance(brain, label, weighted_vector, weights)
        return fallback, fallback, "centroid_fallback"
