"""Build visible, non-fused foundation evidence panels."""

from __future__ import annotations

from typing import Any

from .foundation_contracts import FoundationEvidencePanel, FoundationLaneEvidence, FoundationScore
from .panns_mapping import PannsEventScore, PannsMappingRegistry
from .runtime import NeuralRuntimePrediction


def build_foundation_panel(
    prediction: NeuralRuntimePrediction,
    *,
    taxonomy_version: str,
    brain_version: str,
    panns_mapping: PannsMappingRegistry | None = None,
) -> FoundationEvidencePanel:
    """Build one source-name-blind panel while keeping every lane separate.

    Args:
        prediction: Combined subprocess transport containing distinct lane
            outputs. No scores are averaged here.
        taxonomy_version: Canonical registry version.
        brain_version: Active Aaron prototype index identity.
        panns_mapping: Optional broad AudioSet mapping for explicit support and
            contradiction diagnostics.

    Returns:
        Versioned per-hash evidence with conflicts and review reason.

    Side Effects:
        None.
    """
    lanes = {
        "aaron_prototype": _prototype_lane(prediction, brain_version),
        "clap_broad": _clap_broad_lane(prediction),
        "clap_prompt": _clap_prompt_lane(prediction),
        "panns": _panns_lane(prediction),
        "human_memory": _human_memory_lane(prediction),
    }
    conflicts: list[str] = []
    if prediction.ownership_ready and prediction.semantic_family:
        conflicts.extend(_visible_semantic_conflicts(prediction))
    if panns_mapping is not None and prediction.panns_events and prediction.predicted_label:
        mapped = panns_mapping.evaluate(
            tuple(PannsEventScore(event.label, event.score) for event in prediction.panns_events),
            prediction.predicted_label,
        )
        if mapped.contradiction_score > mapped.support_score and mapped.contradicting_events:
            event_names = ", ".join(event.label for event in mapped.contradicting_events[:3])
            conflicts.append(f"PANNs broad contradiction for {prediction.predicted_label}: {event_names}")
    review_reason = ""
    if conflicts:
        review_reason = "Independent foundation lanes disagree; human review is required."
    elif not prediction.ownership_ready:
        review_reason = prediction.ownership_block_reason or "Aaron prototype memory is not ready for ownership."
    return FoundationEvidencePanel(
        audio_sha256=prediction.file_sha256,
        taxonomy_version=taxonomy_version,
        brain_version=brain_version,
        lanes=lanes,
        conflicts=tuple(conflicts),
        suggested_category=prediction.predicted_label,
        review_reason=review_reason,
    )


def _prototype_lane(prediction: NeuralRuntimePrediction, brain_version: str) -> FoundationLaneEvidence:
    evidence: dict[str, Any] = {
        "second_label": prediction.second_label,
        "known_distribution": prediction.known_distribution,
        "radius_ratio": prediction.radius_ratio,
        "label_example_count": prediction.label_example_count,
        "ownership_ready": prediction.ownership_ready,
        "ownership_reason": prediction.ownership_block_reason,
        "exact_training_match": prediction.exact_training_match,
    }
    return FoundationLaneEvidence(
        lane_id="aaron_prototype",
        model_id=brain_version,
        status="available",
        scores=(FoundationScore(prediction.predicted_label, prediction.top_similarity, prediction.margin, evidence),),
    )


def _clap_broad_lane(prediction: NeuralRuntimePrediction) -> FoundationLaneEvidence:
    scores = tuple(
        FoundationScore(family, score)
        for family, score in sorted(
            prediction.semantic_family_scores.items(),
            key=lambda row: (-row[1], row[0]),
        )[:5]
    )
    status = "available" if scores else "unavailable"
    return FoundationLaneEvidence("clap_broad", "same-pinned-clap-audio-text-model", status, scores)


def _clap_prompt_lane(prediction: NeuralRuntimePrediction) -> FoundationLaneEvidence:
    scores = tuple(
        FoundationScore(
            suggestion.path,
            suggestion.positive_score,
            suggestion.prompt_margin,
            {
                "negative_score": suggestion.negative_score,
                "top_positive_prompt": suggestion.top_positive_prompt,
                "top_positive_similarity": suggestion.top_positive_similarity,
                "top_negative_prompt": suggestion.top_negative_prompt,
                "top_negative_similarity": suggestion.top_negative_similarity,
                "production_ownership_enabled": False,
            },
        )
        for suggestion in prediction.prompt_suggestions
    )
    status = "available" if prediction.prompt_brain_status == "advisory_only" else prediction.prompt_brain_status
    return FoundationLaneEvidence("clap_prompt", "same-pinned-clap-audio-text-model", status, scores)


def _panns_lane(prediction: NeuralRuntimePrediction) -> FoundationLaneEvidence:
    scores = tuple(FoundationScore(event.label, event.score) for event in prediction.panns_events)
    status = "available" if prediction.panns_status == "advisory_only" else prediction.panns_status
    return FoundationLaneEvidence("panns", prediction.panns_model_id, status, scores)


def _human_memory_lane(prediction: NeuralRuntimePrediction) -> FoundationLaneEvidence:
    scores = (
        (FoundationScore(prediction.predicted_label, 1.0, evidence={"exact_content_hash_match": True}),)
        if prediction.exact_training_match
        else ()
    )
    return FoundationLaneEvidence(
        "human_memory",
        "active-gui-training-manifest",
        "available" if scores else "no_exact_match",
        scores,
    )


def _visible_semantic_conflicts(prediction: NeuralRuntimePrediction) -> list[str]:
    from .authority import neural_semantic_compatibility

    assessment = neural_semantic_compatibility(prediction, prediction.predicted_label)
    if not assessment.contradictory:
        return []
    return [
        "CLAP broad family contradiction: "
        f"{prediction.semantic_family} versus {prediction.predicted_label} "
        f"(gap={assessment.contradiction_gap:.3f})"
    ]
