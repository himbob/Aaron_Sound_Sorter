"""Build interpretable calibration feedback from separated neural lanes."""

from __future__ import annotations

from datetime import datetime, timezone

from aaron_sound_sorter.taxonomy_contracts import canonicalize_taxonomy_label

from .calibration import ReviewFeedback
from .runtime import NeuralRuntimePrediction
from .semantic_panel import assess_semantic_label_compatibility


def review_feedback_from_runtime(
    prediction: NeuralRuntimePrediction,
    *,
    final_approved_label: str,
    structure_agreement: float,
    label_prototype_count: int = 0,
    duplicate_conflict: bool = False,
    brain_version: str = "",
    taxonomy_version: str = "",
) -> ReviewFeedback:
    """Create one reviewed calibration event from visible foundation evidence.

    Args:
        prediction: Separated runtime lane output for the reviewed audio.
        final_approved_label: Explicit human-approved canonical target.
        structure_agreement: Measured category/shape agreement in ``[0, 1]``.
        label_prototype_count: Prototype count for the predicted category.
        duplicate_conflict: Whether corpus identity checks found a conflict.
        brain_version: Active local prototype version.
        taxonomy_version: Canonical taxonomy version.

    Returns:
        Append-only feedback suitable for held-out calibration rebuilding.

    Raises:
        ValueError: If ``structure_agreement`` is outside ``[0, 1]``.

    Side Effects:
        None.
    """
    if not 0.0 <= float(structure_agreement) <= 1.0:
        raise ValueError("structure agreement must be in [0, 1]")
    predicted_label = canonicalize_taxonomy_label(prediction.predicted_label)
    approved_label = canonicalize_taxonomy_label(final_approved_label)
    prompt = prediction.prompt_suggestions[0] if prediction.prompt_suggestions else None
    semantic = assess_semantic_label_compatibility(
        predicted_label,
        predicted_family=prediction.semantic_family,
        top_score=prediction.semantic_top_score,
        family_scores=prediction.semantic_family_scores,
    )
    accepted = bool(predicted_label and predicted_label == approved_label)
    return ReviewFeedback(
        file_sha256=prediction.file_sha256,
        provider_id="foundation_panel",
        predicted_label=predicted_label,
        accepted=accepted,
        top_similarity=float(prediction.top_similarity),
        margin=float(prediction.margin),
        radius_ratio=float(prediction.radius_ratio),
        label_example_count=int(prediction.label_example_count),
        label_prototype_count=max(0, int(label_prototype_count)),
        structure_agreement=float(structure_agreement),
        created_utc=datetime.now(timezone.utc).isoformat(),
        final_approved_label=approved_label,
        parent_family_accepted=_top_family(predicted_label) == _top_family(approved_label),
        prompt_top_similarity=float(prompt.positive_score) if prompt else 0.0,
        prompt_margin=float(prompt.prompt_margin) if prompt else 0.0,
        panns_support_score=float(prediction.panns_support_score),
        panns_contradiction_score=float(prediction.panns_contradiction_score),
        broad_family_agreement=0.0 if semantic.contradictory else 1.0,
        out_of_distribution=0.0 if prediction.known_distribution else 1.0,
        duplicate_conflict=1.0 if duplicate_conflict else 0.0,
        brain_version=str(brain_version),
        taxonomy_version=str(taxonomy_version),
    )


def _top_family(label: str) -> str:
    return str(label).split("/", 1)[0] if label else ""
