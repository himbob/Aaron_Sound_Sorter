"""Conservative production-ownership policy for neural audio predictions."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import LabelPrediction


@dataclass(frozen=True)
class OwnershipAssessment:
    """Explain whether one prototype prediction may own a GUI decision.

    Args:
        ready: Whether the neural label may replace the current GUI proposal.
        reason: Compact machine-readable reason for the decision.
        label_example_count: Distinct training examples behind the winning
            label.
        exact_training_match: Whether the audio hash maps to an unambiguous
            explicit label in the active training manifest.

    Side Effects:
        None.
    """

    ready: bool
    reason: str
    label_example_count: int
    exact_training_match: bool


def assess_prototype_ownership(
    prediction: LabelPrediction,
    *,
    exact_training_match: bool,
    minimum_margin: float,
    minimum_label_examples: int,
) -> OwnershipAssessment:
    """Decide whether learned audio evidence is strong enough for ownership.

    Exact content/label matches honor human-approved training immediately.
    Generalization to unseen audio additionally requires a learned-radius
    match, adequate label support, and separation from the runner-up label.

    Args:
        prediction: Source-name-blind prototype evidence for one audio file.
        exact_training_match: True when the content hash maps to an unambiguous
            human-approved label in the active training manifest. The exact
            label is selected before this ownership check.
        minimum_margin: Required cosine-similarity lead for unseen audio.
        minimum_label_examples: Required distinct examples for unseen audio.

    Returns:
        An ownership assessment with a stable diagnostic reason.

    Raises:
        ValueError: If a threshold is negative or the example minimum is less
            than one.

    Side Effects:
        None.

    Important Constraints:
        Filenames, source paths, and sample-pack names are never inputs.
    """
    if minimum_margin < 0.0:
        raise ValueError("minimum_margin cannot be negative")
    if minimum_label_examples < 1:
        raise ValueError("minimum_label_examples must be positive")

    label_example_count = int(prediction.evidence.get("label_example_count", 0))
    if exact_training_match:
        return OwnershipAssessment(
            ready=True,
            reason="exact_human_training_match",
            label_example_count=label_example_count,
            exact_training_match=True,
        )
    if not prediction.known_distribution:
        reason = "outside_learned_radius"
    elif label_example_count < minimum_label_examples:
        reason = "insufficient_label_examples"
    elif prediction.margin < minimum_margin:
        reason = "ambiguous_nearest_labels"
    else:
        reason = "supported_separated_neighborhood"
    return OwnershipAssessment(
        ready=reason == "supported_separated_neighborhood",
        reason=reason,
        label_example_count=label_example_count,
        exact_training_match=False,
    )
