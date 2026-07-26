"""Turn explicit GUI approvals into durable confidence-learning outcomes."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from aaron_sound_sorter.gui.models import PreviewRow, SortPreviewSession
from aaron_sound_sorter.neural_audio.authority import neural_structure_conflicts
from aaron_sound_sorter.neural_audio.calibration import ReviewFeedback, ReviewFeedbackStore
from aaron_sound_sorter.neural_audio.hashing import sha256_file
from aaron_sound_sorter.neural_audio.semantic_panel import compatible_semantic_families
from aaron_sound_sorter.taxonomy_contracts import normalize_taxonomy_path


def calibration_feedback_from_preview_row(
    row: PreviewRow,
    *,
    duplicate_conflict: bool = False,
    brain_version: str = "",
    taxonomy_version: str = "2026-07-25",
) -> ReviewFeedback | None:
    """Convert one explicit GUI approval into separated calibration evidence.

    Args:
        row: Preview row after the user approves or corrects its folder.
        duplicate_conflict: Whether corpus intake found conflicting identity.
        brain_version: Prototype/brain identity active during prediction.
        taxonomy_version: Canonical taxonomy version.

    Returns:
        Review feedback, or ``None`` when the neural lane was unavailable.

    Side Effects:
        Hashes decoded-audio source bytes; never uses its name as evidence.
    """
    predicted_label = normalize_taxonomy_path(row.neural_folder)
    approved_label = normalize_taxonomy_path(row.approved_folder)
    if row.neural_known_distribution is None or not predicted_label or not approved_label:
        return None
    prompt = row.neural_prompt_suggestions[0] if row.neural_prompt_suggestions else {}
    compatible_families = compatible_semantic_families(predicted_label)
    if not row.neural_semantic_family or not compatible_families:
        broad_family_agreement = 0.5
    else:
        broad_family_agreement = 1.0 if row.neural_semantic_family in compatible_families else 0.0
    structure_agreement = 1.0
    if row.result is not None and neural_structure_conflicts(row.result, predicted_label):
        structure_agreement = 0.0
    return ReviewFeedback(
        file_sha256=sha256_file(row.source_path),
        provider_id="foundation_panel",
        predicted_label=predicted_label,
        accepted=predicted_label == approved_label,
        top_similarity=float(row.neural_similarity),
        margin=float(row.neural_margin),
        radius_ratio=float(row.neural_radius_ratio),
        label_example_count=max(0, int(row.neural_label_example_count)),
        label_prototype_count=0,
        structure_agreement=structure_agreement,
        created_utc=datetime.now(timezone.utc).isoformat(),
        final_approved_label=approved_label,
        parent_family_accepted=predicted_label.split("/", 1)[0] == approved_label.split("/", 1)[0],
        prompt_top_similarity=float(prompt.get("positive_score", 0.0)),
        prompt_margin=float(prompt.get("prompt_margin", 0.0)),
        panns_support_score=float(row.panns_support_score),
        panns_contradiction_score=float(row.panns_contradiction_score),
        broad_family_agreement=broad_family_agreement,
        out_of_distribution=0.0 if row.neural_known_distribution else 1.0,
        duplicate_conflict=1.0 if duplicate_conflict else 0.0,
        brain_version=str(brain_version),
        taxonomy_version=str(taxonomy_version),
    )


def record_approved_session_feedback(
    project_root: Path,
    session: SortPreviewSession,
) -> int:
    """Record accepted and corrected predictions when an approved plan exports.

    Exporting the plan is the explicit human action that means every unchanged
    folder is accepted for this review session. Corrected rows remain rejected
    predictions with their final approved label preserved.

    Args:
        project_root: Repository root owning the durable local feedback log.
        session: Fully approved GUI preview session.

    Returns:
        Number of neural outcomes appended.

    Side Effects:
        Appends source-name-blind JSONL feedback and fsyncs each event.
    """
    store = ReviewFeedbackStore(
        Path(project_root) / "neural_artifacts" / "calibration_feedback" / "review_feedback.jsonl"
    )
    appended = 0
    for row in session.rows:
        feedback = calibration_feedback_from_preview_row(row, brain_version=str(session.brain_path))
        if feedback is None:
            continue
        store.append(feedback)
        appended += 1
    return appended
