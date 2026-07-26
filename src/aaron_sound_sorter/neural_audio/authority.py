"""Shared neural production authority for GUI and command-line sorting."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from aaron_sound_sorter.domain.models import ConsensusDecision, SortFileResult
from aaron_sound_sorter.engine.learned_memory_contracts import (
    EXACT_HUMAN_TEACHER_OWNER_CLAIM_SOURCE,
)
from aaron_sound_sorter.taxonomy_contracts import (
    canonicalize_taxonomy_label,
    is_valid_taxonomy_contract_label,
    taxonomy_label_contract,
)

from .runtime import (
    NeuralRuntimeBatch,
    NeuralRuntimePrediction,
    run_configured_neural_predictions,
)
from .semantic_panel import (
    SemanticCompatibilityAssessment,
    assess_semantic_label_compatibility,
)

MEASURED_CONFLICT_REVIEW = "_TO_REVIEW/Measured Role Conflict"


def apply_configured_neural_authority(
    *,
    project_root: Path,
    results: list[SortFileResult],
    report_dir: Path,
) -> tuple[list[SortFileResult], NeuralRuntimeBatch]:
    """Run configured waveform inference and apply category-wide authority.

    Args:
        project_root: Repository root containing neural runtime configuration.
        results: Legacy/measured classification results before placement.
        report_dir: Explicit output folder for runtime diagnostics.

    Returns:
        Updated results in original order plus the neural runtime batch.

    Side Effects:
        Runs the isolated neural subprocess and writes diagnostics below
        ``report_dir``.

    Important Constraints:
        Source paths cross the subprocess boundary only for audio I/O. The
        model receives decoded waveforms, never filename or folder tokens.
    """
    if not results:
        return [], NeuralRuntimeBatch("skipped", "No classified files were supplied.", ())
    row_ids = [f"{index:05d}" for index in range(1, len(results) + 1)]
    batch = run_configured_neural_predictions(
        project_root,
        list(zip(row_ids, (result.source_path for result in results))),
        report_dir,
    )
    predictions = {prediction.row_id: prediction for prediction in batch.predictions}
    updated = [
        apply_neural_prediction_to_result(result, predictions.get(row_id)) for row_id, result in zip(row_ids, results)
    ]
    return updated, batch


def apply_neural_prediction_to_result(
    result: SortFileResult,
    prediction: NeuralRuntimePrediction | None,
) -> SortFileResult:
    """Apply one neural prediction to an unplaced domain result.

    Args:
        result: Measured/legacy result before output placement.
        prediction: Source-name-blind runtime prediction, if available.

    Returns:
        The original result when no neural action is justified, otherwise a
        copy with a neural-owned or Review decision.

    Side Effects:
        Adds neural diagnostics to the result's in-memory evidence mapping.
    """
    if prediction is None:
        return result
    neural_label = canonicalize_taxonomy_label(prediction.predicted_label)
    evidence = _prediction_evidence(prediction)
    result.facts.evidence["neural_runtime"] = evidence
    if not is_valid_taxonomy_contract_label(neural_label):
        return result

    legacy_label = canonicalize_taxonomy_label(result.decision.folder_path or result.decision.final_label)
    if prediction.ownership_ready:
        if neural_panns_contradicts(prediction):
            event_names = ", ".join(event.label for event in prediction.panns_contradicting_events[:3])
            reason = (
                "Learned memory matched, but independent PANNs audio events "
                f"contradicted its source family ({event_names}); forcing human review."
            )
            return _replace_decision(
                result,
                folder_path=MEASURED_CONFLICT_REVIEW,
                final_top="_TO_REVIEW",
                consensus_status="neural_panns_family_conflict_review",
                reason=reason,
                neural_evidence=evidence,
            )
        semantic_assessment = neural_semantic_compatibility(prediction, neural_label)
        evidence["semantic_compatibility"] = {
            "contradictory": semantic_assessment.contradictory,
            "reason": semantic_assessment.reason,
            "compatible_families": list(semantic_assessment.compatible_families),
            "best_compatible_score": semantic_assessment.best_compatible_score,
            "contradiction_gap": semantic_assessment.contradiction_gap,
        }
        if semantic_assessment.contradictory:
            reason = (
                "Learned memory matched, but independent audio-only semantics "
                f"contradicted its source family ({prediction.semantic_family} vs {neural_label}); "
                "forcing human review."
            )
            return _replace_decision(
                result,
                folder_path=MEASURED_CONFLICT_REVIEW,
                final_top="_TO_REVIEW",
                consensus_status="neural_semantic_family_conflict_review",
                reason=reason,
                neural_evidence=evidence,
            )
        if neural_structure_conflicts(result, neural_label):
            reason = (
                "Neural identity had production-ready evidence, but its terminal contradicted measured audio structure."
            )
            return _replace_decision(
                result,
                folder_path=MEASURED_CONFLICT_REVIEW,
                final_top="_TO_REVIEW",
                consensus_status="neural_measured_structure_conflict_review",
                reason=reason,
                neural_evidence=evidence,
            )
        reason = (
            "Source-name-blind neural audio had production-ready learned evidence "
            f"and took ownership from the legacy proposal ({legacy_label})."
        )
        return _replace_decision(
            result,
            folder_path=neural_label,
            final_top=neural_label.split("/", 1)[0],
            consensus_status="neural_production_owner",
            reason=reason,
            neural_evidence=evidence,
        )

    if neural_defers_to_exact_human_teacher(result, prediction):
        evidence["authority_action"] = "defer_to_exact_human_teacher"
        return result

    if neural_disagreement_requires_review(
        prediction,
        legacy_label,
        neural_label,
        result=result,
    ):
        neural_family, legacy_family = neural_conflict_families(neural_label, legacy_label)
        reason = (
            "Neural audio was not ready for production ownership and disagreed "
            f"with the legacy source family ({neural_family} vs {legacy_family}); "
            "forcing human review."
        )
        return _replace_decision(
            result,
            folder_path=MEASURED_CONFLICT_REVIEW,
            final_top="_TO_REVIEW",
            consensus_status="neural_legacy_owner_conflict_review",
            reason=reason,
            neural_evidence=evidence,
        )
    return result


def neural_defers_to_exact_human_teacher(
    result: SortFileResult,
    prediction: NeuralRuntimePrediction,
) -> bool:
    """Return whether weak neural evidence must preserve exact human memory."""
    return bool(
        not prediction.ownership_ready and result.decision.consensus_status == EXACT_HUMAN_TEACHER_OWNER_CLAIM_SOURCE
    )


def neural_semantic_compatibility(
    prediction: NeuralRuntimePrediction,
    neural_label: str,
) -> SemanticCompatibilityAssessment:
    """Compare a learned label with independent audio-only source evidence.

    Args:
        prediction: Runtime prediction carrying zero-shot family scores.
        neural_label: Canonical learned taxonomy label.

    Returns:
        Semantic compatibility assessment used only as a Review guardrail.

    Side Effects:
        None.
    """
    assessment = assess_semantic_label_compatibility(
        neural_label,
        predicted_family=prediction.semantic_family,
        top_score=prediction.semantic_top_score,
        family_scores=prediction.semantic_family_scores,
    )
    if (
        assessment.contradictory
        and prediction.panns_support_score > prediction.panns_contradiction_score
        and prediction.panns_supporting_events
    ):
        return SemanticCompatibilityAssessment(
            False,
            "panns_independent_support_overrules_clap_veto",
            assessment.compatible_families,
            assessment.best_compatible_score,
            assessment.contradiction_gap,
        )
    return assessment


def neural_panns_contradicts(prediction: NeuralRuntimePrediction) -> bool:
    """Return whether mapped PANNs events independently reject the label."""
    return bool(
        prediction.panns_contradicting_events and prediction.panns_contradiction_score > prediction.panns_support_score
    )


def neural_disagreement_requires_review(
    prediction: NeuralRuntimePrediction,
    legacy_label: str,
    neural_label: str,
    *,
    result: SortFileResult | None = None,
) -> bool:
    """Return whether weak neural and legacy source identities require review.

    Unknown neural predictions cannot own a label, but their disagreement is
    still useful negative evidence. Broad family conflicts and meaningful
    within-family source conflicts therefore go to Review. Closely related
    tonal families, such as keys and synths, remain together so sparse neural
    evidence does not create unnecessary review churn.
    """
    if prediction.ownership_ready:
        return False
    neural_family, legacy_family = neural_conflict_families(neural_label, legacy_label)
    if not neural_family or not legacy_family or neural_family == legacy_family:
        return False
    neural_owner = neural_owner_family(neural_label)
    legacy_owner = neural_owner_family(legacy_label)
    if neural_owner != legacy_owner:
        return True
    if neural_family in {"instruments_broad", "fx_broad"} or legacy_family in {
        "instruments_broad",
        "fx_broad",
    }:
        return False
    if neural_source_family(prediction.second_label) == legacy_family:
        return False
    if result is not None and _measured_trace_supports_source_family(result, neural_family):
        return False
    return True


def neural_conflict_families(neural_label: str, legacy_label: str) -> tuple[str, str]:
    """Return comparable source-family names for a neural/legacy disagreement."""
    neural_owner = neural_owner_family(neural_label)
    legacy_owner = neural_owner_family(legacy_label)
    if neural_owner and legacy_owner and neural_owner != legacy_owner:
        return neural_owner, legacy_owner
    neural_source = neural_source_family(neural_label)
    legacy_source = neural_source_family(legacy_label)
    return neural_source, legacy_source


def neural_owner_family(label: str) -> str:
    """Return the broad source-family contract used for conflict Review."""
    normalized = canonicalize_taxonomy_label(label)
    lowered = normalized.lower()
    if normalized.startswith("_TO_REVIEW/"):
        return "review"
    if normalized.startswith("Instruments/Voice/") or normalized.startswith("FX/Human and Voice FX/"):
        return "voice"
    if normalized.startswith("Drums/"):
        return "drums"
    if normalized.startswith("FX/"):
        return "fx"
    if normalized.startswith("Instruments/"):
        return "instruments_nonvoice"
    if "voice" in lowered or "vocal" in lowered:
        return "voice"
    return ""


def neural_source_family(label: str) -> str:
    """Return a source-identity group used only to detect unsafe disagreement."""
    normalized = canonicalize_taxonomy_label(label)
    parts = normalized.split("/")
    if normalized.startswith("Instruments/Voice/"):
        return "voice_musical"
    if normalized.startswith("FX/Human and Voice FX/"):
        return "voice_effect"
    if normalized.startswith("Drums/") and len(parts) > 1:
        drum_family = parts[1]
        drum_groups = {
            "Claps Snaps Slaps": "drums_backbeat",
            "Cymbals": "drums_cymbal",
            "Drum Fills and Rolls": "drums_ensemble",
            "Drum Loops": "drums_ensemble",
            "Hi Hats": "drums_cymbal",
            "Kick Drums": "drums_kick",
            "Percussion": "drums_percussion",
            "Percussion Loops": "drums_percussion",
            "Rims and Sticks": "drums_backbeat",
            "Snares": "drums_backbeat",
            "Toms": "drums_tom",
            "World Percussion": "drums_percussion",
        }
        return drum_groups.get(drum_family, f"drums_{drum_family.casefold().replace(' ', '_')}")
    if normalized.startswith("Instruments/") and len(parts) > 1:
        instrument_family = parts[1]
        instrument_groups = {
            "Bass": "instruments_bass",
            "Brass": "instruments_winds",
            "Guitar": "instruments_strings",
            "Instrument Loops": "instruments_broad",
            "Keys": "instruments_keyed_tonal",
            "Mallets and Bells": "instruments_keyed_tonal",
            "Mixed Musical Loops": "instruments_mixed",
            "Plucked Strings": "instruments_strings",
            "Strings": "instruments_strings",
            "Strings Bowed": "instruments_strings",
            "Synths": "instruments_keyed_tonal",
            "Winds": "instruments_winds",
            "Woodwinds": "instruments_winds",
        }
        return instrument_groups.get(
            instrument_family,
            f"instruments_{instrument_family.casefold().replace(' ', '_')}",
        )
    if normalized.startswith("FX/") and len(parts) > 1:
        fx_family = parts[1]
        if fx_family == "Hybrid Designed FX":
            return "fx_broad"
        return f"fx_{fx_family.casefold().replace(' ', '_')}"
    return neural_owner_family(normalized)


def _measured_trace_supports_source_family(result: SortFileResult, source_family: str) -> bool:
    """Return whether strong pre-neural measured evidence supports a source family."""
    trace = result.decision.authority_trace
    if not isinstance(trace, dict):
        return False
    for claim_name in ("raw_claim", "winner_after_pick", "final_claim"):
        claim = trace.get(claim_name)
        if not isinstance(claim, dict):
            continue
        claim_path = str(claim.get("path", ""))
        try:
            strength = float(claim.get("strength", 0.0))
        except (TypeError, ValueError):
            strength = 0.0
        if strength >= 0.85 and neural_source_family(claim_path) == source_family:
            return True
    return False


def neural_structure_conflicts(result: SortFileResult, neural_label: str) -> bool:
    """Return whether decisive measured structure rejects a neural terminal."""
    terminal = taxonomy_label_contract(neural_label).structure_terminal
    if has_decisive_loop_structure(result) and terminal == "One Shots":
        return True
    return bool(result.facts.is_single_event_like and terminal == "Loops")


def has_decisive_loop_structure(result: SortFileResult) -> bool:
    """Return whether measured structure rules out one-shot placement."""
    facts = result.facts
    if not facts.is_loop_like or facts.is_single_event_like:
        return False
    evidence = facts.evidence if isinstance(facts.evidence, dict) else {}
    shape = evidence.get("shape_vote") if isinstance(evidence.get("shape_vote"), dict) else {}
    roles = evidence.get("measured_roles") if isinstance(evidence.get("measured_roles"), dict) else {}
    structure = evidence.get("structure_facts") if isinstance(evidence.get("structure_facts"), dict) else {}

    shape_name = str(shape.get("primary_shape") or "")
    shape_confidence = _number(shape.get("confidence"))
    event_count = max(
        _number(shape.get("onset_count")),
        _number(structure.get("event_count_estimate")),
        _number(evidence.get("event_count_estimate")),
    )
    onset_span = max(
        _number(shape.get("onset_span_ratio")),
        _number(structure.get("onset_span_ratio")),
        _number(evidence.get("onset_span_ratio")),
    )
    repetition = _number(shape.get("true_repetition_score"))
    loop_population = max(
        _number(shape.get("pitched_event_ratio")),
        _number(shape.get("percussive_event_ratio")),
        _number(shape.get("drumlike_frame_ratio")),
    )
    loop_role = max(
        *(
            _number(roles.get(role_name))
            for role_name in (
                "bass_loop",
                "pitched_music_loop",
                "bright_drum_loop",
                "percussive_drum_loop",
                "low_rhythmic_drum_loop",
            )
        ),
        0.0,
    )
    repeated_shape_names = {
        "bass_phrase",
        "beat_loop",
        "compound_musical_loop",
        "designed_motion_fx_loop",
        "instrument_plus_fx_loop",
        "layered_phrase",
        "mixed_instrument_loop",
        "pitched_phrase",
        "pitched_phrase_shape",
        "pitched_repetition_phrase",
        "repeated_phrase_loop",
        "solo_phrase",
        "vocal_phrase",
    }
    repeated_shape = bool(
        shape_name in repeated_shape_names
        and shape_confidence >= 0.70
        and event_count >= 4.0
        and onset_span >= 0.45
        and (repetition >= 0.45 or loop_population >= 0.72)
    )
    measured_loop_role = bool(loop_role >= 0.74 and event_count >= 4.0 and onset_span >= 0.40)
    sustained_loop = bool(
        facts.is_long
        and shape_name == "sustained_pad"
        and shape_confidence >= 0.80
        and _number(shape.get("pitched_event_ratio")) >= 0.72
        and _number(shape.get("sustained_tonal_frame_ratio")) >= 0.72
        and _number(shape.get("percussive_event_ratio")) <= 0.25
    )
    return bool(repeated_shape or measured_loop_role or sustained_loop)


def _replace_decision(
    result: SortFileResult,
    *,
    folder_path: str,
    final_top: str,
    consensus_status: str,
    reason: str,
    neural_evidence: dict[str, Any],
) -> SortFileResult:
    trace = dict(result.decision.authority_trace) if isinstance(result.decision.authority_trace, dict) else {}
    trace["neural_runtime"] = neural_evidence
    trace["pre_neural_decision"] = {
        "folder_path": result.decision.folder_path,
        "final_top": result.decision.final_top,
        "consensus_status": result.decision.consensus_status,
    }
    trace["final_source"] = consensus_status
    decision: ConsensusDecision = replace(
        result.decision,
        final_label=folder_path,
        final_top=final_top,
        folder_path=folder_path,
        consensus_status=consensus_status,
        reason=reason,
        authority_trace=trace,
    )
    return replace(result, decision=decision)


def _prediction_evidence(prediction: NeuralRuntimePrediction) -> dict[str, Any]:
    return {
        "predicted_label": prediction.predicted_label,
        "second_label": prediction.second_label,
        "top_similarity": prediction.top_similarity,
        "second_similarity": prediction.second_similarity,
        "margin": prediction.margin,
        "radius_ratio": prediction.radius_ratio,
        "known_distribution": prediction.known_distribution,
        "label_example_count": prediction.label_example_count,
        "exact_training_match": prediction.exact_training_match,
        "ownership_ready": prediction.ownership_ready,
        "ownership_reason": prediction.ownership_block_reason,
        "semantic_family": prediction.semantic_family,
        "semantic_second_family": prediction.semantic_second_family,
        "semantic_top_score": prediction.semantic_top_score,
        "semantic_second_score": prediction.semantic_second_score,
        "semantic_margin": prediction.semantic_margin,
        "semantic_family_scores": dict(prediction.semantic_family_scores),
        "panns_support_score": prediction.panns_support_score,
        "panns_contradiction_score": prediction.panns_contradiction_score,
        "panns_supporting_events": [
            {"label": event.label, "score": event.score} for event in prediction.panns_supporting_events
        ],
        "panns_contradicting_events": [
            {"label": event.label, "score": event.score} for event in prediction.panns_contradicting_events
        ],
        "source_name_policy": "audio waveform and content hash only",
    }


def _number(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
