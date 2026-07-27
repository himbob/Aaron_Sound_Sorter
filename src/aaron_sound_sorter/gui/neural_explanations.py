"""Plain-language neural evidence explanations shared by both GUIs."""

from __future__ import annotations

from .models import PreviewRow

SEMANTIC_FAMILY_NAMES = {
    "human_voice": "human voice",
    "drums": "drums",
    "drum_kick": "kick drum",
    "drum_backbeat": "snare, clap, snap, or rimshot",
    "drum_cymbal": "cymbal or hi-hat",
    "drum_percussion": "percussion",
    "drum_full": "full drum beat",
    "bass": "bass instrument",
    "keys": "piano or keyboard",
    "guitar_plucked": "guitar or plucked strings",
    "strings_bowed": "bowed strings",
    "woodwind_reed": "woodwind or reed instrument",
    "brass": "brass instrument",
    "mallet_bell": "mallet, bell, or chime",
    "synth": "synthesizer",
    "fx_impact": "impact or boom FX",
    "fx_transition": "transition, riser, or whoosh FX",
    "fx_glitch": "glitch or digital FX",
    "fx_alert": "alarm or alert FX",
    "fx_machine": "machine or vehicle FX",
    "fx_nature": "nature or weather FX",
    "fx_texture": "texture, drone, or ambience FX",
    "fx_foley": "foley or real-world object FX",
    "animal_creature": "animal or creature sound",
}


def neural_evidence_lines(row: PreviewRow) -> list[str]:
    """Return one independent status line for every neural evidence lane."""
    return [
        neural_memory_summary(row),
        broad_clap_summary(row),
        *detailed_clap_lines(row),
        panns_summary(row),
        calibration_summary(row),
        neural_result_summary(row),
    ]


def neural_memory_summary(row: PreviewRow) -> str:
    """Explain the trained prototype-memory result without model jargon."""
    if row.neural_decision_state == "provisional":
        return "Your trained memory: Analyzing..."
    if row.neural_known_distribution is None:
        return _missing_lane_summary(row, "Your trained memory")
    label = row.neural_folder or "No Result"
    if row.neural_exact_training_match:
        return f"Your trained memory: exact human-approved audio match → {label}."
    if row.neural_ownership_ready:
        count = row.neural_label_example_count
        example_text = f" from {count} approved examples" if count else ""
        return f"Your trained memory: ready to classify{example_text} → {label}."
    if row.neural_ownership_reason in {
        "confidence_calibration_not_promoted",
        "category_group_not_promoted",
    }:
        return (
            "Your trained memory: found a familiar match, but this category group "
            f"has not passed the automatic-placement safety gate → {label}."
        )
    if row.neural_known_distribution:
        return f"Your trained memory: recognizes nearby audio, but is not ready to own the folder → {label}."
    return f"Your trained memory: this audio is outside its familiar neighborhood; nearest guess → {label}."


def broad_clap_summary(row: PreviewRow) -> str:
    """Show CLAP's top broad families instead of hiding the runner-ups."""
    if row.neural_decision_state == "provisional":
        return "CLAP broad hearing: Analyzing..."
    if not row.neural_semantic_family:
        return _missing_lane_summary(
            row,
            "CLAP broad hearing",
            lane_status=row.neural_semantic_status,
        )
    ranked = sorted(
        row.neural_semantic_family_scores.items(),
        key=lambda item: (-float(item[1]), item[0]),
    )[:3]
    if not ranked:
        ranked = [(row.neural_semantic_family, row.neural_semantic_score)]
        if row.neural_semantic_second_family:
            ranked.append((row.neural_semantic_second_family, row.neural_semantic_second_score))
    family_text = "; ".join(
        f"{SEMANTIC_FAMILY_NAMES.get(family, family)} {float(score):.3f}"
        for family, score in ranked
    )
    if row.neural_semantic_score < 0.10:
        signal = "weak"
    elif row.neural_semantic_margin >= 0.06:
        signal = "clear"
    elif row.neural_semantic_margin >= 0.025:
        signal = "mixed"
    else:
        signal = "close call"
    return (
        f"CLAP broad hearing (top families; raw similarity, not probability): {family_text}; "
        f"{signal} signal."
    )


def detailed_clap_lines(row: PreviewRow) -> list[str]:
    """Format experimental detailed prompt suggestions for human review."""
    if row.neural_prompt_status != "advisory_only" or not row.neural_prompt_suggestions:
        if row.neural_prompt_status in {"unavailable", "invalid_index", "model_mismatch"}:
            return [f"CLAP detailed suggestions: Unavailable ({row.neural_prompt_status}); they did not affect sorting."]
        return [
            f"{_missing_lane_summary(row, 'CLAP detailed suggestions', lane_status=row.neural_prompt_status)}; "
            "they did not affect sorting."
        ]
    lines = ["CLAP detailed suggestions (experimental; advice only):"]
    for suggestion in row.neural_prompt_suggestions[:3]:
        path = str(suggestion.get("path", ""))
        audible_prompt = str(suggestion.get("top_positive_prompt", ""))
        explanation = f' — matched "{audible_prompt}"' if audible_prompt else ""
        lines.append(f"  - {path}{explanation}")
    return lines


def panns_summary(row: PreviewRow) -> str:
    """Show raw PANNs events plus candidate-independent grouped families."""
    if row.neural_decision_state == "provisional":
        return "PANNs broad events: Analyzing..."
    if not row.panns_events:
        if row.panns_status == "unavailable":
            return "PANNs broad events: Unavailable; the model did not return an event list."
        return _missing_lane_summary(row, "PANNs broad events", lane_status=row.panns_status)
    event_text = "; ".join(
        f"{event.get('label', '')} {float(event.get('score', 0.0)):.2f}" for event in row.panns_events[:5]
    )
    family_text = "; ".join(
        f"{SEMANTIC_FAMILY_NAMES.get(family, family)} {float(score):.2f}"
        for family, score in sorted(
            row.panns_family_scores.items(),
            key=lambda item: (-float(item[1]), item[0]),
        )[:3]
    )
    if row.panns_contradiction_score > row.panns_support_score and row.panns_contradicting_events:
        role = "broad contradiction; may force Review"
    elif row.panns_support_score > 0.0 and row.panns_supporting_events:
        role = "broad support for trained memory"
    else:
        role = "advice only"
    grouped = f" Grouped families: {family_text}." if family_text else ""
    return f"PANNs broad events (raw scores; {role}): {event_text}.{grouped}"


def _missing_lane_summary(
    row: PreviewRow,
    lane_name: str,
    *,
    lane_status: str = "",
) -> str:
    """Explain why a lane is empty without mislabeling failures as No Result."""
    if row.neural_decision_state == "provisional":
        return f"{lane_name}: Analyzing..."
    if lane_status == "runtime_error":
        return f"{lane_name}: Runtime Error."
    if lane_status in {"unavailable", "invalid_index", "model_mismatch"}:
        return f"{lane_name}: Unavailable ({lane_status})."
    if row.neural_row_error:
        detail = _compact_runtime_detail(row.neural_row_error)
        return f"{lane_name}: Runtime Error ({detail})."
    if row.neural_runtime_status == "not_completed_timeout":
        return f"{lane_name}: Not Completed before the neural runtime timeout."
    if row.neural_runtime_status in {"error", "unavailable"}:
        detail = _compact_runtime_detail(row.neural_runtime_message)
        suffix = f" ({detail})" if detail else ""
        return f"{lane_name}: Runtime {row.neural_runtime_status.title()}{suffix}."
    return f"{lane_name}: No Result."


def _compact_runtime_detail(value: str, limit: int = 180) -> str:
    """Keep runtime diagnostics readable in the GUI."""
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def calibration_summary(row: PreviewRow) -> str:
    """Explain reviewed confidence learning without exposing feature jargon."""
    if row.result is None or not isinstance(row.result.facts.evidence, dict):
        return "Confidence learning: not fitted yet; conservative safety rules remain active."
    neural = row.result.facts.evidence.get("neural_runtime", {})
    calibration = neural.get("confidence_calibration", {}) if isinstance(neural, dict) else {}
    if not isinstance(calibration, dict) or calibration.get("status") != "available":
        return "Confidence learning: not fitted yet; conservative safety rules remain active."
    probability = calibration.get("prediction_correct_probability")
    if probability is None:
        return "Confidence learning: fitted model did not produce a usable score; Review safety remains active."
    return f"Confidence learning: reviewed outcomes estimate {float(probability):.0%} correctness for this decision."


def neural_result_summary(row: PreviewRow) -> str:
    """Explain whether neural evidence controlled or merely advised."""
    if row.neural_decision_state == "provisional":
        return "Neural result: Pending; the visible category is not final yet."
    if row.neural_decision_state == "finalized_without_neural":
        return "Neural result: Not applied; the row was finalized without neural evidence."
    status = row.consensus_status
    if status == "neural_semantic_family_conflict_review":
        return "Neural result: sent to Review because trained memory and independent CLAP hearing strongly disagreed."
    if status == "neural_panns_family_conflict_review":
        return "Neural result: sent to Review because trained memory and independent PANNs events strongly disagreed."
    if status == "neural_measured_structure_conflict_review":
        return "Neural result: sent to Review because the trained label contradicted the measured audio shape."
    if status == "neural_legacy_owner_conflict_review":
        return "Neural result: sent to Review because the available systems disagreed and memory was not ready."
    if status == "neural_calibration_below_threshold_review":
        return (
            "Neural result: sent to Review because confidence learned from past reviews was below the ownership gate."
        )
    if status in {"neural_known_distribution_owner", "neural_production_owner"}:
        return "Neural result: trained memory controlled the proposed folder; detailed CLAP suggestions did not."
    return "Neural result: evidence was advisory; the normal sorter controlled the proposed folder."
