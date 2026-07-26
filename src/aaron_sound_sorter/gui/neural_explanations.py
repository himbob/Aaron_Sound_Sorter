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
    """Return concise explanations of memory, CLAP, and final authority."""
    if row.neural_known_distribution is None:
        return ["Neural audio: unavailable for this preview."]
    return [
        neural_memory_summary(row),
        broad_clap_summary(row),
        *detailed_clap_lines(row),
        panns_summary(row),
        neural_result_summary(row),
    ]


def neural_memory_summary(row: PreviewRow) -> str:
    """Explain the trained prototype-memory result without model jargon."""
    label = row.neural_folder or "no category"
    if row.neural_exact_training_match:
        return f"Your trained memory: exact human-approved audio match → {label}."
    if row.neural_ownership_ready:
        count = row.neural_label_example_count
        example_text = f" from {count} approved examples" if count else ""
        return f"Your trained memory: ready to classify{example_text} → {label}."
    if row.neural_known_distribution:
        return f"Your trained memory: recognizes nearby audio, but is not ready to own the folder → {label}."
    return f"Your trained memory: this audio is outside its familiar neighborhood; nearest guess → {label}."


def broad_clap_summary(row: PreviewRow) -> str:
    """Explain independent broad CLAP evidence as a signal, not confidence."""
    family = SEMANTIC_FAMILY_NAMES.get(row.neural_semantic_family, row.neural_semantic_family or "no opinion")
    if row.neural_semantic_score < 0.10:
        signal = "weak"
    elif row.neural_semantic_margin >= 0.06:
        signal = "clear"
    elif row.neural_semantic_margin >= 0.025:
        signal = "mixed"
    else:
        signal = "close call"
    return f"CLAP broad hearing: {family} ({signal} signal; this is not a probability)."


def detailed_clap_lines(row: PreviewRow) -> list[str]:
    """Format experimental detailed prompt suggestions for human review."""
    if row.neural_prompt_status != "advisory_only":
        return ["CLAP detailed suggestions: unavailable; they did not affect sorting."]
    lines = ["CLAP detailed suggestions (experimental; advice only):"]
    for suggestion in row.neural_prompt_suggestions[:3]:
        path = str(suggestion.get("path", ""))
        audible_prompt = str(suggestion.get("top_positive_prompt", ""))
        explanation = f' — matched "{audible_prompt}"' if audible_prompt else ""
        lines.append(f"  - {path}{explanation}")
    if len(lines) == 1:
        lines.append("  - no detailed suggestion")
    return lines


def panns_summary(row: PreviewRow) -> str:
    """Format independent broad AudioSet events without implying probability."""
    if row.panns_status != "advisory_only" or not row.panns_events:
        return "PANNs broad events: unavailable; they did not affect sorting."
    event_text = "; ".join(
        f"{event.get('label', '')} {float(event.get('score', 0.0)):.2f}" for event in row.panns_events[:5]
    )
    if row.panns_contradiction_score > row.panns_support_score and row.panns_contradicting_events:
        role = "broad contradiction; may force Review"
    elif row.panns_support_score > 0.0 and row.panns_supporting_events:
        role = "broad support for trained memory"
    else:
        role = "advice only"
    return f"PANNs broad events (raw scores; {role}): {event_text}."


def neural_result_summary(row: PreviewRow) -> str:
    """Explain whether neural evidence controlled or merely advised."""
    status = row.consensus_status
    if status == "neural_semantic_family_conflict_review":
        return "Neural result: sent to Review because trained memory and independent CLAP hearing strongly disagreed."
    if status == "neural_panns_family_conflict_review":
        return "Neural result: sent to Review because trained memory and independent PANNs events strongly disagreed."
    if status == "neural_measured_structure_conflict_review":
        return "Neural result: sent to Review because the trained label contradicted the measured audio shape."
    if status == "neural_legacy_owner_conflict_review":
        return "Neural result: sent to Review because the available systems disagreed and memory was not ready."
    if status == "neural_known_distribution_owner":
        return "Neural result: trained memory controlled the proposed folder; detailed CLAP suggestions did not."
    return "Neural result: evidence was advisory; the normal sorter controlled the proposed folder."
