"""Parent-role audit diagnostics for sorter reports."""

from __future__ import annotations

from typing import Any

from aaron_sound_sorter.domain.models import CategoryGuess, ConsensusDecision, SharedAudioFacts, VoterResult
from aaron_sound_sorter.voters.scoring_tools import compatible_tops_for_role, detected_parent_role_name, role_strength

ROLE_KEYS = (
    "bass_loop",
    "voiced_one_shot",
    "percussive_one_shot",
    "pitched_music_phrase",
    "bright_drum_loop",
    "percussive_drum_loop",
    "pitched_music_loop",
    "low_rhythmic_drum_loop",
    "vocal_music_phrase",
)


def build_parent_role_audit(
    *,
    facts: SharedAudioFacts,
    brain_votes: VoterResult,
    physics_votes: VoterResult,
    decision: ConsensusDecision,
) -> dict[str, Any]:
    """Return a compact audit for broad role and family agreement."""
    roles = facts.evidence.get("measured_roles", {}) if isinstance(facts.evidence, dict) else {}
    if not isinstance(roles, dict):
        roles = {}
    gate = facts.evidence.get("dynamic_role_gate", {}) if isinstance(facts.evidence, dict) else {}
    if not isinstance(gate, dict):
        gate = {}
    parent_role = detected_parent_role_name(roles)
    compatible_tops = compatible_tops_for_role(parent_role)
    final_agrees = not compatible_tops or decision.final_top in compatible_tops
    return {
        "detected_parent_role": parent_role,
        "measured_role_vector": {key: role_strength(roles, key) for key in ROLE_KEYS},
        "compatible_top_families": compatible_tops,
        "final_label": decision.final_label,
        "final_top": decision.final_top,
        "final_agrees_with_measured_role": bool(final_agrees),
        "role_gate_mode": str(gate.get("mode", "missing")),
        "role_gate_final_effects_enabled": bool(gate.get("final_effects_enabled", False)),
        "role_gate_reason": str(gate.get("reason", "")),
        "top_family_scores": list(gate.get("top_scores", []) or [])[:8],
        "top_branch_scores": list(gate.get("branch_scores", []) or [])[:12],
        "brain_top_20": summarize_guesses(brain_votes.guesses[:20]),
        "physics_top_20": summarize_guesses(physics_votes.guesses[:20]),
    }


def summarize_guesses(guesses: list[CategoryGuess]) -> list[dict[str, Any]]:
    """Return audit-friendly rows from ranked guesses."""
    rows: list[dict[str, Any]] = []
    for guess in guesses:
        evidence = guess.evidence if isinstance(guess.evidence, dict) else {}
        rows.append(
            {
                "rank": guess.rank,
                "label": guess.label,
                "top_family": guess.top_family,
                "score": round(float(guess.score), 6),
                "confidence": round(float(guess.confidence), 6),
                "role_adjustment": evidence.get("role_adjustment"),
                "role_adjustment_applied": evidence.get("role_adjustment_applied"),
                "role_recommended_adjustment": evidence.get("role_recommended_adjustment"),
                "role_adjustment_mode": evidence.get("role_adjustment_mode"),
                "role_adjustment_reasons": evidence.get("role_adjustment_reasons", []),
                "candidate_role_signature": evidence.get("candidate_role_signature", {}),
                "candidate_role_distance": evidence.get("candidate_role_distance"),
                "family_compatibility": evidence.get("family_compatibility", {}),
            }
        )
    return rows
