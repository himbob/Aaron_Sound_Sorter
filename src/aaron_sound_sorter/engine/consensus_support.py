"""Small support helpers for consensus candidate decisions."""

from __future__ import annotations

from typing import Any

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.voters.scoring_tools import role_strength

VOICE_CANDIDATE_FRAGMENTS = (
    "human and voice",
    "voice",
    "vocal",
    "vox",
    "spoken",
    "choir",
    "breath",
    "crowd",
    "mouth",
)


def _row_path_text(row: dict[str, Any]) -> str:
    """Return a normalized internal candidate path for structural checks."""
    return str(row.get("folder_path") or row.get("label") or "").replace("\\", "/").lower()


def _candidate_role_signature(row: dict[str, Any]) -> dict[str, Any]:
    """Return a candidate role signature from a shared candidate row."""
    signature = row.get("candidate_role_signature")
    if isinstance(signature, dict):
        return signature
    for key in ("brain_evidence", "physics_evidence", "evidence"):
        evidence = row.get(key)
        if isinstance(evidence, dict) and isinstance(evidence.get("candidate_role_signature"), dict):
            return evidence["candidate_role_signature"]
    return {}


def _candidate_voice_fit(row: dict[str, Any]) -> float:
    """Return candidate-internal support for voice/human-vocal identity."""
    signature = _candidate_role_signature(row)
    return max(
        role_strength(signature, "vocal_music_phrase"),
        role_strength(signature, "vocal_phrase"),
        role_strength(signature, "vocal_one_shot"),
        role_strength(signature, "voiced_one_shot"),
    )


def _is_human_voice_candidate_row(row: dict[str, Any]) -> bool:
    """Return True when a shared candidate is a Human/Voice-like path."""
    path = _row_path_text(row)
    return any(fragment in path for fragment in VOICE_CANDIDATE_FRAGMENTS)


def _nested_role_strength(source: dict[str, Any], role: str) -> float:
    """Read a role value from a dict-like role vector."""
    if not isinstance(source, dict):
        return 0.0
    return _role_value(source, role)


def _strong_vocal_shape_true_bucket_evidence(facts: SharedAudioFacts, shape_confidence: float) -> bool:
    """Return True when measured audio strongly supports broad Human/Voice."""
    evidence = facts.evidence if isinstance(facts.evidence, dict) else {}
    measured_roles = evidence.get("measured_roles", {}) if isinstance(evidence, dict) else {}
    direct_view = evidence.get("direct_body_view", {}) if isinstance(evidence, dict) else {}
    direct_roles = direct_view.get("measured_roles", {}) if isinstance(direct_view, dict) else {}
    eligibility = evidence.get("parent_eligibility_v2", {}) if isinstance(evidence, dict) else {}

    role_name = str(eligibility.get("role_name", "") or "") if isinstance(eligibility, dict) else ""
    try:
        eligibility_confidence = (
            float(eligibility.get("confidence", 0.0) or 0.0) if isinstance(eligibility, dict) else 0.0
        )
    except Exception:
        eligibility_confidence = 0.0

    measured_voice = max(
        _nested_role_strength(measured_roles, "vocal_music_phrase"),
        _nested_role_strength(measured_roles, "vocal_phrase"),
        _nested_role_strength(measured_roles, "vocal_one_shot"),
        _nested_role_strength(measured_roles, "voiced_one_shot"),
    )
    direct_voice = max(
        _nested_role_strength(direct_roles, "vocal_music_phrase"),
        _nested_role_strength(direct_roles, "vocal_phrase"),
        _nested_role_strength(direct_roles, "vocal_one_shot"),
        _nested_role_strength(direct_roles, "voiced_one_shot"),
    )

    role_evidence = {}
    if isinstance(measured_roles, dict) and isinstance(measured_roles.get("evidence"), dict):
        role_evidence.update(measured_roles.get("evidence") or {})
    if isinstance(direct_roles, dict) and isinstance(direct_roles.get("evidence"), dict):
        role_evidence.update(direct_roles.get("evidence") or {})
    try:
        formant_identity = float(role_evidence.get("formant_light_voice_identity", 0.0) or 0.0)
    except Exception:
        formant_identity = 0.0

    vocal_eligibility = role_name in {"vocal_phrase", "vocal_one_shot", "voiced_one_shot", "vocal_music_phrase"}
    decisive_vocal_eligibility = vocal_eligibility and eligibility_confidence >= 0.74 and shape_confidence >= 0.88
    direct_vocal_hit = direct_voice >= 0.84 and shape_confidence >= 0.88 and formant_identity >= 0.70
    measured_vocal_phrase = measured_voice >= 0.84 and shape_confidence >= 0.90 and formant_identity >= 0.70
    measured_vocal_one_shot = (
        vocal_eligibility
        and eligibility_confidence >= 0.80
        and measured_voice >= 0.78
        and shape_confidence >= 0.78
        and formant_identity >= 0.68
    )
    return bool(decisive_vocal_eligibility or direct_vocal_hit or measured_vocal_phrase or measured_vocal_one_shot)


def _fact_number(facts: SharedAudioFacts, name: str, default: float = 0.0) -> float:
    """Return a numeric measured fact without throwing on missing data."""
    try:
        if name in facts.feature_values_by_name:
            return float(facts.feature_values_by_name.get(name, default) or default)
        evidence = facts.evidence if isinstance(facts.evidence, dict) else {}
        if name in evidence:
            return float(evidence.get(name, default) or default)
        feature_values = evidence.get("feature_values_by_name", {}) if isinstance(evidence, dict) else {}
        if isinstance(feature_values, dict):
            return float(feature_values.get(name, default) or default)
    except Exception:
        return default
    return default


def _direct_body_roles(facts: SharedAudioFacts) -> dict[str, float]:
    """Return measured roles from the direct/body view, if present."""
    evidence = facts.evidence if isinstance(facts.evidence, dict) else {}
    direct_view = evidence.get("direct_body_view", {}) if isinstance(evidence, dict) else {}
    roles = direct_view.get("measured_roles", {}) if isinstance(direct_view, dict) else {}
    return roles if isinstance(roles, dict) else {}


def _role_value(roles: dict[str, float], role: str) -> float:
    """Read a role score from a role dictionary."""
    try:
        return float(roles.get(role, 0.0) or 0.0)
    except Exception:
        return 0.0


def _shape_name_and_confidence(facts: SharedAudioFacts) -> tuple[str, float]:
    """Return the primary shape vote and confidence."""
    evidence = facts.evidence if isinstance(facts.evidence, dict) else {}
    shape = evidence.get("shape_vote", {}) if isinstance(evidence, dict) else {}
    if not isinstance(shape, dict):
        return "", 0.0
    try:
        confidence = float(shape.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    return str(shape.get("primary_shape", "") or ""), confidence


def _has_near_human_voice_candidate(shared: list[dict[str, Any]], winner_score: float) -> bool:
    """Return True only for a nearby Human/Voice candidate with role support."""
    for row in shared:
        if not _is_human_voice_candidate_row(row):
            continue
        try:
            score = float(row.get("combined_rank_score", 9999.0) or 9999.0)
        except Exception:
            score = 9999.0
        if score > winner_score + 8.0:
            continue
        signature = role_signature_from_row(row)
        role_fit = max(
            role_strength(signature, "vocal_music_phrase"),
            role_strength(signature, "vocal_phrase"),
            role_strength(signature, "vocal_one_shot"),
            role_strength(signature, "voiced_one_shot"),
        )
        if role_fit >= 0.55:
            return True
    return False


def _should_promote_direct_body_bass_loop(facts: SharedAudioFacts, parent_role: str) -> bool:
    """Return True when the direct/body view proves a sub-heavy bass loop."""
    if parent_role == "bass_loop":
        return False
    shape, shape_confidence = _shape_name_and_confidence(facts)
    if shape != "bass_phrase" or shape_confidence < 0.86:
        return False
    direct_roles = _direct_body_roles(facts)
    if _role_value(direct_roles, "bass_loop") < 0.88:
        return False
    sub_ratio = _fact_number(facts, "sub_bass_ratio_lt_150hz")
    high_ratio = _fact_number(facts, "presence_ratio_2000_8000hz") + _fact_number(facts, "air_ratio_gt_8000hz")
    percussive_ratio = _fact_number(facts, "loop_percussive_event_ratio")
    drumlike_ratio = _fact_number(facts, "loop_drumlike_frame_ratio")
    return sub_ratio >= 0.70 and high_ratio <= 0.08 and percussive_ratio <= 0.12 and drumlike_ratio <= 0.18


def _drum_loop_evidence_strength(facts: SharedAudioFacts) -> float:
    """Return measured drum-loop strength across full and direct/body facts."""
    roles = facts.evidence.get("measured_roles", {}) if isinstance(facts.evidence, dict) else {}
    direct_roles = _direct_body_roles(facts)
    if not isinstance(roles, dict):
        roles = {}
    return max(
        _role_value(roles, "low_rhythmic_drum_loop"),
        _role_value(roles, "percussive_drum_loop"),
        _role_value(roles, "bright_drum_loop"),
        _role_value(direct_roles, "low_rhythmic_drum_loop"),
        _role_value(direct_roles, "percussive_drum_loop"),
        _role_value(direct_roles, "bright_drum_loop"),
    )


def _best_broad_instrument_loop_row(shared: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the best broad Instrument Loops shared row, if present."""
    rows = [
        row
        for row in shared
        if str(row.get("top_family", "")) == "Instruments"
        and "instrument loops" in str(row.get("folder_path") or row.get("label") or "").lower()
    ]
    if not rows:
        return None
    rows.sort(
        key=lambda row: (
            float(row.get("combined_rank_score", 9999.0)),
            int(row.get("physics_rank", 9999)),
            int(row.get("brain_rank", 9999)),
            str(row.get("label", "")),
        )
    )
    return rows[0]


def role_signature_from_row(row: dict[str, Any]) -> dict[str, float]:
    """Return a candidate role signature from a shared-candidate row."""
    signature = row.get("candidate_role_signature", {})
    if not isinstance(signature, dict):
        return {}
    return {str(key): role_strength(signature, str(key)) for key in signature}


def label_depth(row: dict[str, Any]) -> int:
    """Return folder depth for one shared-candidate row."""
    return len([part for part in str(row.get("label", "")).split("/") if part])


def first_finite(*values: Any) -> float | None:
    """Return the first finite float, or None."""
    for value in values:
        try:
            number = float(value)
        except Exception:
            continue
        if number == number and number not in (float("inf"), float("-inf")):
            return number
    return None
