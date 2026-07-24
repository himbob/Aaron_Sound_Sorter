# SOURCE-NAME BLINDNESS INVARIANT:
# This module inspects measured audio facts, internal voter evidence, and
# human-trained memory only. It must never inspect producer filenames, source
# folders, ZIP member names, or sample-pack labels as classification evidence.
"""Shared measured contract for human-voice source ownership.

Voice, sax/reeds, synth/formant FX, strings, and mixed melodic loops can all
light up formant-like feature panels.  A final Voice claim therefore needs
actual source ownership, not just a rank-one folder-brain guess.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.engine.decision_helpers import (
    _direct_body_role_strength_from_facts,
    _feature_number_from_facts,
    _shape_confidence_from_facts,
    _shape_vote_from_facts,
)
from aaron_sound_sorter.engine.learned_memory_contracts import has_voice_memory_match

VOCAL_SOURCE_ROLES = frozenset(
    {
        "vocal_music_phrase",
        "vocal_phrase",
        "vocal_one_shot",
        "voiced_one_shot",
    }
)
EXPLICIT_VOCAL_SHAPES = frozenset({"vocal_phrase", "vocal_one_shot"})


def processed_voice_source_owned(facts: SharedAudioFacts | None) -> bool:
    """Return whether measured/memory evidence owns a real Voice source.

    Args:
        facts: Shared source-blind audio facts for the current sample.

    Returns:
        True when learned voice memory, measured vocal roles, explicit vocal
        shapes, or the physics Voice branch claims the source.

    Side Effects:
        None.

    Important Constraints:
        Rank-one folder-brain Voice is deliberately not enough here. Folder
        brains provide candidate evidence; they do not establish source
        ownership for broad pitched/formant loops without memory or measured
        vocal role support.
    """
    if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
        return False
    if has_voice_memory_match(facts):
        return True
    if measured_vocal_role_strength(facts) >= 0.62:
        return True
    if detected_parent_role(facts) in VOCAL_SOURCE_ROLES:
        return True
    if physics_voice_branch_owns_source(facts):
        return True
    return explicit_vocal_shape_owns_source(facts)


def measured_vocal_role_strength(facts: SharedAudioFacts | None) -> float:
    """Return the strongest measured vocal role score."""
    return max(role_strength_from_evidence(facts, role_name) for role_name in VOCAL_SOURCE_ROLES)


def detected_parent_role(facts: SharedAudioFacts | None) -> str:
    """Return the measured parent role name when present."""
    if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
        return ""
    roles = facts.evidence.get("measured_roles")
    if isinstance(roles, Mapping):
        for key in ("detected_parent_role", "parent_role", "primary_role", "role"):
            value = roles.get(key)
            if value:
                return str(value)
    return ""


def physics_voice_branch_owns_source(facts: SharedAudioFacts | None) -> bool:
    """Return whether the physics layer explicitly owns the Voice branch."""
    layer = physics_layer_mapping(facts)
    if not layer:
        return False
    branch = str(layer.get("instrument_branch_selected") or layer.get("physics_layer_branch") or "")
    branch_confidence = safe_float(
        layer.get("instrument_branch_selected_confidence", layer.get("physics_layer_branch_confidence")),
        0.0,
    )
    rap_voice_texture = safe_float(layer.get("instrument_rap_voice_texture"), 0.0)
    human_voice_texture = safe_float(layer.get("instrument_human_voice_texture"), 0.0)
    source_owner = bool(layer.get("instrument_voice_source_owner_confirmed"))
    return bool(
        branch == "Voice"
        and branch_confidence >= 0.56
        and (source_owner or rap_voice_texture >= 0.58 or human_voice_texture >= 0.66)
    )


def explicit_vocal_shape_owns_source(facts: SharedAudioFacts | None) -> bool:
    """Return whether ShapeVoter explicitly detected vocal material."""
    return bool(
        _shape_vote_from_facts(facts) in EXPLICIT_VOCAL_SHAPES
        and _shape_confidence_from_facts(facts) >= 0.74
        and max(
            score_from_facts(facts, "voice_score"),
            score_from_facts(facts, "human_spoken_voice_score"),
            score_from_facts(facts, "human_breath_mouth_score"),
        )
        >= 0.74
    )


def role_strength_from_evidence(facts: SharedAudioFacts | None, role_name: str) -> float:
    """Return one role score from full-file or direct-body measured roles."""
    if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
        return 0.0
    roles = facts.evidence.get("measured_roles")
    full_file_strength = safe_float(roles.get(role_name) if isinstance(roles, Mapping) else 0.0, 0.0)
    direct_strength = _direct_body_role_strength_from_facts(facts, role_name)
    return max(full_file_strength, direct_strength)


def score_from_facts(facts: SharedAudioFacts | None, score_name: str) -> float:
    """Return a measured score from feature values or flat physics subpanels."""
    return max(
        _feature_number_from_facts(facts, score_name),
        flat_subpanel_score(facts, score_name),
    )


def flat_subpanel_score(facts: SharedAudioFacts | None, score_name: str) -> float:
    """Return one score from ``physics_subpanels.flat``."""
    if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
        return 0.0
    subpanels = facts.evidence.get("physics_subpanels")
    if not isinstance(subpanels, Mapping):
        return 0.0
    flat = subpanels.get("flat")
    if not isinstance(flat, Mapping):
        return 0.0
    return safe_float(flat.get(score_name), 0.0)


def physics_layer_mapping(facts: SharedAudioFacts | None) -> Mapping[str, Any]:
    """Return the physics layer decision mapping."""
    if facts is None or not isinstance(getattr(facts, "evidence", None), dict):
        return {}
    layer = facts.evidence.get("physics_layer_decision")
    return layer if isinstance(layer, Mapping) else {}


def safe_float(value: object, default: float = 0.0) -> float:
    """Return a finite float-like value with a fallback."""
    try:
        number = float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)
    if number != number:
        return float(default)
    return number
