# SOURCE-NAME BLINDNESS INVARIANT:
# Decision helper code must never use producer filenames, source folder names,
# ZIP member names, path tokens, or sample-pack labels as classification evidence.
"""Small measured-evidence helpers for DecisionCoreV2 and claim producers.

These helpers are intentionally pure and boring.  Keeping them outside the
giant decision core lets policy classes reuse the same measured-role and
shape parsing without importing DecisionCoreV2 or creating circular coupling.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts

CONCRETE_FX_PATH_FRAGMENTS = (
    "siren",
    "alarm",
    "glitch",
    "stutter",
    "whoosh",
    "swoosh",
    "swish",
    "sweep",
    "riser",
    "build",
    "drop",
    "downlifter",
    "transition",
    "impact",
    "boom",
    "reverse",
    "motor",
    "machine",
    "radio",
    "electrical",
    "glass",
    "metal",
    "crash",
)


LOOP_PHRASE_ROLES = {
    "bass_loop",
    "pitched_music_loop",
    "pitched_music_phrase",
    "pitched_reed_or_instrument_loop",
    "pitched_reed_or_instrument_phrase",
    "mixed_music_loop",
    "clean_sustained_tonal_instrument_loop",
}

DRUM_LOOP_STRUCTURE_ROLES = {
    "drum_loop",
    "percussive_drum_loop",
    "bright_drum_loop",
    "low_rhythmic_drum_loop",
}

DRUM_RESCUE_BLOCKING_PHRASE_SHAPES = {"vocal_phrase", "pitched_phrase", "pitched_phrase_shape", "sustained_pad"}


def _is_measured_loop_phrase_context(shape: str, role: str, measured_role: str) -> bool:
    """Return True when a phrase-like shape is known to describe a loop.

    The short-hit guard exists to stop one-shots from being promoted to loops,
    but ``bass_phrase`` is also a valid loop/phrase shape.  Treat it as a
    short-hit warning only when the measured parent role is not already a loop.
    """
    return shape == "bass_phrase" and (role in LOOP_PHRASE_ROLES or measured_role in LOOP_PHRASE_ROLES)


def _shape_blocks_percussion_loop_rescue(shape: str, shape_confidence: float) -> bool:
    """Return True when phrase shape should block a cross-family drum-loop rescue."""
    return shape in DRUM_RESCUE_BLOCKING_PHRASE_SHAPES and shape_confidence >= 0.70


def _has_drum_loop_structure_support(role: str, measured_role: str, shape: str, shape_confidence: float) -> bool:
    """Return True only when measured structure really supports Drum Loops.

    ``pitched_percussion_loop`` is a weak bridge role.  It can be valid for real
    tonal percussion loops, but it must not override strong vocal, sax/reed, or
    sustained-pad phrase shapes by itself.
    """
    if shape in {"beat_loop", "top_loop", "drum_loop"} and shape_confidence >= 0.70:
        return True
    if role in DRUM_LOOP_STRUCTURE_ROLES or measured_role in DRUM_LOOP_STRUCTURE_ROLES:
        return True
    has_pitched_percussion_role = role == "pitched_percussion_loop" or measured_role == "pitched_percussion_loop"
    if has_pitched_percussion_role:
        return not _shape_blocks_percussion_loop_rescue(shape, shape_confidence)
    return False


def _stable_instrument_claim_strength(
    role: str,
    measured_role: str,
    shape: str,
    shape_confidence: float,
    facts: SharedAudioFacts | None,
) -> float:
    """Return strength of a clean pitched-instrument claim against Human/Voice.

    Voice, sax, piano, synth, and keys can all look formant-like in raw spectral
    facts.  This score is intentionally broad: it only says that the sound has
    stable pitched instrument evidence, so a weak Human/Voice path should not
    steal it.  Actual sax/keys/bass depth is still chosen elsewhere.
    """
    full_pitched = max(
        _role_strength_from_facts(facts, "pitched_music_loop"),
        _role_strength_from_facts(facts, "pitched_music_phrase"),
    )
    direct_pitched = max(
        _direct_body_role_strength_from_facts(facts, "pitched_music_loop"),
        _direct_body_role_strength_from_facts(facts, "pitched_music_phrase"),
    )
    reed_or_music_role = role in LOOP_PHRASE_ROLES or measured_role in LOOP_PHRASE_ROLES
    phrase_shape = shape in {"pitched_phrase", "sustained_pad", "bass_phrase"} and shape_confidence >= 0.70
    score = max(full_pitched, direct_pitched)
    if reed_or_music_role:
        score = max(score, 0.78)
    if phrase_shape:
        score = max(score, 0.72)
    return min(1.0, max(0.0, score))


def _norm_path(path: str) -> str:
    return str(path or "").replace("\\", "/").lower()


def _path_has_any(path: str, fragments: tuple[str, ...]) -> bool:
    low = _norm_path(path)
    return any(fragment.lower() in low for fragment in fragments)


def _candidate_combined_score(candidate: dict) -> float:
    """Read a candidate combined-rank score safely."""
    try:
        return float(candidate.get("combined_rank_score", 9999.0) or 9999.0)
    except Exception:
        return 9999.0


def _is_concrete_fx_path(path: str) -> bool:
    """Return True for specific FX candidate paths, not source filenames."""
    low = _norm_path(path)
    if not low.startswith("fx/"):
        return False
    return _path_has_any(low, CONCRETE_FX_PATH_FRAGMENTS)


def _feature_number_from_facts(facts: SharedAudioFacts | None, name: str, default: float = 0.0) -> float:
    """Return a named feature value from shared facts or nested evidence."""
    if facts is None:
        return default
    try:
        if name in facts.feature_values_by_name:
            return float(facts.feature_values_by_name.get(name, default) or default)
    except Exception:
        pass
    if not isinstance(facts.evidence, dict):
        return default
    try:
        if name in facts.evidence:
            return float(facts.evidence.get(name, default) or default)
    except Exception:
        pass
    feature_values = facts.evidence.get("feature_values_by_name")
    if isinstance(feature_values, dict) and name in feature_values:
        try:
            return float(feature_values.get(name, default) or default)
        except Exception:
            return default
    measured_roles = facts.evidence.get("measured_roles")
    if isinstance(measured_roles, dict):
        role_evidence = measured_roles.get("evidence")
        if isinstance(role_evidence, dict):
            try:
                return float(role_evidence.get(name, default) or default)
            except Exception:
                return default
    return default


def _voice_identity_from_facts(facts: SharedAudioFacts | None) -> float:
    """Return formant/light-voice identity evidence from measured role payloads."""
    if facts is None or not isinstance(facts.evidence, dict):
        return 0.0
    best = 0.0
    for container_name in ("measured_roles",):
        roles = facts.evidence.get(container_name)
        if isinstance(roles, dict) and isinstance(roles.get("evidence"), dict):
            best = max(best, _safe_float(roles["evidence"].get("formant_light_voice_identity")))
    direct_view = facts.evidence.get("direct_body_view")
    if isinstance(direct_view, dict):
        roles = direct_view.get("measured_roles")
        if isinstance(roles, dict) and isinstance(roles.get("evidence"), dict):
            best = max(best, _safe_float(roles["evidence"].get("formant_light_voice_identity")))
    return best


def _role_strength_from_facts(facts: SharedAudioFacts | None, role_name: str) -> float:
    """Return one measured role score from full-file shared facts."""
    if facts is None or not isinstance(facts.evidence, dict):
        return 0.0
    roles = facts.evidence.get("measured_roles")
    if not isinstance(roles, dict):
        return 0.0
    return _safe_float(roles.get(role_name))


def _direct_body_role_strength_from_facts(facts: SharedAudioFacts | None, role_name: str) -> float:
    """Return one measured role score from the tail-reduced direct/body view."""
    if facts is None or not isinstance(facts.evidence, dict):
        return 0.0
    view = facts.evidence.get("direct_body_view")
    if not isinstance(view, dict) or not view.get("available"):
        return 0.0
    roles = view.get("measured_roles")
    if not isinstance(roles, dict):
        return 0.0
    return _safe_float(roles.get(role_name))


def _measured_role_from_facts(facts: SharedAudioFacts | None) -> str:
    """Return the measured parent role stored in shared evidence, if present."""
    if facts is None or not isinstance(facts.evidence, dict):
        return ""
    roles = facts.evidence.get("measured_roles")
    if isinstance(roles, dict):
        for key in ("detected_parent_role", "parent_role", "primary_role", "role"):
            value = roles.get(key)
            if value:
                return str(value)
        # Some test/diagnostic payloads store role scores directly.
        best_name = ""
        best_value = 0.0
        for key, value in roles.items():
            if key == "evidence" or isinstance(value, dict):
                continue
            number = _safe_float(value)
            if number > best_value:
                best_name = str(key)
                best_value = number
        if best_value > 0.0:
            return best_name
    for key in ("detected_parent_role", "parent_role", "primary_role"):
        value = facts.evidence.get(key)
        if value:
            return str(value)
    return ""


def _shape_metric_from_facts(facts: SharedAudioFacts | None, metric_name: str) -> float:
    """Return a numeric metric from the shared ShapeVoter evidence."""
    if facts is None or not isinstance(facts.evidence, dict):
        return 0.0
    for container_name in ("shape_vote", "shape_vote_json"):
        container = facts.evidence.get(container_name)
        if isinstance(container, dict):
            value = _safe_float(container.get(metric_name))
            if value:
                return value
    result = facts.evidence.get("shape_vote_result")
    if isinstance(result, dict):
        guesses = result.get("guesses")
        if isinstance(guesses, list) and guesses and isinstance(guesses[0], dict):
            return _safe_float(guesses[0].get(metric_name))
    return 0.0


def _shape_vote_from_facts(facts: SharedAudioFacts | None) -> str:
    if facts is None or not isinstance(facts.evidence, dict):
        return ""
    shape_vote = facts.evidence.get("shape_vote")
    if isinstance(shape_vote, str):
        return shape_vote
    if isinstance(shape_vote, dict):
        value = (
            shape_vote.get("primary_shape")
            or shape_vote.get("shape")
            or shape_vote.get("shape_vote")
            or shape_vote.get("label")
        )
        if value:
            return str(value)
    result = facts.evidence.get("shape_vote_result")
    if isinstance(result, dict):
        guesses = result.get("guesses")
        if isinstance(guesses, list) and guesses and isinstance(guesses[0], dict):
            first_guess = guesses[0]
            value = (
                first_guess.get("primary_shape")
                or first_guess.get("shape")
                or first_guess.get("shape_vote")
                or first_guess.get("label")
            )
            if value:
                return str(value)
    for key in ("detected_shape", "shape_label"):
        value = facts.evidence.get(key)
        if value:
            return str(value)
    shape_json = facts.evidence.get("shape_vote_json")
    if isinstance(shape_json, dict):
        value = (
            shape_json.get("primary_shape")
            or shape_json.get("shape")
            or shape_json.get("shape_vote")
            or shape_json.get("label")
        )
        if value:
            return str(value)
    return ""


def _shape_confidence_from_facts(facts: SharedAudioFacts | None) -> float:
    if facts is None or not isinstance(facts.evidence, dict):
        return 0.0
    for key in ("shape_confidence", "shape_score", "shape_vote_confidence"):
        try:
            value = facts.evidence.get(key)
            if value is not None:
                return float(value)
        except Exception:
            pass
    shape_vote = facts.evidence.get("shape_vote")
    if isinstance(shape_vote, dict):
        for key in ("confidence", "shape_confidence", "score"):
            try:
                value = shape_vote.get(key)
                if value is not None:
                    return float(value)
            except Exception:
                pass
    shape_json = facts.evidence.get("shape_vote_json")
    if isinstance(shape_json, dict):
        for key in ("confidence", "shape_confidence", "score"):
            try:
                value = shape_json.get(key)
                if value is not None:
                    return float(value)
            except Exception:
                pass
    return 0.0


def _candidate_role_strength(candidate: dict[str, object], role_name: str) -> float:
    """Extract a candidate role strength from consensus shared-candidate evidence."""
    for source_key in ("candidate_role_signature", "brain_evidence", "physics_evidence"):
        value = candidate.get(source_key)
        if isinstance(value, dict):
            if role_name in value:
                return _safe_float(value.get(role_name))
            nested = value.get("candidate_role_signature")
            if isinstance(nested, dict) and role_name in nested:
                return _safe_float(nested.get(role_name))
    return 0.0


def _safe_float(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except Exception:
        return 0.0
    if number != number:
        return 0.0
    return number


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except Exception:
        return int(default)


# v31.90_repeated_tonal_hit_broad_instrument_loop_claim
