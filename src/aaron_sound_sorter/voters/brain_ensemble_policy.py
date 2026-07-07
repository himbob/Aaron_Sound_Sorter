"""Lane weighting policy for the multi-brain recall ensemble."""

from __future__ import annotations

from typing import Any

from aaron_sound_sorter.domain.models import CategoryGuess


def choose_representative_brain_guess(
    label: str,
    present_lanes: list[str],
    lane_guesses: dict[str, dict[str, CategoryGuess]],
) -> CategoryGuess | None:
    """Choose the lane guess whose metadata represents the ensemble label."""
    preferred_order = [
        "core_baby",
        "spread_baby",
        "full",
        "harmonic_core_baby",
        "harmonic_spread_baby",
        "outlier_baby",
        "harmonic_outlier_baby",
    ]
    for lane in preferred_order:
        if lane in present_lanes and label in lane_guesses.get(lane, {}):
            return lane_guesses[lane][label]
    for lane in present_lanes:
        guess = lane_guesses.get(lane, {}).get(label)
        if guess is not None:
            return guess
    return None


def brain_lane_vote_weight(lane: str, *, weight_profile: str = "generic", label: str = "") -> float:
    """Return product-vote weight for a brain lane under a role profile."""
    profile_weights: dict[str, dict[str, float]] = {
        "drum_loop_full_priority": {
            "full": 1.72,
            "core_baby": 0.26,
            "spread_baby": 0.26,
            "baby": 0.26,
            "outlier_baby": 0.10,
            "harmonic_core_baby": 0.18,
            "harmonic_spread_baby": 0.16,
            "harmonic_outlier_baby": 0.06,
        },
        "bass_loop_baby_priority": {
            "full": 0.86,
            "core_baby": 1.38,
            "spread_baby": 1.38,
            "baby": 1.10,
            "outlier_baby": 0.55,
            "harmonic_core_baby": 0.38,
            "harmonic_spread_baby": 0.34,
            "harmonic_outlier_baby": 0.16,
        },
        "vocal_core_spread_priority": {
            "full": 0.78,
            "core_baby": 1.30,
            "spread_baby": 1.30,
            "baby": 1.10,
            "outlier_baby": 0.22,
            "harmonic_core_baby": 0.40,
            "harmonic_spread_baby": 0.36,
            "harmonic_outlier_baby": 0.12,
        },
        "generic_pitched_full_guarded": {
            "full": 1.10,
            "core_baby": 0.82,
            "spread_baby": 0.88,
            "baby": 0.80,
            "outlier_baby": 0.22,
            "harmonic_core_baby": 0.32,
            "harmonic_spread_baby": 0.28,
            "harmonic_outlier_baby": 0.10,
        },
        "generic": {
            "core_baby": 1.00,
            "spread_baby": 0.96,
            "baby": 0.90,
            "full": 1.00,
            "outlier_baby": 0.30,
            "harmonic_core_baby": 0.42,
            "harmonic_spread_baby": 0.36,
            "harmonic_outlier_baby": 0.14,
        },
    }
    weights = profile_weights.get(weight_profile, profile_weights["generic"])
    base = float(weights.get(lane, 0.30))
    if weight_profile == "generic_pitched_full_guarded" and is_reed_or_brass_label(label):
        if lane == "outlier_baby":
            return 0.58
        if lane in {"core_baby", "spread_baby"}:
            return max(base, 0.90)
        if lane == "full":
            return max(base, 1.10)
    return base


def rank_support(rank: float) -> float:
    """Convert a lane rank to a bounded support score."""
    try:
        rank = max(1.0, float(rank))
    except Exception:
        rank = 9999.0
    return 1.0 / (rank**0.72)


def brain_role_fit_multiplier(
    guess: CategoryGuess | None,
    *,
    weight_profile: str = "generic",
    label: str = "",
) -> float:
    """Use measured-role diagnostics to damp unsupported cross-family guesses."""
    if guess is None or not isinstance(guess.evidence, dict):
        return 1.0
    if bool(guess.evidence.get("human_override_exact_audio_match")):
        return 1.85
    if bool(guess.evidence.get("human_override_generalized_audio_match")):
        return 1.35
    family = guess.evidence.get("family_compatibility", {})
    compatible = True
    if isinstance(family, dict):
        compatible = bool(family.get("is_family_compatible", True))
    try:
        parent_strength = float(guess.evidence.get("measured_parent_role_strength", 0.0) or 0.0)
    except Exception:
        parent_strength = 0.0
    try:
        candidate_strength = float(guess.evidence.get("candidate_parent_role_strength", 0.0) or 0.0)
    except Exception:
        candidate_strength = 0.0
    label_norm = normalize_label_text(label or guess.label or guess.folder_path)

    if weight_profile == "drum_loop_full_priority":
        if "drum loop" in label_norm or "drum loops" in label_norm:
            return 1.28
        if compatible and guess.top_family == "Drums":
            return 1.08
        return 0.42

    if weight_profile == "bass_loop_baby_priority":
        if "bass" in label_norm or "808" in label_norm:
            return 1.24
        if compatible and parent_strength >= 0.70:
            return 0.92
        return 0.45

    if weight_profile == "vocal_core_spread_priority":
        if "human and voice" in label_norm or "vocal" in label_norm or "voice" in label_norm:
            return 1.24
        if guess.top_family == "FX" and compatible:
            return 0.90
        return 0.50

    if weight_profile == "generic_pitched_full_guarded":
        is_instrument_label = label_norm.startswith("instruments/")
        if is_reed_or_brass_label(label_norm):
            if compatible and parent_strength >= 0.70:
                return 0.92
            return 0.60
        if is_instrument_label and (
            "synth" in label_norm or "keys" in label_norm or "rhodes" in label_norm or "piano" in label_norm
        ):
            return 1.16
        if is_instrument_label and (
            "instrument loops" in label_norm or "guitar" in label_norm or "strings" in label_norm
        ):
            return 1.05

    if not compatible and parent_strength >= 0.70:
        return 0.32
    if compatible and parent_strength >= 0.70 and candidate_strength >= 0.55:
        return 1.18
    if compatible:
        return 1.04
    return 0.82


def brain_ensemble_weight_profile(*, facts: Any | None, representative: CategoryGuess | None) -> tuple[str, str]:
    """Choose the lane weighting profile from measured audio evidence."""
    evidence: dict[str, Any] = {}
    if representative is not None and isinstance(representative.evidence, dict):
        evidence.update(representative.evidence)
    facts_evidence = getattr(facts, "evidence", {}) if facts is not None else {}
    measured_roles: dict[str, Any] = {}
    if isinstance(facts_evidence, dict) and isinstance(facts_evidence.get("measured_roles"), dict):
        measured_roles = dict(facts_evidence.get("measured_roles") or {})
    elif isinstance(evidence.get("measured_role_strengths"), dict):
        measured_roles = dict(evidence.get("measured_role_strengths") or {})

    detected_role = str(evidence.get("detected_parent_role") or "")
    if isinstance(evidence.get("family_compatibility"), dict):
        detected_role = str(evidence["family_compatibility"].get("detected_parent_role") or detected_role)
    primary_roles = measured_roles.get("primary_roles", []) if isinstance(measured_roles, dict) else []
    primary_role_text = (
        " ".join(str(item) for item in primary_roles) if isinstance(primary_roles, list) else str(primary_roles)
    )

    direct_roles: dict[str, Any] = {}
    if isinstance(facts_evidence, dict):
        direct_view = facts_evidence.get("direct_body_view")
        if isinstance(direct_view, dict) and isinstance(direct_view.get("measured_roles"), dict):
            direct_roles = dict(direct_view.get("measured_roles") or {})
    drum_strength = max(
        safe_float(measured_roles.get("low_rhythmic_drum_loop")),
        safe_float(measured_roles.get("percussive_drum_loop")),
        safe_float(measured_roles.get("bright_drum_loop")),
        safe_float(direct_roles.get("low_rhythmic_drum_loop")),
        safe_float(direct_roles.get("percussive_drum_loop")),
        safe_float(direct_roles.get("bright_drum_loop")),
    )
    bass_strength = max(safe_float(measured_roles.get("bass_loop")), safe_float(direct_roles.get("bass_loop")))
    shape_vote = facts_evidence.get("shape_vote", {}) if isinstance(facts_evidence, dict) else {}
    primary_shape = str(shape_vote.get("primary_shape", "") or "") if isinstance(shape_vote, dict) else ""
    shape_confidence = safe_float(shape_vote.get("confidence", 0.0)) if isinstance(shape_vote, dict) else 0.0
    eligibility = facts_evidence.get("parent_eligibility_v2", {}) if isinstance(facts_evidence, dict) else {}
    eligibility_role = str(eligibility.get("role_name", "") or "") if isinstance(eligibility, dict) else ""
    eligibility_confidence = safe_float(eligibility.get("confidence", 0.0)) if isinstance(eligibility, dict) else 0.0
    vocal_strength = max(
        safe_float(measured_roles.get("vocal_music_phrase")),
        safe_float(measured_roles.get("vocal_phrase")),
        safe_float(measured_roles.get("vocal_one_shot")),
        safe_float(measured_roles.get("voiced_one_shot")),
        safe_float(direct_roles.get("vocal_music_phrase")),
        safe_float(direct_roles.get("vocal_phrase")),
        safe_float(direct_roles.get("vocal_one_shot")),
        safe_float(direct_roles.get("voiced_one_shot")),
    )
    pitched_strength = max(
        safe_float(measured_roles.get("pitched_music_loop")),
        safe_float(measured_roles.get("pitched_music_phrase")),
    )

    if "drum_loop" in detected_role or drum_strength >= 0.50 or "drum_loop" in primary_role_text:
        return (
            "drum_loop_full_priority",
            "measured drum-loop role: full brain is weighted highest based on run evidence",
        )
    if "bass_loop" in detected_role or bass_strength >= 0.72 or "bass_loop" in primary_role_text:
        return "bass_loop_baby_priority", "measured bass-loop role: core/spread baby lanes are weighted highest"
    if (
        "vocal" in detected_role
        or vocal_strength >= 0.58
        or "vocal" in primary_role_text
        or ("vocal" in eligibility_role and eligibility_confidence >= 0.70)
        or (
            primary_shape in {"pitched_phrase_shape", "vocal_phrase"}
            and shape_confidence >= 0.82
            and vocal_strength >= 0.35
        )
    ):
        return (
            "vocal_core_spread_priority",
            "measured/direct vocal role: core/spread baby lanes can rescue Human/Voice candidates",
        )
    if pitched_strength >= 0.72 or "pitched" in detected_role or "pitched" in primary_role_text:
        return (
            "generic_pitched_full_guarded",
            "generic pitched role: full brain is guarded and outlier reed recall is limited",
        )
    return "generic", "no dominant measured role: balanced conservative lane weights"


def safe_float(value: Any, default: float = 0.0) -> float:
    """Return a finite float-like value without raising."""
    try:
        result = float(value)
    except Exception:
        return default
    if result != result:
        return default
    return result


def normalize_label_text(value: Any) -> str:
    """Normalize an internal label/folder string for category-fragment checks."""
    return str(value or "").replace("\\", "/").lower()


def is_reed_or_brass_label(label: Any) -> bool:
    """Return whether an internal label/folder belongs to brass/woodwind identity."""
    text = normalize_label_text(label)
    return any(
        fragment in text
        for fragment in (
            "sax",
            "saxophone",
            "woodwind",
            "woodwinds",
            "brass",
            "trumpet",
            "horn",
            "reed",
            "flute",
            "clarinet",
            "trombone",
        )
    )
