"""Reusable scoring tools for voters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from aaron_sound_sorter.core import FEATURE_NAMES, FP_SIZE
from aaron_sound_sorter.domain.facts import fingerprint_array
from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.domain.policies import PhysicsVoterPolicy
from aaron_sound_sorter.domain.roles import measured_roles_from_features
from aaron_sound_sorter.engine.brain_runtime_cache import get_brain_runtime_cache
from aaron_sound_sorter.taxonomy_contracts import taxonomy_label_contract
from aaron_sound_sorter.training_labels import label_default_structure, public_label, top_for_public_label

PROFILE_FEATURES = tuple(str(name) for name in FEATURE_NAMES[:FP_SIZE])


def label_folder_path(label: str) -> str:
    """Return a public folder path for a label."""
    return public_label(label)


def label_top_family(brain: dict[str, Any], label: str) -> str:
    """Return the learned top family for a label."""
    top_by_label = brain.get("top_by_label", {}) if isinstance(brain.get("top_by_label", {}), dict) else {}
    return str(top_by_label.get(label, "") or top_for_public_label(label) or "_TO_REVIEW")


def label_structure(brain: dict[str, Any], label: str) -> str:
    """Return learned/default structure for a label."""
    structure_by_label = (
        brain.get("structure_by_label", {}) if isinstance(brain.get("structure_by_label", {}), dict) else {}
    )
    return str(structure_by_label.get(label, "") or label_default_structure(label) or "one_shot")


def label_parts(label: str) -> list[str]:
    """Return normalized folder path parts for a learned label."""
    return [part.strip() for part in public_label(label).replace("\\", "/").split("/") if part.strip()]


def label_maps(brain: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Return folder and top maps for every known label."""
    runtime = get_brain_runtime_cache(brain)
    return runtime.folder_map, runtime.top_map


def canonical_candidate_labels(labels: list[str]) -> list[str]:
    """Return valid canonical labels from a brain candidate list.

    Args:
        labels: Raw labels from an active brain JSON.

    Returns:
        Candidate labels whose internal taxonomy grammar is valid and already
        canonical.

    Side Effects:
        None.

    Important Constraints:
        This inspects only internal trained labels. It never reads source audio
        filenames or sample-pack paths.
    """
    candidates: list[str] = []
    seen: set[str] = set()
    for label in labels:
        contract = taxonomy_label_contract(label)
        if not contract.valid:
            continue
        if contract.canonical_label != contract.normalized_label:
            continue
        if contract.canonical_label in seen:
            continue
        seen.add(contract.canonical_label)
        candidates.append(contract.canonical_label)
    return candidates


def feature_weights(brain: dict[str, Any]) -> np.ndarray:
    """Return finite feature weights."""
    return get_brain_runtime_cache(brain).feature_weights


def scaler_for_label(brain: dict[str, Any], label: str) -> tuple[np.ndarray, np.ndarray]:
    """Return mean/std scaler for a label's structure lane."""
    return get_brain_runtime_cache(brain).scaler_for_label(label)


def pad_vector(values: np.ndarray, fill: float) -> np.ndarray:
    """Return a finite vector with FP_SIZE values."""
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.size < FP_SIZE:
        vector = np.pad(vector, (0, FP_SIZE - vector.size), mode="constant", constant_values=fill)
    return np.nan_to_num(vector[:FP_SIZE], nan=fill, posinf=fill, neginf=fill).astype(np.float32, copy=False)


def centroid_distance(brain: dict[str, Any], label: str, weighted_vector: np.ndarray, weights: np.ndarray) -> float:
    """Return nearest centroid distance for a label."""
    centroids = brain.get("centroids", {}) if isinstance(brain.get("centroids", {}), dict) else {}
    label_centroids = np.asarray(centroids.get(label, []), dtype=np.float32)
    if label_centroids.ndim == 1 and label_centroids.size:
        label_centroids = label_centroids.reshape(1, -1)
    if label_centroids.ndim != 2 or not label_centroids.size:
        return float("inf")
    usable = min(label_centroids.shape[1], weighted_vector.size)
    centroid_weighted = label_centroids[:, :usable] * weights[:usable][None, :]
    return float(np.min(np.linalg.norm(centroid_weighted - weighted_vector[:usable][None, :], axis=1)))


def profile_residual_score(
    fingerprint: Sequence[float],
    profile: dict[str, Any],
    policy: PhysicsVoterPolicy,
) -> tuple[float, dict[str, Any]]:
    """Return sample-count-neutral full-feature bucket membership score.

    Each label/folder gets one equal-status physics bucket.  The score is
    based only on how the new sound's named fingerprint values fit that
    bucket's stored feature statistics.  Training sample count, reliability,
    and valid_count are reported for diagnostics but do not boost or punish
    a bucket.  This deliberately prevents large training folders from having
    more physics-voter authority than small folders.
    """
    stats = profile.get("feature_stats", {}) if isinstance(profile, dict) else {}
    if not isinstance(stats, dict) or not stats:
        return float("inf"), {"used_feature_count": 0, "severe_count": 0}

    values = fingerprint_array(np.asarray(fingerprint, dtype=np.float32))

    reliability = profile_reliability(profile)
    scale_multiplier = reliability.get("scale_multiplier", 1.0)
    range_penalty_multiplier = reliability.get("range_penalty_multiplier", 1.0)
    severe_penalty_multiplier = reliability.get("severe_penalty_multiplier", 1.0)

    residuals: list[float] = []
    range_violations = 0
    severe_count = 0
    inside_range_count = 0
    missing_feature_count = 0

    for index, name in enumerate(PROFILE_FEATURES):
        stat = stats.get(name)
        if not isinstance(stat, dict):
            missing_feature_count += 1
            continue
        if index >= values.size:
            missing_feature_count += 1
            continue

        value = float(values[index])
        median = float(stat.get("median", 0.0) or 0.0)
        scale = profile_scale(stat, policy.minimum_scale) * float(scale_multiplier)
        z = abs(value - median) / scale

        p10 = float(stat.get("p10", median) if stat.get("p10", None) is not None else median)
        p90 = float(stat.get("p90", median) if stat.get("p90", None) is not None else median)
        low, high = (p10, p90) if p10 <= p90 else (p90, p10)
        if low <= value <= high:
            inside_range_count += 1
            range_penalty = 0.0
        else:
            range_violations += 1
            distance_outside = (low - value) if value < low else (value - high)
            range_penalty = 1.0 + (distance_outside / scale)

        if z >= 3.5 or range_penalty >= 3.5:
            severe_count += 1

        # Count-neutral bucket membership:
        # - every label gets the same one vote unit
        # - every available feature contributes equally
        # - reliability/valid_count do not weight the score
        # - inside-range values are rewarded, but typicality around median still matters
        residuals.append(float(0.35 * z + 0.65 * range_penalty * float(range_penalty_multiplier)))

    if not residuals:
        return float("inf"), {"used_feature_count": 0, "severe_count": 0}

    mean_residual = float(np.mean(np.asarray(residuals, dtype=np.float32)))
    used_count = len(residuals)
    coverage_ratio = used_count / max(1, FP_SIZE)
    severe_ratio = severe_count / max(1, used_count)
    outside_ratio = range_violations / max(1, used_count)
    inside_ratio = inside_range_count / max(1, used_count)

    # Missing features are a bucket-definition problem, not sample-count imbalance.
    # Penalize missing knobs so a one-feature bucket cannot beat a full bucket.
    missing_penalty = (1.0 - coverage_ratio) * 8.0
    score = (
        mean_residual
        + policy.severe_profile_penalty * severe_ratio * float(severe_penalty_multiplier)
        + missing_penalty
    )

    return float(score), {
        "used_feature_count": used_count,
        "feature_coverage_ratio": round(float(coverage_ratio), 6),
        "inside_range_count": inside_range_count,
        "inside_range_ratio": round(float(inside_ratio), 6),
        "range_violation_count": range_violations,
        "range_violation_ratio": round(float(outside_ratio), 6),
        "missing_feature_count": missing_feature_count,
        "severe_count": severe_count,
        "severe_ratio": round(float(severe_ratio), 6),
        "mean_residual": round(float(mean_residual), 6),
        "bucket_vote_weight": 1.0,
        "sample_count_weighting": "disabled",
        "reliability_weighting": "range_softening_only",
        "profile_raw_count": int(reliability.get("raw_count", 0)),
        "profile_effective_count": int(reliability.get("effective_count", 0)),
        "profile_fact_profile_strength": str(reliability.get("fact_profile_strength", "")),
        "profile_reliability_tier": str(reliability.get("tier", "unknown")),
        "profile_scale_multiplier": round(float(scale_multiplier), 6),
        "profile_range_penalty_multiplier": round(float(range_penalty_multiplier), 6),
        "profile_severe_penalty_multiplier": round(float(severe_penalty_multiplier), 6),
    }


def profile_reliability(profile: dict[str, Any]) -> dict[str, Any]:
    """Return support tier and softening policy for one fact profile.

    Count-neutral voting is still preserved: this does not give large folders
    more score power.  It only prevents tiny profiles from acting like brittle
    exclusion zones with unrealistically tight p10-p90 ranges.
    """

    def read_int(name: str, default: int = 0) -> int:
        try:
            return int(float(profile.get(name, default) or default))
        except Exception:
            return default

    raw_count = read_int("raw_count")
    effective_count = read_int("effective_count", raw_count)
    eligible_count = read_int("eligible_count", effective_count)
    strength = str(profile.get("fact_profile_strength", "") or "").lower()
    usable_count = max(raw_count, effective_count, eligible_count)
    if usable_count <= 4 or strength == "tentative":
        tier = "tiny"
        scale_multiplier = 3.0
        range_penalty_multiplier = 0.42
        severe_penalty_multiplier = 0.45
    elif usable_count <= 12 or strength == "weak":
        tier = "weak"
        scale_multiplier = 2.0
        range_penalty_multiplier = 0.65
        severe_penalty_multiplier = 0.70
    elif usable_count <= 30:
        tier = "moderate"
        scale_multiplier = 1.35
        range_penalty_multiplier = 0.85
        severe_penalty_multiplier = 0.90
    else:
        tier = "strong"
        scale_multiplier = 1.0
        range_penalty_multiplier = 1.0
        severe_penalty_multiplier = 1.0
    return {
        "tier": tier,
        "raw_count": raw_count,
        "effective_count": effective_count,
        "eligible_count": eligible_count,
        "fact_profile_strength": strength,
        "scale_multiplier": scale_multiplier,
        "range_penalty_multiplier": range_penalty_multiplier,
        "severe_penalty_multiplier": severe_penalty_multiplier,
    }


def profile_scale(stat: dict[str, Any], minimum_scale: float) -> float:
    """Return robust scale for one feature statistic."""
    iqr = abs(float(stat.get("iqr", 0.0) or 0.0))
    mad = abs(float(stat.get("mad", 0.0) or 0.0))
    return max(iqr / 1.349 if iqr > 0 else 0.0, mad * 1.4826 if mad > 0 else 0.0, minimum_scale)


def outside_profile_range(value: float, stat: dict[str, Any], scale: float) -> bool:
    """Return whether a feature is outside the learned central range."""
    median = float(stat.get("median", 0.0) or 0.0)
    p10 = float(stat.get("p10", median) if stat.get("p10") is not None else median)
    p90 = float(stat.get("p90", median) if stat.get("p90") is not None else median)
    low, high = (p10, p90) if p10 <= p90 else (p90, p10)
    return bool(value < low - scale or value > high + scale)


def structure_penalty(
    label: str, brain: dict[str, Any], facts: SharedAudioFacts, policy: PhysicsVoterPolicy
) -> tuple[float, str]:
    """Return a score penalty when candidate structure disagrees with facts."""
    structure = label_structure(brain, label)
    if structure == "loop" and not facts.is_loop_like:
        return policy.structure_mismatch_penalty, "label_expects_loop_but_audio_not_loop_like"
    if structure == "long_fx" and (facts.is_single_event_like or facts.is_short_hit_like):
        return policy.structure_mismatch_penalty, "label_expects_long_fx_but_audio_is_single_event"
    if structure == "one_shot" and facts.is_loop_like:
        return policy.structure_mismatch_penalty, "label_expects_one_shot_but_audio_loop_like"
    return 0.0, "structure_compatible"


def role_compatibility_adjustment(
    label: str,
    brain: dict[str, Any],
    facts: SharedAudioFacts,
    *,
    voter_name: str,
) -> tuple[float, dict[str, Any]]:
    """Return measured-role diagnostics without changing voter scores.

    Architecture rule: brain and physics voters must report their own raw
    category evidence.  Role and shape facts are measured behavior evidence
    for the arbiter; they are not allowed to pre-rank, boost, penalize, or
    filter identity candidates.

    Older v31 builds used this function for score shaping: compatible measured
    roles received negative score deltas and cross-family candidates received
    positive penalties.  That recreated rule-tree behavior before the arbiter
    could compare the real brain lanes.  The function now preserves the same
    diagnostics and the same recommended adjustment values for reports, but
    always returns an applied delta of 0.0.
    """
    roles = facts.evidence.get("measured_roles", {}) if isinstance(facts.evidence, dict) else {}
    if not isinstance(roles, dict):
        roles = {}
    top = label_top_family(brain, label)
    candidate_roles = candidate_role_signature(brain, label)
    role_distance = measured_to_candidate_role_distance(roles, candidate_roles)
    family = role_family_compatibility(top, roles)
    parent_role = str(family.get("detected_parent_role", "unknown"))
    parent_strength = role_strength(roles, parent_role) if parent_role != "unknown" else 0.0
    candidate_strength = role_strength(candidate_roles, parent_role) if parent_role != "unknown" else 0.0
    compatible = bool(family.get("is_family_compatible", True))
    recommended_adjustment = 0.0
    applied_adjustment = 0.0
    reasons: list[str]
    if parent_role == "unknown":
        reasons = ["role_diagnostic_no_strong_measured_role"]
    elif compatible:
        if candidate_strength >= 0.55:
            recommended_adjustment = -0.65 * parent_strength * candidate_strength
            reasons = ["role_diagnostic_profile_compatible"]
        else:
            recommended_adjustment = -0.15 * parent_strength
            reasons = ["role_diagnostic_top_family_compatible"]
    else:
        recommended_adjustment = 0.95 * parent_strength
        reasons = ["role_diagnostic_family_conflict"]

    return applied_adjustment, {
        "role_adjustment": round(float(applied_adjustment), 6),
        "role_adjustment_applied": round(float(applied_adjustment), 6),
        "role_recommended_adjustment": round(float(recommended_adjustment), 6),
        "role_adjustment_mode": "diagnostic_only",
        "role_adjustment_reasons": reasons,
        "candidate_role_signature": candidate_roles,
        "candidate_role_distance": role_distance,
        "family_compatibility": family,
        "detected_parent_role": parent_role,
        "measured_parent_role_strength": round(float(parent_strength), 6),
        "candidate_parent_role_strength": round(float(candidate_strength), 6),
        "role_diagnostics_voter": str(voter_name),
        "measured_role_strengths": {
            "bass_loop": role_strength(roles, "bass_loop"),
            "voiced_one_shot": role_strength(roles, "voiced_one_shot"),
            "percussive_one_shot": role_strength(roles, "percussive_one_shot"),
            "pitched_music_phrase": role_strength(roles, "pitched_music_phrase"),
            "bright_drum_loop": role_strength(roles, "bright_drum_loop"),
            "percussive_drum_loop": role_strength(roles, "percussive_drum_loop"),
            "pitched_music_loop": role_strength(roles, "pitched_music_loop"),
            "low_rhythmic_drum_loop": role_strength(roles, "low_rhythmic_drum_loop"),
            "vocal_music_phrase": role_strength(roles, "vocal_music_phrase"),
        },
    }


def role_strength(roles: dict[str, Any], name: str) -> float:
    try:
        return max(0.0, min(1.0, float(roles.get(name, 0.0) or 0.0)))
    except Exception:
        return 0.0


def profile_median(brain: dict[str, Any], label: str, feature_name: str, default: float = 0.0) -> float:
    profiles = (
        brain.get("category_fact_profiles", {}) if isinstance(brain.get("category_fact_profiles", {}), dict) else {}
    )
    profile = profiles.get(label, {}) if isinstance(profiles, dict) else {}
    stats = profile.get("feature_stats", {}) if isinstance(profile, dict) else {}
    stat = stats.get(feature_name, {}) if isinstance(stats, dict) else {}
    try:
        return float(stat.get("median", default) if isinstance(stat, dict) else default)
    except Exception:
        return float(default)


def profile_feature_values(brain: dict[str, Any], label: str) -> dict[str, float]:
    """Return profile medians by feature name for one candidate label."""
    return {name: profile_median(brain, label, str(name)) for name in FEATURE_NAMES[:FP_SIZE]}


def candidate_role_signature(brain: dict[str, Any], label: str) -> dict[str, float]:
    """Return a profile-derived functional role vector for diagnostics."""
    cache = brain.setdefault("_candidate_role_signature_cache", {})
    if isinstance(cache, dict):
        cached = cache.get(label)
        if isinstance(cached, dict):
            return cached
    structure = label_structure(brain, label)
    roles = measured_roles_from_features(
        is_loop_like=structure == "loop",
        is_single_event_like=structure == "one_shot",
        is_short_hit_like=structure == "one_shot",
        is_long=structure in {"loop", "long_fx"},
        feature_values=profile_feature_values(brain, label),
    )
    evidence = roles.to_evidence()
    signature = {
        "bass_loop": role_strength(evidence, "bass_loop"),
        "voiced_one_shot": role_strength(evidence, "voiced_one_shot"),
        "percussive_one_shot": role_strength(evidence, "percussive_one_shot"),
        "pitched_music_phrase": role_strength(evidence, "pitched_music_phrase"),
        "bright_drum_loop": role_strength(evidence, "bright_drum_loop"),
        "percussive_drum_loop": role_strength(evidence, "percussive_drum_loop"),
        "pitched_music_loop": role_strength(evidence, "pitched_music_loop"),
        "low_rhythmic_drum_loop": role_strength(evidence, "low_rhythmic_drum_loop"),
        "vocal_music_phrase": role_strength(evidence, "vocal_music_phrase"),
    }
    if isinstance(cache, dict):
        cache[label] = signature
    return signature


def measured_to_candidate_role_distance(measured: dict[str, Any], candidate: dict[str, float]) -> float:
    """Return Euclidean distance between measured and candidate role vectors."""
    keys = [
        "bass_loop",
        "voiced_one_shot",
        "percussive_one_shot",
        "pitched_music_phrase",
        "bright_drum_loop",
        "percussive_drum_loop",
        "pitched_music_loop",
        "low_rhythmic_drum_loop",
        "vocal_music_phrase",
    ]
    total = 0.0
    for key in keys:
        total += (role_strength(measured, key) - float(candidate.get(key, 0.0) or 0.0)) ** 2
    return round(float(np.sqrt(total)), 6)


def role_family_compatibility(top_family: str, roles: dict[str, Any]) -> dict[str, Any]:
    """Return broad family compatibility for the measured parent role."""
    parent_role = detected_parent_role_name(roles)
    compatible_tops = compatible_tops_for_role(parent_role)
    compatible = not compatible_tops or str(top_family) in compatible_tops
    return {
        "detected_parent_role": parent_role,
        "candidate_top_family": str(top_family),
        "compatible_top_families": compatible_tops,
        "is_family_compatible": bool(compatible),
    }


def detected_parent_role_name(roles: dict[str, Any]) -> str:
    """Return the strongest measured parent role name."""
    keys = [
        "bass_loop",
        "voiced_one_shot",
        "percussive_one_shot",
        "pitched_music_phrase",
        "bright_drum_loop",
        "percussive_drum_loop",
        "pitched_music_loop",
        "low_rhythmic_drum_loop",
        "vocal_music_phrase",
    ]
    strengths = [(key, role_strength(roles, key)) for key in keys]
    best_name, best_value = max(strengths, key=lambda item: item[1])
    return best_name if best_value >= 0.55 else "unknown"


def compatible_tops_for_role(parent_role: str) -> list[str]:
    """Return broad compatible destination families for a measured role."""
    if parent_role == "bass_loop":
        return ["Instruments"]
    if parent_role == "voiced_one_shot":
        return ["Instruments", "FX"]
    if parent_role == "vocal_music_phrase":
        return ["Instruments", "FX"]
    if parent_role == "percussive_one_shot":
        return ["Drums", "FX"]
    if parent_role == "pitched_music_phrase":
        return ["Instruments"]
    if parent_role in {"bright_drum_loop", "percussive_drum_loop", "low_rhythmic_drum_loop"}:
        return ["Drums"]
    if parent_role == "pitched_music_loop":
        return ["Instruments"]
    return []


def label_profile_voice_score(brain: dict[str, Any], label: str) -> float:
    """Return how much a label profile behaves like a voiced/formant one-shot."""
    f0 = profile_median(brain, label, "f0_voiced_ratio")
    pitch = profile_median(brain, label, "pitch_confidence")
    formant = profile_median(brain, label, "formant_like_peak_spacing")
    mid = profile_median(brain, label, "mid_ratio_500_2000hz")
    presence = profile_median(brain, label, "presence_ratio_2000_8000hz")
    return float(
        np.mean(
            [
                clamp01((f0 - 0.65) / 0.30),
                clamp01((pitch - 0.35) / 0.45),
                clamp01(formant / 0.60),
                clamp01(((mid + presence) - 0.35) / 0.45),
            ]
        )
    )


def label_profile_percussive_one_shot_score(brain: dict[str, Any], label: str) -> float:
    """Return how much a label profile behaves like a short percussive one-shot."""
    log_crest = profile_median(brain, label, "log_crest")
    attack = profile_median(brain, label, "attack_rise_time_norm", 1.0)
    temporal = profile_median(brain, label, "temporal_centroid_ratio", 0.5)
    tail = profile_median(brain, label, "tail_energy_ratio", 0.5)
    log_transients = profile_median(brain, label, "log_transient_count", 0.0)
    f0_voiced = profile_median(brain, label, "f0_voiced_ratio", 0.0)
    return float(
        np.mean(
            [
                clamp01((log_crest - 1.10) / 1.20),
                clamp01((0.30 - attack) / 0.30),
                clamp01((0.48 - temporal) / 0.48),
                clamp01((0.50 - tail) / 0.50),
                clamp01((2.20 - log_transients) / 2.20),
                clamp01((0.72 - f0_voiced) / 0.72),
            ]
        )
    )


def label_profile_bright_drum_loop_score(brain: dict[str, Any], label: str) -> float:
    """Return how much a label profile behaves like a bright drum/hat loop."""
    presence = profile_median(brain, label, "presence_ratio_2000_8000hz")
    air = profile_median(brain, label, "air_ratio_gt_8000hz")
    loop_percussive = profile_median(brain, label, "loop_percussive_event_ratio")
    loop_drumlike = profile_median(brain, label, "loop_drumlike_frame_ratio")
    loop_sustained = profile_median(brain, label, "loop_sustained_tonal_frame_ratio")
    high_total = min(1.0, presence + air)
    return float(
        np.mean(
            [
                clamp01((high_total - 0.35) / 0.45),
                clamp01((loop_percussive - 0.45) / 0.45),
                clamp01((loop_drumlike - 0.35) / 0.40),
                clamp01((0.40 - loop_sustained) / 0.40),
            ]
        )
    )


def clamp01(number: float) -> float:
    return max(0.0, min(1.0, float(number)))
