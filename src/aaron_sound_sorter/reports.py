# Auto-split from Aaron_Sound_Sorter.py.
# This is a component module, not a legacy wrapper.
from __future__ import annotations

from .brain import (
    label_model_distance_score,
    scaler_for_label,
    score_sample_against_label_facts,
    structure_distance_scores,
    structure_penalty_for_label,
    top_penalty_for_label,
    top_prefilter_allowed_tops,
)
from .committee import frontend_router_scores_for_fingerprint, linear_ridge_scores_for_fingerprint
from .core import *


def _committee_strength_score(value: str) -> float:
    rank = _membership_profile_strength_rank(str(value or ""))
    # weak/tentative/ok/good/strong -> 0..1-ish
    return max(0.0, min(1.0, rank / 4.0))


def _committee_minmax_good_low(value: float, values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return 0.0
    lo = min(finite)
    hi = max(finite)
    if hi <= lo + 1e-9:
        return 1.0
    return max(0.0, min(1.0, 1.0 - ((float(value) - lo) / (hi - lo))))


def _committee_minmax_good_high(value: float, values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return 0.0
    lo = min(finite)
    hi = max(finite)
    if hi <= lo + 1e-9:
        return 1.0
    return max(0.0, min(1.0, (float(value) - lo) / (hi - lo)))


def _committee_depth_score(label: str) -> float:
    public = public_label(str(label))
    # Deep is useful, but it should be a small tie breaker, not the main vote.
    depth = len([p for p in public.split("/") if p.strip()])
    return max(0.0, min(1.0, depth / 6.0))


def _committee_support_score(brain: dict, label: str) -> float:
    profiles = brain.get("category_fact_profiles", {}) if isinstance(brain.get("category_fact_profiles"), dict) else {}
    p = profiles.get(label, {}) if isinstance(profiles, dict) else {}
    eff = (
        float(
            p.get(
                "effective_count", brain.get("effective_counts", {}).get(label, brain.get("counts", {}).get(label, 0))
            )
            or 0.0
        )
        if isinstance(p, dict)
        else 0.0
    )
    strength = _committee_strength_score(p.get("fact_profile_strength", "") if isinstance(p, dict) else "")
    count_score = max(0.0, min(1.0, math.log1p(eff) / math.log1p(64.0)))
    return 0.60 * count_score + 0.40 * strength


def _committee_router_score_for_label(brain: dict, label: str, top_scores: Sequence[Tuple[str, float]]) -> float:
    if not top_scores:
        return 0.5
    label_top = brain.get("top_by_label", {}).get(label, top_for_public_label(label))
    values = [float(s) for _t, s in top_scores]
    label_score = None
    for top, score in top_scores:
        if str(top) == str(label_top):
            label_score = float(score)
            break
    if label_score is None:
        return 0.0
    return _committee_minmax_good_low(label_score, values)


def _committee_normalized_full_100_profile_score(
    brain: dict,
    fingerprint: Sequence[float],
    label: str,
    feature_names_list: Sequence[str],
) -> Tuple[float, Dict[str, float]]:
    """Normalized label fit across all reliable feature stats.

    This is the committee's full-data helper.  It compares the sample against the
    candidate label's learned medians/IQR/MAD for every reliable feature in the
    100-feature profile.  Training counts are not a positive vote here.  They only
    determine whether a feature statistic is reliable enough to use.
    """
    profiles = (
        brain.get("category_fact_profiles", {}) if isinstance(brain.get("category_fact_profiles", {}), dict) else {}
    )
    profile = profiles.get(label) if isinstance(profiles, dict) else None
    if not isinstance(profile, dict):
        return 0.50, {"used": 0.0, "severe": 0.0, "mean_fit": 0.50}
    stats = profile.get("feature_stats", {})
    if not isinstance(stats, dict) or not stats:
        return 0.50, {"used": 0.0, "severe": 0.0, "mean_fit": 0.50}
    values = np.asarray(fingerprint, dtype=np.float32)
    names = list(feature_names_list or brain.get("feature_names", FEATURE_NAMES))
    idx_by_name = {str(n): i for i, n in enumerate(names)}
    min_reliability = float(brain.get("committee_full100_min_feature_reliability", 0.15) or 0.15)
    min_valid = int(brain.get("committee_full100_min_feature_valid_count", 3) or 3)
    fits: List[float] = []
    weights_used: List[float] = []
    severe = 0
    used = 0
    for name, st in stats.items():
        if not isinstance(st, dict):
            continue
        idx = idx_by_name.get(str(name))
        if idx is None or idx >= values.size:
            continue
        reliability = float(st.get("reliability", 0.0) or 0.0)
        valid_count = int(st.get("valid_count", 0) or 0)
        if reliability < min_reliability or valid_count < min_valid:
            continue
        x = float(values[idx])
        if not math.isfinite(x):
            continue
        median = float(st.get("median", 0.0) or 0.0)
        iqr = abs(float(st.get("iqr", 0.0) or 0.0))
        mad = abs(float(st.get("mad", 0.0) or 0.0))
        p10 = float(st.get("p10", median) if st.get("p10", None) is not None else median)
        p90 = float(st.get("p90", median) if st.get("p90", None) is not None else median)
        scale = max(iqr / 1.349 if iqr > 0 else 0.0, mad * 1.4826 if mad > 0 else 0.0, 0.035)
        z = abs(x - median) / scale
        # Smooth robust fit.  z=0 -> 1.0, z=2 -> ~0.61, z=4 -> ~0.14.
        fit = math.exp(-0.5 * (z / 2.0) ** 2)
        # Extra violation for being well outside the learned central range.
        lo, hi = (p10, p90) if p10 <= p90 else (p90, p10)
        outside_range = x < (lo - scale) or x > (hi + scale)
        if z >= 3.5 or outside_range:
            severe += 1
        # Weight by learned reliability and the project feature weight if present.
        try:
            feat_w = float(FEATURE_WEIGHTS[idx]) if idx < len(FEATURE_WEIGHTS) else 1.0
        except Exception:
            feat_w = 1.0
        w = max(0.05, min(2.5, reliability * feat_w))
        fits.append(float(fit))
        weights_used.append(float(w))
        used += 1
    if not fits:
        return 0.50, {"used": 0.0, "severe": 0.0, "mean_fit": 0.50}
    mean_fit = float(np.average(np.asarray(fits, dtype=np.float32), weights=np.asarray(weights_used, dtype=np.float32)))
    severe_ratio = severe / max(1, used)
    score = max(0.0, min(1.0, mean_fit - 0.45 * severe_ratio))
    return score, {
        "used": float(used),
        "severe": float(severe),
        "mean_fit": float(mean_fit),
        "severe_ratio": float(severe_ratio),
    }


def _committee_label_full_100_profile_score(
    brain: dict,
    fingerprint: Sequence[float],
    label: str,
    feature_names_list: Sequence[str],
) -> float:
    """Label-specific full-profile router vote using the complete feature vector.

    This intentionally uses normalized 100-feature statistics instead of broad
    family counts.  A common label such as Guitar must match its own learned
    feature profile; it does not get a free boost because Instruments was a
    plausible top family.
    """
    normalized, _details = _committee_normalized_full_100_profile_score(brain, fingerprint, label, feature_names_list)
    # Blend in the older fact scorer lightly because it captures existing range
    # logic, but the normalized all-feature fit is the authority.
    profiles = (
        brain.get("category_fact_profiles", {}) if isinstance(brain.get("category_fact_profiles", {}), dict) else {}
    )
    profile = profiles.get(label) if isinstance(profiles, dict) else None
    old_score = 0.50
    if isinstance(profile, dict):
        try:
            raw = score_sample_against_label_facts(
                np.asarray(fingerprint, dtype=np.float32),
                profile,
                list(feature_names_list or brain.get("feature_names", FEATURE_NAMES)),
            )
            old_score = max(0.0, min(1.0, 0.5 + 0.5 * float(raw)))
        except Exception:
            old_score = 0.50
    return max(0.0, min(1.0, 0.80 * float(normalized) + 0.20 * float(old_score)))


def _committee_informed_router_score_for_label(
    brain: dict,
    fingerprint: Sequence[float],
    label: str,
    top_scores: Sequence[Tuple[str, float]],
    feature_names_list: Sequence[str],
) -> Tuple[float, float, float]:
    """Return combined router, top-family prior, and label-specific profile vote.

    The combined router is deliberately label-specific now.  Broad top-family
    evidence still counts, but it is not allowed to hand a perfect 1.0 score to
    every child label in that family.  A candidate such as Acoustic Guitar must
    match its own 100-feature label profile, not merely the broad Instruments top.
    """
    top_prior = _committee_router_score_for_label(brain, label, top_scores)
    label_full = _committee_label_full_100_profile_score(brain, fingerprint, label, feature_names_list)
    top_w = float(brain.get("committee_router_top_prior_weight", 0.12) or 0.12)
    label_w = float(brain.get("committee_router_label_full_100_weight", 0.88) or 0.88)
    total = max(1e-9, max(0.0, top_w) + max(0.0, label_w))
    combined = ((max(0.0, top_w) * top_prior) + (max(0.0, label_w) * label_full)) / total
    return max(0.0, min(1.0, combined)), max(0.0, min(1.0, top_prior)), max(0.0, min(1.0, label_full))


def _short_low_drum_hit_signature(
    brain: dict, fingerprint: Sequence[float], duration_sec: Optional[float] = None
) -> Tuple[bool, str]:
    """Detect a short low/front-loaded one-shot that should favor drum families.

    This is a broad physics role detector, not a tom-specific or filename rule.
    Pitched drums can have strong f0 confidence, so pitch alone must not push a
    short low transient into Guitar, Choir, or Rhodes.  The signature requires a
    short single-hit shape: fast attack, early temporal centroid, strong low/sub
    energy, low tail energy, and no loop evidence.
    """
    try:
        duration = float(duration_sec) if duration_sec is not None else 0.0
    except Exception:
        duration = 0.0
    if duration <= 0.0:
        duration = 0.0
    sub = _feature_value_by_name(fingerprint, "sub_bass_ratio_lt_150hz", 0.0)
    bass = _feature_value_by_name(fingerprint, "bass_ratio_150_500hz", 0.0)
    low_total = sub + bass
    temporal = _feature_value_by_name(fingerprint, "temporal_centroid_ratio", 1.0)
    attack = _feature_value_by_name(fingerprint, "attack_rise_time_norm", 1.0)
    tail = _feature_value_by_name(fingerprint, "tail_energy_ratio", 1.0)
    span = _feature_value_by_name(fingerprint, "onset_span_ratio", 0.0)
    regularity = _feature_value_by_name(fingerprint, "onset_interval_regularity", 0.0)
    event_rate = _feature_value_by_name(fingerprint, "event_rate_hz", 0.0)
    loop_like, loop_reason = (
        fingerprint_loop_evidence_detail(fingerprint, duration) if duration > 0.0 else (False, "no_duration")
    )
    max_dur = float(brain.get("short_low_drum_hit_max_duration", 1.60) or 1.60)
    sub_floor = float(brain.get("short_low_drum_hit_sub_floor", 0.35) or 0.35)
    low_floor = float(brain.get("short_low_drum_hit_low_total_floor", 0.60) or 0.60)
    temporal_ceiling = float(brain.get("short_low_drum_hit_temporal_centroid_ceiling", 0.24) or 0.24)
    attack_ceiling = float(brain.get("short_low_drum_hit_attack_ceiling", 0.09) or 0.09)
    tail_ceiling = float(brain.get("short_low_drum_hit_tail_ceiling", 0.22) or 0.22)

    ok = (
        (duration <= 0.0 or duration <= max_dur)
        and not loop_like
        and (sub >= sub_floor or low_total >= low_floor)
        and temporal <= temporal_ceiling
        and attack <= attack_ceiling
        and tail <= tail_ceiling
    )
    reason = (
        f"short_low_drum_hit={ok} duration={duration:.3f} sub={sub:.3f} low_total={low_total:.3f} "
        f"temporal={temporal:.3f} attack={attack:.3f} tail={tail:.3f} "
        f"span={span:.3f} regularity={regularity:.3f} event_rate={event_rate:.3f} loop_like={loop_like} {loop_reason}"
    )
    return bool(ok), reason


def _short_low_drum_hit_adjustment_for_label(
    brain: dict,
    label: str,
    fingerprint: Sequence[float],
    duration_sec: Optional[float] = None,
) -> float:
    """Ranking adjustment for short low drum-hit physics.

    When the measured physics screams short low drum/percussion hit, Drums should
    get a meaningful boost and broad Instruments labels should not receive a free
    pass from pitch or top-family priors.  This is intentionally broad: it does
    not force Tom, but it keeps the contest in the correct musical family.
    """
    is_drum_hit, _reason = _short_low_drum_hit_signature(brain, fingerprint, duration_sec)
    if not is_drum_hit:
        return 0.0
    top = str((brain.get("top_by_label", {}) or {}).get(label, top_for_public_label(label)))
    public = public_label(label).lower()
    if top == "Drums":
        bonus = float(brain.get("short_low_drum_hit_drum_bonus", 1.45) or 1.10)
        if any(
            word in public
            for word in ["tom", "tabla", "percussion", "kick", "bongo", "conga", "snare", "clap", "rim", "stick"]
        ):
            bonus += float(brain.get("short_low_drum_hit_low_drum_leaf_bonus", 0.55) or 0.35)
        return -bonus
    if top == "Instruments":
        return float(brain.get("short_low_drum_hit_instrument_penalty", 2.35) or 1.65)
    if top in {"FX", "Textures"}:
        return float(brain.get("short_low_drum_hit_non_drum_penalty", 0.45) or 0.45)
    return 0.0


def _committee_structure_score_for_label(
    brain: dict, label: str, structure_scores: Sequence[Tuple[str, float]]
) -> float:
    if not structure_scores:
        return 0.5
    label_structure = brain.get("structure_by_label", {}).get(label, label_default_structure(label))
    values = [float(s) for _st, s in structure_scores]
    label_score = None
    for st, score in structure_scores:
        if str(st) == str(label_structure):
            label_score = float(score)
            break
    if label_score is None:
        return 0.0
    return _committee_minmax_good_low(label_score, values)


def _committee_rival_score_for_label(
    brain: dict,
    label: str,
    rows: Sequence[Dict[str, float]],
    raw_x: np.ndarray,
    feature_names_list: Sequence[str],
    rival_contrast_facts: dict,
) -> float:
    """Average learned contrast vote for one candidate against nearby rivals.

    Positive means the candidate's own learned contrast facts are satisfied more
    often than its rivals' facts.  This is generic and uses only learned
    feature stats, never filenames or category-name rescue rules.
    """
    if not isinstance(rival_contrast_facts, dict) or not rival_contrast_facts:
        return 0.5
    rival_rows = [r for r in rows if str(r.get("label", "")) != str(label)][:4]
    if not rival_rows:
        return 0.5
    vals: List[float] = []
    for r in rival_rows:
        rival = str(r.get("label", ""))
        info = (
            rival_contrast_facts.get(label, {}).get(rival)
            if isinstance(rival_contrast_facts.get(label, {}), dict)
            else None
        )
        if not isinstance(info, dict):
            continue
        matched: List[str] = []
        failed: List[str] = []
        sc = score_sample_against_rival_contrast_facts(raw_x, info, feature_names_list, matched, failed)
        vals.append(float(sc))
    if not vals:
        return 0.5
    # Contrast scores are usually around -1..1; map conservatively to 0..1.
    return max(0.0, min(1.0, 0.5 + (float(np.mean(vals)) * 0.25)))


def committee_agreement_score_candidate(
    brain: dict,
    fingerprint: Sequence[float],
    row: Dict[str, float],
    rows: Sequence[Dict[str, float]],
    feature_names_list: Sequence[str],
    top_scores: Sequence[Tuple[str, float]],
    structure_scores: Sequence[Tuple[str, float]],
    rival_contrast_facts: dict,
) -> Dict[str, object]:
    """Score one learned candidate by committee agreement.

    The raw brain remains a committee member, but it is not the dictator.  A
    candidate can win only if the learned folder physics supports it.  All
    signals here are dynamic products of the training tree and measured audio.
    """
    label = str(row.get("label", ""))
    raw_x = np.asarray(fingerprint, dtype=np.float32)
    candidate_rows = list(rows)[
        : max(1, int(brain.get("committee_max_candidates", COMMITTEE_MAX_CANDIDATES) or COMMITTEE_MAX_CANDIDATES))
    ]
    current_scores = [float(r.get("current_score", 0.0)) for r in candidate_rows]
    raw_distances = [float(r.get("raw_distance", r.get("current_score", 0.0))) for r in candidate_rows]
    fact_scores = [float(r.get("fact_score", 0.0)) for r in candidate_rows]

    prototype = _committee_minmax_good_low(float(row.get("current_score", 0.0)), current_scores)
    exemplar = _committee_minmax_good_low(float(row.get("raw_distance", row.get("current_score", 0.0))), raw_distances)
    membership = folder_membership_evidence(brain, fingerprint, label)
    membership_score = float(membership.get("membership_score", 0.0) or 0.0) if membership.get("enabled") else 0.0
    if bool(membership.get("blocked", False)):
        membership_score = min(membership_score, 0.08)
    fact_profile = _committee_minmax_good_high(float(row.get("fact_score", 0.0)), fact_scores) if fact_scores else 0.5
    router, top_family_prior, label_full_100_router = _committee_informed_router_score_for_label(
        brain, fingerprint, label, top_scores, feature_names_list
    )
    normalized_full100, normalized_full100_details = _committee_normalized_full_100_profile_score(
        brain, fingerprint, label, feature_names_list
    )
    # Keep the public label100 vote equal to the normalized full-profile evidence,
    # not the broad-family prior.
    label_full_100_router = normalized_full100
    router = max(0.0, min(1.0, 0.12 * top_family_prior + 0.88 * label_full_100_router))
    structure = _committee_structure_score_for_label(brain, label, structure_scores)
    rival = _committee_rival_score_for_label(
        brain, label, candidate_rows, raw_x, feature_names_list, rival_contrast_facts
    )
    support = _committee_support_score(brain, label)
    depth = _committee_depth_score(label)

    weights = {
        "prototype": float(
            brain.get("committee_weight_prototype", COMMITTEE_WEIGHT_PROTOTYPE) or COMMITTEE_WEIGHT_PROTOTYPE
        ),
        "membership": float(
            brain.get("committee_weight_membership", COMMITTEE_WEIGHT_MEMBERSHIP) or COMMITTEE_WEIGHT_MEMBERSHIP
        ),
        "fact_profile": float(
            brain.get("committee_weight_fact_profile", COMMITTEE_WEIGHT_FACT_PROFILE) or COMMITTEE_WEIGHT_FACT_PROFILE
        ),
        "router": float(brain.get("committee_weight_router", COMMITTEE_WEIGHT_ROUTER) or COMMITTEE_WEIGHT_ROUTER),
        "rival_contrast": float(
            brain.get("committee_weight_rival_contrast", COMMITTEE_WEIGHT_RIVAL_CONTRAST)
            or COMMITTEE_WEIGHT_RIVAL_CONTRAST
        ),
        "support": float(brain.get("committee_weight_support", COMMITTEE_WEIGHT_SUPPORT) or COMMITTEE_WEIGHT_SUPPORT),
        "depth": float(brain.get("committee_weight_depth", COMMITTEE_WEIGHT_DEPTH) or COMMITTEE_WEIGHT_DEPTH),
        "structure": float(
            brain.get("committee_weight_structure", COMMITTEE_WEIGHT_STRUCTURE) or COMMITTEE_WEIGHT_STRUCTURE
        ),
    }
    votes = {
        "prototype": prototype,
        "exemplar": exemplar,
        "membership": membership_score,
        "fact_profile": fact_profile,
        "router": router,
        "top_family_prior": top_family_prior,
        "label_full_100_router": label_full_100_router,
        "full100_used": float(normalized_full100_details.get("used", 0.0)),
        "full100_severe": float(normalized_full100_details.get("severe", 0.0)),
        "full100_mean_fit": float(normalized_full100_details.get("mean_fit", 0.0)),
        "rival_contrast": rival,
        "support": support,
        "depth": depth,
        "structure": structure,
    }
    total_w = sum(max(0.0, float(v)) for v in weights.values()) or 1.0
    score = sum(max(0.0, float(weights[k])) * max(0.0, min(1.0, float(votes[k]))) for k in weights) / total_w
    severe_count = int(membership.get("severe_count", 0) or 0) if isinstance(membership, dict) else 0
    return {
        "label": label,
        "score": float(score),
        "votes": votes,
        "membership": membership,
        "blocked": bool(membership.get("blocked", False)) if isinstance(membership, dict) else False,
        "severe_count": severe_count,
        "current_score": float(row.get("current_score", 0.0)),
        "raw_distance": float(row.get("raw_distance", 0.0)),
    }


def _committee_cross_family_physics_override_allowed(
    brain: dict, current: Dict[str, object], candidate: Dict[str, object]
) -> Tuple[bool, str]:
    """Allow a cross-family committee switch only when the full committee is decisive.

    The old committee was same-family-only because a broad-family committee can
    become a dangerous reclassifier.  That safety rule is still the default.  This
    override is narrower: it only permits a cross-family switch when the incumbent
    is weak and the challenger has strong label-specific 100-feature support,
    membership support, structure support, and a large committee advantage.
    """
    cand_votes = candidate.get("votes", {}) if isinstance(candidate.get("votes"), dict) else {}
    cur_votes = current.get("votes", {}) if isinstance(current.get("votes"), dict) else {}
    cand_score = float(candidate.get("score", 0.0) or 0.0)
    cur_score = float(current.get("score", 0.0) or 0.0)
    advantage = cand_score - cur_score
    cand_label100 = float(cand_votes.get("label_full_100_router", 0.0) or 0.0)
    cur_label100 = float(cur_votes.get("label_full_100_router", 0.0) or 0.0)
    cand_membership = float(cand_votes.get("membership", 0.0) or 0.0)
    cand_fact = float(cand_votes.get("fact_profile", 0.0) or 0.0)
    cand_structure = float(cand_votes.get("structure", 0.0) or 0.0)
    cur_membership = float(cur_votes.get("membership", 0.0) or 0.0)
    cur_fact = float(cur_votes.get("fact_profile", 0.0) or 0.0)
    cand_top = top_for_public_label(str(candidate.get("label", "")))
    cur_top = top_for_public_label(str(current.get("label", "")))
    if bool(candidate.get("blocked", False)):
        return False, "candidate_membership_blocked"
    # Do not let the generalized full-100 override recreate the old failure
    # where an Instrument raw pick becomes an FX/Drums label unless a specific
    # role guard supports that family.  Drums-over-Instruments is handled below
    # by the drum-over-instrument physics branch; FX-over-Instruments needs a
    # separate transition/impact role guard and is not allowed here.
    if cur_top == "Instruments" and cand_top == "FX":
        return False, "instrument_to_fx_cross_family_requires_explicit_fx_role_guard"
    min_best = float(brain.get("committee_cross_family_override_min_best_score", 0.74) or 0.74)
    max_current = float(brain.get("committee_cross_family_override_max_current_score", 0.72) or 0.48)
    min_adv = float(brain.get("committee_cross_family_override_min_advantage", 0.07) or 0.24)
    min_label100 = float(brain.get("committee_cross_family_override_min_label100", 0.72) or 0.72)
    min_membership = float(brain.get("committee_cross_family_override_min_membership", 0.65) or 0.65)
    min_fact = float(brain.get("committee_cross_family_override_min_fact", 0.65) or 0.65)
    min_structure = float(brain.get("committee_cross_family_override_min_structure", 0.75) or 0.75)
    float(brain.get("committee_cross_family_override_label100_gap", 0.02) or 0.15)
    incumbent_weak = (
        cur_score <= max_current or bool(current.get("blocked", False)) or (cur_membership <= 0.15 and cur_fact <= 0.15)
    )
    challenger_strong = (
        cand_score >= min_best
        and advantage >= min_adv
        and cand_label100 >= min_label100
        and cand_membership >= min_membership
        and cand_fact >= min_fact
        and cand_structure >= min_structure
    )
    label100_not_worse = cand_label100 >= cur_label100 - float(
        brain.get("committee_cross_family_override_label100_tolerance", 0.08) or 0.08
    )
    physics_better_than_incumbent = (
        cand_membership >= cur_membership - 0.03 and cand_fact >= cur_fact - 0.03 and cand_score >= cur_score + min_adv
    )
    drum_over_instrument_physics = (
        cand_top == "Drums"
        and cur_top == "Instruments"
        and cand_score >= cur_score + min_adv
        and cand_membership >= 0.80
        and cand_fact >= 0.60
        and cand_structure >= min_structure
    )
    if (
        incumbent_weak
        and challenger_strong
        and (label100_not_worse or physics_better_than_incumbent or drum_over_instrument_physics)
    ):
        return True, (
            "cross_family_full_100_committee_override: "
            f"candidate_score={cand_score:.3f} current_score={cur_score:.3f} advantage={advantage:.3f} "
            f"candidate_label100={cand_label100:.3f} current_label100={cur_label100:.3f} "
            f"candidate_membership={cand_membership:.3f} candidate_fact={cand_fact:.3f} "
            f"candidate_structure={cand_structure:.3f} cand_top={cand_top} cur_top={cur_top}"
        )
    return False, (
        "cross_family_override_not_decisive: "
        f"candidate_score={cand_score:.3f} current_score={cur_score:.3f} advantage={advantage:.3f} "
        f"candidate_label100={cand_label100:.3f} current_label100={cur_label100:.3f} "
        f"candidate_membership={cand_membership:.3f} candidate_fact={cand_fact:.3f} candidate_structure={cand_structure:.3f} "
        f"cand_top={cand_top} cur_top={cur_top}"
    )


def committee_agreement_choose_label(
    brain: dict,
    fingerprint: Sequence[float],
    rows: Sequence[Dict[str, float]],
    current_label: str,
    feature_names_list: Sequence[str],
    top_scores: Sequence[Tuple[str, float]],
    structure_scores: Sequence[Tuple[str, float]],
    rival_contrast_facts: dict,
) -> Tuple[str, Dict[str, object]]:
    """Let the full committee choose the deepest physically approved candidate.

    A lower-ranked candidate can beat the raw nearest candidate when the folder
    membership/physics/router/rival facts agree.  This function never invents a
    label and never reads filenames.  It only chooses among labels already
    nominated by learned brain scoring.
    """
    if not bool(brain.get("committee_agreement_enabled", COMMITTEE_AGREEMENT_ENABLED_DEFAULT)):
        return current_label, {"mode": "committee_agreement", "switched": False, "reason": "disabled"}
    if not rows:
        return current_label, {"mode": "committee_agreement", "switched": False, "reason": "no_rows"}
    max_candidates = max(
        1, int(brain.get("committee_max_candidates", COMMITTEE_MAX_CANDIDATES) or COMMITTEE_MAX_CANDIDATES)
    )
    candidate_rows = list(rows)[:max_candidates]
    row_by_label = {str(r.get("label", "")): r for r in candidate_rows}
    # The current label may have been chosen by ensemble/rival logic and may not
    # be inside the top-N current_score rows after committee-specific physics
    # adjustments.  Include the actual current row, not rows[0].  The old code
    # inserted rows[0], which could make the committee compare a candidate
    # against itself and then keep the wrong current label with zero advantage.
    if current_label not in row_by_label and rows:
        all_rows_by_label = {str(r.get("label", "")): r for r in rows}
        current_row = all_rows_by_label.get(str(current_label))
        if current_row is not None:
            candidate_rows = [dict(current_row)] + candidate_rows
        else:
            candidate_rows = [dict(rows[0])] + candidate_rows
    candidate_rows = candidate_rows[:max_candidates]
    scored = [
        committee_agreement_score_candidate(
            brain,
            fingerprint,
            r,
            candidate_rows,
            feature_names_list,
            top_scores,
            structure_scores,
            rival_contrast_facts,
        )
        for r in candidate_rows
    ]
    if not scored:
        return current_label, {"mode": "committee_agreement", "switched": False, "reason": "no_scored_candidates"}
    # Prefer candidates that pass membership. A blocked candidate can only remain
    # if everything is blocked and no alternative is remotely plausible.
    unblocked = [s for s in scored if not bool(s.get("blocked", False))]
    pool = unblocked if unblocked else scored
    pool.sort(
        key=lambda s: (
            -float(s.get("score", 0.0)),
            -_committee_depth_score(str(s.get("label", ""))),
            float(s.get("current_score", 0.0)),
            public_label(str(s.get("label", ""))),
        )
    )
    current = next((s for s in scored if str(s.get("label", "")) == str(current_label)), scored[0])

    # v0.6.3: committee is not allowed to be a broad-family reclassifier.
    # It can refine within the raw brain's family.  Cross-family committee
    # candidates are diagnostics/review triggers, not final placement winners,
    # unless a new explicit override flag is set.  Old brains cannot opt in.
    committee_requested_cross_family = bool(brain.get("committee_allow_cross_family_switch", False))
    committee_explicit_cross_family_opt_in = bool(brain.get("committee_explicit_cross_family_override_enabled", False))
    committee_allow_cross_family = bool(committee_requested_cross_family and committee_explicit_cross_family_opt_in)
    committee_rejected: List[str] = []
    best = current
    best_cross_family_override_allowed = False
    for candidate in pool:
        cand_label = str(candidate.get("label", ""))
        if cand_label == str(current_label):
            best = candidate
            break
        if (not committee_allow_cross_family) and not _model_ensemble_same_top_family(brain, current_label, cand_label):
            override_ok, override_reason = _committee_cross_family_physics_override_allowed(brain, current, candidate)
            cand_top = str((brain.get("top_by_label", {}) or {}).get(cand_label, top_for_public_label(cand_label)))
            cur_top = str(
                (brain.get("top_by_label", {}) or {}).get(str(current_label), top_for_public_label(str(current_label)))
            )
            # Explicit broad-family rescue for measured percussive one-shots.
            # If the committee itself ranks a Drums candidate above a current
            # Instrument/FX leaf and the 100-feature physics has the short low
            # hit signature, let the committee switch. This is not a count vote
            # and not a category-name rescue; it requires the measured role and
            # the candidate's own committee score to agree.
            short_drum_ok, short_drum_reason = _short_low_drum_hit_signature(brain, fingerprint, None)
            if (
                (not override_ok)
                and cand_top == "Drums"
                and cur_top in {"Instruments", "FX", "Textures"}
                and short_drum_ok
            ):
                cand_votes = candidate.get("votes", {}) if isinstance(candidate.get("votes"), dict) else {}
                if (
                    float(candidate.get("score", 0.0) or 0.0) >= 0.72
                    and (float(candidate.get("score", 0.0) or 0.0) - float(current.get("score", 0.0) or 0.0)) >= 0.045
                    and float(cand_votes.get("membership", 0.0) or 0.0) >= 0.70
                    and float(cand_votes.get("structure", 0.0) or 0.0) >= 0.80
                ):
                    override_ok = True
                    override_reason = f"short_low_percussive_one_shot_committee_override: {short_drum_reason}"
            if not override_ok:
                committee_rejected.append(f"{cand_label}:cross_family_blocked:{override_reason}")
                continue
            committee_rejected.append(f"{cand_label}:cross_family_override_allowed:{override_reason}")
            best_cross_family_override_allowed = True
        best = candidate
        break

    # Log nearby unsafe cross-family candidates even if current wins first.
    for candidate in pool:
        if len(committee_rejected) >= 8:
            break
        cand_label = str(candidate.get("label", ""))
        if cand_label == str(current_label):
            continue
        if (not committee_allow_cross_family) and not _model_ensemble_same_top_family(brain, current_label, cand_label):
            reason = f"{cand_label}:cross_family_blocked"
            if reason not in committee_rejected:
                committee_rejected.append(reason)

    advantage = float(best.get("score", 0.0)) - float(current.get("score", 0.0))
    raw_score_gap = float(best.get("current_score", 0.0)) - float(current.get("current_score", 0.0))
    min_adv = float(
        brain.get("committee_switch_min_advantage", COMMITTEE_SWITCH_MIN_ADVANTAGE) or COMMITTEE_SWITCH_MIN_ADVANTAGE
    )
    max_gap = float(
        brain.get("committee_raw_max_score_gap", COMMITTEE_RAW_MAX_SCORE_GAP) or COMMITTEE_RAW_MAX_SCORE_GAP
    )
    min_best = float(
        brain.get("committee_review_min_best_score", COMMITTEE_REVIEW_MIN_BEST_SCORE) or COMMITTEE_REVIEW_MIN_BEST_SCORE
    )
    effective_min_adv = min_adv
    if bool(best_cross_family_override_allowed):
        effective_min_adv = min(
            effective_min_adv, float(brain.get("committee_cross_family_override_switch_min_advantage", 0.05) or 0.05)
        )
    switched = (
        str(best.get("label", "")) != str(current_label)
        and advantage >= effective_min_adv
        and (raw_score_gap <= max_gap or bool(best_cross_family_override_allowed))
        and float(best.get("score", 0.0)) >= min_best
    )
    chosen = str(best.get("label", current_label)) if switched else current_label

    def summarize(s: Dict[str, object]) -> str:
        v = s.get("votes", {}) if isinstance(s.get("votes"), dict) else {}
        return (
            f"{public_label(str(s.get('label', '')))} score={float(s.get('score', 0.0)):.3f} "
            f"proto={float(v.get('prototype', 0.0)):.2f} memb={float(v.get('membership', 0.0)):.2f} "
            f"fact={float(v.get('fact_profile', 0.0)):.2f} router={float(v.get('router', 0.0)):.2f} "
            f"topprior={float(v.get('top_family_prior', 0.0)):.2f} label100={float(v.get('label_full_100_router', 0.0)):.2f} "
            f"full100_bad={float(v.get('full100_severe', 0.0)):.0f} rival={float(v.get('rival_contrast', 0.0)):.2f} support={float(v.get('support', 0.0)):.2f} "
            f"struct={float(v.get('structure', 0.0)):.2f} blocked={bool(s.get('blocked', False))}"
        )

    meta = {
        "mode": "committee_agreement",
        "enabled": True,
        "switched": bool(switched),
        "from_label": str(current_label),
        "to_label": str(chosen),
        "advantage": float(advantage),
        "effective_min_advantage": float(effective_min_adv),
        "raw_score_gap": float(raw_score_gap),
        "best_score": float(best.get("score", 0.0)),
        "current_score": float(current.get("score", 0.0)),
        "reason": "committee_switched_to_deepest_physically_approved_candidate"
        if switched
        else "committee_kept_current_candidate",
        "top_committee": [summarize(s) for s in sorted(scored, key=lambda s: -float(s.get("score", 0.0)))[:5]],
        "chosen_votes": best.get("votes", {}) if switched else current.get("votes", {}),
        "chosen_membership": best.get("membership", {}) if switched else current.get("membership", {}),
        "rejected_proposals": "; ".join(committee_rejected[:8]),
        "same_family_switch_only": bool(not committee_allow_cross_family),
        "best_cross_family_override_allowed": bool(best_cross_family_override_allowed),
        "committee_requested_cross_family_switch": bool(committee_requested_cross_family),
        "committee_explicit_cross_family_opt_in": bool(committee_explicit_cross_family_opt_in),
    }
    return chosen, meta


def _measured_structure_contract(
    brain: dict, fingerprint: Sequence[float], duration_sec: Optional[float] = None
) -> Dict[str, object]:
    """Return strong measured structure evidence for prediction-time ranking.

    This is the early structure guard Aaron asked for: a clearly repeating loop
    should not let one-shot labels win just because their timbre centroid is close.
    It uses the same 100-feature fingerprint plus duration that the Stage 4 brain
    was trained with.  It does not inspect filenames or folder names.
    """
    meta: Dict[str, object] = {
        "observed_structure": "",
        "confidence": "none",
        "reason": "duration_not_available",
        "one_shot_mismatch_penalty": 0.0,
        "loop_mismatch_penalty": 0.0,
        "matching_structure_bonus": 0.0,
    }
    if duration_sec is None:
        return meta
    try:
        duration = float(duration_sec)
    except Exception:
        return meta
    if not math.isfinite(duration) or duration <= 0.0:
        meta["reason"] = "bad_duration"
        return meta

    loop_like, loop_reason = fingerprint_loop_evidence_detail(fingerprint, duration)
    if loop_like:
        meta.update(
            {
                "observed_structure": "loop",
                "confidence": "hard",
                "reason": loop_reason,
                # Big enough to move obvious repeating patterns above timbre-similar
                # one-shot labels, but still finite so diagnostics can show the contest.
                "one_shot_mismatch_penalty": float(brain.get("measured_loop_vs_one_shot_penalty", 8.0) or 8.0),
                "matching_structure_bonus": float(brain.get("measured_structure_match_bonus", 0.35) or 0.35),
            }
        )
        return meta

    # Symmetric front-door structure proof for obvious short one-shots.
    # Older builds only hard-filtered obvious loops out of one-shot labels. That
    # left very short single events free to waste ranking slots on loop labels,
    # even though the same measured structure head had already proved there is
    # no loop-like event spread. This is deliberately generic: duration plus
    # event topology only, no folder words, filenames, or instrument categories.
    try:
        fp = np.asarray(fingerprint, dtype=np.float32)
        transients = fingerprint_primary_event_count(fp)
        temporal_center = float(fp[42]) if fp.size > 42 else 1.0
        regularity = float(fp[43]) if fp.size > 43 else 0.0
        attack = float(fp[44]) if fp.size > 44 else 1.0
        onset_span = float(fp[45]) if fp.size > 45 else 1.0
        event_rate = float(fp[46]) if fp.size > 46 else 0.0
    except Exception:
        transients = 999.0
        temporal_center = 1.0
        regularity = 0.0
        attack = 1.0
        onset_span = 1.0
        event_rate = 999.0

    short_single_event = (
        duration <= float(brain.get("measured_short_one_shot_max_duration", 1.25) or 1.25)
        and transients <= float(brain.get("measured_short_one_shot_max_events", 2.0) or 2.0)
        and onset_span <= float(brain.get("measured_short_one_shot_max_span", 0.22) or 0.22)
        and temporal_center <= float(brain.get("measured_short_one_shot_max_temporal_centroid", 0.42) or 0.42)
    )
    tiny_nonloop = (
        duration <= float(brain.get("measured_tiny_one_shot_max_duration", 0.75) or 0.75)
        and transients <= 3.0
        and onset_span <= 0.30
    )
    if short_single_event or tiny_nonloop:
        meta.update(
            {
                "observed_structure": "one_shot",
                "confidence": "hard",
                "reason": (
                    "short_single_event_one_shot_proof "
                    f"duration={duration:.3f} transients={transients:.1f} "
                    f"temporal={temporal_center:.3f} regularity={regularity:.3f} "
                    f"span={onset_span:.3f} rate={event_rate:.3f} attack={attack:.3f}; {loop_reason}"
                ),
                "loop_mismatch_penalty": float(brain.get("measured_one_shot_vs_loop_penalty", 5.0) or 5.0),
                "matching_structure_bonus": float(brain.get("measured_structure_match_bonus", 0.35) or 0.35),
            }
        )
        return meta

    # A long file can be a one-shot only when measured as a single front-loaded
    # event with tail.  Do not punish loop labels for ambiguous phrases here.
    try:
        front_tail = bool(fingerprint_front_loaded_tail_evidence(np.asarray(fingerprint, dtype=np.float32), duration))
    except Exception:
        front_tail = False
    if front_tail:
        meta.update(
            {
                "observed_structure": "one_shot",
                "confidence": "hard",
                "reason": loop_reason,
                "loop_mismatch_penalty": float(brain.get("measured_one_shot_vs_loop_penalty", 5.0) or 5.0),
                "matching_structure_bonus": float(brain.get("measured_structure_match_bonus", 0.35) or 0.35),
            }
        )
        return meta

    meta["reason"] = loop_reason
    return meta


def _coherent_feature_group_outlier_penalty(
    brain: dict,
    fingerprint: Sequence[float],
    label: str,
    fact_profile: Dict[str, object] | None,
    feature_names_list: List[str],
) -> Tuple[float, str]:
    """Penalty when a whole physics group contradicts a label profile.

    Per-feature fact scoring can be too forgiving when a candidate wins from one
    broad shape cue while several related features all disagree. This helper is
    still brain-first and label-agnostic: it uses learned p10/p90/MAD ranges from
    the folder atlas and only asks whether a coherent descriptor group is outside
    the label's own learned profile. It does not inspect filenames or branch on
    category names.
    """
    if not isinstance(fact_profile, dict):
        return 0.0, ""
    if not bool(brain.get("coherent_group_outlier_penalty_enabled", True)):
        return 0.0, "disabled"
    try:
        eligible = int(fact_profile.get("eligible_count", 0) or 0)
    except Exception:
        eligible = 0
    if eligible < int(brain.get("coherent_group_outlier_min_count", 10) or 10):
        return 0.0, "profile_too_small"

    feature_stats = fact_profile.get("feature_stats", {})
    if not isinstance(feature_stats, dict):
        return 0.0, "no_feature_stats"
    raw_x = np.asarray(fingerprint, dtype=np.float32)
    index_by_name = {str(name): i for i, name in enumerate(feature_names_list)}

    groups: Dict[str, Tuple[str, ...]] = {
        "pitch_harmonic": (
            "pitch_confidence",
            "f0_voiced_ratio",
            "harmonic_to_noise_ratio",
            "harmonic_energy_ratio",
            "attack_pitch_confidence",
            "body_pitch_confidence",
            "tail_pitch_confidence",
            "fundamental_dominance_ratio",
            "formant_like_peak_spacing",
        ),
        "envelope_event_shape": (
            "log_crest",
            "temporal_centroid_ratio",
            "attack_rise_time_norm",
            "onset_span_ratio",
            "event_rate_hz",
            "tail_energy_ratio",
            "noise_burst_duration_ms",
            "log_decay_ratio",
        ),
        "spectral_noise_shape": (
            "spectral_flatness_mean",
            "spectral_entropy_mean",
            "zcr_mean",
            "attack_flatness",
            "body_flatness",
            "tail_flatness",
            "attack_entropy",
            "body_entropy",
            "tail_entropy",
        ),
        "band_balance": (
            "sub_bass_ratio_lt_150hz",
            "bass_ratio_150_500hz",
            "mid_ratio_500_2000hz",
            "presence_ratio_2000_8000hz",
            "air_ratio_gt_8000hz",
            "attack_high_ratio",
            "body_high_ratio",
            "tail_high_ratio",
        ),
    }

    total_penalty = 0.0
    reasons: List[str] = []
    min_features = int(brain.get("coherent_group_outlier_min_features", 3) or 3)
    max_group_penalty = float(brain.get("coherent_group_outlier_max_group_penalty", 1.25) or 1.25)
    overall_cap = float(brain.get("coherent_group_outlier_total_cap", 2.25) or 2.25)
    rel_floor = float(brain.get("coherent_group_outlier_min_reliability", 0.45) or 0.45)

    for group_name, names in groups.items():
        high_count = low_count = usable = 0
        high_sev = low_sev = 0.0
        shown: List[str] = []
        for fname in names:
            idx = index_by_name.get(fname)
            fs = feature_stats.get(fname)
            if idx is None or idx >= raw_x.size or not isinstance(fs, dict):
                continue
            rel = float(fs.get("reliability", 0.0) or 0.0)
            if rel < rel_floor:
                continue
            value = float(raw_x[idx])
            if not math.isfinite(value):
                continue
            p10 = float(fs.get("p10", fs.get("lower", fs.get("median", 0.0))) or 0.0)
            p90 = float(fs.get("p90", fs.get("upper", fs.get("median", 0.0))) or 0.0)
            float(fs.get("median", 0.0) or 0.0)
            mad = abs(float(fs.get("mad", 0.0) or 0.0))
            scale = max(mad * 2.5, abs(p90 - p10) * 0.35, 1e-6)
            usable += 1
            if value > p90:
                sev = min(2.5, (value - p90) / scale)
                high_count += 1
                high_sev += sev * max(0.5, rel)
                if len(shown) < 4:
                    shown.append(f"{fname}=high({value:.3g}>{p90:.3g})")
            elif value < p10:
                sev = min(2.5, (p10 - value) / scale)
                low_count += 1
                low_sev += sev * max(0.5, rel)
                if len(shown) < 4:
                    shown.append(f"{fname}=low({value:.3g}<{p10:.3g})")

        # Coherent means several related features disagree in the same direction.
        # Mixed high/low scatter is less meaningful and left to normal fact scoring.
        if usable >= min_features and high_count >= min_features and high_count >= low_count + 1:
            p = min(max_group_penalty, 0.28 * high_count + 0.22 * high_sev)
            total_penalty += p
            reasons.append(f"{group_name}:high:{high_count}/{usable}:penalty={p:.2f}:{';'.join(shown)}")
        elif usable >= min_features and low_count >= min_features and low_count >= high_count + 1:
            p = min(max_group_penalty, 0.28 * low_count + 0.22 * low_sev)
            total_penalty += p
            reasons.append(f"{group_name}:low:{low_count}/{usable}:penalty={p:.2f}:{';'.join(shown)}")

    return min(overall_cap, float(total_penalty)), " | ".join(reasons[:3])


def _measured_structure_adjustment_for_label(brain: dict, label: str, structure_contract: Dict[str, object]) -> float:
    observed = str(structure_contract.get("observed_structure", "") or "")
    if not observed:
        return 0.0
    label_structure = str(
        brain.get("structure_by_label", {}).get(label, label_default_structure(label)) or label_default_structure(label)
    )
    if observed == "loop":
        if label_structure != "loop":
            return float(structure_contract.get("one_shot_mismatch_penalty", 8.0) or 8.0)
        return -float(structure_contract.get("matching_structure_bonus", 0.35) or 0.35)
    if observed == "one_shot":
        if label_structure == "loop":
            return float(structure_contract.get("loop_mismatch_penalty", 5.0) or 5.0)
        return -float(structure_contract.get("matching_structure_bonus", 0.35) or 0.35)
    return 0.0


def _feature_value_by_name(fingerprint: Sequence[float], name: str, default: float = 0.0) -> float:
    """Read a 100-feature value by name without relying on mystery indexes."""
    try:
        idx = FEATURE_NAMES.index(name)
        values = np.asarray(fingerprint, dtype=np.float32)
        if idx < values.size:
            value = float(values[idx])
            return value if math.isfinite(value) else float(default)
    except Exception:
        pass
    return float(default)


def _loop_family_tiebreak_adjustment_for_label(
    brain: dict,
    label: str,
    fingerprint: Sequence[float],
    structure_contract: Dict[str, object],
) -> float:
    """Prefer the right broad loop family after hard loop detection.

    Once the measured 100-feature structure says "hard loop", the candidate set
    may contain only the broad loop labels currently present in the brain:
    Drums/Drum Loops/Loops and Instruments/Instrument Loops/Loops.  A pitched,
    harmonic, resonant loop should not tie with or lose to Drum Loops just
    because it has repeated attacks.  This is not a filename rule and it does
    not invent a Bell Loop label; it only resolves the broad loop-family choice.
    """
    observed = str(structure_contract.get("observed_structure", "") or "")
    confidence = str(structure_contract.get("confidence", "") or "")
    if observed != "loop" or confidence != "hard":
        return 0.0

    public = public_label(label).replace("\\", "/").strip("/")
    if not public.endswith("/Loops"):
        return 0.0

    pitch_conf = _feature_value_by_name(fingerprint, "pitch_confidence", 0.0)
    voiced = _feature_value_by_name(fingerprint, "f0_voiced_ratio", 0.0)
    harmonic_ratio = _feature_value_by_name(fingerprint, "harmonic_energy_ratio", 0.0)
    hnr = _feature_value_by_name(fingerprint, "harmonic_to_noise_ratio", 0.0)
    flatness = _feature_value_by_name(fingerprint, "spectral_flatness_mean", 1.0)
    entropy = _feature_value_by_name(fingerprint, "spectral_entropy_mean", 1.0)
    event_rate = _feature_value_by_name(fingerprint, "event_rate_hz", 0.0)
    regularity = _feature_value_by_name(fingerprint, "onset_interval_regularity", 0.0)

    tonal_loop = (
        pitch_conf >= float(brain.get("tonal_loop_pitch_confidence_floor", 0.55) or 0.55)
        or voiced >= float(brain.get("tonal_loop_voiced_ratio_floor", 0.35) or 0.35)
        or harmonic_ratio >= float(brain.get("tonal_loop_harmonic_ratio_floor", 0.28) or 0.28)
        or hnr >= float(brain.get("tonal_loop_hnr_floor", 0.20) or 0.20)
    )
    low_noise_tone = flatness <= 0.38 and entropy <= 0.92
    drum_grid_like = event_rate >= 2.0 and regularity >= 0.45 and not tonal_loop

    if public.startswith("Instruments/") and tonal_loop and low_noise_tone:
        return -float(brain.get("instrument_loop_tonal_bonus", 1.25) or 1.25)
    if public.startswith("Drums/") and tonal_loop and low_noise_tone:
        return float(brain.get("drum_loop_tonal_penalty", 0.85) or 0.85)
    if public.startswith("Drums/") and drum_grid_like:
        return -float(brain.get("drum_loop_grid_bonus", 0.35) or 0.35)
    return 0.0


def predict(
    brain: dict, fingerprint: Sequence[float], structure: str = "", duration_sec: Optional[float] = None
) -> Tuple[str, str, float, float, List[Tuple[str, float]]]:
    labels = list(brain["labels"])
    if structure:
        structure_by_label = brain.get("structure_by_label", {})
        filtered = [
            label for label in labels if structure_by_label.get(label, label_default_structure(label)) == structure
        ]
        labels = filtered or labels

    # Learned top prefilter. This reduces cross-family mistakes without giving
    # the predictor expected structure or filename help.
    top_by_label = brain.get("top_by_label", {})
    original_labels = list(labels)
    top_n = max(3, int(brain.get("top_prefilter_n", 3) or 3))
    allowed_tops, _top_scores = top_prefilter_allowed_tops(brain, fingerprint, top_n=top_n)
    if allowed_tops:
        filtered = [label for label in labels if top_by_label.get(label, top_for_public_label(label)) in allowed_tops]
        labels = filtered if filtered else original_labels

    weights = np.asarray(brain["feature_weights"], dtype=np.float32)
    raw_x = np.asarray(fingerprint, dtype=np.float32)
    structure_scores = structure_distance_scores(brain, fingerprint)
    measured_structure_contract = _measured_structure_contract(brain, fingerprint, duration_sec)

    # Hard structure filter for obvious measured loops. A clearly repeating long
    # file should not return one-shot labels in top5 at all. Penalties alone still
    # let one-shot labels appear as confusing near-misses, which makes debugging
    # and review output misleading.
    #
    # This is intentionally asymmetric: only a hard measured loop removes one-shot
    # candidates. Long single events with tails remain eligible as one-shots, and
    # ambiguous phrases are left to the normal scorer.
    measured_structure = str(measured_structure_contract.get("observed_structure", "") or "")
    measured_structure_confidence = str(measured_structure_contract.get("confidence", "") or "")
    if measured_structure == "loop" and measured_structure_confidence == "hard":
        structure_by_label = (
            brain.get("structure_by_label", {}) if isinstance(brain.get("structure_by_label", {}), dict) else {}
        )
        loop_filtered = [
            label
            for label in labels
            if str(structure_by_label.get(label, label_default_structure(label)) or label_default_structure(label))
            == "loop"
        ]
        if loop_filtered:
            labels = loop_filtered
    elif measured_structure == "one_shot" and measured_structure_confidence == "hard":
        structure_by_label = (
            brain.get("structure_by_label", {}) if isinstance(brain.get("structure_by_label", {}), dict) else {}
        )
        one_shot_filtered = [
            label
            for label in labels
            if str(structure_by_label.get(label, label_default_structure(label)) or label_default_structure(label))
            != "loop"
        ]
        if one_shot_filtered:
            labels = one_shot_filtered

    linear_scores = linear_ridge_scores_for_fingerprint(brain, fingerprint)
    frontend_router_scores = frontend_router_scores_for_fingerprint(brain, fingerprint)

    # v0.4.82: Pre-load fact profiles and feature names once for the whole contest.
    fact_profiles = (
        brain.get("category_fact_profiles", {}) if isinstance(brain.get("category_fact_profiles"), dict) else {}
    )
    feature_names_list: List[str] = list(brain.get("feature_names", FEATURE_NAMES))
    fact_enabled = bool(brain.get("fact_scoring_enabled", FACT_SCORING_ENABLED_DEFAULT))
    fact_weight = float(brain.get("fact_score_weight", FACT_SCORE_WEIGHT) or FACT_SCORE_WEIGHT)

    rows: List[Dict[str, float]] = []
    for label in labels:
        mean, std = scaler_for_label(brain, label)
        x = (raw_x - mean) / std
        xw = x * weights
        raw_score, raw_distance, _model_mode = label_model_distance_score(brain, label, xw)
        # Count-neutral ranking: do not apply support-balance bonuses here.
        # Category count effects belong in the trained brain, not in helper logic.
        support_bonus = 0.0
        structure_penalty = structure_penalty_for_label(brain, fingerprint, label, structure_scores)
        measured_structure_penalty = _measured_structure_adjustment_for_label(brain, label, measured_structure_contract)
        loop_family_tiebreak = _loop_family_tiebreak_adjustment_for_label(
            brain, label, fingerprint, measured_structure_contract
        )
        short_low_drum_hit_adjustment = _short_low_drum_hit_adjustment_for_label(
            brain, label, fingerprint, duration_sec
        )
        top_penalty = top_penalty_for_label(brain, label, _top_scores)
        frontend_router_penalty, frontend_router_rank_score, frontend_router_note = _frontend_route_penalty_for_label(
            brain, label, frontend_router_scores
        )

        # v0.4.82: Fact score shifts primary ranking so the first guess already
        # uses measured audio facts. Positive fact score = good match, lowers current_score.
        # Negative = atypical, raises current_score. Weight is small so facts refine
        # rather than override the distance contest.
        fact_score = 0.0
        if fact_enabled and fact_profiles:
            profile = fact_profiles.get(label)
            if isinstance(profile, dict):
                fact_score = score_sample_against_label_facts(raw_x, profile, feature_names_list)
        fact_penalty = -fact_weight * fact_score  # good match → negative → helps
        coherent_group_penalty, coherent_group_reason = _coherent_feature_group_outlier_penalty(
            brain,
            raw_x,
            label,
            fact_profiles.get(label) if isinstance(fact_profiles, dict) else None,
            feature_names_list,
        )

        current_score = (
            max(0.0, float(raw_score) - float(support_bonus))
            + float(structure_penalty)
            + float(measured_structure_penalty)
            + float(loop_family_tiebreak)
            + float(short_low_drum_hit_adjustment)
            + float(top_penalty)
            + float(frontend_router_penalty)
            + float(fact_penalty)
            + float(coherent_group_penalty)
        )
        support_neutral_score = (
            float(raw_score)
            + float(structure_penalty)
            + float(measured_structure_penalty)
            + float(loop_family_tiebreak)
            + float(short_low_drum_hit_adjustment)
            + float(top_penalty)
            + float(frontend_router_penalty)
        )
        label_top = top_by_label.get(label, top_for_public_label(label))
        spread_scale = _label_spread_scale_for_ensemble(brain, label, label_top)
        spread_norm_score = float(raw_distance) / max(0.20, float(spread_scale))
        anchor_score = _weighted_anchor_distance_for_label(brain, label, fingerprint)
        if not math.isfinite(anchor_score):
            anchor_score = float(raw_score)
        linear_score = float(linear_scores.get(label, 0.0)) if isinstance(linear_scores, dict) else 0.0
        membership_for_tournament = folder_membership_evidence(brain, raw_x, label)
        membership_blocked = (
            bool(membership_for_tournament.get("blocked", False))
            if isinstance(membership_for_tournament, dict)
            else False
        )
        membership_score_for_tournament = (
            float(membership_for_tournament.get("membership_score", 0.0) or 0.0)
            if isinstance(membership_for_tournament, dict) and membership_for_tournament.get("enabled")
            else 0.0
        )
        rows.append(
            {
                "label": label,
                "current_score": float(current_score),
                "support_neutral_score": float(support_neutral_score),
                "spread_norm_score": float(spread_norm_score),
                "anchor_score": float(anchor_score),
                "linear_score": float(linear_score),
                "linear_rank_score": float(-linear_score),
                "frontend_router_penalty": float(frontend_router_penalty),
                "frontend_router_rank_score": float(frontend_router_rank_score),
                "frontend_router_note": str(frontend_router_note),
                "raw_score": float(raw_score),
                "raw_distance": float(raw_distance),
                "support_bonus": float(support_bonus),
                "structure_penalty": float(structure_penalty),
                "measured_structure_penalty": float(measured_structure_penalty),
                "loop_family_tiebreak": float(loop_family_tiebreak),
                "short_low_drum_hit_adjustment": float(short_low_drum_hit_adjustment),
                "top_penalty": float(top_penalty),
                "fact_score": float(fact_score),
                "fact_penalty": float(fact_penalty),
                "coherent_group_penalty": float(coherent_group_penalty),
                "coherent_group_reason": str(coherent_group_reason),
                "membership_score": float(membership_score_for_tournament),
                "membership_blocked": bool(membership_blocked),
            }
        )

    if not rows:
        raise RuntimeError("No labels available for prediction")
    rows.sort(key=lambda r: (float(r["current_score"]), str(r["label"])))
    current_label = _raw_brain_winner_label(rows) or str(rows[0]["label"])
    final_label, ensemble_meta = ensemble_tournament_choose_label(brain, rows, current_label)

    # v0.5.1: rival contrast becomes a cautious learned second-pass chooser.
    # It can only swap between close labels already learned from the training
    # tree and only through dynamic feature facts. No filename or category-name
    # rescue logic is allowed here.
    rival_contrast_enabled = bool(brain.get("rival_contrast_enabled", RIVAL_CONTRAST_ENABLED_DEFAULT))
    rival_contrast_facts = (
        brain.get("category_rival_contrast_facts", {})
        if isinstance(brain.get("category_rival_contrast_facts"), dict)
        else {}
    )
    rival_second_pass_meta: Dict[str, object] = {
        "mode": "rival_contrast_second_pass",
        "swapped": False,
        "reason": "not_run",
    }
    if rival_contrast_enabled and rival_contrast_facts:
        final_label, rival_second_pass_meta = rival_contrast_second_pass_choose_label(
            brain=brain,
            rows=rows,
            current_label=final_label,
            raw_x=raw_x,
            feature_names_list=feature_names_list,
            fact_profiles=fact_profiles,
            rival_contrast_facts=rival_contrast_facts,
        )

    weak_rival_meta: Dict[str, object] = {"mode": "weak_profile_rival_override", "swapped": False, "reason": "not_run"}
    if not bool(rival_second_pass_meta.get("swapped", False)) and fact_profiles:
        final_label, weak_rival_meta = weak_profile_rival_override_choose_label(
            brain=brain,
            rows=rows,
            current_label=final_label,
            raw_x=raw_x,
            feature_names_list=feature_names_list,
            fact_profiles=fact_profiles,
        )

    committee_meta: Dict[str, object] = {"mode": "committee_agreement", "switched": False, "reason": "not_run"}
    if bool(brain.get("committee_agreement_enabled", COMMITTEE_AGREEMENT_ENABLED_DEFAULT)):
        final_label, committee_meta = committee_agreement_choose_label(
            brain=brain,
            fingerprint=fingerprint,
            rows=rows,
            current_label=final_label,
            feature_names_list=feature_names_list,
            top_scores=_top_scores,
            structure_scores=structure_scores,
            rival_contrast_facts=rival_contrast_facts,
        )

    row_by_label = {str(r["label"]): r for r in rows}
    best_row = row_by_label.get(final_label, rows[0])
    current_sorted = rows
    best_current_score = float(best_row["current_score"])
    other_scores = [float(r["current_score"]) for r in current_sorted if str(r["label"]) != final_label]
    second_score = min(other_scores) if other_scores else best_current_score + 99.0

    predicted_label = final_label
    predicted_top = brain.get("top_by_label", {}).get(predicted_label, top_for_public_label(predicted_label))
    top_scale = float(brain.get("typical_dist_mean_by_top", {}).get(predicted_top, 1.0) or 1.0)
    # Similarity stays tied to the final label's raw distance, not to penalties.
    sim = 1.0 / (1.0 + (float(best_row.get("raw_distance", best_current_score)) / max(0.1, top_scale)))
    margin = max(0.0, float(second_score) - float(best_current_score))

    # v0.5.1: Rival contrast scoring is now both a controlled second-pass
    # chooser above and a manifest explanation here.  If the second pass swapped,
    # report the challenger-vs-former-winner facts that caused the swap.
    rival_contrast_score_val = 0.0
    matched_key_facts: List[str] = []
    failed_key_facts: List[str] = []
    top_rival_label = ""
    if bool(rival_second_pass_meta.get("swapped", False)):
        top_rival_label = str(rival_second_pass_meta.get("from_label", ""))
        rival_contrast_score_val = float(rival_second_pass_meta.get("challenger_contrast_score", 0.0) or 0.0)
        matched_key_facts = list(rival_second_pass_meta.get("matched_key_facts", []) or [])
        failed_key_facts = list(rival_second_pass_meta.get("failed_key_facts", []) or [])
    elif rival_contrast_enabled and rival_contrast_facts and len(rows) >= 2:
        runner_up_rows = [r for r in rows if str(r["label"]) != predicted_label]
        if runner_up_rows:
            top_rival_label = str(runner_up_rows[0]["label"])
            label_rivalries = rival_contrast_facts.get(predicted_label, {})
            if isinstance(label_rivalries, dict) and top_rival_label in label_rivalries:
                rival_info = label_rivalries[top_rival_label]
                if isinstance(rival_info, dict):
                    rival_contrast_score_val = score_sample_against_rival_contrast_facts(
                        raw_x, rival_info, feature_names_list, matched_key_facts, failed_key_facts
                    )

    # Store fact and rival contrast evidence for manifest writers.
    brain["_last_fact_meta"] = {
        "fact_score": float(best_row.get("fact_score", 0.0)),
        "fact_penalty": float(best_row.get("fact_penalty", 0.0)),
        "rival_contrast_score": float(rival_contrast_score_val),
        "top_rival_label": top_rival_label,
        "matched_key_facts": matched_key_facts[:8],
        "failed_key_facts": failed_key_facts[:8],
        "fact_profile_strength": str(
            (brain.get("category_fact_profiles", {}) or {})
            .get(predicted_label, {})
            .get("fact_profile_strength", "unknown")
            if isinstance(brain.get("category_fact_profiles"), dict)
            else "unknown"
        ),
        "rival_second_pass_swapped": str(bool(rival_second_pass_meta.get("swapped", False))),
        "rival_second_pass_from_label": str(rival_second_pass_meta.get("from_label", "")),
        "rival_second_pass_to_label": str(rival_second_pass_meta.get("to_label", "")),
        "rival_second_pass_reason": str(rival_second_pass_meta.get("reason", "")),
        "rival_second_pass_score_gap": float(rival_second_pass_meta.get("score_gap", 0.0) or 0.0),
        "weak_profile_rival_swapped": bool(weak_rival_meta.get("swapped", False)),
        "weak_profile_rival_from_label": str(weak_rival_meta.get("from_label", predicted_label)),
        "weak_profile_rival_to_label": str(weak_rival_meta.get("to_label", predicted_label)),
        "weak_profile_rival_reason": str(weak_rival_meta.get("reason", "")),
        "weak_profile_rival_score_gap": float(weak_rival_meta.get("score_gap", 0.0) or 0.0),
        "committee_agreement_switched": bool(committee_meta.get("switched", False)),
        "committee_agreement_from_label": str(committee_meta.get("from_label", predicted_label)),
        "committee_agreement_to_label": str(committee_meta.get("to_label", predicted_label)),
        "committee_agreement_reason": str(committee_meta.get("reason", "")),
        "committee_agreement_advantage": float(committee_meta.get("advantage", 0.0) or 0.0),
        "committee_agreement_top": " | ".join(str(x) for x in list(committee_meta.get("top_committee", []) or [])[:5]),
        # v0.5.6: committee_agreement_score is the final chosen label's committee
        # score (0-1). When switched=True this is the challenger's score; when
        # switched=False it is the incumbent's score. Exposes whether the committee
        # had high or low confidence in its pick, independent of whether it switched.
        "committee_agreement_score": float(
            committee_meta.get("best_score", 0.0)
            if committee_meta.get("switched")
            else committee_meta.get("current_score", 0.0)
        )
        if isinstance(committee_meta, dict)
        else 0.0,
        # committee_votes is the per-member vote breakdown for the chosen label.
        # Format: "proto=0.72 memb=0.65 fact=0.58 router=0.71 rival=0.65 supp=0.80 depth=0.75 struct=0.60"
        # Each value is the normalized [0,1] vote that member cast for this label.
        "committee_votes": " ".join(
            f"{k[:5]}={float(v):.3f}"
            for k, v in sorted((committee_meta.get("chosen_votes", {}) or {}).items())
            if isinstance(v, (int, float)) and math.isfinite(float(v))
        )
        if isinstance(committee_meta, dict)
        else "",
        # committee_top_candidates: the ranked list of all scored candidates with
        # their breakdown — suitable for pasting into a spreadsheet for audit.
        "committee_top_candidates": " | ".join(str(x) for x in list(committee_meta.get("top_committee", []) or [])[:5]),
        "measured_structure": str(measured_structure_contract.get("observed_structure", "")),
        "measured_structure_confidence": str(measured_structure_contract.get("confidence", "")),
        "measured_structure_reason": str(measured_structure_contract.get("reason", "")),
        "loop_family_tiebreak": float(best_row.get("loop_family_tiebreak", 0.0) or 0.0),
        "short_low_drum_hit_adjustment": float(best_row.get("short_low_drum_hit_adjustment", 0.0) or 0.0),
        "short_low_drum_hit_reason": _short_low_drum_hit_signature(brain, fingerprint, duration_sec)[1],
        "coherent_group_penalty": float(best_row.get("coherent_group_penalty", 0.0) or 0.0),
        "coherent_group_reason": str(best_row.get("coherent_group_reason", "")),
    }

    brain["_last_committee_meta"] = (
        committee_meta if isinstance(committee_meta, dict) else {"mode": "committee_agreement", "reason": "bad_meta"}
    )

    # Top matches report the prediction contest after measured structure penalties.
    # Raw distances are still in reports, but top_match should answer Aaron's
    # practical debugging question: which labels were actually allowed to compete
    # after the 100-feature structure evidence was considered?
    top_rows = sorted(rows, key=lambda r: (float(r.get("current_score", r.get("raw_distance", 0.0))), str(r["label"])))[
        :8
    ]
    top5: List[Tuple[str, float]] = [
        (str(r["label"]), float(r.get("current_score", r.get("raw_distance", 0.0)))) for r in top_rows
    ]
    if predicted_label not in [label_name for label_name, _score in top5]:
        top5 = [(predicted_label, float(best_current_score))] + top5[:4]
    if isinstance(ensemble_meta, dict) and ensemble_meta.get("mode"):
        brain["_last_ensemble_meta"] = ensemble_meta
    try:
        top_sorted = sorted(
            (frontend_router_scores.get("top", {}) or {}).items(), key=lambda kv: (-float(kv[1]), str(kv[0]))
        )[:3]
        struct_sorted = sorted(
            (frontend_router_scores.get("structure", {}) or {}).items(), key=lambda kv: (-float(kv[1]), str(kv[0]))
        )[:3]
        parent_sorted = sorted(
            (frontend_router_scores.get("parent", {}) or {}).items(), key=lambda kv: (-float(kv[1]), str(kv[0]))
        )[:3]
        brain["_last_frontend_router_meta"] = {
            "mode": "frontend_router_brain",
            "top3_top": "; ".join(f"{k}:{float(v):.3f}" for k, v in top_sorted),
            "top3_structure": "; ".join(f"{k}:{float(v):.3f}" for k, v in struct_sorted),
            "top3_parent": "; ".join(f"{k}:{float(v):.3f}" for k, v in parent_sorted),
            "final_label_router_note": str(best_row.get("frontend_router_note", "")),
            "final_label_router_penalty": float(best_row.get("frontend_router_penalty", 0.0)),
        }
    except Exception:
        brain["_last_frontend_router_meta"] = {"mode": "frontend_router_brain", "error": "router_meta_failed"}
    return predicted_label, predicted_top, float(sim), float(margin), top5[:5]


def safe_folder_name(text: str) -> str:
    text = str(text or "Unknown").strip().replace("/", " ").replace("\\", " ")
    text = re.sub(r"[^\w .()#&+,'-]+", "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "Unknown"


def materialize_preview_audio(src: Path, dest: Path) -> str:
    """Create a lightweight preview artifact.

    Phase 3 runs can touch thousands of large samples. Preview folders are for
    listening/debugging only, so use symlinks by default instead of duplicating
    audio and filling Aaron's drive. If symlinks fail on a volume, fall back to
    copying so the run still completes.
    """
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        os.symlink(str(src), str(dest))
        return str(dest)
    except Exception:
        # v0.4.67: default to symlink-only so tests cannot silently duplicate
        # hundreds of MB/GB of audio. A real copy requires an explicit override.
        if os.environ.get("AARON_SOUND_SORTER_ALLOW_AUDIO_COPY", "0") != "1":
            return ""
        try:
            shutil.copy2(src, dest)
            return str(dest)
        except Exception:
            return ""


def gate_prediction(
    predicted_label: str,
    predicted_top: str,
    similarity: float,
    margin: float,
    read_status: str,
    duration_sec: float,
    min_similarity: float,
    min_margin: float,
    weak_similarity_floor: float = 0.20,
    strong_margin: float = 1.50,
    allow_low_similarity_clear_margin: bool = False,
) -> Tuple[str, str, str, str]:
    """Return final_top, final_label, confidence_status, decision_reason.

    v0.4.77 decisive product policy:
      - Broken/unreadable audio still goes to real _TO_REVIEW.
      - Usable audio gets the best learned-folder pick.
      - Weak similarity, narrow margins, and rival ambiguity become audit reasons
        in the manifest, not the main destination.

    This keeps the product useful as a sorter while still exposing uncertainty.
    """
    if read_status != "ok":
        return "_TO_REVIEW", "Unreadable or Broken", "review", f"read_status={read_status}"

    try:
        sim = float(similarity or 0.0)
        gap = float(margin or 0.0)
    except Exception:
        sim = 0.0
        gap = 0.0

    # Only truly meaningless matches go to real review. Everything else is a
    # forced best-pick with an audit reason. Later safety gates may turn that
    # audited pick into review when the physics/family evidence is unsafe.
    no_match_floor = max(0.08, float(weak_similarity_floor or 0.20) * 0.50)
    if sim < no_match_floor and gap < 0.25:
        return (
            "_TO_REVIEW",
            "No Meaningful Match",
            "review",
            (f"no_meaningful_match similarity {sim:.3f} < {no_match_floor:.3f} and margin {gap:.3f} < 0.250"),
        )

    if sim < float(min_similarity):
        return (
            predicted_top,
            predicted_label,
            "auto_place",
            (
                f"AUDIT weak_pick: similarity {sim:.3f} < {float(min_similarity):.3f}; "
                f"forced_best_learned_folder_pick margin={gap:.3f}"
            ),
        )
    if gap < float(min_margin):
        return (
            predicted_top,
            predicted_label,
            "auto_place",
            (
                f"AUDIT ambiguous_pick: margin {gap:.3f} < {float(min_margin):.3f}; "
                f"forced_best_learned_folder_pick similarity={sim:.3f}"
            ),
        )
    return predicted_top, predicted_label, "auto_place", (f"strong_pick similarity={sim:.3f} margin={gap:.3f}")


def write_committee_agreement_report(manifest_rows: List[Dict[str, str]], reports_dir: Path) -> Path:
    """Write a dedicated report of every file where the committee acted.

    The committee columns are present in every manifest row, but this report
    extracts only the rows where the committee did something notable:
      - switched the label (highest value — the committee overrode the raw brain)
      - confirmed a low-confidence pick (score < 0.40)
      - kept a high-confidence pick (score >= 0.70, good validation)

    Columns are exactly the user-requested set plus the raw-brain context so you
    can see what the committee changed and why.  The report is sorted: switches
    first (most important for debugging), then low-confidence confirmations, then
    high-confidence confirmations.
    """
    out_rows: List[Dict[str, str]] = []
    for r in manifest_rows:
        switched = str(r.get("committee_agreement_switched", "False")).lower() in ("true", "1", "yes")
        score_str = str(r.get("committee_agreement_score", "0.0"))
        try:
            score = float(score_str)
        except (ValueError, TypeError):
            score = 0.0
        adv_str = str(r.get("committee_agreement_advantage", "0.0"))
        try:
            advantage = float(adv_str)
        except (ValueError, TypeError):
            advantage = 0.0

        if switched:
            action_type = "SWITCHED"
        elif score < 0.40:
            action_type = "LOW_CONFIDENCE_CONFIRMED"
        elif score >= 0.70:
            action_type = "HIGH_CONFIDENCE_CONFIRMED"
        else:
            continue  # skip routine mid-range confirmations to keep report focused

        out_rows.append(
            {
                "action_type": action_type,
                "source_path": str(r.get("source_path", "")),
                "final_label": str(r.get("final_label", r.get("raw_predicted_label", ""))),
                "final_top": str(r.get("final_top", "")),
                "confidence_status": str(r.get("confidence_status", "")),
                "committee_agreement_score": f"{score:.6f}",
                "committee_agreement_switched": str(switched),
                "committee_agreement_from_label": str(r.get("committee_agreement_from_label", "")),
                "committee_agreement_to_label": str(r.get("committee_agreement_to_label", "")),
                "committee_agreement_reason": str(r.get("committee_agreement_reason", "")),
                "committee_agreement_advantage": f"{advantage:.6f}",
                "committee_votes": str(r.get("committee_votes", "")),
                "committee_top_candidates": str(r.get("committee_top_candidates", ""))[:400],
                "raw_predicted_label": str(r.get("raw_predicted_label", r.get("forced_guess_label", ""))),
                "similarity": str(r.get("similarity", "")),
                "margin_distance_gap": str(r.get("margin_distance_gap", "")),
                "fact_score": str(r.get("fact_score", "")),
                "rival_contrast_score": str(r.get("rival_contrast_score", "")),
                "matched_key_facts": str(r.get("matched_key_facts", "")),
                "failed_key_facts": str(r.get("failed_key_facts", "")),
                "review_reason": str(r.get("review_reason", "")),
            }
        )

    # Sort: switches first (most important), then low-confidence, then high-confidence.
    action_order = {"SWITCHED": 0, "LOW_CONFIDENCE_CONFIRMED": 1, "HIGH_CONFIDENCE_CONFIRMED": 2}
    out_rows.sort(
        key=lambda row: (
            action_order.get(str(row.get("action_type", "")), 9),
            -abs(float(row.get("committee_agreement_advantage", "0.0") or 0.0)),
            str(row.get("source_path", "")),
        )
    )

    fields = [
        "action_type",
        "source_path",
        "final_label",
        "final_top",
        "confidence_status",
        "committee_agreement_score",
        "committee_agreement_switched",
        "committee_agreement_from_label",
        "committee_agreement_to_label",
        "committee_agreement_reason",
        "committee_agreement_advantage",
        "committee_votes",
        "committee_top_candidates",
        "raw_predicted_label",
        "similarity",
        "margin_distance_gap",
        "fact_score",
        "rival_contrast_score",
        "matched_key_facts",
        "failed_key_facts",
        "review_reason",
    ]
    out = reports_dir / "committee_agreement_decisions.csv"
    write_csv(out, out_rows, fields)
    switched_count = sum(1 for r in out_rows if r["action_type"] == "SWITCHED")
    low_conf_count = sum(1 for r in out_rows if r["action_type"] == "LOW_CONFIDENCE_CONFIRMED")
    high_conf_count = sum(1 for r in out_rows if r["action_type"] == "HIGH_CONFIDENCE_CONFIRMED")
    print(
        f"Committee report: {switched_count} switches, {low_conf_count} low-confidence confirmations, "
        f"{high_conf_count} high-confidence confirmations -> {out}"
    )
    return out


def write_gate_sweep_report(
    brain: dict,
    prediction_records: List[Dict[str, object]],
    reports_dir: Path,
    allow_low_similarity_clear_margin: bool = False,
) -> Dict[str, object]:
    """Report-only threshold sweep for pure-brain gate calibration.

    This does not change placements. It shows how coverage/accuracy would move
    under several similarity/margin settings while keeping the same learned
    structure/top conflict gates. This is the safe way to tune Phase 3 without
    adding semantic label hacks.
    """
    sim_values = [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65]
    margin_values = [0.40, 0.50, 0.60, 0.80, 1.00, 1.25, 1.50]
    rows: List[Dict[str, str]] = []
    total = len(prediction_records)

    for sim_gate in sim_values:
        for margin_gate in margin_values:
            auto = 0
            review = 0
            label_pass = 0
            top_pass = 0
            label_fail = 0
            top_fail = 0
            for rec in prediction_records:
                final_top, final_label, confidence_status, review_reason = gate_prediction(
                    str(rec.get("pred_label", "")),
                    str(rec.get("pred_top", "")),
                    float(rec.get("similarity", 0.0) or 0.0),
                    float(rec.get("margin", 0.0) or 0.0),
                    str(rec.get("read_status", "")),
                    float(rec.get("duration_sec", 0.0) or 0.0),
                    sim_gate,
                    margin_gate,
                    0.20,
                    1.50,
                    allow_low_similarity_clear_margin,
                )
                final_top, final_label, confidence_status, review_reason = apply_learned_conflict_gates(
                    brain,
                    rec.get("fingerprint", []),
                    str(rec.get("pred_label", "")),
                    str(rec.get("pred_top", "")),
                    final_top,
                    final_label,
                    confidence_status,
                    review_reason,
                    float(rec.get("similarity", 0.0) or 0.0),
                    float(rec.get("margin", 0.0) or 0.0),
                    float(rec.get("duration_sec", 0.0) or 0.0),
                    rec.get("top5", []),
                )
                if confidence_status == "auto_place":
                    auto += 1
                    if final_label == rec.get("expected_label"):
                        label_pass += 1
                    else:
                        label_fail += 1
                    if final_top == rec.get("expected_top"):
                        top_pass += 1
                    else:
                        top_fail += 1
                else:
                    review += 1
            rows.append(
                {
                    "min_similarity": f"{sim_gate:.3f}",
                    "min_margin": f"{margin_gate:.3f}",
                    "total_eval_files": str(total),
                    "auto_placed": str(auto),
                    "sent_to_review": str(review),
                    "auto_place_correct": str(label_pass),
                    "auto_place_wrong": str(label_fail),
                    "auto_place_precision": f"{(label_pass / auto) if auto else 0.0:.6f}",
                    "auto_place_label_pass": str(label_pass),
                    "auto_place_label_fail": str(label_fail),
                    "auto_place_label_accuracy": f"{(label_pass / auto) if auto else 0.0:.6f}",
                    "auto_place_top_pass": str(top_pass),
                    "auto_place_top_fail": str(top_fail),
                    "auto_place_top_accuracy": f"{(top_pass / auto) if auto else 0.0:.6f}",
                }
            )

    fields = [
        "min_similarity",
        "min_margin",
        "total_eval_files",
        "auto_placed",
        "sent_to_review",
        "auto_place_correct",
        "auto_place_wrong",
        "auto_place_precision",
        "auto_place_label_pass",
        "auto_place_label_fail",
        "auto_place_label_accuracy",
        "auto_place_top_pass",
        "auto_place_top_fail",
        "auto_place_top_accuracy",
    ]
    write_csv(reports_dir / "pure_brain_gate_sweep_report_only.csv", rows, fields)

    zero_fail = [
        r
        for r in rows
        if int(r["auto_placed"]) > 0 and int(r["auto_place_label_fail"]) == 0 and int(r["auto_place_top_fail"]) == 0
    ]
    best_zero = max(zero_fail, key=lambda r: int(r["auto_placed"])) if zero_fail else {}
    return {
        "report": str(reports_dir / "pure_brain_gate_sweep_report_only.csv"),
        "best_zero_failure_gate_report_only": best_zero,
        "policy": "report-only; placements are not changed by this sweep",
    }


def copy_sorted_preview_file(
    source_path: str,
    sorted_root: Path,
    structure: str,
    final_top: str,
    final_label: str,
    eval_index: int,
    expected_label: str,
    predicted_label: str,
) -> str:
    """Copy one eval file into the visible sorted preview folder.

    v0.4.5 uses normal producer folders directly:
      Drums/.../Kick/One Shots
      Drums/.../Kick/Loops

    There are no top-level _ONE_SHOTS or _LOOPS preview lanes.  If a file goes
    to review, the review folder includes the brain's top guess.
    """
    src = Path(source_path)
    if not src.exists() and not src.is_symlink():
        return ""

    if final_top == "_TO_REVIEW":
        guess = public_label(predicted_label)
        review_name = f"guess_{safe_folder_name(guess)}__{safe_folder_name(public_label(final_label))}"
        dest_dir = sorted_root / "_TO_REVIEW" / review_name
    else:
        parts = folder_parts_for_public_label(final_label)
        if parts:
            dest_dir = sorted_root.joinpath(*parts)
        else:
            dest_dir = sorted_root / safe_folder_name(final_top) / safe_folder_name(public_label(final_label))

    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower() or ".wav"
    original_safe = safe_folder_name(src.stem)[:52]
    digest = hashlib.sha1(
        (str(src) + "|" + expected_label + "|" + predicted_label).encode("utf-8", errors="replace")
    ).hexdigest()[:10]
    # Keep the filename short. Expected/predicted labels are already in CSV reports and folder paths.
    dest = dest_dir / f"eval_{eval_index:05d}__{digest}__{original_safe}{suffix}"
    i = 2
    base = dest
    while dest.exists():
        dest = base.with_name(base.stem + f"__{i}" + base.suffix)
        i += 1
    try:
        return materialize_preview_audio(src, dest)
    except Exception:
        return ""


def copy_training_preview(
    train_features: List[FeatureRow], run_dir: Path, reports_dir: Path, max_per_label: int = 3
) -> None:
    """Copy a small visible sample of training rows by structure lane and label.

    This is not used for training. It lets Aaron audit what the brain actually learned from.
    """
    preview_root = run_dir / "training_preview"
    counts = Counter()
    rows = []
    for idx, row in enumerate(train_features, 1):
        key = row.label
        if counts[key] >= max_per_label:
            continue
        src = Path(row.path)
        if not src.exists() and not src.is_symlink():
            continue
        top = top_for_public_label(row.label)
        parts = folder_parts_for_public_label(row.label)
        dest_dir = preview_root
        if parts:
            dest_dir = dest_dir.joinpath(*parts)
        else:
            dest_dir = dest_dir / safe_folder_name(top) / safe_folder_name(public_label(row.label))
        dest_dir.mkdir(parents=True, exist_ok=True)
        suffix = src.suffix.lower() or ".wav"
        dest = dest_dir / f"train_{idx:04d}__{safe_folder_name(src.stem)[:90]}{suffix}"
        try:
            made = materialize_preview_audio(src, dest)
            if not made:
                continue
            counts[key] += 1
            rows.append(
                {
                    "source_path": row.path,
                    "preview_copy": str(dest),
                    "structure": row.structure,
                    "internal_label": row.label,
                    "public_label": public_label(row.label),
                    "top": top,
                    "group_key": row.group_key,
                    "duration_sec": f"{row.duration_sec:.6f}",
                    "physics_tags": fingerprint_physics_tags(row.fingerprint, row.duration_sec),
                    "physics_summary": fingerprint_physics_summary(row.fingerprint, row.duration_sec),
                    **fingerprint_physics_dict(row.fingerprint, row.duration_sec),
                }
            )
        except Exception:
            pass
    if rows:
        write_csv(
            reports_dir / "training_preview_manifest.csv",
            rows,
            [
                "source_path",
                "preview_copy",
                "structure",
                "internal_label",
                "public_label",
                "top",
                "group_key",
                "duration_sec",
                *PHYSICS_REPORT_FIELDS,
            ],
        )
        lines = []
        for p in sorted(preview_root.rglob("*")):
            rel = p.relative_to(preview_root)
            depth = len(rel.parts) - 1
            indent = "  " * depth
            suffix = "/" if p.is_dir() else ""
            lines.append(f"{indent}{p.name}{suffix}")
        (reports_dir / "training_preview_tree.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sorted_preview_tree(sorted_root: Path, reports_dir: Path) -> None:
    lines: List[str] = []
    if not sorted_root.exists():
        return
    for p in sorted(sorted_root.rglob("*")):
        rel = p.relative_to(sorted_root)
        depth = len(rel.parts) - 1
        indent = "  " * depth
        suffix = "/" if p.is_dir() else ""
        lines.append(f"{indent}{p.name}{suffix}")
    (reports_dir / "sorted_preview_tree.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(
    brain: dict,
    eval_features: List[FeatureRow],
    reports_dir: Path,
    sorted_root: Path,
    min_similarity: float,
    min_margin: float,
    weak_similarity_floor: float = 0.20,
    strong_margin: float = 1.50,
    allow_low_similarity_clear_margin: bool = False,
) -> dict:
    rows = []
    label_total = Counter()
    raw_label_pass = Counter()
    final_auto_label_pass = Counter()
    top_total = Counter()
    raw_top_pass = Counter()
    final_top_pass = Counter()
    failure_pairs = Counter()
    final_failure_pairs = Counter()
    review_reasons = Counter()
    auto_count = 0
    review_count = 0
    prediction_records: List[Dict[str, object]] = []
    active_brain_labels = set(str(x) for x in brain.get("labels", []))
    active_eval_total = 0
    active_raw_label_good = 0
    active_auto_count = 0
    active_auto_label_good = 0
    inactive_eval_total = 0
    inactive_auto_count = 0

    sorted_root.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(eval_features, 1):
        expected_label_active = row.label in active_brain_labels
        if expected_label_active:
            active_eval_total += 1
        else:
            inactive_eval_total += 1
        label_total[row.label] += 1
        top_total[row.top] += 1
        if row.read_status != "ok":
            pred_label = "review_unreadable"
            pred_top = "_TO_REVIEW"
            sim = 0.0
            margin = 0.0
            top5 = []
            fact_meta = {}
        else:
            pred_label, pred_top, sim, margin, top5 = predict(brain, row.fingerprint, "", row.duration_sec)
            fact_meta = brain.get("_last_fact_meta", {}) if isinstance(brain.get("_last_fact_meta", {}), dict) else {}

        raw_label_ok = pred_label == row.label
        raw_top_ok = pred_top == row.top
        if raw_label_ok:
            raw_label_pass[row.label] += 1
            if expected_label_active:
                active_raw_label_good += 1
        else:
            failure_pairs[(row.label, pred_label)] += 1
        if raw_top_ok:
            raw_top_pass[row.top] += 1

        final_top, final_label, confidence_status, review_reason = gate_prediction(
            pred_label,
            pred_top,
            sim,
            margin,
            row.read_status,
            row.duration_sec,
            min_similarity,
            min_margin,
            weak_similarity_floor,
            strong_margin,
            allow_low_similarity_clear_margin,
        )
        final_top, final_label, confidence_status, review_reason = apply_learned_conflict_gates(
            brain,
            row.fingerprint,
            pred_label,
            pred_top,
            final_top,
            final_label,
            confidence_status,
            review_reason,
            sim,
            margin,
            row.duration_sec,
            top5,
        )
        stage_diag = prediction_stage_diagnostics(brain, row.fingerprint, pred_label)
        rel = brain.get("label_reliability_by_label", {})
        rel_info = rel.get(pred_label, {}) if isinstance(rel, dict) else {}
        if not isinstance(rel_info, dict):
            rel_info = {}
        nearest_predicted_examples = format_nearest_training_examples(
            nearest_training_examples_for_label(brain, pred_label, row.fingerprint, limit=3)
        )
        nearest_expected_examples = format_nearest_training_examples(
            nearest_training_examples_for_label(brain, row.label, row.fingerprint, limit=3)
        )
        current_physics = {
            "physics_tags": fingerprint_physics_tags(row.fingerprint, row.duration_sec),
            "physics_summary": fingerprint_physics_summary(row.fingerprint, row.duration_sec),
            **fingerprint_physics_dict(row.fingerprint, row.duration_sec),
        }
        if confidence_status == "auto_place":
            auto_count += 1
            if expected_label_active:
                active_auto_count += 1
            else:
                inactive_auto_count += 1
            if final_label == row.label:
                final_auto_label_pass[row.label] += 1
                if expected_label_active:
                    active_auto_label_good += 1
            else:
                final_failure_pairs[(row.label, final_label)] += 1
            if final_top == row.top:
                final_top_pass[row.top] += 1
        else:
            review_count += 1
            review_reasons[review_reason] += 1

        preview_copy = copy_sorted_preview_file(
            row.path, sorted_root, row.structure, final_top, final_label, idx, row.label, pred_label
        )

        sanity_warnings = fingerprint_sanity_warnings(pred_label, row.fingerprint)
        matched_key_facts = fact_meta.get("matched_key_facts", []) if isinstance(fact_meta, dict) else []
        failed_key_facts = fact_meta.get("failed_key_facts", []) if isinstance(fact_meta, dict) else []
        if not isinstance(matched_key_facts, (list, tuple)):
            matched_key_facts = [str(matched_key_facts)] if str(matched_key_facts) else []
        if not isinstance(failed_key_facts, (list, tuple)):
            failed_key_facts = [str(failed_key_facts)] if str(failed_key_facts) else []

        status = "PASS" if raw_label_ok else "FAIL"
        final_status = (
            "AUTO_PASS"
            if confidence_status == "auto_place" and final_label == row.label
            else ("AUTO_FAIL" if confidence_status == "auto_place" else "REVIEW")
        )

        rows.append(
            {
                "status": status,
                "final_status": final_status,
                "confidence_status": confidence_status,
                "review_reason": review_reason,
                "sanity_warnings_report_only": sanity_warnings,
                "eval_index": str(idx),
                "source_path": row.path,
                "preview_copy_path": preview_copy,
                "group_key": row.group_key,
                "eval_reused_training_source": "1" if row.eval_reused_training_source else "0",
                "eval_selection_method": row.eval_selection_method,
                "sampling_pool_size": str(row.sampling_pool_size or ""),
                "expected_structure": row.structure,
                "expected_top": row.top,
                "expected_internal_label": row.label,
                "expected_label": public_label(row.label),
                "expected_label_active_in_brain": "1" if expected_label_active else "0",
                "expected_label_training_status": "active" if expected_label_active else "inactive_or_discovery_only",
                "raw_predicted_top": pred_top,
                "raw_predicted_internal_label": pred_label,
                "raw_predicted_label": public_label(pred_label),
                "final_top": final_top,
                "final_internal_label": final_label,
                "final_label": public_label(final_label),
                "predicted_label_training_count": str(rel_info.get("training_count", "")),
                "predicted_label_support_tier": str(rel_info.get("support_tier", "")),
                "fact_score": f"{float(fact_meta.get('fact_score', 0.0) or 0.0):.6f}"
                if isinstance(fact_meta, dict)
                else "0.000000",
                "fact_penalty": f"{float(fact_meta.get('fact_penalty', 0.0) or 0.0):.6f}"
                if isinstance(fact_meta, dict)
                else "0.000000",
                "coherent_group_penalty": f"{float(fact_meta.get('coherent_group_penalty', 0.0) or 0.0):.6f}"
                if isinstance(fact_meta, dict)
                else "0.000000",
                "coherent_group_reason": str(
                    fact_meta.get("coherent_group_reason", "") if isinstance(fact_meta, dict) else ""
                ),
                "rival_contrast_score": f"{float(fact_meta.get('rival_contrast_score', 0.0) or 0.0):.6f}"
                if isinstance(fact_meta, dict)
                else "0.000000",
                "top_rival_internal_label": str(
                    fact_meta.get("top_rival_label", "") if isinstance(fact_meta, dict) else ""
                ),
                "top_rival_label": public_label(
                    str(fact_meta.get("top_rival_label", "") if isinstance(fact_meta, dict) else "")
                ),
                "matched_key_facts": "; ".join(str(x) for x in matched_key_facts if str(x)),
                "failed_key_facts": "; ".join(str(x) for x in failed_key_facts if str(x)),
                "fact_profile_strength": str(
                    fact_meta.get("fact_profile_strength", "") if isinstance(fact_meta, dict) else ""
                ),
                "rival_second_pass_swapped": str(
                    fact_meta.get("rival_second_pass_swapped", "") if isinstance(fact_meta, dict) else ""
                ),
                "rival_second_pass_from_label": str(
                    fact_meta.get("rival_second_pass_from_label", "") if isinstance(fact_meta, dict) else ""
                ),
                "rival_second_pass_to_label": str(
                    fact_meta.get("rival_second_pass_to_label", "") if isinstance(fact_meta, dict) else ""
                ),
                "rival_second_pass_reason": str(
                    fact_meta.get("rival_second_pass_reason", "") if isinstance(fact_meta, dict) else ""
                ),
                "rival_second_pass_score_gap": f"{float(fact_meta.get('rival_second_pass_score_gap', 0.0) or 0.0):.6f}"
                if isinstance(fact_meta, dict)
                else "0.000000",
                "weak_profile_rival_swapped": str(
                    fact_meta.get("weak_profile_rival_swapped", "") if isinstance(fact_meta, dict) else ""
                ),
                "weak_profile_rival_from_label": str(
                    fact_meta.get("weak_profile_rival_from_label", "") if isinstance(fact_meta, dict) else ""
                ),
                "weak_profile_rival_to_label": str(
                    fact_meta.get("weak_profile_rival_to_label", "") if isinstance(fact_meta, dict) else ""
                ),
                "weak_profile_rival_reason": str(
                    fact_meta.get("weak_profile_rival_reason", "") if isinstance(fact_meta, dict) else ""
                ),
                "weak_profile_rival_score_gap": f"{float(fact_meta.get('weak_profile_rival_score_gap', 0.0) or 0.0):.6f}"
                if isinstance(fact_meta, dict)
                else "0.000000",
                "committee_agreement_score": f"{float(fact_meta.get('committee_agreement_score', 0.0) or 0.0):.6f}"
                if isinstance(fact_meta, dict)
                else "0.000000",
                "committee_agreement_switched": str(
                    fact_meta.get("committee_agreement_switched", False) if isinstance(fact_meta, dict) else False
                ),
                "committee_agreement_from_label": public_label(
                    str(fact_meta.get("committee_agreement_from_label", "") if isinstance(fact_meta, dict) else "")
                ),
                "committee_agreement_to_label": public_label(
                    str(fact_meta.get("committee_agreement_to_label", "") if isinstance(fact_meta, dict) else "")
                ),
                "committee_agreement_reason": str(
                    fact_meta.get("committee_agreement_reason", "") if isinstance(fact_meta, dict) else ""
                ),
                "committee_agreement_advantage": f"{float(fact_meta.get('committee_agreement_advantage', 0.0) or 0.0):.6f}"
                if isinstance(fact_meta, dict)
                else "0.000000",
                "committee_votes": str(fact_meta.get("committee_votes", "") if isinstance(fact_meta, dict) else ""),
                "committee_top_candidates": str(
                    fact_meta.get("committee_top_candidates", "") if isinstance(fact_meta, dict) else ""
                ),
                "similarity": f"{sim:.6f}",
                "margin_distance_gap": f"{margin:.6f}",
                **stage_diag,
                "read_status": row.read_status,
                "duration_sec": f"{row.duration_sec:.6f}",
                **current_physics,
                "nearest_predicted_training_examples": nearest_predicted_examples,
                "nearest_expected_training_examples": nearest_expected_examples,
                "top_match_1": public_label(top5[0][0]) if len(top5) > 0 else "",
                "top_match_1_internal": top5[0][0] if len(top5) > 0 else "",
                "top_match_1_dist": f"{top5[0][1]:.6f}" if len(top5) > 0 else "",
                "top_match_2": public_label(top5[1][0]) if len(top5) > 1 else "",
                "top_match_2_internal": top5[1][0] if len(top5) > 1 else "",
                "top_match_2_dist": f"{top5[1][1]:.6f}" if len(top5) > 1 else "",
                "top_match_3": public_label(top5[2][0]) if len(top5) > 2 else "",
                "top_match_3_internal": top5[2][0] if len(top5) > 2 else "",
                "top_match_3_dist": f"{top5[2][1]:.6f}" if len(top5) > 2 else "",
                "top_match_4": public_label(top5[3][0]) if len(top5) > 3 else "",
                "top_match_4_internal": top5[3][0] if len(top5) > 3 else "",
                "top_match_4_dist": f"{top5[3][1]:.6f}" if len(top5) > 3 else "",
                "top_match_5": public_label(top5[4][0]) if len(top5) > 4 else "",
                "top_match_5_internal": top5[4][0] if len(top5) > 4 else "",
                "top_match_5_dist": f"{top5[4][1]:.6f}" if len(top5) > 4 else "",
            }
        )
        prediction_records.append(
            {
                "expected_label": row.label,
                "expected_top": row.top,
                "pred_label": pred_label,
                "pred_top": pred_top,
                "similarity": sim,
                "margin": margin,
                "read_status": row.read_status,
                "duration_sec": row.duration_sec,
                "fingerprint": row.fingerprint,
                "top5": top5,
            }
        )

    fields = [
        "status",
        "final_status",
        "confidence_status",
        "review_reason",
        "sanity_warnings_report_only",
        "eval_index",
        "source_path",
        "preview_copy_path",
        "group_key",
        "eval_reused_training_source",
        "eval_selection_method",
        "sampling_pool_size",
        "expected_structure",
        "expected_top",
        "expected_internal_label",
        "expected_label",
        "expected_label_active_in_brain",
        "expected_label_training_status",
        "raw_predicted_top",
        "raw_predicted_internal_label",
        "raw_predicted_label",
        "final_top",
        "final_internal_label",
        "final_label",
        "predicted_label_training_count",
        "predicted_label_support_tier",
        "fact_score",
        "fact_penalty",
        "coherent_group_penalty",
        "coherent_group_reason",
        "rival_contrast_score",
        "top_rival_internal_label",
        "top_rival_label",
        "matched_key_facts",
        "failed_key_facts",
        "fact_profile_strength",
        "rival_second_pass_swapped",
        "rival_second_pass_from_label",
        "rival_second_pass_to_label",
        "rival_second_pass_reason",
        "rival_second_pass_score_gap",
        "weak_profile_rival_swapped",
        "weak_profile_rival_from_label",
        "weak_profile_rival_to_label",
        "weak_profile_rival_reason",
        "weak_profile_rival_score_gap",
        "committee_agreement_score",
        "committee_agreement_switched",
        "committee_agreement_from_label",
        "committee_agreement_to_label",
        "committee_agreement_reason",
        "committee_agreement_advantage",
        "committee_votes",
        "committee_top_candidates",
        "similarity",
        "margin_distance_gap",
        "learned_top_1",
        "learned_top_1_score",
        "learned_top_2",
        "learned_top_2_score",
        "learned_top_gap",
        "learned_top_gap_ratio",
        "learned_structure",
        "learned_structure_score",
        "label_structure",
        "label_structure_score",
        "structure_gap_raw",
        "structure_gap_normalized",
        "read_status",
        "duration_sec",
        *PHYSICS_REPORT_FIELDS,
        "nearest_predicted_training_examples",
        "nearest_expected_training_examples",
        "top_match_1",
        "top_match_1_internal",
        "top_match_1_dist",
        "top_match_2",
        "top_match_2_internal",
        "top_match_2_dist",
        "top_match_3",
        "top_match_3_internal",
        "top_match_3_dist",
        "top_match_4",
        "top_match_4_internal",
        "top_match_4_dist",
        "top_match_5",
        "top_match_5_internal",
        "top_match_5_dist",
    ]
    write_csv(reports_dir / "pure_brain_eval_results.csv", rows, fields)
    write_csv(reports_dir / "pure_brain_eval_failures_only.csv", [r for r in rows if r["status"] != "PASS"], fields)
    write_csv(
        reports_dir / "pure_brain_confusion_pairs_by_feature.csv", [r for r in rows if r["status"] != "PASS"], fields
    )
    write_csv(
        reports_dir / "pure_brain_auto_place_failures_only.csv",
        [r for r in rows if r["final_status"] == "AUTO_FAIL"],
        fields,
    )
    write_csv(reports_dir / "pure_brain_review_only.csv", [r for r in rows if r["final_status"] == "REVIEW"], fields)

    label_rows = []
    reliability = brain.get("label_reliability_by_label", {})
    for label in sorted(label_total):
        total = label_total[label]
        raw_passed = raw_label_pass[label]
        auto_passed = final_auto_label_pass[label]
        auto_total = sum(1 for r in rows if r["expected_label"] == label and r["confidence_status"] == "auto_place")
        rel_info = reliability.get(label, {}) if isinstance(reliability, dict) else {}
        if not isinstance(rel_info, dict):
            rel_info = {}
        label_rows.append(
            {
                "expected_internal_label": label,
                "expected_label": public_label(label),
                "expected_structure": label_default_structure(label),
                "training_count": str(rel_info.get("training_count", "")),
                "support_tier": str(rel_info.get("support_tier", "")),
                "intra_spread_mean": f"{float(rel_info.get('intra_spread_mean', 0.0) or 0.0):.6f}",
                "intra_spread_p90": f"{float(rel_info.get('intra_spread_p90', 0.0) or 0.0):.6f}",
                "intra_spread_relative_to_top_mean": f"{float(rel_info.get('intra_spread_relative_to_top_mean', 0.0) or 0.0):.6f}",
                "raw_passed": str(raw_passed),
                "total": str(total),
                "raw_accuracy": f"{(raw_passed / total) if total else 0:.4f}",
                "auto_placed": str(auto_total),
                "auto_passed": str(auto_passed),
                "auto_accuracy": f"{(auto_passed / auto_total) if auto_total else 0:.4f}",
                "reviewed": str(total - auto_total),
            }
        )
    write_csv(
        reports_dir / "pure_brain_accuracy_by_label.csv",
        label_rows,
        [
            "expected_structure",
            "expected_internal_label",
            "expected_label",
            "training_count",
            "support_tier",
            "intra_spread_mean",
            "intra_spread_p90",
            "intra_spread_relative_to_top_mean",
            "raw_passed",
            "total",
            "raw_accuracy",
            "auto_placed",
            "auto_passed",
            "auto_accuracy",
            "reviewed",
        ],
    )

    support_risk_rows = []
    for label in sorted(label_total):
        label_eval_rows = [r for r in rows if r.get("expected_internal_label") == label]
        holdout_count = sum(1 for r in label_eval_rows if r.get("eval_reused_training_source") != "1")
        reused_count = sum(1 for r in label_eval_rows if r.get("eval_reused_training_source") == "1")
        auto_fail_count = sum(1 for r in label_eval_rows if r.get("final_status") == "AUTO_FAIL")
        auto_place_count = sum(1 for r in label_eval_rows if r.get("confidence_status") == "auto_place")
        low_support_reviews = sum(
            1 for r in label_eval_rows if str(r.get("review_reason", "")).startswith("Low Label Support")
        )
        rel_info = reliability.get(label, {}) if isinstance(reliability, dict) else {}
        if not isinstance(rel_info, dict):
            rel_info = {}
        risk_flags = []
        train_count = int(rel_info.get("training_count", 0) or 0)
        if train_count <= MEDIUM_LABEL_COUNT:
            risk_flags.append("low_training_support")
        spread_rel = float(rel_info.get("intra_spread_relative_to_top_mean", 0.0) or 0.0)
        if spread_rel >= 1.35:
            risk_flags.append("wide_label_spread")
        isolation_score = float(rel_info.get("label_isolation_score", 0.0) or 0.0)
        boundary_fraction = float(rel_info.get("boundary_conflict_fraction", 0.0) or 0.0)
        if isolation_score >= 0.90:
            risk_flags.append("crowded_label_space")
        if boundary_fraction >= 0.20:
            risk_flags.append("many_cross_boundary_teachers")
        if reused_count and not holdout_count:
            risk_flags.append("no_true_holdout")
        elif reused_count:
            risk_flags.append("mixed_holdout_and_reused_eval")
        if auto_fail_count:
            risk_flags.append("auto_place_failure")
        if low_support_reviews:
            risk_flags.append("low_support_reviews")
        support_risk_rows.append(
            {
                "expected_internal_label": label,
                "expected_label": public_label(label),
                "top": top_for_public_label(label),
                "structure": label_default_structure(label),
                "training_count": str(train_count),
                "support_tier": str(rel_info.get("support_tier", "")),
                "intra_spread_relative_to_top_mean": f"{spread_rel:.6f}",
                "label_isolation_score": f"{isolation_score:.6f}",
                "boundary_conflict_fraction": f"{boundary_fraction:.6f}",
                "most_common_nearest_other_label": str(rel_info.get("most_common_nearest_other_label", "")),
                "eval_total": str(len(label_eval_rows)),
                "holdout_eval": str(holdout_count),
                "reused_training_eval": str(reused_count),
                "auto_placed": str(auto_place_count),
                "auto_place_failures": str(auto_fail_count),
                "low_support_reviews": str(low_support_reviews),
                "risk_flags": ";".join(risk_flags),
            }
        )
    write_csv(
        reports_dir / "pure_brain_label_support_risk_report.csv",
        support_risk_rows,
        [
            "expected_internal_label",
            "expected_label",
            "top",
            "structure",
            "training_count",
            "support_tier",
            "intra_spread_relative_to_top_mean",
            "label_isolation_score",
            "boundary_conflict_fraction",
            "most_common_nearest_other_label",
            "eval_total",
            "holdout_eval",
            "reused_training_eval",
            "auto_placed",
            "auto_place_failures",
            "low_support_reviews",
            "risk_flags",
        ],
    )

    failure_rows = []
    for (exp, pred), count in failure_pairs.most_common():
        failure_rows.append(
            {
                "expected_internal_label": exp,
                "expected_label": public_label(exp),
                "raw_predicted_internal_label": pred,
                "raw_predicted_label": public_label(pred),
                "count": str(count),
            }
        )
    write_csv(
        reports_dir / "pure_brain_failure_pairs.csv",
        failure_rows,
        ["expected_internal_label", "expected_label", "raw_predicted_internal_label", "raw_predicted_label", "count"],
    )

    confusion_summary_rows = []
    confusion_groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        exp = r.get("expected_internal_label", "")
        pred = r.get("raw_predicted_internal_label", "")
        if exp and pred and exp != pred:
            confusion_groups[(exp, pred)].append(r)
    for (exp, pred), group in sorted(confusion_groups.items(), key=lambda item: len(item[1]), reverse=True):
        tag_counter = Counter()
        review_counter = Counter()
        final_counter = Counter()
        example_files = []
        sims = []
        margins = []
        for r in group:
            tag_counter.update([t for t in str(r.get("physics_tags", "")).split(";") if t])
            review_counter[str(r.get("review_reason", ""))] += 1
            final_counter[str(r.get("final_status", ""))] += 1
            try:
                sims.append(float(r.get("similarity", 0.0) or 0.0))
                margins.append(float(r.get("margin_distance_gap", 0.0) or 0.0))
            except Exception:
                pass
            if len(example_files) < 4:
                example_files.append(Path(str(r.get("source_path", ""))).name)
        exp_structure = (
            group[0].get("expected_structure", label_default_structure(exp)) if group else label_default_structure(exp)
        )
        pred_structure = label_default_structure(pred)
        structure_conflict = "1" if exp_structure != pred_structure else "0"
        confusion_summary_rows.append(
            {
                "expected_internal_label": exp,
                "expected_label": public_label(exp),
                "confused_with_internal_label": pred,
                "confused_with_label": public_label(pred),
                "count": str(len(group)),
                "expected_top": top_for_public_label(exp),
                "confused_with_top": top_for_public_label(pred),
                "expected_structure": exp_structure,
                "confused_with_structure": pred_structure,
                "structure_conflict": structure_conflict,
                "avg_similarity": f"{(sum(sims) / len(sims)) if sims else 0.0:.6f}",
                "avg_margin": f"{(sum(margins) / len(margins)) if margins else 0.0:.6f}",
                "final_status_counts": ";".join(f"{k}:{v}" for k, v in final_counter.most_common()),
                "top_review_reasons": " || ".join(f"{k}:{v}" for k, v in review_counter.most_common(3) if k),
                "top_physics_tags": " || ".join(f"{k}:{v}" for k, v in tag_counter.most_common(8)),
                "example_files": " || ".join(example_files),
            }
        )
    write_csv(
        reports_dir / "pure_brain_confusion_partner_summary.csv",
        confusion_summary_rows,
        [
            "expected_internal_label",
            "expected_label",
            "confused_with_internal_label",
            "confused_with_label",
            "count",
            "expected_top",
            "confused_with_top",
            "expected_structure",
            "confused_with_structure",
            "structure_conflict",
            "avg_similarity",
            "avg_margin",
            "final_status_counts",
            "top_review_reasons",
            "top_physics_tags",
            "example_files",
        ],
    )

    final_failure_rows = []
    for (exp, pred), count in final_failure_pairs.most_common():
        final_failure_rows.append(
            {
                "expected_internal_label": exp,
                "expected_label": public_label(exp),
                "final_internal_label": pred,
                "final_label": public_label(pred),
                "count": str(count),
            }
        )
    write_csv(
        reports_dir / "pure_brain_auto_place_failure_pairs.csv",
        final_failure_rows,
        ["expected_internal_label", "expected_label", "final_internal_label", "final_label", "count"],
    )

    review_rows = [{"review_reason": k, "count": str(v)} for k, v in review_reasons.most_common()]
    write_csv(reports_dir / "pure_brain_review_reason_counts.csv", review_rows, ["review_reason", "count"])
    gate_sweep_summary = write_gate_sweep_report(
        brain,
        prediction_records,
        reports_dir,
        allow_low_similarity_clear_margin=allow_low_similarity_clear_margin,
    )
    write_committee_agreement_report(rows, reports_dir)

    write_sorted_preview_tree(sorted_root, reports_dir)

    total = sum(label_total.values())
    raw_label_good = sum(raw_label_pass.values())
    raw_top_good = sum(raw_top_pass.values())
    auto_label_good = sum(final_auto_label_pass.values())
    auto_top_good = sum(final_top_pass.values())

    def subset_summary(subrows: List[Dict[str, str]]) -> Dict[str, object]:
        subset_total = len(subrows)
        subset_raw_label = sum(1 for r in subrows if r.get("status") == "PASS")
        subset_raw_top = sum(1 for r in subrows if r.get("raw_predicted_top") == r.get("expected_top"))
        subset_auto = [r for r in subrows if r.get("confidence_status") == "auto_place"]
        subset_auto_label = sum(
            1 for r in subset_auto if r.get("final_internal_label") == r.get("expected_internal_label")
        )
        subset_auto_top = sum(1 for r in subset_auto if r.get("final_top") == r.get("expected_top"))
        return {
            "total_eval_files": subset_total,
            "raw_label_pass": subset_raw_label,
            "raw_label_accuracy": (subset_raw_label / subset_total) if subset_total else 0.0,
            "raw_top_pass": subset_raw_top,
            "raw_top_accuracy": (subset_raw_top / subset_total) if subset_total else 0.0,
            "auto_placed": len(subset_auto),
            "sent_to_review": subset_total - len(subset_auto),
            "auto_place_label_pass": subset_auto_label,
            "auto_place_label_accuracy": (subset_auto_label / len(subset_auto)) if subset_auto else 0.0,
            "auto_place_top_pass": subset_auto_top,
            "auto_place_top_accuracy": (subset_auto_top / len(subset_auto)) if subset_auto else 0.0,
            "auto_place_failures": len(subset_auto) - subset_auto_label,
        }

    holdout_eval_rows = [r for r in rows if r.get("eval_reused_training_source") != "1"]
    reused_eval_rows = [r for r in rows if r.get("eval_reused_training_source") == "1"]
    active_expected_rows = [r for r in rows if r.get("expected_label_active_in_brain") == "1"]
    inactive_expected_rows = [r for r in rows if r.get("expected_label_active_in_brain") != "1"]
    holdout_summary = subset_summary(holdout_eval_rows)
    reused_summary = subset_summary(reused_eval_rows)
    active_expected_summary = subset_summary(active_expected_rows)
    inactive_expected_summary = subset_summary(inactive_expected_rows)
    summary = {
        "total_eval_files": total,
        "raw_label_pass": raw_label_good,
        "raw_label_accuracy": (raw_label_good / total) if total else 0.0,
        "raw_top_pass": raw_top_good,
        "raw_top_accuracy": (raw_top_good / total) if total else 0.0,
        "auto_placed": auto_count,
        "sent_to_review": review_count,
        "auto_place_label_pass": auto_label_good,
        "auto_place_label_accuracy": (auto_label_good / auto_count) if auto_count else 0.0,
        "auto_place_top_pass": auto_top_good,
        "auto_place_top_accuracy": (auto_top_good / auto_count) if auto_count else 0.0,
        "min_similarity": min_similarity,
        "min_margin": min_margin,
        "weak_similarity_floor": weak_similarity_floor,
        "strong_margin": strong_margin,
        "allow_low_similarity_clear_margin": bool(allow_low_similarity_clear_margin),
        "raw_failures": total - raw_label_good,
        "auto_place_failures": auto_count - auto_label_good,
        "accuracy_by_label": label_rows,
        "raw_failure_pairs": failure_rows,
        "auto_place_failure_pairs": final_failure_rows,
        "review_reasons": review_rows,
        "holdout_only_summary": holdout_summary,
        "reused_training_preview_summary": reused_summary,
        "active_expected_label_summary": active_expected_summary,
        "inactive_or_discovery_expected_label_summary": inactive_expected_summary,
        "active_expected_eval_files": active_eval_total,
        "inactive_or_discovery_expected_eval_files": inactive_eval_total,
        "active_expected_raw_label_pass": active_raw_label_good,
        "active_expected_auto_placed": active_auto_count,
        "active_expected_auto_label_pass": active_auto_label_good,
        "inactive_or_discovery_expected_auto_placed": inactive_auto_count,
        "gate_sweep_report_only": gate_sweep_summary,
        "sorted_preview_folder": str(sorted_root),
    }
    write_json(reports_dir / "pure_brain_eval_summary.json", summary)

    with (reports_dir / "pure_brain_eval_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Aaron Sound Sorter Stage 4 v0.5.7 COMMITTEE_PHYSICS_GAP_LOCKS Evaluation Summary\n\n")
        f.write("This run did not call Aaron_Sound_Sorter.py.\n")
        f.write("Predicted labels can only come from this new scratch brain.\n")
        f.write("v0.4.5+ uses One Shots / Loops as terminal leaf folders, not top-level lanes.\n")
        f.write(
            "Eval prediction is pure brain matching across all labels. Expected structure is not passed into predict().\n\n"
        )
        f.write(f"Total eval files: {total}\n")
        f.write(f"Raw label pass: {raw_label_good}/{total} = {summary['raw_label_accuracy']:.2%}\n")
        f.write(f"Raw top pass: {raw_top_good}/{total} = {summary['raw_top_accuracy']:.2%}\n")
        f.write("\nHoldout honesty:\n")
        f.write(
            f"  True holdout eval files: {holdout_summary['total_eval_files']} "
            f"(auto {holdout_summary['auto_placed']}, "
            f"auto label pass {holdout_summary['auto_place_label_pass']}/{holdout_summary['auto_placed']})\n"
        )
        f.write(
            f"  Reused-training preview files: {reused_summary['total_eval_files']} "
            f"(auto {reused_summary['auto_placed']}, "
            f"auto label pass {reused_summary['auto_place_label_pass']}/{reused_summary['auto_placed']})\n"
        )
        f.write("\nActive-label honesty:\n")
        f.write(
            f"  Active expected-label eval files: {active_expected_summary['total_eval_files']} "
            f"(auto {active_expected_summary['auto_placed']}, "
            f"auto label pass {active_expected_summary['auto_place_label_pass']}/{active_expected_summary['auto_placed']})\n"
        )
        f.write(
            f"  Discovery-only/inactive expected-label eval files: {inactive_expected_summary['total_eval_files']} "
            f"(auto {inactive_expected_summary['auto_placed']}, "
            f"auto label pass {inactive_expected_summary['auto_place_label_pass']}/{inactive_expected_summary['auto_placed']})\n"
        )
        f.write("\nStrict gate:\n")
        f.write(f"  min_similarity: {min_similarity:.3f}\n")
        f.write(f"  min_margin: {min_margin:.3f}\n")
        f.write(f"  weak_similarity_floor: {weak_similarity_floor:.3f}\n")
        f.write(f"  strong_margin_for_low_similarity: {strong_margin:.3f}\n")
        f.write(f"  allow_low_similarity_clear_margin: {allow_low_similarity_clear_margin}\n")
        f.write(f"  Auto-placed: {auto_count}/{total}\n")
        f.write(f"  Sent to review: {review_count}/{total}\n")
        f.write(
            f"  Auto-place label pass: {auto_label_good}/{auto_count} = {summary['auto_place_label_accuracy']:.2%}\n"
            if auto_count
            else "  Auto-place label pass: 0/0\n"
        )
        f.write(
            f"  Auto-place top pass: {auto_top_good}/{auto_count} = {summary['auto_place_top_accuracy']:.2%}\n"
            if auto_count
            else "  Auto-place top pass: 0/0\n"
        )
        f.write(f"\nSorted preview folder:\n{sorted_root}\n")
        best_gate = (
            gate_sweep_summary.get("best_zero_failure_gate_report_only", {})
            if isinstance(gate_sweep_summary, dict)
            else {}
        )
        f.write("\nGate sweep report-only:\n")
        f.write(f"  Report: {gate_sweep_summary.get('report', '') if isinstance(gate_sweep_summary, dict) else ''}\n")
        if best_gate:
            f.write(
                "  Best zero-failure gate in sweep: "
                f"min_similarity={best_gate.get('min_similarity')} min_margin={best_gate.get('min_margin')} "
                f"auto_placed={best_gate.get('auto_placed')}/{best_gate.get('total_eval_files')}\n"
            )
            try:
                best_sim = float(best_gate.get("min_similarity", min_similarity))
                best_margin = float(best_gate.get("min_margin", min_margin))
                if min_similarity < best_sim or min_margin < best_margin:
                    f.write(
                        "  WARNING: this run used a looser gate than the zero-failure sweep point. "
                        "Treat auto-placed folders as diagnostic, not safe output.\n"
                    )
            except Exception:
                pass
        else:
            f.write("  Best zero-failure gate in sweep: none\n")
        f.write("\nAccuracy by expected label:\n")
        for r in label_rows:
            f.write(
                f"  raw {int(r['raw_passed']):3d}/{int(r['total']):3d} {float(r['raw_accuracy']):.1%}"
                f" | auto {int(r['auto_passed']):3d}/{int(r['auto_placed']):3d} {float(r['auto_accuracy']):.1%}"
                f" | review {int(r['reviewed']):3d}  {r['expected_structure']} {r['expected_label']}\n"
            )
        f.write("\nTop raw failure pairs:\n")
        for r in failure_rows[:40]:
            f.write(f"  {r['count']:>3}  {r['expected_label']} -> {r['raw_predicted_label']}\n")
        f.write("\nAuto-place failure pairs:\n")
        for r in final_failure_rows[:40]:
            f.write(f"  {r['count']:>3}  {r['expected_label']} -> {r['final_label']}\n")
        f.write("\nReview reasons:\n")
        for r in review_rows:
            f.write(f"  {r['count']:>3}  {r['review_reason']}\n")

    return summary


def copy_audio_samples_for_listening(eval_features: List[FeatureRow], reports_dir: Path, max_files: int = 40) -> None:
    """Copy a tiny review pack of eval files, grouped by expected label.

    This is intentionally small. It helps Aaron listen to the test set without
    uploading or moving the source library.
    """
    review_dir = reports_dir.parent / "listen_review_pack"
    review_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    copied = []
    for row in eval_features:
        if counts[row.label] >= 2 or len(copied) >= max_files:
            continue
        src = Path(row.path)
        if not src.exists() and not src.is_symlink():
            continue
        dest_dir = review_dir / row.label
        dest_dir.mkdir(parents=True, exist_ok=True)
        safe_name = f"eval_{len(copied) + 1:03d}{src.suffix.lower() or '.wav'}"
        dest = dest_dir / safe_name
        try:
            made = materialize_preview_audio(src, dest)
            if not made:
                continue
            copied.append(
                {
                    "source_path": str(src),
                    "review_copy": str(dest),
                    "expected_label": row.label,
                    "group_key": row.group_key,
                }
            )
            counts[row.label] += 1
        except Exception:
            pass
    if copied:
        write_csv(
            reports_dir / "listen_review_pack_manifest.csv",
            copied,
            ["source_path", "review_copy", "expected_label", "group_key"],
        )
