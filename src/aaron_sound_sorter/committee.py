# Auto-split from Aaron_Sound_Sorter.py.
# This is a component module, not a legacy wrapper.
from __future__ import annotations

import math

from .core import *


def learned_label_reliability_reason(brain: dict, predicted_label: str, similarity: float, margin: float) -> str:
    """Review weak auto-placements from labels with poor learned support.

    Professional classifiers do this as confidence calibration: a label with 5
    examples is not trusted as much as a label with 100 examples.  No label names
    or filename hints are used here.
    """
    reliability = brain.get("label_reliability_by_label", {})
    info = reliability.get(predicted_label, {}) if isinstance(reliability, dict) else {}
    if not isinstance(info, dict):
        return ""
    count = int(info.get("training_count", brain.get("counts", {}).get(predicted_label, 0)) or 0)
    tier = str(info.get("support_tier", label_reliability_requirements(count)[0]))
    active_status = str(info.get("active_status", ""))
    req_similarity = float(info.get("min_similarity_for_auto_place", 0.0) or 0.0)
    req_margin = float(info.get("min_margin_for_auto_place", 0.0) or 0.0)
    if req_similarity <= 0.0 and req_margin <= 0.0:
        return ""
    misses = []
    if similarity < req_similarity:
        misses.append(f"similarity {similarity:.3f} < {req_similarity:.3f}")
    if margin < req_margin:
        misses.append(f"margin {margin:.3f} < {req_margin:.3f}")
    if not misses:
        return ""
    return f"Low Label Support: label_training_count={count} tier={tier} active_status={active_status}; " + "; ".join(
        misses
    )


def label_anchor_consensus_reason(brain: dict, predicted_label: str, fingerprint: Sequence[float]) -> str:
    """Review predictions supported by only one nearby teacher anchor.

    This fixes the large/uneven-folder failure mode: a label with many anchors can
    win because one training file happens to sit near the mystery sound while the
    rest of that learned folder is far away.  That is not certainty.  It is a
    single-neighbor accident.  The rule uses only learned vectors and training
    counts.  It does not inspect filenames or category names.
    """
    try:
        anchors = np.asarray(brain.get("anchors_by_label", {}).get(predicted_label, []), dtype=np.float32)
        if anchors.ndim == 1 and anchors.size:
            anchors = anchors.reshape(1, -1)
        n = int(anchors.shape[0]) if anchors.ndim == 2 else 0
        if n < 5:
            return ""
        mean, std = scaler_for_label(brain, predicted_label)
        weights = np.asarray(brain.get("feature_weights", FEATURE_WEIGHTS), dtype=np.float32)
        xw = ((np.asarray(fingerprint, dtype=np.float32) - mean) / std) * weights
        aw = anchors * weights[None, :]
        # Use RMS distance so this agrees with nearest_training_examples_for_label().
        d = np.sqrt(np.mean((aw - xw[None, :]) ** 2, axis=1))
        d = np.sort(np.nan_to_num(d, nan=999.0, posinf=999.0, neginf=999.0))
        if d.size < 5:
            return ""
        nearest = float(d[0])
        second = float(d[1])
        third = float(d[2])
        fifth = float(d[4])
        # v0.4.71: anchor consensus is for larger labels. Small labels are
        # already protected by stricter support and margin requirements, and
        # using a 5-anchor neighborhood on a 5-example folder breaks the basic
        # training-recall contract. Also, an exact teacher recall should never
        # be blocked by neighborhood heuristics.
        rel = (
            brain.get("label_reliability_by_label", {})
            if isinstance(brain.get("label_reliability_by_label", {}), dict)
            else {}
        )
        rel_info = rel.get(predicted_label, {}) if isinstance(rel, dict) else {}
        train_count = int(rel_info.get("training_count", n) or n) if isinstance(rel_info, dict) else n
        if train_count <= SMALL_LABEL_COUNT or nearest <= 1e-5:
            return ""
        # A true confident folder match should have a small neighborhood of
        # similar teachers, not one isolated hit and then a big cliff.
        if nearest <= 0.70 and second >= 0.90 and (second / max(nearest, 1e-6)) >= 1.55:
            return (
                "Single Anchor Match: nearest teacher is close but label neighborhood is not; "
                f"anchor_count={n} nearest={nearest:.3f} second={second:.3f} third={third:.3f} fifth={fifth:.3f}"
            )
        if nearest <= 0.85 and third >= 1.10 and (third / max(nearest, 1e-6)) >= 1.60:
            return (
                "Weak Anchor Consensus: prediction depends on too few nearby teachers; "
                f"anchor_count={n} nearest={nearest:.3f} second={second:.3f} third={third:.3f} fifth={fifth:.3f}"
            )
    except Exception as exc:
        return f"Anchor Consensus Check Error: {str(exc)[:80]}"
    return ""


def distributed_phrase_one_shot_conflict_reason(fingerprint: Sequence[float], duration_sec: float) -> str:
    """Review long distributed phrases that the brain wants to place as One Shots.

    This is not a loop detector and it is not a category rule.  It simply says a
    long, multi-event sound with events spread across most of the file is not a
    safe one-shot auto-place.  It may be a musical phrase, speech phrase, build,
    texture, or irregular loop.  The sorter should not bury it in a One Shots
    leaf without stronger structure support.
    """
    try:
        fp = np.asarray(fingerprint, dtype=np.float32)
        if fp.size < FP_SIZE:
            return ""
        events = fingerprint_primary_event_count(fp)
        temporal = float(fp[42])
        regularity = float(fp[43])
        attack = float(fp[44])
        span = float(fp[45])
        rate = float(fp[46])
        tail = float(fp[47])
        pitch = float(fp[49])
    except Exception:
        return ""
    if duration_sec >= 4.0 and events >= 8.0 and span >= 0.55 and temporal >= 0.24:
        return (
            "Physical Structure Conflict: long distributed multi-event audio is not safe for One Shots; "
            f"duration={duration_sec:.3f} events={events:.1f} span={span:.3f} temporal={temporal:.3f} "
            f"regularity={regularity:.3f} rate={rate:.3f} tail={tail:.3f} attack={attack:.3f} pitch={pitch:.3f}"
        )
    if duration_sec >= 6.0 and events >= 4.0 and span >= 0.48 and temporal >= 0.32 and tail >= 0.35:
        return (
            "Physical Structure Conflict: long distributed phrase with multiple events is not safe for One Shots; "
            f"duration={duration_sec:.3f} events={events:.1f} span={span:.3f} temporal={temporal:.3f} "
            f"regularity={regularity:.3f} rate={rate:.3f} tail={tail:.3f} attack={attack:.3f} pitch={pitch:.3f}"
        )
    if duration_sec >= 7.0 and events >= 5.0 and span >= 0.65 and tail >= 0.45:
        return (
            "Physical Structure Conflict: long phrase/texture with spread events and tail is not safe for One Shots; "
            f"duration={duration_sec:.3f} events={events:.1f} span={span:.3f} temporal={temporal:.3f} "
            f"regularity={regularity:.3f} rate={rate:.3f} tail={tail:.3f} attack={attack:.3f} pitch={pitch:.3f}"
        )
    return ""


def physical_structure_conflict_reason(
    brain: dict, fingerprint: Sequence[float], predicted_label: str, duration_sec: float
) -> str:
    """Review audio-structure contradictions before final auto-place.

    This is product safety, not category routing.  The brain may still offer a
    candidate label, but it cannot confidently place long distributed phrases or
    strong loop-like material into a One Shots leaf.
    """
    label_structure = brain.get("structure_by_label", {}).get(predicted_label, label_default_structure(predicted_label))
    if label_structure != "one_shot":
        return ""

    # A musical/speech phrase can be irregular and still not be a one-shot.
    phrase_reason = distributed_phrase_one_shot_conflict_reason(fingerprint, duration_sec)
    if phrase_reason:
        return phrase_reason

    is_loop, loop_reason = fingerprint_loop_evidence_detail(fingerprint, duration_sec)
    if not is_loop:
        return ""
    structure_scores = structure_distance_scores(brain, fingerprint)
    learned_structure = structure_scores[0][0] if structure_scores else ""
    # v0.4.65: raw physics can veto One Shots even when the learned structure
    # head is also confused.  The old version trusted the structure head and let
    # obvious long multi-event phrases land in One Shots.
    return (
        "Physical Structure Conflict: strong_audio_loop_evidence=True "
        f"label_structure=one_shot learned_structure={learned_structure or 'unknown'}; {loop_reason}"
    )


def strip_terminal_structure_from_public_label(label: str) -> str:
    """Remove only the trailing One Shots/Loops leaf from a producer label."""
    base = public_label(label).replace("\\", "/").strip("/")
    low = base.lower()
    for suffix in ("/one shots", "/one shot", "/loops", "/loop"):
        if low.endswith(suffix):
            return base[: -len(suffix)].strip("/")
    return base


def top5_public_labels(top5: Sequence[Tuple[str, float]]) -> List[str]:
    return [public_label(label) for label, _score in (top5 or [])]


def top5_has_any(top5: Sequence[Tuple[str, float]], fragments: Sequence[str]) -> bool:
    text = " || ".join(top5_public_labels(top5)).lower()
    return any(fragment.lower() in text for fragment in fragments)


def review_loop_evidence_detail(fingerprint: Sequence[float], duration_sec: float) -> Tuple[bool, str]:
    """Looser loop proof for review grouping only.

    This is not an auto-place rule and not a training-label repair rule.  It is a
    listening-review aid: if the physics clearly says repeated material, the
    preview should not make Aaron audit it only under One Shots.
    """
    strict, strict_reason = fingerprint_loop_evidence_detail(fingerprint, duration_sec)
    if strict:
        return True, "strict_" + strict_reason
    try:
        fp = np.asarray(fingerprint, dtype=np.float32)
        if fp.size < FP_SIZE:
            return False, "fingerprint_too_small"
        transients = fingerprint_primary_event_count(fp)
        regularity = float(fp[43])
        temporal_center = float(fp[42])
        attack = float(fp[44])
        onset_span = float(fp[45]) if fp.size > 45 else 0.0
        event_rate = float(fp[46]) if fp.size > 46 else 0.0
    except Exception as exc:
        return False, f"fingerprint_error:{str(exc)[:80]}"
    if fingerprint_front_loaded_tail_evidence(fingerprint, duration_sec):
        return False, "front_loaded_tail_veto_for_review_grouping"
    if (
        duration_sec >= 4.0
        and transients >= 6
        and onset_span >= 0.60
        and temporal_center >= 0.30
        and event_rate >= 0.50
        and regularity <= 0.85
    ):
        return True, (
            "review_loop_by_distributed_repeated_events "
            f"duration={duration_sec:.3f} transients={transients:.1f} regularity={regularity:.3f} "
            f"temporal={temporal_center:.3f} span={onset_span:.3f} rate={event_rate:.3f} attack={attack:.3f}"
        )
    if duration_sec >= 2.0 and transients >= 8 and onset_span >= 0.55 and event_rate >= 1.0 and regularity <= 0.80:
        return True, (
            "review_loop_by_dense_repeated_events "
            f"duration={duration_sec:.3f} transients={transients:.1f} regularity={regularity:.3f} span={onset_span:.3f} rate={event_rate:.3f}"
        )
    return False, (
        "no_review_loop_proof "
        f"duration={duration_sec:.3f} transients={transients:.1f} regularity={regularity:.3f} temporal={temporal_center:.3f} span={onset_span:.3f} rate={event_rate:.3f}"
    )


def same_identity_loop_label(predicted_label: str) -> str:
    base = strip_terminal_structure_from_public_label(predicted_label)
    if not base:
        return "Instruments/Mixed Musical Loops/Loops"
    return label_with_terminal_structure(base, "loop")


def predicted_label_contains_any(predicted_label: str, fragments: Sequence[str]) -> bool:
    text = public_label(predicted_label).lower()
    return any(fragment.lower() in text for fragment in fragments)


def physics_says_not_cymbal_or_hat(fp: np.ndarray, duration_sec: float) -> Tuple[bool, str]:
    """Detect cymbal/hat raw guesses that lack high-air/metal evidence.

    This does not choose a final label.  It protects review routing from treating
    every bright/noisy transient or long machine bed as cymbal just because a
    weak cymbal prototype is nearby.
    """
    if fp.size < FP_SIZE:
        return False, "fingerprint_too_small"
    transients = fingerprint_primary_event_count(fp)
    presence = float(fp[39])
    air = float(fp[40])
    flatness = float(fp[29])
    entropy = float(fp[30])
    tail = float(fp[47])
    event_rate = float(fp[46])
    # Cymbals/hats need real air/top-end or a metal-like bright tail.  A long
    # low-air noisy machine bed or a short low-air clap should not be grouped
    # under cymbal in the review folders.
    if air <= 0.050 and (duration_sec >= 1.5 or transients <= 4.0):
        return True, (
            "low_air_cymbal_veto "
            f"duration={duration_sec:.3f} events={transients:.1f} presence={presence:.3f} air={air:.3f} "
            f"flatness={flatness:.3f} entropy={entropy:.3f} tail={tail:.3f} rate={event_rate:.3f}"
        )
    if air <= 0.080 and duration_sec >= 4.0 and event_rate >= 0.40:
        return True, (
            "long_repeating_low_air_cymbal_veto "
            f"duration={duration_sec:.3f} events={transients:.1f} air={air:.3f} rate={event_rate:.3f}"
        )
    return False, ""


def physics_says_not_animal_candidate(fp: np.ndarray, duration_sec: float) -> Tuple[bool, str]:
    """Detect animal/dog raw guesses that are actually long mechanical/build beds."""
    if fp.size < FP_SIZE:
        return False, "fingerprint_too_small"
    transients = fingerprint_primary_event_count(fp)
    slope = float(fp[34])
    regularity = float(fp[43])
    temporal = float(fp[42])
    attack = float(fp[44])
    onset_span = float(fp[45])
    event_rate = float(fp[46])
    tail = float(fp[47])
    pitch = float(fp[49])
    moving_build_like = (
        duration_sec >= 2.0
        and (slope >= 0.045 or attack >= 0.10 or tail >= 0.45)
        and transients >= 4.0
        and onset_span >= 0.45
        and temporal >= 0.25
    )
    machine_loop_like = duration_sec >= 3.0 and event_rate >= 1.0 and regularity <= 0.75 and onset_span >= 0.50
    if moving_build_like or machine_loop_like:
        return True, (
            "animal_label_veto_by_long_repeated_or_build_physics "
            f"duration={duration_sec:.3f} slope={slope:.3f} events={transients:.1f} regularity={regularity:.3f} "
            f"temporal={temporal:.3f} attack={attack:.3f} span={onset_span:.3f} rate={event_rate:.3f} tail={tail:.3f} pitch={pitch:.3f}"
        )
    return False, ""


def _label_text(label: str) -> str:
    return public_label(label).lower().replace("\\", "/")


def _is_drum_loop_label(label: str) -> bool:
    text = _label_text(label)
    return text.startswith("drums/") and "/loops" in text


def _is_global_drum_loop_label(label: str) -> bool:
    text = _label_text(label)
    return text.startswith("drums/drum loops")


def global_drum_loop_review_label(brain: dict, top5: Sequence[Tuple[str, float]] = ()) -> str:
    """Deprecated compatibility wrapper. No global category fallback is fabricated."""
    return ""


def _drum_component_from_label(label: str) -> str:
    return ""


def _top5_drum_component_set(top5: Sequence[Tuple[str, float]]) -> Set[str]:
    return set()


def _physics_matches_local_drum_loop_component(
    component: str, low_total: float, mid: float, high_total: float, pitch: float
) -> bool:
    return False


def drum_local_loop_conflict_reason(
    brain: dict,
    fingerprint: Sequence[float],
    duration_sec: float,
    predicted_label: str,
    predicted_top: str,
    top5: Sequence[Tuple[str, float]] = (),
) -> str:
    """No named local/global category conflict is inferred in folder-brain mode."""
    return ""


def bright_single_crash_cymbal_candidate_reason(fingerprint: Sequence[float], duration_sec: float) -> str:
    """Detect normal crash-cymbal one-shots so they do not look like white noise, coins, or open hats."""
    fp = np.asarray(fingerprint, dtype=np.float32)
    if fp.size < FP_SIZE:
        return ""
    transients = fingerprint_primary_event_count(fp)
    sub = float(fp[36])
    bass = float(fp[37])
    mid = float(fp[38])
    presence = float(fp[39])
    air = float(fp[40])
    temporal = float(fp[42])
    onset_span = float(fp[45])
    tail = float(fp[47])
    pitch = float(fp[49])
    low_total = sub + bass
    high_total = presence + air
    # Long crash tails often have a few envelope wiggles. They should still
    # be treated as one-shots when energy is front-loaded and high-band dominant.
    if (
        duration_sec >= 0.80
        and high_total >= 0.65
        and low_total <= 0.18
        and temporal <= 0.22
        and (transients <= 6.0 or onset_span <= 0.25)
        and tail <= 0.22
        and pitch <= 0.45
    ):
        return (
            "bright_single_crash_cymbal_physics: high-band front-loaded one-shot with decay tail; "
            f"duration={duration_sec:.3f} events={transients:.1f} low_total={low_total:.3f} "
            f"mid={mid:.3f} high={high_total:.3f} temporal={temporal:.3f} span={onset_span:.3f} tail={tail:.3f}"
        )
    return ""


def voice_or_vocal_fx_candidate_reason(fingerprint: Sequence[float], duration_sec: float, predicted_top: str) -> str:
    """Report-only guard for vocal/voice-FX-like audio that raw brain calls drums.

    This deliberately does not use filename words. It only creates a review
    candidate when mid-band/pitch/noise evidence looks more like processed voice
    than a clean drum hit.
    """
    if predicted_top != "Drums":
        return ""
    fp = np.asarray(fingerprint, dtype=np.float32)
    if fp.size < FP_SIZE:
        return ""
    transients = fingerprint_primary_event_count(fp)
    sub = float(fp[36])
    bass = float(fp[37])
    mid = float(fp[38])
    presence = float(fp[39])
    air = float(fp[40])
    temporal = float(fp[42])
    onset_span = float(fp[45])
    tail = float(fp[47])
    pitch = float(fp[49])
    flatness = float(fp[29])
    low_total = sub + bass
    high_total = presence + air
    # Processed voice often has mid-band dominance and pitch/formant behavior,
    # while drum/clap hits tend to be sharper, shorter, and less sustained.
    if (
        duration_sec >= 0.45
        and low_total <= 0.35
        and mid >= 0.45
        and high_total <= 0.50
        and pitch >= 0.28
        and (tail >= 0.04 or duration_sec >= 1.20 or onset_span >= 0.15)
        and not (duration_sec <= 0.45 and transients <= 2.0)
    ):
        return (
            "voice_or_vocal_fx_physics_candidate: mid-band pitched/noisy material is a poor drum/clap fit; "
            f"duration={duration_sec:.3f} events={transients:.1f} low_total={low_total:.3f} mid={mid:.3f} "
            f"high={high_total:.3f} pitch={pitch:.3f} flatness={flatness:.3f} temporal={temporal:.3f} "
            f"span={onset_span:.3f} tail={tail:.3f}"
        )
    return ""


def one_shot_fx_candidate_reason(fingerprint: Sequence[float], duration_sec: float, predicted_top: str) -> str:
    """Report-only guard for impact/sweep/effect one-shots raw brain calls drums.

    This is intentionally conservative and only affects the physical-role review
    folder, not final auto-placement.
    """
    if predicted_top != "Drums":
        return ""
    fp = np.asarray(fingerprint, dtype=np.float32)
    if fp.size < FP_SIZE:
        return ""
    transients = fingerprint_primary_event_count(fp)
    sub = float(fp[36])
    bass = float(fp[37])
    mid = float(fp[38])
    presence = float(fp[39])
    air = float(fp[40])
    temporal = float(fp[42])
    onset_span = float(fp[45])
    tail = float(fp[47])
    slope = float(fp[34])
    width = float(fp[32])
    low_total = sub + bass
    high_total = presence + air
    if (
        duration_sec >= 1.20
        and temporal <= 0.22
        and onset_span <= 0.35
        and transients <= 8.0
        and (width >= 0.35 or abs(slope) >= 0.08 or tail >= 0.18 or high_total >= 0.55)
        and not (low_total >= 0.55 and high_total <= 0.18 and duration_sec <= 1.50)
    ):
        return (
            "one_shot_fx_physics_candidate: longer/wide/moving/tailed one-shot is a poor dry drum fit; "
            f"duration={duration_sec:.3f} events={transients:.1f} low_total={low_total:.3f} mid={mid:.3f} "
            f"high={high_total:.3f} width={width:.3f} slope={slope:.3f} temporal={temporal:.3f} span={onset_span:.3f} tail={tail:.3f}"
        )
    return ""


def physical_role_recommendation(
    brain: dict,
    fingerprint: Sequence[float],
    duration_sec: float,
    predicted_label: str,
    predicted_top: str,
    top5: Sequence[Tuple[str, float]],
) -> Dict[str, str]:
    """Return no named-category recommendation.

    Stage 4 folder-brain mode no longer fabricates alternate terminal category
    paths. Physics can still force review through learned conflict gates, but
    any alternate destination must be learned from the training folder tree.
    """
    return {
        "physical_role_candidate_top": "",
        "physical_role_candidate_label": "",
        "physical_role_candidate_reason": "",
        "physical_role_action": "disabled_no_named_category_recommendations",
    }


def borderline_auto_place_reason(similarity: float, margin: float) -> str:
    """Block barely-passing auto placements.

    This is a product certainty gate. It does not know categories or filenames.
    A deep folder placement that only barely clears the old threshold is not a
    confident placement, especially in an uneven training tree. The brain can
    still report its best candidate, but the product should review the file.
    """
    try:
        sim = float(similarity or 0.0)
        gap = float(margin or 0.0)
    except Exception:
        return ""
    if sim < 0.560:
        return f"Borderline Auto Place: similarity {sim:.3f} < 0.560"
    if sim < 0.600 and gap < 1.100:
        return f"Borderline Auto Place: similarity {sim:.3f} and margin {gap:.3f} are not strong enough"
    return ""


def _label_parent_without_terminal(label: str) -> str:
    """Return a broad learned sibling parent without hard-coded category names."""
    parts = [p.strip().lower() for p in public_label(label).replace("\\", "/").split("/") if p.strip()]
    if not parts:
        return ""
    # Drop terminal structure leaf if present.
    if parts[-1] in {"one shots", "one shot", "loops", "loop", "long fx"}:
        parts = parts[:-1]
    # Drop the final identity leaf so sibling labels like Bongo/Conga compare
    # as the same learned parent bucket. This is dynamic path geometry, not a
    # category-specific rule.
    if len(parts) >= 2:
        parts = parts[:-1]
    return "/".join(parts)


def dynamic_ambiguous_sibling_label(first_label: str, second_label: str) -> str:
    """Create a learned-tree placement that is sorted but not a false leaf.

    This does not use a fixed category vocabulary.  It takes the common learned
    folder path shared by the top two rival labels and creates a dynamic
    ambiguous-leaf folder from their actual learned leaf names.
    """
    p1 = [p.strip() for p in public_label(first_label).replace("\\", "/").split("/") if p.strip()]
    p2 = [p.strip() for p in public_label(second_label).replace("\\", "/").split("/") if p.strip()]
    if not p1 or not p2:
        return public_label(first_label)
    term = ""
    if p1[-1].lower() in {"one shots", "one shot", "loops", "loop", "long fx"} and p2[-1].lower() == p1[-1].lower():
        term = p1[-1]
        p1_core, p2_core = p1[:-1], p2[:-1]
    else:
        p1_core, p2_core = p1, p2
    common: List[str] = []
    for a, b in zip(p1_core, p2_core):
        if a.strip().lower() == b.strip().lower():
            common.append(a)
        else:
            break
    if not common:
        common = [p1[0]]
    leaf1 = p1_core[len(common)] if len(p1_core) > len(common) else p1_core[-1]
    leaf2 = p2_core[len(common)] if len(p2_core) > len(common) else p2_core[-1]
    pair = " vs ".join(sorted({safe_folder_name(leaf1), safe_folder_name(leaf2)}))
    parts = common + ["_Ambiguous Leaf", pair]
    if term:
        parts.append(term)
    return "/".join(parts)


def adaptive_depth_placement_label(
    brain: dict, pred_label: str, similarity: float, margin: float, top5: Sequence[Tuple[str, float]] | None = None
) -> Tuple[str, str]:
    """Return dynamic parent/ambiguous placement label when leaf identity is risky.

    This is the product compromise between two bad choices: wrong leaf vs review
    dumping. It still sorts the file under the learned top/parent neighborhood,
    but it does not pretend the exact sibling leaf is certain.
    """
    if not bool(brain.get("adaptive_depth_placement_enabled", ADAPTIVE_DEPTH_PLACEMENT_ENABLED_DEFAULT)):
        return "", ""
    if not top5 or len(top5) < 2:
        return "", ""
    try:
        first_label = str(top5[0][0])
        second_label = str(top5[1][0])
        if first_label != pred_label or not second_label or first_label == second_label:
            return "", ""
        top_by = brain.get("top_by_label", {}) if isinstance(brain.get("top_by_label", {}), dict) else {}
        if top_by.get(first_label, top_for_public_label(first_label)) != top_by.get(
            second_label, top_for_public_label(second_label)
        ):
            return "", ""
        struct = brain.get("structure_by_label", {}) if isinstance(brain.get("structure_by_label", {}), dict) else {}
        if struct.get(first_label, label_default_structure(first_label)) != struct.get(
            second_label, label_default_structure(second_label)
        ):
            return "", ""
        try:
            sim = float(similarity or 0.0)
            gap = float(margin or 0.0)
        except Exception:
            sim, gap = 0.0, 0.0
        # Trigger on low/moderate confidence or close-ish leaf contests. Also
        # trigger when the sibling imbalance checker flags a coverage advantage.
        sibling_reason = sibling_imbalance_ambiguity_reason(brain, pred_label, sim, gap, top5)
        max_sim = float(
            brain.get("adaptive_depth_max_similarity", ADAPTIVE_DEPTH_MAX_SIMILARITY) or ADAPTIVE_DEPTH_MAX_SIMILARITY
        )
        max_margin = float(
            brain.get("adaptive_depth_max_margin", ADAPTIVE_DEPTH_MAX_MARGIN) or ADAPTIVE_DEPTH_MAX_MARGIN
        )
        if sibling_reason or sim < max_sim or gap < max_margin:
            dyn = dynamic_ambiguous_sibling_label(first_label, second_label)
            reason = (
                f"adaptive_depth_pick: leaf_identity_risky; placed_under_dynamic_learned_parent "
                f"instead_of_false_leaf; top1={public_label(first_label)} top2={public_label(second_label)} "
                f"similarity={sim:.3f} margin={gap:.3f}"
            )
            return dyn, reason
    except Exception as exc:
        return "", f"adaptive_depth_error:{str(exc)[:80]}"
    return "", ""


def sibling_imbalance_ambiguity_reason(
    brain: dict, pred_label: str, similarity: float, margin: float, top5: Sequence[Tuple[str, float]] | None = None
) -> str:
    """Review fragile wins where a better-supported sibling beats a tiny sibling.

    This catches the micro-regression failure mode: two related learned folders
    share top family and structure, one has many more teachers, and the winner's
    absolute similarity is only moderate. In that case the model may be using
    support/coverage rather than identity. The safe product behavior is review,
    not a confident wrong sibling folder.
    """
    if not top5 or len(top5) < 2:
        return ""
    try:
        first_label = str(top5[0][0])
        second_label = str(top5[1][0])
        if first_label != pred_label or not second_label or second_label == first_label:
            return ""
        top_by = brain.get("top_by_label", {}) if isinstance(brain.get("top_by_label", {}), dict) else {}
        if top_by.get(first_label, top_for_public_label(first_label)) != top_by.get(
            second_label, top_for_public_label(second_label)
        ):
            return ""
        struct = brain.get("structure_by_label", {}) if isinstance(brain.get("structure_by_label", {}), dict) else {}
        if struct.get(first_label, label_default_structure(first_label)) != struct.get(
            second_label, label_default_structure(second_label)
        ):
            return ""
        p1 = _label_parent_without_terminal(first_label)
        p2 = _label_parent_without_terminal(second_label)
        rel = (
            brain.get("label_reliability_by_label", {})
            if isinstance(brain.get("label_reliability_by_label", {}), dict)
            else {}
        )
        counts = brain.get("counts", {}) if isinstance(brain.get("counts", {}), dict) else {}
        c1 = int((rel.get(first_label, {}) or {}).get("training_count", counts.get(first_label, 0)) or 0)
        c2 = int((rel.get(second_label, {}) or {}).get("training_count", counts.get(second_label, 0)) or 0)
        if c1 <= 0 or c2 <= 0:
            return ""
        # First guard: broad rival inside the same learned top family and structure.
        # This catches related leaves that do not share the same immediate parent
        # path, such as two high-frequency percussion folders.  It only reviews
        # moderate-similarity wins where the runner-up is severely undertrained.
        if c1 >= max(4, c2 * 3) and c2 <= TINY_LABEL_COUNT and float(similarity) < 0.58:
            return (
                "Broad Imbalanced Rival Ambiguity: better-supported same-family label won with only moderate similarity; "
                f"winner_count={c1} runner_up_count={c2} similarity={float(similarity):.3f} margin={float(margin):.3f} "
                f"winner={public_label(first_label)} runner_up={public_label(second_label)}"
            )
        # Stronger sibling guard: when labels also share the same immediate learned
        # parent, be more conservative because the model is choosing between close
        # variants of the same role. Exact/near-exact teacher recall stays untouched
        # because similarity will be near 1.0.
        if p1 and p1 == p2 and c1 >= max(4, c2 * 3) and c2 <= TINY_LABEL_COUNT and float(similarity) < 0.64:
            return (
                "Sibling Imbalance Ambiguity: better-supported learned sibling won with only moderate similarity; "
                f"winner_count={c1} runner_up_count={c2} similarity={float(similarity):.3f} margin={float(margin):.3f} "
                f"winner={public_label(first_label)} runner_up={public_label(second_label)}"
            )
        # If both labels are small and the winner has at least 4x the support,
        # demand stronger absolute similarity before auto-placement.
        if (
            p1
            and p1 == p2
            and c1 <= SMALL_LABEL_COUNT
            and c2 <= SMALL_LABEL_COUNT
            and c1 >= c2 * 4
            and float(similarity) < 0.60
        ):
            return (
                "Small Sibling Imbalance Ambiguity: related small labels with uneven support; "
                f"winner_count={c1} runner_up_count={c2} similarity={float(similarity):.3f} margin={float(margin):.3f} "
                f"winner={public_label(first_label)} runner_up={public_label(second_label)}"
            )
    except Exception as exc:
        return f"Sibling Imbalance Check Error: {str(exc)[:80]}"
    return ""


def _safe_label_structure(brain: dict, label: str) -> str:
    struct = brain.get("structure_by_label", {}) if isinstance(brain.get("structure_by_label", {}), dict) else {}
    return str(struct.get(label, label_default_structure(label)) or label_default_structure(label))


def _score_label_for_dynamic_structure_gate(brain: dict, label: str, fingerprint: Sequence[float]) -> float:
    """Score one learned label for a structure-gate fallback.

    This is deliberately label-name agnostic. It reuses the same learned scaler,
    feature weights, adaptive label model, and fact score used by prediction. It
    does not inspect filenames and it does not contain category-specific rescues.
    """
    try:
        raw_x = np.asarray(fingerprint, dtype=np.float32)
        weights = np.asarray(brain.get("feature_weights", FEATURE_WEIGHTS), dtype=np.float32)
        mean, std = scaler_for_label(brain, label)
        xw = ((raw_x - mean) / std) * weights
        raw_score, _raw_distance, _mode = label_model_distance_score(brain, label, xw)
        structure_penalty = structure_penalty_for_label(
            brain, fingerprint, label, structure_distance_scores(brain, fingerprint)
        )
        fact_bonus = 0.0
        fact_profiles = (
            brain.get("category_fact_profiles", {}) if isinstance(brain.get("category_fact_profiles", {}), dict) else {}
        )
        profile = fact_profiles.get(label)
        if isinstance(profile, dict):
            feature_names_list: List[str] = list(brain.get("feature_names", FEATURE_NAMES))
            fact_score = score_sample_against_label_facts(raw_x, profile, feature_names_list)
            fact_bonus = float(brain.get("fact_score_weight", FACT_SCORE_WEIGHT) or FACT_SCORE_WEIGHT) * float(
                fact_score
            )
        return float(raw_score) + float(structure_penalty) - float(fact_bonus)
    except Exception:
        return float("inf")


def dynamic_structure_gate_destination(
    brain: dict,
    fingerprint: Sequence[float],
    predicted_label: str,
    duration_sec: float,
) -> Tuple[str, str]:
    """Return a dynamic non-one-shot destination for loop/phrase evidence.

    Unknown sort input is different from training curation. Training folder truth
    remains untouched, but product placement must not bury long distributed or
    repeated material in a One Shots leaf. This gate uses only measured audio
    structure and learned folder geometry.
    """
    conflict = physical_structure_conflict_reason(brain, fingerprint, predicted_label, duration_sec)
    if not conflict:
        return "", ""
    labels = [str(x) for x in brain.get("labels", [])]
    predicted_top = str(
        (brain.get("top_by_label", {}) or {}).get(predicted_label, top_for_public_label(predicted_label))
    )
    predicted_parent = _label_parent_without_terminal(predicted_label)
    struct_by = brain.get("structure_by_label", {}) if isinstance(brain.get("structure_by_label", {}), dict) else {}

    all_loop_candidates = [
        label for label in labels if str(struct_by.get(label, label_default_structure(label))) == "loop"
    ]
    same_top_candidates = [
        label
        for label in all_loop_candidates
        if str((brain.get("top_by_label", {}) or {}).get(label, top_for_public_label(label))) == predicted_top
    ]
    same_parent = [label for label in same_top_candidates if _label_parent_without_terminal(label) == predicted_parent]

    # v0.5.1: Never escape to a different top family just because the brain has
    # some generic loop label elsewhere.  The v0.4.83 gate correctly blocked long
    # distributed audio from One Shots, but when the learned top had no loop leaf
    # it fell through to all_loop_candidates and sent FX vinyl/noise material into
    # Drums or Instruments loops.  That is a structure fix turning into an identity
    # error.  This gate may choose a learned loop only inside the same learned
    # parent/top.  Otherwise it preserves the predicted identity path and changes
    # only the terminal structure leaf from One Shots to Loops.
    candidate_pool = same_parent or same_top_candidates
    if candidate_pool:
        scored = sorted(
            ((label, _score_label_for_dynamic_structure_gate(brain, label, fingerprint)) for label in candidate_pool),
            key=lambda item: (float(item[1]), public_label(item[0])),
        )
        chosen, score = scored[0]
        scope = "same_learned_parent" if same_parent else "same_learned_top"
        return chosen, (
            "dynamic_structure_gate: measured_loop_or_phrase_evidence_blocked_one_shot_leaf; "
            f"rerouted_to_best_learned_loop_label scope={scope} score={float(score):.4f}; "
            f"from={public_label(predicted_label)} to={public_label(chosen)}; {conflict}"
        )

    # No same-top learned loop exists. Preserve the identity path and change only
    # the terminal structure. This creates a useful dynamic destination such as
    # FX/Textures/Noise and Static/Vinyl Noise/Loops instead of cross-routing to
    # Instruments/Instrument Loops or Drums/Drum Loops.
    base = strip_terminal_structure_from_public_label(predicted_label)
    if base:
        fallback = label_with_terminal_structure(base, "loop")
        return fallback, (
            "dynamic_structure_gate: measured_loop_or_phrase_evidence_blocked_one_shot_leaf; "
            "no_same_top_learned_loop_label_available; preserved_predicted_identity_with_loop_terminal; "
            f"from={public_label(predicted_label)} to={fallback}; {conflict}"
        )
    return "", ""


def membership_feature_absolute_tolerance(fname: str, lower: float, upper: float, p25: float, p75: float) -> float:
    """Minimum learned-membership tolerance for zero-spread physics profiles.

    This is feature-scale math, not category logic.  Robust p10/p90 envelopes
    are still primary, but tiny folders often have identical values. Without a
    minimum tolerance, a harmless 0.01 pitch-confidence difference or 5 Hz F0
    difference becomes an infinite severe violation.
    """
    name = str(fname or "").lower()
    observed_scale = max(abs(float(lower)), abs(float(upper)), abs(float(p25)), abs(float(p75)), 1.0)
    if name.endswith("_hz") or "frequency_hz" in name:
        return max(12.0, 0.025 * observed_scale)
    if name.endswith("_ms") or "duration_ms" in name:
        return max(12.0, 0.08 * observed_scale)
    if "cents" in name:
        return max(20.0, 0.08 * observed_scale)
    if (
        "ratio" in name
        or "confidence" in name
        or "flatness" in name
        or "entropy" in name
        or "zcr" in name
        or "stability" in name
    ):
        return 0.035
    if "count" in name:
        return 1.5
    if "slope" in name:
        return 0.08 * observed_scale
    if "harmonic_to_noise" in name:
        return 0.40
    return max(0.025, 0.03 * observed_scale)


def _membership_profile_strength_rank(strength: str) -> int:
    order = {"weak": 0, "tentative": 1, "ok": 2, "good": 3, "strong": 4}
    return order.get(str(strength or "").lower(), 0)


def membership_feature_discriminating_weight(brain: dict, label: str, feature_name: str) -> Tuple[float, float, str]:
    """Return how important one feature is for this label against learned rivals.

    The membership gate is intentionally human-readable and ignores MFCC slots,
    but explicit physics features are not all equally diagnostic.  A violation
    on a feature that strongly separates this label from nearby rivals should
    count more than a violation on a generic feature shared by every rival.

    This uses only category_rival_contrast_facts learned from the training tree.
    No filenames, folder-name rescue words, or category-specific constants are
    used.  Multiplier range is deliberately modest so membership remains a
    safety gate rather than a new hidden classifier.
    """
    rival_facts = brain.get("category_rival_contrast_facts", {})
    if not isinstance(rival_facts, dict):
        return 1.0, 0.0, "no_rival_facts"
    label_facts = rival_facts.get(label, {})
    if not isinstance(label_facts, dict) or not label_facts:
        return 1.0, 0.0, "no_label_rivals"

    best_score = 0.0
    best_reason = "non_discriminating"
    for rival, info in label_facts.items():
        if not isinstance(info, dict):
            continue
        discs = info.get("best_discriminators", [])
        if not isinstance(discs, list):
            continue
        for disc in discs:
            if not isinstance(disc, dict):
                continue
            if str(disc.get("feature", "")) != str(feature_name):
                continue
            try:
                effect = abs(float(disc.get("robust_effect_size", 0.0) or 0.0))
                reliability = max(0.0, min(1.0, float(disc.get("reliability", 0.0) or 0.0)))
                rank = max(1, int(disc.get("rank", 10) or 10))
            except Exception:
                continue
            rank_factor = max(0.35, 1.0 - 0.06 * float(rank - 1))
            score = min(1.0, (effect * reliability * rank_factor) / 3.0)
            if score > best_score:
                best_score = score
                best_reason = f"vs={public_label(str(rival))} rank={rank} effect={effect:.2f} rel={reliability:.2f}"

    # If this label has rival facts but this feature is not a discriminator,
    # down-weight the membership penalty.  If it is a strong discriminator,
    # up-weight it.  Keep the bounds conservative.
    multiplier = 0.55 + 1.35 * best_score
    multiplier = max(0.55, min(1.90, multiplier))
    return float(multiplier), float(best_score), best_reason


def folder_membership_evidence(
    brain: dict,
    fingerprint: Sequence[float],
    label: str,
) -> Dict[str, object]:
    """Evaluate whether a sample fits one learned folder's physics atlas.

    This is the missing v0.5.0 product gate. It uses the already-built
    category_fact_profiles: robust p10/p90 ranges, IQR, MAD, and reliability.
    It does not inspect filenames, source names, or category-specific fragments.

    Returns a manifest-safe evidence dict. Severe violations mean the sample is
    outside a reliable feature envelope for the learned folder. A nearest folder
    with multiple severe violations is not a safe exact auto-place.
    """
    profiles = brain.get("category_fact_profiles", {}) if isinstance(brain.get("category_fact_profiles"), dict) else {}
    profile = profiles.get(label)
    if not isinstance(profile, dict):
        return {
            "enabled": False,
            "label": str(label),
            "reason": "no_fact_profile",
            "profile_strength": "missing",
            "eligible_count": 0,
            "severe_count": 0,
            "violation_count": 0,
            "matched_count": 0,
            "weighted_severity": 0.0,
            "blocked": False,
            "severe_features": [],
            "violation_features": [],
            "matched_features": [],
        }
    try:
        eligible_count = int(
            profile.get("eligible_count", profile.get("effective_count", profile.get("raw_count", 0))) or 0
        )
    except Exception:
        eligible_count = 0
    strength = str(
        profile.get("fact_profile_strength", _fact_profile_strength(eligible_count))
        or _fact_profile_strength(eligible_count)
    )
    if eligible_count < int(
        brain.get("membership_min_profile_count", MEMBERSHIP_MIN_PROFILE_COUNT) or MEMBERSHIP_MIN_PROFILE_COUNT
    ):
        return {
            "enabled": False,
            "label": str(label),
            "reason": f"profile_too_small eligible_count={eligible_count}",
            "profile_strength": strength,
            "eligible_count": eligible_count,
            "severe_count": 0,
            "violation_count": 0,
            "matched_count": 0,
            "weighted_severity": 0.0,
            "blocked": False,
            "severe_features": [],
            "violation_features": [],
            "matched_features": [],
        }

    fp = np.asarray(fingerprint, dtype=np.float32)
    feature_names_list: List[str] = list(brain.get("feature_names", FEATURE_NAMES))
    feature_stats = profile.get("feature_stats", {})
    if not isinstance(feature_stats, dict):
        return {
            "enabled": False,
            "label": str(label),
            "reason": "bad_feature_stats",
            "profile_strength": strength,
            "eligible_count": eligible_count,
            "severe_count": 0,
            "violation_count": 0,
            "matched_count": 0,
            "weighted_severity": 0.0,
            "blocked": False,
            "severe_features": [],
            "violation_features": [],
            "matched_features": [],
        }

    min_reliability = float(
        brain.get("membership_min_reliability", MEMBERSHIP_MIN_RELIABILITY) or MEMBERSHIP_MIN_RELIABILITY
    )
    severe_iqr_mult = float(
        brain.get("membership_severe_iqr_multiplier", MEMBERSHIP_SEVERE_IQR_MULTIPLIER)
        or MEMBERSHIP_SEVERE_IQR_MULTIPLIER
    )
    severe_mad_mult = float(
        brain.get("membership_severe_mad_multiplier", MEMBERSHIP_SEVERE_MAD_MULTIPLIER)
        or MEMBERSHIP_SEVERE_MAD_MULTIPLIER
    )
    severe_range_frac = float(
        brain.get("membership_severe_range_fraction", MEMBERSHIP_SEVERE_RANGE_FRACTION)
        or MEMBERSHIP_SEVERE_RANGE_FRACTION
    )

    reliable_feature_count = 0
    matched: List[str] = []
    violations: List[str] = []
    severe: List[str] = []
    weighted_severity = 0.0
    weighted_violation = 0.0
    total_reliable_weight = 0.0

    # MFCC features are useful for ranking but poor human-readable membership
    # facts.  The gate focuses on explicit physics features where robust
    # envelope/band/noise/pitch violations are meaningful.
    explicit_start = 26

    for fi, fname in enumerate(feature_names_list):
        if fi < explicit_start or fi >= fp.size:
            continue
        stats = feature_stats.get(fname)
        if not isinstance(stats, dict):
            continue
        try:
            rel = float(stats.get("reliability", 0.0) or 0.0)
            valid_count = int(stats.get("valid_count", 0) or 0)
        except Exception:
            continue
        if rel < min_reliability or valid_count < MEMBERSHIP_MIN_PROFILE_COUNT:
            continue
        try:
            val = float(fp[fi])
            p10 = float(stats.get("p10", 0.0) or 0.0)
            p25 = float(stats.get("p25", p10) or p10)
            p75 = float(stats.get("p75", p25) or p25)
            p90 = float(stats.get("p90", p75) or p75)
            iqr = abs(float(stats.get("iqr", p75 - p25) or (p75 - p25)))
            mad = abs(float(stats.get("mad", 0.0) or 0.0))
        except Exception:
            continue
        if not all(math.isfinite(x) for x in (val, p10, p25, p75, p90, iqr, mad)):
            continue
        disc_multiplier, disc_score, disc_reason = membership_feature_discriminating_weight(
            brain, str(label), str(fname)
        )
        feature_weight = max(0.05, rel) * disc_multiplier
        reliable_feature_count += 1
        total_reliable_weight += feature_weight
        lower = min(p10, p90)
        upper = max(p10, p90)
        if lower <= val <= upper:
            if p25 <= val <= p75 or p75 <= val <= p25:
                matched.append(fname)
            continue

        delta = lower - val if val < lower else val - upper
        spread = max(abs(p90 - p10), iqr, mad * 2.0, 1e-6)
        abs_tol = membership_feature_absolute_tolerance(fname, lower, upper, p25, p75)
        # Tiny or perfectly uniform baby profiles are common in tests and small
        # folders.  A value that is only microscopically outside p10/p90 should
        # not become a severe violation merely because spread is zero.
        if delta <= abs_tol:
            matched.append(fname)
            continue
        severe_delta = max(
            iqr * severe_iqr_mult, mad * severe_mad_mult, spread * severe_range_frac, abs_tol * 2.0, 1e-5
        )
        severity_ratio = float(delta / max(1e-6, severe_delta))
        weighted_violation += feature_weight * min(1.5, severity_ratio)
        short = (
            f"{fname}={val:.4g} outside [{lower:.4g},{upper:.4g}] "
            f"rel={rel:.2f} disc={disc_score:.2f} w={feature_weight:.2f} ratio={severity_ratio:.2f}"
        )
        violations.append(short)
        if delta >= severe_delta:
            severe.append(short)
            weighted_severity += feature_weight * min(2.0, severity_ratio)

    block_severe_count = int(
        brain.get("membership_block_severe_count", MEMBERSHIP_BLOCK_SEVERE_COUNT) or MEMBERSHIP_BLOCK_SEVERE_COUNT
    )
    block_weighted = float(
        brain.get("membership_block_weighted_severity", MEMBERSHIP_BLOCK_WEIGHTED_SEVERITY)
        or MEMBERSHIP_BLOCK_WEIGHTED_SEVERITY
    )
    blocked = bool(len(severe) >= block_severe_count or (len(severe) >= 2 and weighted_severity >= block_weighted))
    score = 1.0
    if total_reliable_weight > 0.0:
        score = float(max(-1.0, min(1.0, 1.0 - (weighted_violation / max(0.1, total_reliable_weight)))))

    return {
        "enabled": True,
        "label": str(label),
        "reason": "ok",
        "profile_strength": strength,
        "eligible_count": eligible_count,
        "reliable_feature_count": reliable_feature_count,
        "matched_count": len(matched),
        "violation_count": len(violations),
        "severe_count": len(severe),
        "weighted_violation": float(weighted_violation),
        "weighted_severity": float(weighted_severity),
        "membership_score": float(score),
        "blocked": blocked,
        "severe_features": severe[:12],
        "violation_features": violations[:16],
        "matched_features": matched[:12],
    }


def format_membership_evidence(evidence: Dict[str, object]) -> str:
    severe = "; ".join(str(x) for x in (evidence.get("severe_features") or [])[:8])
    return (
        f"membership profile_strength={evidence.get('profile_strength', '')} eligible={evidence.get('eligible_count', 0)} "
        f"reliable={evidence.get('reliable_feature_count', 0)} severe={evidence.get('severe_count', 0)} "
        f"violations={evidence.get('violation_count', 0)} weighted_severity={float(evidence.get('weighted_severity', 0.0) or 0.0):.3f} "
        f"score={float(evidence.get('membership_score', 0.0) or 0.0):.3f} severe_features=[{severe}]"
    )


def choose_membership_safe_alternative(
    brain: dict,
    fingerprint: Sequence[float],
    current_label: str,
    top5: Sequence[Tuple[str, float]] | None,
    current_evidence: Dict[str, object],
) -> Tuple[str, str]:
    """Choose a safer learned candidate if top alternatives pass membership.

    This only compares labels already proposed by the learned chooser. It does
    not invent categories and does not look at filenames.  If no candidate passes
    with a clear membership improvement, the caller should send to review rather
    than pretend the blocked label is correct.
    """
    if not top5:
        return "", ""
    current_score = None
    for label, score in top5:
        if str(label) == str(current_label):
            current_score = float(score)
            break
    if current_score is None:
        current_score = float(top5[0][1]) if top5 else 0.0
    current_severity = float(current_evidence.get("weighted_severity", 0.0) or 0.0)
    max_gap = float(
        brain.get("membership_alt_max_score_gap", MEMBERSHIP_ALT_MAX_SCORE_GAP) or MEMBERSHIP_ALT_MAX_SCORE_GAP
    )
    min_improvement = float(
        brain.get("membership_alt_min_severity_improvement", MEMBERSHIP_ALT_MIN_SEVERITY_IMPROVEMENT)
        or MEMBERSHIP_ALT_MIN_SEVERITY_IMPROVEMENT
    )
    top_by = brain.get("top_by_label", {}) if isinstance(brain.get("top_by_label", {}), dict) else {}
    current_top = str(top_by.get(current_label, top_for_public_label(current_label)))
    allow_cross_family_alt = bool(
        brain.get("membership_alt_allow_cross_family_switch", False)
        and brain.get("membership_alt_explicit_cross_family_override_enabled", False)
    )
    candidates: List[Tuple[float, float, str, Dict[str, object], float]] = []
    for label, score in top5:
        label = str(label)
        if label == current_label:
            continue
        label_top = str(top_by.get(label, top_for_public_label(label)))
        if (not allow_cross_family_alt) and label_top != current_top:
            continue
        score = float(score)
        if score - current_score > max_gap:
            continue
        ev = folder_membership_evidence(brain, fingerprint, label)
        if not ev.get("enabled"):
            continue
        sev = float(ev.get("weighted_severity", 0.0) or 0.0)
        blocked = bool(ev.get("blocked", False))
        improvement = current_severity - sev
        if blocked or improvement < min_improvement:
            continue
        candidates.append((sev, score, label, ev, improvement))
    if not candidates:
        return "", ""
    candidates.sort(key=lambda item: (item[0], item[1], public_label(item[2])))
    sev, score, label, ev, improvement = candidates[0]
    return label, (
        "membership_gate_switched_to_safer_learned_candidate: "
        f"from={public_label(current_label)} to={public_label(label)} score={score:.3f} "
        f"severity_improvement={improvement:.3f}; {format_membership_evidence(ev)}"
    )


def learned_membership_gate_decision(
    brain: dict,
    fingerprint: Sequence[float],
    final_label: str,
    final_top: str,
    top5: Sequence[Tuple[str, float]] | None,
    predicted_top: str,
) -> Tuple[str, str, str, Dict[str, object]]:
    """Return final_label/final_top action after learned membership testing.

    Action values: "pass", "switch", or "review".
    """
    if not bool(brain.get("membership_gate_enabled", MEMBERSHIP_GATE_ENABLED_DEFAULT)):
        return final_label, final_top, "pass", {"enabled": False, "reason": "membership_gate_disabled"}
    ev = folder_membership_evidence(brain, fingerprint, final_label)
    if not ev.get("enabled"):
        return final_label, final_top, "pass", ev
    if not bool(ev.get("blocked", False)):
        warn_count = int(
            brain.get("membership_warn_severe_count", MEMBERSHIP_WARN_SEVERE_COUNT) or MEMBERSHIP_WARN_SEVERE_COUNT
        )
        if int(ev.get("severe_count", 0) or 0) >= warn_count:
            ev = dict(ev)
            ev["warning"] = True
        return final_label, final_top, "pass", ev
    alt_label, alt_reason = choose_membership_safe_alternative(brain, fingerprint, final_label, top5, ev)
    if alt_label:
        ev = dict(ev)
        ev["alternative_reason"] = alt_reason
        return alt_label, top_for_public_label(alt_label), "switch", ev
    ev = dict(ev)
    ev["review_reason"] = (
        "learned_membership_gate_blocked_exact_leaf: "
        f"label={public_label(final_label)} predicted_top={predicted_top} final_top={final_top}; "
        + format_membership_evidence(ev)
    )
    return "_TO_REVIEW/Conflicting Evidence/Learned Folder Membership Failed", "_TO_REVIEW", "review", ev


def learned_top_family_safety_decision(
    brain: dict,
    fingerprint: Sequence[float],
    final_label: str,
    final_top: str,
    predicted_top: str,
    top5: Sequence[Tuple[str, float]] | None,
) -> Tuple[str, str, str, str]:
    """Switch or review when learned top-family evidence contradicts the final label.

    This is a committee-level safety check, not a category rule.  The top-family
    head is learned from the trusted folder tree.  If it says the current final
    top is the wrong broad family by the calibrated review gap, we either switch
    to the best already-nominated label inside the learned top family or send the
    file to review.  We never invent a category and we never read filenames.
    """
    _allowed, scores = top_prefilter_allowed_tops(
        brain, fingerprint, top_n=max(3, int(brain.get("top_prefilter_n", 3) or 3))
    )
    if len(scores) < 2:
        return "", "", "pass", "no_top_scores"
    best_top, best_score = scores[0]
    if str(best_top) == str(final_top):
        return "", "", "pass", "final_top_matches_learned_top"
    final_score = None
    for top, score in scores:
        if str(top) == str(final_top):
            final_score = float(score)
            break
    if final_score is None:
        return "", "", "pass", "final_top_absent_from_top_scores"
    raw_review_gap = float(brain.get("top_review_gap", 0.25) or 0.25)
    review_gap = min(max(raw_review_gap, 0.0), 0.25)
    gap = float(final_score) - float(best_score)
    if gap < review_gap:
        return "", "", "pass", f"top_gap_below_threshold gap={gap:.3f} threshold={review_gap:.3f}"

    candidates = list(top5 or [])
    first_top = ""
    if candidates:
        first_label = str(candidates[0][0])
        first_top = str((brain.get("top_by_label", {}) or {}).get(first_label, top_for_public_label(first_label)))
    # Do not let a noisy broad top-family head overrule a coherent leaf/family
    # prediction by itself.  It must be corroborated either by the raw predicted
    # top or by the nearest learned label.
    if str(best_top) not in {str(predicted_top), str(first_top)}:
        return (
            "",
            "",
            "pass",
            (
                f"top_conflict_not_corroborated learned_top={best_top} predicted_top={predicted_top} "
                f"nearest_top={first_top} final_top={final_top} gap={gap:.3f}"
            ),
        )

    for candidate_label, candidate_score in candidates:
        candidate_label = str(candidate_label)
        candidate_top = str(
            (brain.get("top_by_label", {}) or {}).get(candidate_label, top_for_public_label(candidate_label))
        )
        if candidate_top != str(best_top):
            continue
        ev = folder_membership_evidence(brain, fingerprint, candidate_label)
        if bool(ev.get("enabled", False)) and bool(ev.get("blocked", False)):
            continue
        return (
            candidate_label,
            candidate_top,
            "switch",
            (
                "learned_top_family_safety_switched: "
                f"learned_top={best_top} final_top={final_top} gap={gap:.3f} >= {review_gap:.3f}; "
                f"chosen={public_label(candidate_label)} score={float(candidate_score):.3f}"
            ),
        )

    return (
        final_label,
        final_top,
        "review",
        (
            "learned_top_family_safety_review: "
            f"learned_top={best_top} final_top={final_top} gap={gap:.3f} >= {review_gap:.3f}; "
            "no_unblocked_candidate_from_learned_top_in_top5"
        ),
    )


def _first_unblocked_candidate_for_top(
    brain: dict,
    fingerprint: Sequence[float],
    top5: Sequence[Tuple[str, float]] | None,
    wanted_top: str,
) -> Tuple[str, float, str]:
    """Return first learned candidate from a broad top that passes membership.

    This is a generic committee helper. It never inspects filenames or source
    folders. It only searches labels already proposed by the learned model.
    """
    for label, score in list(top5 or []):
        label = str(label)
        top = str((brain.get("top_by_label", {}) or {}).get(label, top_for_public_label(label)))
        if top != str(wanted_top):
            continue
        ev = folder_membership_evidence(brain, fingerprint, label)
        if bool(ev.get("enabled", False)) and bool(ev.get("blocked", False)):
            continue
        return label, float(score), "top5_candidate"
    return "", 0.0, ""


def _is_loop_label_for_guard(brain: dict, label: str) -> bool:
    struct = brain.get("structure_by_label", {}) if isinstance(brain.get("structure_by_label", {}), dict) else {}
    return str(struct.get(str(label), label_default_structure(str(label))) or "").lower() == "loop"


def _drum_loop_physics_supported(fingerprint: Sequence[float], duration_sec: float) -> Tuple[bool, str]:
    """Return whether measured audio has enough drum/percussion-loop physics.

    This is a broad-family physics check, not a leaf-category rule.  A loop can
    be a drum loop only when it has distributed repeated events plus drum-like
    body and/or percussive broadband top.  Long tonal, vocal, or FX-motion loops
    may have many events, but repeated events alone are not sufficient evidence
    for Drums.
    """
    try:
        fp = np.asarray(fingerprint, dtype=np.float32)
        transients = fingerprint_primary_event_count(fp)
        low_total = safe_feature_value(fp, 36) + safe_feature_value(fp, 37)
        mid = safe_feature_value(fp, 38)
        high = safe_feature_value(fp, 39) + safe_feature_value(fp, 40)
        flatness = safe_feature_value(fp, 29)
        entropy = safe_feature_value(fp, 30)
        pitch = safe_feature_value(fp, 49)
        temporal = safe_feature_value(fp, 42)
        attack = safe_feature_value(fp, 44)
        span = safe_feature_value(fp, 45)
        event_rate = safe_feature_value(fp, 46)
        tail = safe_feature_value(fp, 47)
        slope = abs(safe_feature_value(fp, 34))
    except Exception as exc:
        return False, f"drum_loop_support_error:{str(exc)[:80]}"

    # Some older unit tests and minimal synthetic brains only populate structure
    # features.  Do not turn missing band/texture fields into proof against Drums.
    if (low_total + mid + high) < 0.05 and flatness <= 0.02 and entropy <= 0.02:
        return True, (
            f"insufficient_band_texture_evidence_for_drum_loop_veto duration={float(duration_sec):.3f} "
            f"events={transients:.1f} span={span:.3f} rate={event_rate:.3f} temporal={temporal:.3f}"
        )

    distributed = bool(
        float(duration_sec) >= 1.2
        and transients >= 5.0
        and span >= 0.45
        and 0.18 <= temporal <= 0.82
        and event_rate >= 0.70
    )
    low_body_drum = bool(low_total >= 0.32 and entropy >= 0.24 and flatness >= 0.10)
    bright_top_percussion = bool(high >= 0.42 and flatness >= 0.28 and entropy >= 0.46)
    mixed_broadband_drum = bool(low_total >= 0.16 and high >= 0.16 and flatness >= 0.18 and entropy >= 0.36)
    dense_click_percussion = bool(high >= 0.70 and event_rate >= 2.0 and flatness >= 0.30 and entropy >= 0.55)
    low_pulse_drum_loop = bool(
        low_total >= 0.72
        and mid <= 0.18
        and high <= 0.10
        and event_rate >= 1.10
        and attack <= 0.04
        and temporal <= 0.58
        and (pitch >= 0.42 or tail >= 0.30)
    )

    supported = bool(
        distributed
        and (
            low_body_drum
            or bright_top_percussion
            or mixed_broadband_drum
            or dense_click_percussion
            or low_pulse_drum_loop
        )
    )
    reason = (
        f"distributed={distributed} duration={float(duration_sec):.3f} events={transients:.1f} "
        f"span={span:.3f} rate={event_rate:.3f} temporal={temporal:.3f} attack={attack:.3f} "
        f"low_total={low_total:.3f} mid={mid:.3f} high={high:.3f} flatness={flatness:.3f} "
        f"entropy={entropy:.3f} pitch={pitch:.3f} tail={tail:.3f} slope={slope:.3f} "
        f"low_body_drum={low_body_drum} bright_top_percussion={bright_top_percussion} "
        f"mixed_broadband_drum={mixed_broadband_drum} dense_click_percussion={dense_click_percussion} "
        f"low_pulse_drum_loop={low_pulse_drum_loop}"
    )
    return supported, reason


def _low_pulse_drum_loop_supported(fingerprint: Sequence[float], duration_sec: float) -> Tuple[bool, str]:
    """Return whether a loop has sparse sub-heavy kick-pulse physics."""
    try:
        fp = np.asarray(fingerprint, dtype=np.float32)
        transients = fingerprint_primary_event_count(fp)
        low_total = safe_feature_value(fp, 36) + safe_feature_value(fp, 37)
        mid = safe_feature_value(fp, 38)
        high = safe_feature_value(fp, 39) + safe_feature_value(fp, 40)
        pitch = safe_feature_value(fp, 49)
        temporal = safe_feature_value(fp, 42)
        attack = safe_feature_value(fp, 44)
        span = safe_feature_value(fp, 45)
        event_rate = safe_feature_value(fp, 46)
        tail = safe_feature_value(fp, 47)
    except Exception as exc:
        return False, f"low_pulse_drum_loop_error:{str(exc)[:80]}"

    distributed = bool(
        float(duration_sec) >= 1.2
        and transients >= 5.0
        and span >= 0.45
        and 0.18 <= temporal <= 0.82
        and event_rate >= 0.70
    )
    supported = bool(
        distributed
        and low_total >= 0.72
        and mid <= 0.18
        and high <= 0.10
        and event_rate >= 1.10
        and attack <= 0.04
        and temporal <= 0.58
        and (pitch >= 0.42 or tail >= 0.30)
    )
    return supported, (
        f"distributed={distributed} duration={float(duration_sec):.3f} events={transients:.1f} "
        f"span={span:.3f} rate={event_rate:.3f} temporal={temporal:.3f} attack={attack:.3f} "
        f"low_total={low_total:.3f} mid={mid:.3f} high={high:.3f} pitch={pitch:.3f} tail={tail:.3f}"
    )


def _is_fx_transition_label_for_guard(label: str) -> bool:
    """Return whether a learned label describes an FX transition/motion role."""
    text = public_label(str(label)).lower()
    if "structural and transitional" in text:
        return True
    return bool(re.search(r"\b(riser|build|drop|downlifter|reverse|tail|sweep|whoosh)\b", text))


def _fx_transition_motion_supported(fingerprint: Sequence[float], duration_sec: float) -> Tuple[bool, str]:
    """Return whether measured audio has envelope/motion support for transition FX."""
    try:
        fp = np.asarray(fingerprint, dtype=np.float32)
        transients = fingerprint_primary_event_count(fp)
        temporal = safe_feature_value(fp, 42)
        attack = safe_feature_value(fp, 44)
        span = safe_feature_value(fp, 45)
        event_rate = safe_feature_value(fp, 46)
        tail = safe_feature_value(fp, 47)
        slope_signed = safe_feature_value(fp, 34)
        slope = abs(slope_signed)
        flatness = safe_feature_value(fp, 29)
        entropy = safe_feature_value(fp, 30)
        pitch = safe_feature_value(fp, 49)
    except Exception as exc:
        return False, f"fx_transition_motion_error:{str(exc)[:80]}"

    # A structural FX leaf should be supported by time-varying envelope or
    # spectral motion, not merely by being bright, noisy, or short. This is a
    # role contract for transition-like learned labels; it does not inspect the
    # source filename or source folder.
    reverse_or_tail = bool(
        float(duration_sec) >= 0.45
        and tail >= 0.82
        and (attack >= 0.18 or temporal >= 0.62 or span >= 0.32 or slope >= 0.08)
    )
    slow_envelope_swell = bool(float(duration_sec) >= 0.45 and attack >= 0.30 and tail >= 0.55 and temporal >= 0.34)
    directional_motion = bool(
        float(duration_sec) >= 0.45 and slope >= 0.12 and (tail >= 0.20 or attack >= 0.12 or span >= 0.20)
    )
    distributed_sweep = bool(
        float(duration_sec) >= 1.20
        and span >= 0.45
        and temporal >= 0.28
        and tail >= 0.35
        and (slope >= 0.08 or attack >= 0.18 or flatness >= 0.45 or entropy >= 0.70)
        and transients <= max(24.0, float(duration_sec) * 8.0)
    )
    noisy_motion = bool(
        float(duration_sec) >= 0.60
        and flatness >= 0.45
        and entropy >= 0.65
        and tail >= 0.65
        and (attack >= 0.12 or span >= 0.25 or slope >= 0.06)
    )

    supported = bool(reverse_or_tail or slow_envelope_swell or directional_motion or distributed_sweep or noisy_motion)
    reason = (
        f"supported={supported} duration={float(duration_sec):.3f} events={transients:.1f} "
        f"temporal={temporal:.3f} span={span:.3f} rate={event_rate:.3f} attack={attack:.3f} "
        f"tail={tail:.3f} slope={slope_signed:.3f} flatness={flatness:.3f} "
        f"entropy={entropy:.3f} pitch={pitch:.3f} reverse_or_tail={reverse_or_tail} "
        f"slow_envelope_swell={slow_envelope_swell} directional_motion={directional_motion} "
        f"distributed_sweep={distributed_sweep} noisy_motion={noisy_motion}"
    )
    return supported, reason


def _first_unblocked_non_transition_fx_candidate(
    brain: dict,
    fingerprint: Sequence[float],
    top5: Sequence[Tuple[str, float]] | None,
) -> Tuple[str, float]:
    for cand_label, cand_score in list(top5 or []):
        cand_label = str(cand_label)
        cand_top = str((brain.get("top_by_label", {}) or {}).get(cand_label, top_for_public_label(cand_label)))
        if cand_top != "FX":
            continue
        if _is_fx_transition_label_for_guard(cand_label):
            continue
        ev = folder_membership_evidence(brain, fingerprint, cand_label)
        if bool(ev.get("enabled", False)) and bool(ev.get("blocked", False)):
            continue
        return cand_label, float(cand_score)
    return "", 0.0


def _canonical_learned_top_loop_label(brain: dict, final_top: str) -> str:
    """Return the shortest learned loop label for a broad top, if it exists.

    This is used only after measured structure has proved loop behavior and the
    model has already chosen that broad top.  It avoids false precision like a
    mixed beat being buried in a clap/snare/hat leaf when the leaf contest is
    ambiguous.
    """
    labels = [str(x) for x in brain.get("labels", [])]
    top_by = brain.get("top_by_label", {}) if isinstance(brain.get("top_by_label", {}), dict) else {}
    candidates = []
    for lab in labels:
        if str(top_by.get(lab, top_for_public_label(lab))) != str(final_top):
            continue
        if not _is_loop_label_for_guard(brain, lab):
            continue
        parts = [p for p in public_label(lab).replace("\\", "/").split("/") if p]
        if len(parts) >= 3:
            candidates.append((len(parts), public_label(lab).lower(), lab))
    if not candidates:
        return ""
    candidates.sort()
    return candidates[0][2]


def _guard_candidate_has_structure_support(
    brain: dict, fingerprint: Sequence[float], label: str, min_score: float = 0.20
) -> Tuple[bool, str]:
    """Return whether a guard switch target has learned structure support.

    Physical guards are allowed to block catastrophic family errors, but they
    must not switch into an arbitrary rival leaf just because that rival appears
    somewhere in top5.  The target must agree with the learned structure head
    at least weakly.  Otherwise review the conflict instead of creating a new
    wrong category.
    """
    try:
        scores = structure_distance_scores(brain, fingerprint)
        score = _committee_structure_score_for_label(brain, str(label), scores)
        label_structure = str(
            (brain.get("structure_by_label", {}) or {}).get(str(label), label_default_structure(str(label)))
        )
        return bool(
            score >= float(min_score)
        ), f"target_structure={label_structure} structure_score={score:.3f} min={float(min_score):.3f}"
    except Exception as exc:
        return False, f"target_structure_check_error:{str(exc)[:80]}"


def physical_family_guard_decision(
    brain: dict,
    fingerprint: Sequence[float],
    final_label: str,
    final_top: str,
    predicted_top: str,
    top5: Sequence[Tuple[str, float]] | None,
    duration_sec: float,
) -> Tuple[str, str, str, str]:
    """Generic physics guard for catastrophic broad-family leaks.

    Purpose: catch the current Phase 4 failure mode where a physically obvious
    musical loop is auto-placed into FX, or a short front-loaded percussive hit is
    auto-placed into Instruments/FX. This uses only measured features plus
    already-nominated learned candidates. No filenames, source folder names,
    source archive names, or category-specific rescue words are used.

    It is intentionally conservative: switch only when a same-run candidate from
    the physically appropriate broad family already exists and passes membership;
    otherwise review instead of inventing a leaf.
    """
    fp = np.asarray(fingerprint, dtype=np.float32)
    transients = float(np.expm1(max(0.0, safe_feature_value(fp, 35))))
    flatness = safe_feature_value(fp, 29)
    entropy = safe_feature_value(fp, 30)
    slope = abs(safe_feature_value(fp, 34))
    temporal = safe_feature_value(fp, 42)
    attack = safe_feature_value(fp, 44)
    span = safe_feature_value(fp, 45)
    event_rate = safe_feature_value(fp, 46)
    tail = safe_feature_value(fp, 47)
    pitch = safe_feature_value(fp, 49)
    sub = safe_feature_value(fp, 36)
    bass = safe_feature_value(fp, 37)
    mid = safe_feature_value(fp, 38)
    high = safe_feature_value(fp, 39) + safe_feature_value(fp, 40)

    candidates = list(top5 or [])
    candidate_tops = {
        str((brain.get("top_by_label", {}) or {}).get(str(label), top_for_public_label(str(label))))
        for label, _score in candidates
    }

    # Tiny/short hits can miss the normal onset detector because there are too
    # few frames.  Treat a real, front-loaded, short non-silent burst as one
    # effective event for physical guards. This prevents 50-100 ms percussion
    # ticks from escaping into guitar/voice/brass simply because event_count=0.
    tiny_real_hit = bool(
        float(duration_sec) <= 0.20
        and temporal <= 0.34
        and attack <= 0.10
        and tail <= 0.38
        and ((sub + bass) >= 0.18 or high >= 0.16 or flatness >= 0.18 or entropy >= 0.28)
    )
    effective_events = max(float(transients), 1.0 if tiny_real_hit else 0.0)

    # General short-hit shape. This is not a category guess; it is a structure
    # fact. A sub-100 ms tick, a snare body hit, a tom, a rim click, a bell hit,
    # and a tiny designed impact all share this structure. Later family choice
    # still requires an already-nominated candidate from that family.
    front_loaded_short_hit_shape = bool(
        float(duration_sec) <= 1.10
        and effective_events <= 3.0
        and temporal <= 0.26
        and attack <= 0.11
        and tail <= 0.28
        and ((sub + bass) >= 0.18 or high >= 0.12 or flatness >= 0.10 or entropy >= 0.22 or pitch >= 0.45)
    )
    final_label_text = public_label(final_label).lower()
    percussion_leaf_context = bool(
        any(
            word in final_label_text
            for word in [
                "percussion",
                "tom",
                "snare",
                "kick",
                "rim",
                "stick",
                "bell",
                "metallic",
                "tabla",
                "bongo",
                "conga",
            ]
        )
        or ("cymbal" in final_label_text and (high >= 0.25 or flatness >= 0.22))
    )
    pitched_percussion_one_shot = bool(
        float(duration_sec) <= 1.25
        and effective_events <= 2.0
        and temporal <= 0.22
        and attack <= 0.065
        and tail <= 0.18
        and ((sub + bass) >= 0.25 or high >= 0.18 or flatness >= 0.18 or percussion_leaf_context)
        and not (float(duration_sec) >= 1.00 and tail >= 0.24 and temporal >= 0.24)
    )
    short_noisy_multi_percussion = bool(
        float(duration_sec) <= 1.50
        and effective_events >= 3.0
        and temporal <= 0.35
        and attack <= 0.085
        and tail <= 0.50
        and (flatness >= 0.30 or entropy >= 0.52 or event_rate >= 3.0)
    )

    # Long musical loop or phrase: sustained/multi-event, pitch-organized or
    # stable mid-band source material, not a steep transition, not broadband FX
    # noise. Reverb/tail can make sax, brass, voice, and keys look like FX in
    # the full-file view, so this guard treats source/body evidence separately
    # from space/tail. It never reads filenames; it only switches to learned
    # Instrument labels already present in the brain.
    low_air = bool(safe_feature_value(fp, 40) <= 0.10)
    mid_presence = float(mid + safe_feature_value(fp, 39))
    distributed_phrase = bool(
        float(duration_sec) >= 2.0
        and transients >= 5.0
        and span >= 0.60
        and 0.24 <= temporal <= 0.78
        and event_rate >= 0.70
    )
    stable_musical_body = bool(
        distributed_phrase
        and abs(safe_feature_value(fp, 34)) <= 0.14
        and low_air
        and (
            pitch >= 0.30
            or (flatness <= 0.34 and entropy <= 0.68 and mid_presence >= 0.34)
            or (flatness <= 0.14 and entropy <= 0.50 and mid >= 0.28)
        )
        and not (flatness >= 0.55 and entropy >= 0.72 and pitch < 0.30)
    )
    tonal_musical_loop = bool(
        float(duration_sec) >= 2.0
        and (transients >= 4.0 or span >= 0.35 or event_rate >= 1.0)
        and (
            pitch >= 0.42
            or (pitch >= 0.25 and flatness <= 0.35 and entropy <= 0.68)
            or (flatness <= 0.30 and entropy <= 0.58 and high <= 0.28)
            or stable_musical_body
        )
        and flatness <= 0.40
        and entropy <= 0.72
        and slope <= 0.18
    )
    final_fx_transition_motion_ok = False
    if str(final_top) == "FX" and _is_fx_transition_label_for_guard(final_label):
        final_fx_transition_motion_ok, _transition_motion_reason = _fx_transition_motion_supported(fp, duration_sec)
    if tonal_musical_loop and str(final_top) == "FX" and not final_fx_transition_motion_ok:
        label, score, _why = _first_unblocked_candidate_for_top(brain, fp, candidates, "Instruments")
        loop_label = "Instruments/Instrument Loops/Loops"
        # If the raw top-5 did not contain an Instrument, fall back only to a
        # learned generic Instrument Loops label and only for stable musical
        # body evidence. This avoids inventing sax/guitar/etc. while keeping
        # real downlifters and noisy FX in FX.
        if not label and stable_musical_body and loop_label in set(str(x) for x in brain.get("labels", [])):
            ev = folder_membership_evidence(brain, fp, loop_label)
            if not (bool(ev.get("enabled", False)) and bool(ev.get("blocked", False))):
                label, score = loop_label, 0.0
        if label:
            # Prefer the existing generic instrument-loop label when the audio is
            # structurally a loop and the candidate is only a one-shot leaf.
            if (str(label).endswith("/One Shots") or stable_musical_body) and loop_label in set(
                str(x) for x in brain.get("labels", [])
            ):
                ev = folder_membership_evidence(brain, fp, loop_label)
                if not (bool(ev.get("enabled", False)) and bool(ev.get("blocked", False))):
                    label = loop_label
            return (
                label,
                "Instruments",
                "switch",
                (
                    "physical_family_guard_switched_tonal_loop_to_instruments: "
                    f"from_top={final_top} predicted_top={predicted_top} chosen={public_label(label)} "
                    f"score={score:.3f} duration={float(duration_sec):.3f} events={transients:.1f} "
                    f"pitch={pitch:.3f} flatness={flatness:.3f} entropy={entropy:.3f} "
                    f"mid_presence={mid_presence:.3f} air={safe_feature_value(fp, 40):.3f} slope={slope:.3f}"
                ),
            )
        return (
            final_label,
            final_top,
            "review",
            (
                "physical_family_guard_review_tonal_loop_no_instrument_candidate: "
                f"from_top={final_top} duration={float(duration_sec):.3f} events={transients:.1f} "
                f"pitch={pitch:.3f} flatness={flatness:.3f} entropy={entropy:.3f} "
                f"mid_presence={mid_presence:.3f} air={safe_feature_value(fp, 40):.3f} slope={slope:.3f}"
            ),
        )

    # Short low-body pitched/ringing percussion guard.  Real toms, bongos,
    # tablas, low bells, and struck-object percussion can be strongly pitched
    # and have a ringing tail.  Earlier guards were too strict on tail length,
    # which let these sounds escape into Guitar/Bass/FX just because pitch was
    # strong.  This switch is intentionally candidate-led: it only fires when
    # a Drums candidate is already ranked near the top of the learned contest.
    low_body_pitched_percussion_ring = bool(
        str(final_top) != "Drums"
        and float(duration_sec) <= 0.85
        and effective_events <= 5.0
        and temporal <= 0.40
        and attack <= 0.085
        and tail <= 0.68
        and (sub + bass) >= 0.55
        and high <= 0.14
        and pitch >= 0.32
    )
    if low_body_pitched_percussion_ring:
        ranked_drum_candidates = []
        for rank, (cand_label, cand_score) in enumerate(candidates, start=1):
            cand_top = str(
                (brain.get("top_by_label", {}) or {}).get(str(cand_label), top_for_public_label(str(cand_label)))
            )
            if cand_top == "Drums":
                ranked_drum_candidates.append((rank, str(cand_label), float(cand_score)))
        if ranked_drum_candidates and ranked_drum_candidates[0][0] <= 3:
            _rank, drum_label, drum_score = ranked_drum_candidates[0]
            ev = folder_membership_evidence(brain, fp, drum_label)
            # Prefer the ranked drum candidate even when the human-readable
            # membership gate is unsure. The raw 100-feature brain already
            # nominated it near the top; the guard is only stopping catastrophic
            # family leakage.
            return (
                drum_label,
                "Drums",
                "switch",
                (
                    "physical_family_guard_switched_low_body_pitched_ring_to_drums: "
                    f"from_top={final_top} predicted_top={predicted_top} chosen={public_label(drum_label)} "
                    f"rank={_rank} score={drum_score:.3f} duration={float(duration_sec):.3f} "
                    f"events={transients:.1f} effective_events={effective_events:.1f} low_total={(sub + bass):.3f} "
                    f"high={high:.3f} pitch={pitch:.3f} temporal={temporal:.3f} attack={attack:.3f} tail={tail:.3f} "
                    f"membership_blocked={bool(ev.get('blocked', False))}"
                ),
            )

    # Voice/human transient guard.  Processed shouts, screams, breath/mouth
    # sounds, and short vocal FX can be front-loaded enough to resemble claps
    # or rims.  Do not let the generic short-hit drum rescue swallow them when
    # the measured physics has mid-band pitched/formant-like behavior and the
    # candidate list already contains a voice/human label.  This uses no names
    # from the source file; only the learned candidate labels and measured
    # fingerprint are consulted.
    voice_like_short_event = bool(
        str(final_top) == "Drums"
        and float(duration_sec) >= 0.42
        and (sub + bass) <= 0.38
        and mid >= 0.36
        and high <= 0.62
        and pitch >= 0.28
        and (tail >= 0.055 or span >= 0.12 or float(duration_sec) >= 0.85)
        and not (float(duration_sec) <= 0.34 and tail <= 0.10 and temporal <= 0.16)
        and not (short_low_drum_one_shot if "short_low_drum_one_shot" in locals() else False)
    )
    if voice_like_short_event:
        best_voice_label = ""
        best_voice_score = 0.0
        for cand_label, cand_score in candidates:
            cand_text = public_label(str(cand_label)).lower()
            cand_top = str(
                (brain.get("top_by_label", {}) or {}).get(str(cand_label), top_for_public_label(str(cand_label)))
            )
            if cand_top not in {"Instruments", "FX"}:
                continue
            if not any(
                word in cand_text
                for word in ["voice", "vocal", "choir", "scream", "breath", "mouth", "spoken", "crowd"]
            ):
                continue
            ev = folder_membership_evidence(brain, fp, str(cand_label))
            if bool(ev.get("enabled", False)) and bool(ev.get("blocked", False)):
                continue
            best_voice_label = str(cand_label)
            best_voice_score = float(cand_score)
            break
        if best_voice_label:
            chosen_top = str(
                (brain.get("top_by_label", {}) or {}).get(best_voice_label, top_for_public_label(best_voice_label))
            )
            return (
                best_voice_label,
                chosen_top,
                "switch",
                (
                    "physical_family_guard_switched_drum_to_voice_human_candidate: "
                    f"from_top={final_top} predicted_top={predicted_top} chosen={public_label(best_voice_label)} "
                    f"score={best_voice_score:.3f} duration={float(duration_sec):.3f} events={transients:.1f} "
                    f"low_total={(sub + bass):.3f} mid={mid:.3f} high={high:.3f} pitch={pitch:.3f} "
                    f"span={span:.3f} tail={tail:.3f} temporal={temporal:.3f}"
                ),
            )
        # No voice/human candidate was nominated by the brain.  Do not invent
        # a voice label and do not punish a likely drum hit; let the rest of the
        # physical guards continue.

    f0_voiced = safe_feature_value(fp, FEATURE_NAMES.index("f0_voiced_ratio"))
    harmonic_energy = safe_feature_value(fp, FEATURE_NAMES.index("harmonic_energy_ratio"))
    attack_pitch = safe_feature_value(fp, FEATURE_NAMES.index("attack_pitch_confidence"))
    body_pitch = safe_feature_value(fp, FEATURE_NAMES.index("body_pitch_confidence"))
    inharmonicity = safe_feature_value(fp, FEATURE_NAMES.index("inharmonicity"))
    noise_burst_ms = safe_feature_value(fp, FEATURE_NAMES.index("noise_burst_duration_ms"))

    # Voiced/formant-like material can be rhythmic and transient enough to look
    # like drum loops, claps, or rims in the raw distance space.  When measured
    # voicing and mid-band evidence say the audio is a poor drum fit, do not
    # leave it in Drums.  This never reads filenames and does not invent a vocal
    # category; it switches only to an already nominated non-Drums candidate or
    # sends the file to review.
    voiced_formant_non_drum = bool(
        str(final_top) == "Drums"
        and float(duration_sec) >= 0.35
        and (sub + bass) <= 0.38
        and mid >= 0.35
        and high <= 0.72
        and pitch >= 0.38
        and f0_voiced >= 0.62
        and harmonic_energy >= 0.10
        and (attack_pitch >= 0.40 or body_pitch >= 0.40)
        and not (float(duration_sec) <= 0.24 and temporal <= 0.24 and tail <= 0.20 and f0_voiced <= 0.45)
        and not (inharmonicity >= 0.55 and harmonic_energy <= 0.16 and (sub + bass) >= 0.45)
    )
    if voiced_formant_non_drum:
        best_label = ""
        best_top = ""
        best_score = 0.0
        for cand_label, cand_score in candidates:
            cand_label = str(cand_label)
            cand_top = str((brain.get("top_by_label", {}) or {}).get(cand_label, top_for_public_label(cand_label)))
            if cand_top == "Drums":
                continue
            ev = folder_membership_evidence(brain, fp, cand_label)
            if bool(ev.get("enabled", False)) and bool(ev.get("blocked", False)):
                continue
            best_label = cand_label
            best_top = cand_top
            best_score = float(cand_score)
            break
        detail = (
            "physical_family_guard_voiced_formant_non_drum: "
            f"from_top={final_top} predicted_top={predicted_top} duration={float(duration_sec):.3f} "
            f"events={transients:.1f} low_total={(sub + bass):.3f} mid={mid:.3f} high={high:.3f} "
            f"pitch={pitch:.3f} f0_voiced={f0_voiced:.3f} harmonic_energy={harmonic_energy:.3f} "
            f"attack_pitch={attack_pitch:.3f} body_pitch={body_pitch:.3f} inharmonicity={inharmonicity:.3f} "
            f"noise_burst_ms={noise_burst_ms:.1f}"
        )
        if best_label:
            return (
                best_label,
                best_top,
                "switch",
                (
                    "physical_family_guard_switched_voiced_formant_non_drum_to_candidate: "
                    f"chosen={public_label(best_label)} score={best_score:.3f}; {detail}"
                ),
            )
        return (
            "_TO_REVIEW/Conflicting Evidence/Voiced Formant Non Drum",
            "_TO_REVIEW",
            "review",
            (detail + "; no_unblocked_non_drum_candidate"),
        )

    # Repeated events alone do not prove a drum loop.  Rap/vocal phrases,
    # reverby instrument loops, and some FX beds can all be rhythmic.  If the
    # chosen family is Drums/loop but measured body/top/noise evidence is not
    # drum-like, block the drum placement and prefer an already nominated
    # non-drum candidate.  This is intentionally physics-first and does not
    # read filenames or source folders.
    if str(final_top) == "Drums" and _is_loop_label_for_guard(brain, final_label):
        drum_loop_ok, drum_loop_reason = _drum_loop_physics_supported(fp, duration_sec)
        if not drum_loop_ok:
            best_label = ""
            best_top = ""
            best_score = 0.0
            for cand_label, cand_score in candidates:
                cand_label = str(cand_label)
                cand_top = str((brain.get("top_by_label", {}) or {}).get(cand_label, top_for_public_label(cand_label)))
                if cand_top == "Drums":
                    continue
                ev = folder_membership_evidence(brain, fp, cand_label)
                if bool(ev.get("enabled", False)) and bool(ev.get("blocked", False)):
                    continue
                best_label = cand_label
                best_top = cand_top
                best_score = float(cand_score)
                break
            detail = (
                "physical_family_guard_drum_loop_rejected_by_non_drum_physics: "
                f"from_top={final_top} predicted_top={predicted_top}; {drum_loop_reason}"
            )
            if best_label:
                structure_ok, structure_note = _guard_candidate_has_structure_support(
                    brain,
                    fp,
                    best_label,
                    float(brain.get("physical_guard_switch_min_structure_score", 0.20) or 0.20),
                )
                if structure_ok:
                    return (
                        best_label,
                        best_top,
                        "switch",
                        (
                            "physical_family_guard_switched_unsupported_drum_loop_to_candidate: "
                            f"chosen={public_label(best_label)} score={best_score:.3f}; {structure_note}; {detail}"
                        ),
                    )
                return (
                    "_TO_REVIEW/Conflicting Evidence/Unsupported Drum Loop Physics",
                    "_TO_REVIEW",
                    "review",
                    (detail + "; blocked_non_drum_switch_due_to_weak_target_structure; " + structure_note),
                )
            return (
                "_TO_REVIEW/Conflicting Evidence/Unsupported Drum Loop Physics",
                "_TO_REVIEW",
                "review",
                detail + "; no_unblocked_non_drum_candidate",
            )

    # Symmetric loop-family repair for the opposite failure: a measured drum
    # loop, especially a sub-heavy kick pulse loop, can be laundered into a bass
    # or generic Instrument loop because the tone is strong and the top end is
    # sparse. Only use this when the raw broad family already said Drums.
    if str(final_top) != "Drums" and _is_loop_label_for_guard(brain, final_label) and str(predicted_top) == "Drums":
        drum_loop_ok, drum_loop_reason = _drum_loop_physics_supported(fp, duration_sec)
        if drum_loop_ok:
            best_label = ""
            best_score = 0.0
            best_rank = 0
            for rank, (cand_label, cand_score) in enumerate(candidates, start=1):
                cand_label = str(cand_label)
                cand_top = str((brain.get("top_by_label", {}) or {}).get(cand_label, top_for_public_label(cand_label)))
                if cand_top != "Drums":
                    continue
                if not _is_loop_label_for_guard(brain, cand_label):
                    continue
                ev = folder_membership_evidence(brain, fp, cand_label)
                if bool(ev.get("enabled", False)) and bool(ev.get("blocked", False)):
                    continue
                best_label = cand_label
                best_score = float(cand_score)
                best_rank = rank
                break
            if best_label:
                generic_loop_label = _canonical_learned_top_loop_label(brain, "Drums")
                if generic_loop_label:
                    ev = folder_membership_evidence(brain, fp, generic_loop_label)
                    if not (bool(ev.get("enabled", False)) and bool(ev.get("blocked", False))):
                        best_label = generic_loop_label
                return (
                    best_label,
                    "Drums",
                    "switch",
                    (
                        "physical_family_guard_switched_supported_drum_loop_back_to_drums: "
                        f"from_top={final_top} predicted_top={predicted_top} chosen={public_label(best_label)} "
                        f"rank={best_rank} score={best_score:.3f}; {drum_loop_reason}"
                    ),
                )
            low_pulse_ok, low_pulse_reason = _low_pulse_drum_loop_supported(fp, duration_sec)
            if low_pulse_ok:
                generic_loop_label = _canonical_learned_top_loop_label(brain, "Drums")
                if generic_loop_label:
                    ev = folder_membership_evidence(brain, fp, generic_loop_label)
                    if not (bool(ev.get("enabled", False)) and bool(ev.get("blocked", False))):
                        return (
                            generic_loop_label,
                            "Drums",
                            "switch",
                            (
                                "physical_family_guard_switched_supported_low_pulse_loop_back_to_drums: "
                                f"from_top={final_top} predicted_top={predicted_top} "
                                f"chosen={public_label(generic_loop_label)}; {low_pulse_reason}"
                            ),
                        )

    if str(final_top) == "FX" and _is_fx_transition_label_for_guard(final_label):
        transition_ok, transition_reason = _fx_transition_motion_supported(fp, duration_sec)
        if not transition_ok:
            alt_label, alt_score = _first_unblocked_non_transition_fx_candidate(brain, fp, candidates)
            if alt_label:
                return (
                    alt_label,
                    "FX",
                    "switch",
                    (
                        "physical_family_guard_switched_unsupported_fx_transition_to_fx_candidate: "
                        f"from_label={public_label(final_label)} chosen={public_label(alt_label)} "
                        f"score={alt_score:.3f}; {transition_reason}"
                    ),
                )

    # Short, front-loaded hit: this is the generic percussive one-shot shape.
    # Do not let a guitar/voice/dog/siren-style leaf auto-place unless there is
    # stronger non-drum evidence and no drum candidate in the learned top set.
    short_percussive_hit = bool(
        front_loaded_short_hit_shape
        and (effective_events >= 1.0 or event_rate >= 0.5)
        and not (pitch >= 0.86 and flatness <= 0.030 and high <= 0.003 and sub <= 0.02 and tail >= 0.16)
    )
    if short_percussive_hit and str(final_top) != "Drums":
        label, score, _why = _first_unblocked_candidate_for_top(brain, fp, candidates, "Drums")
        if label:
            return (
                label,
                "Drums",
                "switch",
                (
                    "physical_family_guard_switched_short_hit_to_drums: "
                    f"from_top={final_top} predicted_top={predicted_top} chosen={public_label(label)} "
                    f"score={score:.3f} duration={float(duration_sec):.3f} events={transients:.1f} effective_events={effective_events:.1f} "
                    f"tiny_real_hit={tiny_real_hit} temporal={temporal:.3f} attack={attack:.3f} tail={tail:.3f} pitch={pitch:.3f}"
                ),
            )
        if str(final_top) == "FX" or "_Ambiguous Leaf" in str(final_label):
            return (
                final_label,
                final_top,
                "review",
                (
                    "physical_family_guard_review_short_hit_no_unblocked_drum_candidate: "
                    f"from_top={final_top} duration={float(duration_sec):.3f} events={transients:.1f} "
                    f"temporal={temporal:.3f} attack={attack:.3f} tail={tail:.3f} pitch={pitch:.3f}"
                ),
            )

    # Pitched drums exist.  A tom/tabla/low percussion hit can have strong pitch
    # confidence, but if it is short, low-heavy, front-loaded, and has a fast
    # attack with little tail, the later tonal-non-drum rescue must not launder
    # it back into Instruments.
    short_low_drum_one_shot = bool(
        front_loaded_short_hit_shape
        and effective_events <= 5.0
        and (sub >= 0.28 or (sub + bass) >= 0.35 or bass >= 0.18)
    )
    short_drum_or_metallic_one_shot = bool(
        front_loaded_short_hit_shape
        and (
            short_low_drum_one_shot
            or pitched_percussion_one_shot
            or percussion_leaf_context
            or (high >= 0.18 and (flatness >= 0.08 or entropy >= 0.22))
        )
    )

    # Clean tonal one-shots can be real instrument stabs/plucks/chords even if
    # they are short. Keep the old cymbal-regression guard, but require a clean
    # tonal/non-percussive profile: no low-body hit evidence, no metallic/noisy
    # percussive evidence, and no percussion leaf context. This keeps drums from
    # becoming guitar while still rescuing true tonal stabs from cymbal labels.
    clean_tonal_non_drum_one_shot = bool(
        float(duration_sec) <= 1.10
        and pitch >= 0.55
        and flatness <= 0.08
        and entropy >= 0.42
        and (sub + bass) <= 0.18
        and high <= 0.16
        and not short_drum_or_metallic_one_shot
        and not percussion_leaf_context
    )

    # v0.6.1: tonal musical material must not be laundered into Drums just
    # because its attack, shimmer, or rhythm resembles cymbal/snare/percussion.
    # This catches piano/organ/chord/arp/string/brass/guitar-style failures
    # without reading names or folders.  It is intentionally broad-family only:
    # switch to an already nominated Instrument candidate, otherwise review.
    tonal_non_drum_music = bool(
        str(final_top) == "Drums"
        and pitch >= 0.42
        and flatness <= 0.30
        and entropy <= 0.72
        and not short_low_drum_one_shot
        and not pitched_percussion_one_shot
        and not short_drum_or_metallic_one_shot
        and not (
            # preserve genuinely short, bright, dry cymbal/hat/rim-like hits
            float(duration_sec) <= 1.15
            and attack <= 0.018
            and temporal <= 0.18
            and tail <= 0.12
            and high >= 0.42
            and pitch < 0.55
        )
        and (
            float(duration_sec) >= 1.20
            or transients >= 3.0
            or tail >= 0.20
            or temporal >= 0.22
            or mid >= 0.25
            or clean_tonal_non_drum_one_shot
            # Strong pitch in a short hit is not enough to call it an
            # Instrument. Toms, tablas, bells, rims, snares, and tiny impacts
            # can all have pitch. For non-clean tonal material, require some
            # time-based musical behavior before this rescue can fire.
            or (pitch >= 0.55 and high <= 0.12 and float(duration_sec) >= 1.20 and temporal >= 0.22)
        )
    )
    if tonal_non_drum_music:
        label, score, _why = _first_unblocked_candidate_for_top(brain, fp, candidates, "Instruments")
        if label:
            return (
                label,
                "Instruments",
                "switch",
                (
                    "physical_family_guard_switched_tonal_non_drum_to_instruments: "
                    f"from_top={final_top} predicted_top={predicted_top} chosen={public_label(label)} "
                    f"score={score:.3f} duration={float(duration_sec):.3f} events={transients:.1f} "
                    f"pitch={pitch:.3f} flatness={flatness:.3f} entropy={entropy:.3f} "
                    f"mid={mid:.3f} high={high:.3f} tail={tail:.3f} temporal={temporal:.3f}"
                ),
            )
        return (
            final_label,
            final_top,
            "review",
            (
                "physical_family_guard_review_tonal_non_drum_no_instrument_candidate: "
                f"from_top={final_top} duration={float(duration_sec):.3f} events={transients:.1f} "
                f"pitch={pitch:.3f} flatness={flatness:.3f} entropy={entropy:.3f} "
                f"mid={mid:.3f} high={high:.3f} tail={tail:.3f} temporal={temporal:.3f}"
            ),
        )

    # Short noisy multi-hit bursts are often shakers, sticks, metallic ticks,
    # or other percussion.  The Animal/Bird FX label is allowed to win only when
    # the learned evidence beats the percussion candidate on its own merits; if
    # the run already nominated a Drums candidate and the measured shape is a
    # short percussive burst, prefer Drums over Bird/Animal.
    animal_like_fx_label = bool(
        str(final_top) == "FX"
        and re.search(
            r"/(bird|animal|creature|cat|dog)\b|\b(bird|animal|creature|cat|dog)\b", public_label(final_label).lower()
        )
    )
    if animal_like_fx_label and short_noisy_multi_percussion:
        label, score, _why = _first_unblocked_candidate_for_top(brain, fp, candidates, "Drums")
        if label:
            return (
                label,
                "Drums",
                "switch",
                (
                    "physical_family_guard_switched_short_noisy_animal_fx_to_drums: "
                    f"from_top={final_top} predicted_top={predicted_top} chosen={public_label(label)} "
                    f"score={score:.3f} duration={float(duration_sec):.3f} events={transients:.1f} effective_events={effective_events:.1f} "
                    f"rate={event_rate:.3f} flatness={flatness:.3f} entropy={entropy:.3f} temporal={temporal:.3f} tail={tail:.3f}"
                ),
            )
        return (
            final_label,
            final_top,
            "review",
            (
                "physical_family_guard_review_short_noisy_animal_fx_no_drum_candidate: "
                f"duration={float(duration_sec):.3f} events={transients:.1f} effective_events={effective_events:.1f} "
                f"rate={event_rate:.3f} flatness={flatness:.3f} entropy={entropy:.3f} temporal={temporal:.3f} tail={tail:.3f}"
            ),
        )

    # v0.6.0: rhythmic/multi-event drum-roll or beat-like material must not be
    # auto-placed as risers, downlifters, animals, ambience, or generic FX.  This
    # is deliberately broad-family only; it chooses an existing Drums candidate or
    # reviews. It does not invent a fill/snare/kick label.
    rhythmic_drum_candidate = bool(
        str(final_top) == "FX"
        and not final_fx_transition_motion_ok
        and float(duration_sec) >= 1.0
        and (transients >= 4.0 or event_rate >= 1.0)
        and (span >= 0.10 or temporal <= 0.20 or tail <= 0.18)
        and abs(slope) <= 0.16
        and not (flatness >= 0.55 and entropy >= 0.72 and pitch < 0.30)
    )
    if rhythmic_drum_candidate:
        label, score, _why = _first_unblocked_candidate_for_top(brain, fp, candidates, "Drums")
        if label:
            return (
                label,
                "Drums",
                "switch",
                (
                    "physical_family_guard_switched_rhythmic_fx_to_drums: "
                    f"from_top={final_top} predicted_top={predicted_top} chosen={public_label(label)} "
                    f"score={score:.3f} duration={float(duration_sec):.3f} events={transients:.1f} "
                    f"span={span:.3f} rate={event_rate:.3f} slope={slope:.3f} tail={tail:.3f}"
                ),
            )
        return (
            final_label,
            final_top,
            "review",
            (
                "physical_family_guard_review_rhythmic_fx_no_drum_candidate: "
                f"from_top={final_top} duration={float(duration_sec):.3f} events={transients:.1f} "
                f"span={span:.3f} rate={event_rate:.3f} slope={slope:.3f} tail={tail:.3f}"
            ),
        )

    if str(final_top) == "FX" and _is_fx_transition_label_for_guard(final_label):
        transition_ok, transition_reason = _fx_transition_motion_supported(fp, duration_sec)
        if not transition_ok:
            alt_label, alt_score = _first_unblocked_non_transition_fx_candidate(brain, fp, candidates)
            if alt_label:
                return (
                    alt_label,
                    "FX",
                    "switch",
                    (
                        "physical_family_guard_switched_unsupported_fx_transition_to_fx_candidate: "
                        f"from_label={public_label(final_label)} chosen={public_label(alt_label)} "
                        f"score={alt_score:.3f}; {transition_reason}"
                    ),
                )
            return (
                "_TO_REVIEW/Conflicting Evidence/Unsupported FX Transition Physics",
                "_TO_REVIEW",
                "review",
                (
                    "physical_family_guard_review_unsupported_fx_transition: "
                    f"from_label={public_label(final_label)}; {transition_reason}; no_unblocked_non_transition_fx_candidate"
                ),
            )

    return "", "", "pass", "physical_family_guard_no_action"


def apply_learned_conflict_gates(
    brain: dict,
    fingerprint: Sequence[float],
    pred_label: str,
    pred_top: str,
    final_top: str,
    final_label: str,
    confidence_status: str,
    review_reason: str,
    similarity: float = 0.0,
    margin: float = 0.0,
    duration_sec: float = 0.0,
    top5: Sequence[Tuple[str, float]] | None = None,
) -> Tuple[str, str, str, str]:
    """Apply learned safety checks as audit flags, not default review routing.

    v0.4.77 decisive product policy: after the basic read/no-match gate, the
    sorter keeps the best learned-folder placement. Learned conflicts become
    manifest audit reasons so Aaron gets an actually sorted library plus a clear
    list of risky decisions to inspect.
    """
    if confidence_status != "auto_place":
        return final_top, final_label, confidence_status, review_reason

    audit_reasons: List[str] = []

    adaptive_label, adaptive_reason = adaptive_depth_placement_label(brain, pred_label, similarity, margin, top5)
    if adaptive_label:
        final_label = adaptive_label
        final_top = top_for_public_label(adaptive_label)
        audit_reasons.append(adaptive_reason)

    reliability_reason = learned_label_reliability_reason(brain, pred_label, similarity, margin)
    if reliability_reason:
        audit_reasons.append("low_label_support_pick: " + reliability_reason)

    borderline_reason = borderline_auto_place_reason(similarity, margin)
    if borderline_reason:
        try:
            rel = (
                brain.get("label_reliability_by_label", {})
                if isinstance(brain.get("label_reliability_by_label", {}), dict)
                else {}
            )
            info = rel.get(pred_label, {}) if isinstance(rel, dict) else {}
            count = (
                int(info.get("training_count", brain.get("counts", {}).get(pred_label, 0)) or 0)
                if isinstance(info, dict)
                else 0
            )
            req_sim = (
                float(info.get("min_similarity_for_auto_place", TINY_LABEL_MIN_SIMILARITY) or TINY_LABEL_MIN_SIMILARITY)
                if isinstance(info, dict)
                else TINY_LABEL_MIN_SIMILARITY
            )
            req_margin = (
                float(info.get("min_margin_for_auto_place", TINY_LABEL_MIN_MARGIN) or TINY_LABEL_MIN_MARGIN)
                if isinstance(info, dict)
                else TINY_LABEL_MIN_MARGIN
            )
            if (
                count <= SMALL_LABEL_COUNT
                and float(similarity) >= req_sim
                and float(margin) >= max(req_margin * 1.25, req_margin + 0.25)
            ):
                borderline_reason = ""
        except Exception:
            pass
    if borderline_reason:
        audit_reasons.append("borderline_forced_pick: " + borderline_reason)

    ensemble_meta = brain.get("_last_ensemble_meta", {})
    if isinstance(ensemble_meta, dict) and ensemble_meta.get("mode") == "model_tournament_ensemble":
        if ensemble_meta.get("switched"):
            audit_reasons.append(
                "model_tournament_switched_pick: "
                + f"current={public_label(str(ensemble_meta.get('current_label', '')))} "
                + f"final={public_label(str(ensemble_meta.get('final_label', '')))} "
                + f"advantage={float(ensemble_meta.get('ensemble_advantage', 0.0) or 0.0):.3f}"
            )
        else:
            audit_reasons.append(
                "model_tournament_confirmed_pick: " + f"top_votes={str(ensemble_meta.get('top_ensemble', ''))[:180]}"
            )

    committee_meta = brain.get("_last_committee_meta", {})
    if isinstance(committee_meta, dict) and committee_meta.get("mode") == "committee_agreement":
        if committee_meta.get("switched"):
            audit_reasons.append(
                "committee_agreement_switched_pick: "
                + f"from={public_label(str(committee_meta.get('from_label', '')))} "
                + f"to={public_label(str(committee_meta.get('to_label', '')))} "
                + f"advantage={float(committee_meta.get('advantage', 0.0) or 0.0):.3f} "
                + str(committee_meta.get("reason", ""))
            )
        else:
            top_txt = str(committee_meta.get("top_committee", ""))[:240]
            audit_reasons.append("committee_agreement_confirmed_pick: " + top_txt)
        chosen_membership = committee_meta.get("chosen_membership", {}) if isinstance(committee_meta, dict) else {}
        if isinstance(chosen_membership, dict) and bool(chosen_membership.get("blocked", False)):
            audit_reasons.append(
                "committee_chosen_membership_blocked: " + format_membership_evidence(chosen_membership)
            )

    router_meta = brain.get("_last_frontend_router_meta", {})
    if isinstance(router_meta, dict) and router_meta.get("mode") == "frontend_router_brain":
        audit_reasons.append(
            "frontend_router_brain_tiebreaker: "
            + f"top={str(router_meta.get('top3_top', ''))[:90]} "
            + f"structure={str(router_meta.get('top3_structure', ''))[:70]} "
            + f"parent={str(router_meta.get('top3_parent', ''))[:110]} "
            + f"chosen_penalty={float(router_meta.get('final_label_router_penalty', 0.0) or 0.0):.3f}"
        )

    # v0.6.5: Coherent feature-group outliers are not cosmetic audit notes.
    # If the brain's own learned fact profile says an entire physics group is
    # severely outside the selected label, exact auto-placement is unsafe.
    # This remains label-agnostic: the penalty was computed from learned
    # per-folder p10/p90/MAD ranges, not filenames or hand-written category picks.
    fact_meta = brain.get("_last_fact_meta", {})
    if isinstance(fact_meta, dict):
        try:
            coherent_penalty = float(fact_meta.get("coherent_group_penalty", 0.0) or 0.0)
        except Exception:
            coherent_penalty = 0.0
        coherent_reason = str(fact_meta.get("coherent_group_reason", "") or "")
        severe_threshold = float(brain.get("severe_coherent_group_review_threshold", 2.0) or 2.0)
        if (
            bool(brain.get("severe_coherent_group_review_enabled", True))
            and coherent_penalty >= severe_threshold
            and coherent_reason
        ):
            audit_reasons.append(
                "severe_coherent_group_outlier_pick: "
                + f"coherent_group_penalty={coherent_penalty:.3f} >= {severe_threshold:.3f}; "
                + coherent_reason[:320]
            )

    sibling_reason = sibling_imbalance_ambiguity_reason(brain, pred_label, similarity, margin, top5)
    if sibling_reason:
        audit_reasons.append("imbalanced_rival_forced_pick: " + sibling_reason)

    structure_gate_label, structure_gate_reason = dynamic_structure_gate_destination(
        brain, fingerprint, pred_label, duration_sec
    )
    if structure_gate_label:
        final_label = structure_gate_label
        final_top = top_for_public_label(structure_gate_label)
        audit_reasons.append(structure_gate_reason)
    else:
        physical_structure_reason = physical_structure_conflict_reason(brain, fingerprint, pred_label, duration_sec)
        if physical_structure_reason:
            audit_reasons.append("physical_structure_risk_pick: " + physical_structure_reason)

    anchor_reason = label_anchor_consensus_reason(brain, pred_label, fingerprint)
    if anchor_reason:
        audit_reasons.append("weak_label_neighborhood_pick: " + anchor_reason)

    structure_reason = learned_structure_conflict_reason(brain, fingerprint, pred_label)
    if structure_reason:
        audit_reasons.append("learned_structure_risk_pick: " + structure_reason)

    top_reason = learned_top_conflict_reason(brain, fingerprint, final_top)
    committee_switched_to_final_top = False
    if isinstance(committee_meta, dict) and committee_meta.get("switched"):
        committee_to = str(committee_meta.get("to_label", ""))
        committee_to_top = str(
            (brain.get("top_by_label", {}) or {}).get(committee_to, top_for_public_label(committee_to))
        )
        committee_switched_to_final_top = bool(committee_to_top == str(final_top))
    if top_reason:
        audit_reasons.append("learned_top_family_risk_pick: " + top_reason)
        if not committee_switched_to_final_top:
            top_label, top_family, top_action, top_safety_reason = learned_top_family_safety_decision(
                brain, fingerprint, final_label, final_top, pred_top, top5
            )
            if top_action == "switch":
                audit_reasons.append(top_safety_reason)
                final_label = top_label
                final_top = top_family
            elif top_action == "review":
                audit_reasons.append(top_safety_reason)
                base = review_reason if review_reason else "forced_best_learned_folder_pick"
                if not base.startswith("AUDIT"):
                    base = "AUDIT " + base
                return "_TO_REVIEW", top_label or final_label, "review", base + " || " + " || ".join(audit_reasons)
        else:
            audit_reasons.append("learned_top_family_safety_skipped: committee_already_switched_to_final_top")

    # v0.5.1: Learned-folder membership is the product safety gate.  Ranking
    # alone is not enough.  If the selected folder violates too many reliable
    # physics facts from its own learned profile, do not exact-place it.
    membership_label, membership_top, membership_action, membership_evidence = learned_membership_gate_decision(
        brain=brain,
        fingerprint=fingerprint,
        final_label=final_label,
        final_top=final_top,
        top5=top5,
        predicted_top=pred_top,
    )
    brain["_last_membership_meta"] = membership_evidence if isinstance(membership_evidence, dict) else {}
    if membership_action == "switch":
        audit_reasons.append(
            "learned_membership_gate_switched_pick: " + str(membership_evidence.get("alternative_reason", ""))
        )
        final_label = membership_label
        final_top = membership_top
    elif membership_action == "review":
        audit_reasons.append(
            str(membership_evidence.get("review_reason", "learned_membership_gate_blocked_exact_leaf"))
        )
        base = review_reason if review_reason else "forced_best_learned_folder_pick"
        if not base.startswith("AUDIT"):
            base = "AUDIT " + base
        return "_TO_REVIEW", membership_label, "review", base + " || " + " || ".join(audit_reasons)
    elif isinstance(membership_evidence, dict) and membership_evidence.get("warning"):
        audit_reasons.append("learned_membership_gate_warning: " + format_membership_evidence(membership_evidence))

    # Membership can switch to a safer leaf after the broad top-family check
    # above. Re-check broad family agreement after that switch so a membership
    # safety repair does not accidentally become a cross-family misroute.
    post_top_reason = learned_top_conflict_reason(brain, fingerprint, final_top)
    if post_top_reason and not committee_switched_to_final_top:
        top_label, top_family, top_action, top_safety_reason = learned_top_family_safety_decision(
            brain, fingerprint, final_label, final_top, pred_top, top5
        )
        if top_action == "switch":
            audit_reasons.append("post_membership_top_family_safety: " + top_safety_reason)
            final_label = top_label
            final_top = top_family
        elif top_action == "review":
            audit_reasons.append("post_membership_top_family_safety: " + top_safety_reason)
            base = review_reason if review_reason else "forced_best_learned_folder_pick"
            if not base.startswith("AUDIT"):
                base = "AUDIT " + base
            return "_TO_REVIEW", top_label or final_label, "review", base + " || " + " || ".join(audit_reasons)

    phys_label, phys_top, phys_action, phys_reason = physical_family_guard_decision(
        brain, fingerprint, final_label, final_top, pred_top, top5, duration_sec
    )
    if phys_action == "switch":
        audit_reasons.append(phys_reason)
        final_label = phys_label
        final_top = phys_top
    elif phys_action == "review":
        audit_reasons.append(phys_reason)
        base = review_reason if review_reason else "forced_best_learned_folder_pick"
        if not base.startswith("AUDIT"):
            base = "AUDIT " + base
        return "_TO_REVIEW", phys_label or final_label, "review", base + " || " + " || ".join(audit_reasons)

    # v0.6.6: when measured structure proves a loop and the final broad family
    # is already Drums, avoid false leaf precision such as clap/snare/hat/percussion
    # loops for mixed or top-only beat material.  Only collapse when the current
    # placement is already an ambiguous dynamic leaf or the audit trail says the
    # leaf identity was risky.  Specific clean loop leaves can still survive.
    try:
        structure_by = (
            brain.get("structure_by_label", {}) if isinstance(brain.get("structure_by_label", {}), dict) else {}
        )
        final_structure = str(
            structure_by.get(final_label, label_default_structure(final_label)) or label_default_structure(final_label)
        )
        measured_loop, measured_loop_reason = fingerprint_loop_evidence_detail(fingerprint, duration_sec)
        leaf_risky = ("_Ambiguous Leaf" in public_label(final_label)) or any(
            "adaptive_depth_pick" in x for x in audit_reasons
        )
        if final_top == "Drums" and final_structure == "loop" and measured_loop and leaf_risky:
            generic_loop_label = _canonical_learned_top_loop_label(brain, "Drums")
            if generic_loop_label and generic_loop_label != final_label:
                final_label = generic_loop_label
                final_top = "Drums"
                audit_reasons.append(
                    "generic_loop_family_placement: measured loop plus risky drum leaf contest; "
                    f"placed_under_broad_learned_loop_label={public_label(generic_loop_label)}; {measured_loop_reason}"
                )
    except Exception as exc:
        audit_reasons.append(f"generic_loop_family_placement_error:{str(exc)[:80]}")

    if audit_reasons:
        base = review_reason if review_reason else "forced_best_learned_folder_pick"
        if not base.startswith("AUDIT"):
            base = "AUDIT " + base
        audit_text = base + " || " + " || ".join(audit_reasons)

        # v0.6.0: audits that describe unsafe evidence must not still produce an
        # auto-sorted file. These are not cosmetic notes; they are the exact
        # situations that created wrong-family placements in the big real run.
        hard_review_markers = (
            # Restored: these were safety nets for genuinely risky predictions.
            # Removing them caused wrong auto-placements on borderline/weak picks.
            "weak_label_neighborhood_pick",
            "learned_top_family_risk_pick",
            "physical_structure_risk_pick",
            "imbalanced_rival_forced_pick",
            "learned_membership_gate_warning",
            "committee_chosen_membership_blocked",
            # v0.6.5 addition: coherent physics group outlier is a hard blocker.
            "severe_coherent_group_outlier_pick",
        )

        # v0.6.1: the big real run showed that the raw/prototype brain often
        # had the least-bad family while later ensemble/committee heads jumped
        # across top families.  A cross-family switch is not a safe correction;
        # it is a review signal unless a later physical-family guard made an
        # explicit measured-physics switch.  No filenames or folder names are used.
        def _top_for_label_name(_label: str) -> str:
            return str((brain.get("top_by_label", {}) or {}).get(str(_label), top_for_public_label(str(_label))))

        cross_family_switch = False
        if isinstance(ensemble_meta, dict) and ensemble_meta.get("switched"):
            from_top = _top_for_label_name(str(ensemble_meta.get("current_label", "")))
            to_top = _top_for_label_name(str(ensemble_meta.get("final_label", "")))
            if from_top and to_top and from_top != to_top:
                cross_family_switch = True
                audit_text += (
                    f" || safety_policy_cross_family_model_tournament_switch from_top={from_top} to_top={to_top}"
                )
        if isinstance(committee_meta, dict) and committee_meta.get("switched"):
            from_top = _top_for_label_name(str(committee_meta.get("from_label", "")))
            to_top = _top_for_label_name(str(committee_meta.get("to_label", "")))
            if from_top and to_top and from_top != to_top:
                cross_family_switch = True
                audit_text += f" || safety_policy_cross_family_committee_switch from_top={from_top} to_top={to_top}"

        if any(marker in audit_text for marker in hard_review_markers) or cross_family_switch:
            return (
                "_TO_REVIEW",
                final_label,
                "review",
                audit_text + " || safety_policy_review_instead_of_audited_auto_place",
            )
        return final_top, final_label, "auto_place", audit_text

    return final_top, final_label, confidence_status, review_reason


def _weighted_anchor_distance_for_label(brain: dict, label: str, fingerprint: Sequence[float]) -> float:
    """Nearest capped teacher-anchor distance for one label.

    This is one tournament head. It is not allowed to dominate alone; it only
    contributes one vote so giant folders cannot win from unlimited anchors.
    """
    anchors = np.asarray(brain.get("anchors_by_label", {}).get(label, []), dtype=np.float32)
    if anchors.ndim == 1 and anchors.size:
        anchors = anchors.reshape(1, -1)
    if anchors.size == 0:
        return float("inf")
    weights = np.asarray(brain.get("feature_weights", FEATURE_WEIGHTS), dtype=np.float32)
    mean, std = scaler_for_label(brain, label)
    xw = ((np.asarray(fingerprint, dtype=np.float32) - mean) / std) * weights
    aw = anchors * weights[None, :]
    d = np.linalg.norm(aw - xw[None, :], axis=1).astype(np.float32)
    return float(np.min(d)) if d.size else float("inf")


def _label_spread_scale_for_ensemble(brain: dict, label: str, predicted_top: str = "") -> float:
    """Return a safe spread/tolerance scale for support-neutral comparison.

    Small labels have little measured spread, so clamp to a fraction of the
    learned top-family scale. Big diffuse labels keep their wider spread. This
    is a dynamic normalization head, not a category rule.
    """
    scale_values: List[float] = []
    spread = brain.get("label_intra_spread_by_label", {}).get(label, {})
    if isinstance(spread, dict):
        for key in ("p90", "mean", "median"):
            try:
                val = float(spread.get(key, 0.0) or 0.0)
            except Exception:
                val = 0.0
            if math.isfinite(val) and val > 0.0:
                scale_values.append(val)
    model = brain.get("label_models_by_label", {}).get(label, {})
    if isinstance(model, dict):
        for key in ("spread_p90", "spread_mean"):
            try:
                val = float(model.get(key, 0.0) or 0.0)
            except Exception:
                val = 0.0
            if math.isfinite(val) and val > 0.0:
                scale_values.append(val)
    top = predicted_top or brain.get("top_by_label", {}).get(label, top_for_public_label(label))
    try:
        top_scale = float(brain.get("typical_dist_mean_by_top", {}).get(top, 1.0) or 1.0)
    except Exception:
        top_scale = 1.0
    if math.isfinite(top_scale) and top_scale > 0.0:
        scale_values.append(0.35 * top_scale)
    if not scale_values:
        return 1.0
    return float(max(0.20, np.median(np.asarray(scale_values, dtype=np.float32))))


def _rank_map_from_scores(items: List[Tuple[str, float]]) -> Dict[str, int]:
    """Tie-aware rank map for heads where lower scores are better.

    v0.6.2 fix: equal scores must get equal ranks.  Otherwise disabled or
    constant heads, especially the front-end router when it returns 1.0 for
    every label, turn alphabetical label order into fake model evidence.
    """
    clean = [(str(label), float(score)) for label, score in items if math.isfinite(float(score))]
    clean.sort(key=lambda t: (t[1], t[0]))
    ranks: Dict[str, int] = {}
    previous_score: Optional[float] = None
    current_rank = 0
    for position, (label, score) in enumerate(clean, start=1):
        if previous_score is None or not math.isclose(score, previous_score, rel_tol=1e-9, abs_tol=1e-9):
            current_rank = position
            previous_score = score
        ranks[label] = current_rank
    return ranks


def _score_head_has_information(items: List[Tuple[str, float]]) -> bool:
    """Return True only when a tournament head carries non-constant evidence."""
    values = [float(score) for _label, score in items if math.isfinite(float(score))]
    if len(values) <= 1:
        return False
    lo = min(values)
    hi = max(values)
    return bool(math.isfinite(lo) and math.isfinite(hi) and not math.isclose(lo, hi, rel_tol=1e-9, abs_tol=1e-9))


def _model_ensemble_same_top_family(brain: dict, left: str, right: str) -> bool:
    top_by = brain.get("top_by_label", {}) if isinstance(brain.get("top_by_label", {}), dict) else {}
    left_top = str(top_by.get(left, top_for_public_label(left)))
    right_top = str(top_by.get(right, top_for_public_label(right)))
    return left_top == right_top


def _raw_brain_winner_label(rows: Sequence[Dict[str, float]]) -> str:
    """Return the unmodified prototype/exemplar brain winner.

    v0.6.3 fix: the tournament must be anchored to the brain's actual
    nearest-label answer, not to ``current_score`` after helper penalties
    such as top-family, structure, fact, router, support, or anchor heads.
    Those helper heads are allowed to advise, but they must not redefine what
    "the brain picked" before the tournament even starts.
    """
    if not rows:
        return ""
    best = min(
        rows,
        key=lambda r: (
            float(r.get("raw_distance", r.get("raw_score", r.get("current_score", float("inf"))))),
            float(r.get("raw_score", r.get("raw_distance", r.get("current_score", float("inf"))))),
            str(r.get("label", "")),
        ),
    )
    return str(best.get("label", ""))


def _linear_head_scores_for_fingerprint(head: dict, brain: dict, fingerprint: Sequence[float]) -> Dict[str, float]:
    """Return scores from any stored linear-ridge head, higher is better."""
    if not isinstance(head, dict) or not bool(head.get("enabled", False)):
        return {}
    labels = [str(x) for x in head.get("labels", [])]
    W = np.asarray(head.get("weights", []), dtype=np.float64)
    if W.ndim != 2 or not labels or W.shape[1] != len(labels):
        return {}
    # Prediction-side matrix guard.  A previous version sanitized the training
    # ridge solve but still allowed a poisoned stored head to reach z_aug @ W,
    # producing overflow/NaN runtime warnings during diagnostics.  Never score
    # a non-finite or absurdly large head.  The prototype/exemplar brain remains
    # active when this head is disabled.
    if not np.all(np.isfinite(W)):
        return {}
    if W.size and float(np.max(np.abs(W))) > float(LINEAR_RIDGE_MAX_ABS_WEIGHT):
        return {}
    mean = np.asarray(head.get("global_mean", brain.get("scaler_mean", [])), dtype=np.float64)
    std = np.asarray(head.get("global_std", brain.get("scaler_std", [])), dtype=np.float64)
    weights = np.asarray(head.get("feature_weights", brain.get("feature_weights", FEATURE_WEIGHTS)), dtype=np.float64)
    x = np.asarray(fingerprint, dtype=np.float64)
    if x.shape[0] != weights.shape[0] or mean.shape[0] != weights.shape[0] or std.shape[0] != weights.shape[0]:
        return {}
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean, std = _finite_global_stats(mean, std, weights.shape[0])
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    z = ((x - mean) / std) * weights
    z = np.nan_to_num(z, nan=0.0, posinf=LINEAR_RIDGE_FEATURE_CLIP, neginf=-LINEAR_RIDGE_FEATURE_CLIP)
    z = np.clip(z, -float(LINEAR_RIDGE_FEATURE_CLIP), float(LINEAR_RIDGE_FEATURE_CLIP))
    z_aug = np.concatenate([z, np.ones((1,), dtype=np.float64)])
    with np.errstate(all="ignore"):
        scores = z_aug @ W
    if not np.all(np.isfinite(scores)):
        return {}
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return {label: float(scores[i]) for i, label in enumerate(labels)}


def linear_ridge_scores_for_fingerprint(brain: dict, fingerprint: Sequence[float]) -> Dict[str, float]:
    """Return balanced linear-ridge scores by label, higher is better."""
    return _linear_head_scores_for_fingerprint(brain.get("linear_ridge_head", {}), brain, fingerprint)


def frontend_router_scores_for_fingerprint(brain: dict, fingerprint: Sequence[float]) -> Dict[str, Dict[str, float]]:
    """Return front-end router scores for top, structure, and learned parent."""
    router = brain.get("frontend_router_brain", {})
    if not isinstance(router, dict) or not bool(router.get("enabled", False)):
        return {"top": {}, "structure": {}, "parent": {}}
    return {
        "top": _linear_head_scores_for_fingerprint(router.get("top_head", {}), brain, fingerprint),
        "structure": _linear_head_scores_for_fingerprint(router.get("structure_head", {}), brain, fingerprint),
        "parent": _linear_head_scores_for_fingerprint(router.get("parent_head", {}), brain, fingerprint),
    }


def _rank_from_high_score(scores: Dict[str, float]) -> Dict[str, int]:
    """Tie-aware rank map for heads where higher scores are better.

    v0.6.2 fix: the old implementation sorted by label as a tie-breaker and
    assigned different ranks to equal scores.  When a head was disabled or
    returned a constant score, alphabetical label order became fake evidence.
    Equal numeric scores must receive equal ranks.
    """
    clean = [(str(k), float(v)) for k, v in scores.items() if math.isfinite(float(v))]
    clean.sort(key=lambda t: (-t[1], t[0]))
    ranks: Dict[str, int] = {}
    previous_score: Optional[float] = None
    current_rank = 0
    for position, (label, score) in enumerate(clean, start=1):
        if previous_score is None or not math.isclose(score, previous_score, rel_tol=1e-9, abs_tol=1e-9):
            current_rank = position
            previous_score = score
        ranks[label] = current_rank
    return ranks


def _frontend_route_penalty_for_label(
    brain: dict, label: str, router_scores: Dict[str, Dict[str, float]]
) -> Tuple[float, float, str]:
    """Soft front-end-router penalty for a candidate label.

    The router never rejects labels.  It adds a small learned penalty when the
    candidate's top family, structure lane, or parent neighborhood are not
    supported by the second model head.  This lets the front end tie-break
    without turning into review dumping or hard-coded routing.
    """
    if not isinstance(router_scores, dict):
        return 0.0, 1.0, "router_disabled"
    top_scores = router_scores.get("top", {}) or {}
    struct_scores = router_scores.get("structure", {}) or {}
    parent_scores = router_scores.get("parent", {}) or {}
    if not top_scores and not struct_scores and not parent_scores:
        return 0.0, 1.0, "router_disabled"
    top_by = brain.get("top_by_label", {}) if isinstance(brain.get("top_by_label", {}), dict) else {}
    struct_by = brain.get("structure_by_label", {}) if isinstance(brain.get("structure_by_label", {}), dict) else {}
    lab_top = top_by.get(label, top_for_public_label(label))
    lab_struct = struct_by.get(label, label_default_structure(label))
    lab_parent = learned_parent_route_for_label(label)
    rank_top = _rank_from_high_score(top_scores)
    rank_struct = _rank_from_high_score(struct_scores)
    rank_parent = _rank_from_high_score(parent_scores)
    penalty = 0.0
    rank_sum = 0.0
    notes = []
    if top_scores:
        rt = rank_top.get(lab_top, len(rank_top) + 2)
        rank_sum += rt
        if rt == 1:
            p = 0.0
        elif rt == 2:
            p = FRONTEND_ROUTER_TOP_PENALTY * 0.35
        elif rt == 3:
            p = FRONTEND_ROUTER_TOP_PENALTY * 0.70
        else:
            p = FRONTEND_ROUTER_TOP_PENALTY
        penalty += p
        notes.append(f"top={lab_top}:r{rt}:p{p:.2f}")
    if struct_scores:
        rs = rank_struct.get(lab_struct, len(rank_struct) + 2)
        rank_sum += rs
        if rs == 1:
            p = 0.0
        elif rs == 2:
            p = FRONTEND_ROUTER_STRUCTURE_PENALTY * 0.45
        else:
            p = FRONTEND_ROUTER_STRUCTURE_PENALTY
        penalty += p
        notes.append(f"structure={lab_struct}:r{rs}:p{p:.2f}")
    if parent_scores:
        rp = rank_parent.get(lab_parent, len(rank_parent) + 2)
        rank_sum += rp
        if rp == 1:
            p = 0.0
        elif rp == 2:
            p = FRONTEND_ROUTER_PARENT_PENALTY * 0.30
        elif rp <= 4:
            p = FRONTEND_ROUTER_PARENT_PENALTY * 0.60
        else:
            p = FRONTEND_ROUTER_PARENT_PENALTY
        penalty += p
        notes.append(f"parent={lab_parent}:r{rp}:p{p:.2f}")
    return float(penalty), float(rank_sum if rank_sum > 0 else 1.0), ";".join(notes)


def ensemble_tournament_choose_label(
    brain: dict,
    rows: List[Dict[str, float]],
    current_label: str,
) -> Tuple[str, Dict[str, object]]:
    """Choose a final label using multiple learned matching heads.

    v0.6.2 tournament contract:
      * disabled/constant heads do not vote;
      * tied scores receive tied ranks;
      * membership-blocked rivals cannot win;
      * cross-family switches are disabled by default unless the current label is
        itself membership-blocked.

    This keeps the ensemble as a same-family re-ranker instead of a broad-family
    source reclassifier.  A piano-like raw winner must not become a drum merely
    because one anchor or disabled router head gives a fake rank advantage.
    """
    if not bool(brain.get("model_ensemble_enabled", MODEL_ENSEMBLE_ENABLED_DEFAULT)):
        return current_label, {"mode": "prototype_only", "switched": False}
    if not rows or len(rows) <= 1:
        return current_label, {"mode": "single_candidate", "switched": False}

    try:
        max_labels = int(brain.get("model_ensemble_max_labels", MODEL_ENSEMBLE_MAX_LABELS) or MODEL_ENSEMBLE_MAX_LABELS)
    except Exception:
        max_labels = MODEL_ENSEMBLE_MAX_LABELS

    head_keys = [
        "current_score",
        "support_neutral_score",
        "spread_norm_score",
        "anchor_score",
        "linear_rank_score",
        "frontend_router_rank_score",
    ]
    head_items: Dict[str, List[Tuple[str, float]]] = {
        key: [(str(r.get("label", "")), float(r.get(key, float("inf")))) for r in rows] for key in head_keys
    }
    active_heads = [key for key in head_keys if _score_head_has_information(head_items[key])]
    ignored_heads = [key for key in head_keys if key not in active_heads]
    if "current_score" not in active_heads:
        active_heads.insert(0, "current_score")

    # Include best labels by active heads only.  Disabled or all-tied heads must
    # not expand the tournament candidate pool.
    candidate_labels: Set[str] = {current_label}
    per_head_take = max(3, max_labels // 2)
    for key in active_heads:
        ordered = sorted(rows, key=lambda r: (float(r.get(key, float("inf"))), str(r.get("label", ""))))
        for row in ordered[:per_head_take]:
            label = str(row.get("label", ""))
            if label:
                candidate_labels.add(label)

    candidates = [r for r in rows if str(r.get("label", "")) in candidate_labels]
    if len(candidates) <= 1:
        return current_label, {
            "mode": "single_candidate",
            "switched": False,
            "active_heads": ";".join(active_heads),
            "ignored_heads": ";".join(ignored_heads),
        }

    rank_maps: Dict[str, Dict[str, int]] = {}
    for key in active_heads:
        rank_maps[key] = _rank_map_from_scores([(str(r["label"]), float(r.get(key, float("inf")))) for r in candidates])

    default_weights = {
        "current_score": MODEL_ENSEMBLE_CURRENT_RANK_WEIGHT,
        "support_neutral_score": MODEL_ENSEMBLE_NEUTRAL_RANK_WEIGHT,
        "spread_norm_score": MODEL_ENSEMBLE_SPREAD_RANK_WEIGHT,
        "anchor_score": MODEL_ENSEMBLE_ANCHOR_RANK_WEIGHT,
        "linear_rank_score": MODEL_ENSEMBLE_LINEAR_RANK_WEIGHT,
        "frontend_router_rank_score": MODEL_ENSEMBLE_FRONTEND_ROUTER_RANK_WEIGHT,
    }
    brain_weight_keys = {
        "current_score": "model_ensemble_current_rank_weight",
        "support_neutral_score": "model_ensemble_neutral_rank_weight",
        "spread_norm_score": "model_ensemble_spread_rank_weight",
        "anchor_score": "model_ensemble_anchor_rank_weight",
        "linear_rank_score": "model_ensemble_linear_rank_weight",
        "frontend_router_rank_score": "model_ensemble_frontend_router_rank_weight",
    }
    weights_by_head: Dict[str, float] = {}
    for key in active_heads:
        weights_by_head[key] = float(brain.get(brain_weight_keys[key], default_weights[key]) or default_weights[key])

    ensemble_rows = []
    worst_rank = len(candidates) + 2
    row_by_label = {str(r["label"]): r for r in candidates}
    current_row = row_by_label.get(current_label, {})
    bool(current_row.get("membership_blocked", False))

    # v0.6.3 safety: old scratch brains may contain
    # model_ensemble_allow_cross_family_switch=True from the broken tournament
    # era.  Do not let a saved brain silently re-enable broad-family jumps.
    # Cross-family override now requires a new explicit opt-in flag that old
    # brains do not have.  Otherwise a piano/chord can again become Drums/FX
    # even after the code default was fixed.
    brain_requested_cross_family = bool(brain.get("model_ensemble_allow_cross_family_switch", False))
    explicit_cross_family_opt_in = bool(brain.get("model_ensemble_explicit_cross_family_override_enabled", False))
    allow_cross_family = bool(brain_requested_cross_family and explicit_cross_family_opt_in)
    for label in sorted(row_by_label):
        score = 0.0
        for key in active_heads:
            score += weights_by_head[key] * rank_maps[key].get(label, worst_rank)
        ensemble_rows.append((label, float(score)))
    ensemble_rows.sort(key=lambda t: (t[1], t[0]))

    # Pick the best eligible proposal.  Blocked rivals and unsafe cross-family
    # switches are logged but cannot become the model tournament winner.
    #
    # v0.6.3 diagnostic hardening: even when the current/raw winner appears
    # before a bad rival, still record the unsafe rivals near the top of the
    # ensemble.  Otherwise the sorter can look clean while a broken head is
    # still trying to pull a piano/chord into Drums or FX.
    proposed_label = current_label
    proposed_ensemble = next(
        (score for label_name, score in ensemble_rows if label_name == current_label), float("inf")
    )
    rejection_reasons: List[str] = []
    saw_current = False

    def reject_reason_for(label: str) -> str:
        row = row_by_label.get(label, {})
        if bool(row.get("membership_blocked", False)):
            return f"{label}:membership_blocked"
        if (not allow_cross_family) and not _model_ensemble_same_top_family(brain, current_label, label):
            return f"{label}:cross_family_blocked"
        return ""

    for label, score in ensemble_rows:
        if label == current_label:
            saw_current = True
            proposed_label = label
            proposed_ensemble = float(score)
            break
        reason = reject_reason_for(label)
        if reason:
            rejection_reasons.append(reason)
            continue
        proposed_label = label
        proposed_ensemble = float(score)
        break

    # Add diagnostic-only rejections from nearby unsafe proposals after the
    # current label.  Do not use this second pass to choose a winner.
    if saw_current:
        for label, _score in ensemble_rows:
            if len(rejection_reasons) >= 8:
                break
            if label == current_label:
                continue
            reason = reject_reason_for(label)
            if reason and reason not in rejection_reasons:
                rejection_reasons.append(reason)

    current_score = float(row_by_label.get(current_label, {}).get("current_score", float("inf")))

    # v0.6.4: Same-family support/fact/structure refinement must still be able
    # to close a near miss.  v0.6.3 correctly anchored the tournament to the raw
    # prototype winner to stop piano -> drum/FX laundering, but that also
    # accidentally disabled legitimate same-family corrections such as a small
    # underrepresented folder whose support-balanced current_score is clearly
    # better.  This override is deliberately same-family only and still rejects
    # membership-blocked candidates.
    same_family_current_score_override = False
    current_score_improvement = 0.0
    eligible_same_family = [
        r
        for r in row_by_label.values()
        if str(r.get("label", "")) != str(current_label)
        and _model_ensemble_same_top_family(brain, current_label, str(r.get("label", "")))
        and not bool(r.get("membership_blocked", False))
    ]
    if eligible_same_family:
        best_current_row = min(
            eligible_same_family, key=lambda r: (float(r.get("current_score", float("inf"))), str(r.get("label", "")))
        )
        best_current_score = float(best_current_row.get("current_score", float("inf")))
        current_score_improvement = current_score - best_current_score
        min_current_improvement = float(
            brain.get("model_ensemble_same_family_current_score_min_improvement", 0.030) or 0.030
        )
        if current_score_improvement >= min_current_improvement:
            proposed_label = str(best_current_row.get("label", proposed_label))
            proposed_ensemble = next(
                (score for label_name, score in ensemble_rows if label_name == proposed_label), proposed_ensemble
            )
            same_family_current_score_override = True

    proposed_score = float(row_by_label.get(proposed_label, {}).get("current_score", float("inf")))
    current_ensemble = next((score for label_name, score in ensemble_rows if label_name == current_label), float("inf"))
    advantage = current_ensemble - proposed_ensemble
    min_adv = float(
        brain.get("model_ensemble_min_switch_advantage", MODEL_ENSEMBLE_MIN_SWITCH_ADVANTAGE)
        or MODEL_ENSEMBLE_MIN_SWITCH_ADVANTAGE
    )
    max_penalty_ratio = float(
        brain.get("model_ensemble_max_score_penalty_ratio", MODEL_ENSEMBLE_MAX_SCORE_PENALTY_RATIO)
        or MODEL_ENSEMBLE_MAX_SCORE_PENALTY_RATIO
    )

    switched = False
    final_label = current_label
    if proposed_label != current_label:
        not_much_worse = proposed_score <= max(current_score + 0.15, current_score * max_penalty_ratio)
        if ((advantage >= min_adv) or same_family_current_score_override) and not_much_worse:
            final_label = proposed_label
            switched = True

    meta = {
        "mode": "model_tournament_ensemble",
        "switched": bool(switched),
        "current_label": current_label,
        "proposed_label": proposed_label,
        "final_label": final_label,
        "current_ensemble_score": float(current_ensemble),
        "proposed_ensemble_score": float(proposed_ensemble),
        "ensemble_advantage": float(advantage),
        "top_ensemble": "; ".join(f"{label_name}:{score:.2f}" for label_name, score in ensemble_rows[:5]),
        "active_heads": ";".join(active_heads),
        "ignored_heads": ";".join(ignored_heads),
        "rejected_proposals": "; ".join(rejection_reasons[:8]),
        "frontend_router_weight": float(weights_by_head.get("frontend_router_rank_score", 0.0)),
        "same_family_current_score_override": bool(same_family_current_score_override),
        "same_family_current_score_improvement": float(current_score_improvement),
        "same_family_switch_only": bool(not allow_cross_family),
        "brain_requested_cross_family_switch": bool(brain_requested_cross_family),
        "explicit_cross_family_opt_in": bool(explicit_cross_family_opt_in),
    }
    return final_label, meta


def _fact_profile_is_usable(profile: object) -> bool:
    if not isinstance(profile, dict):
        return False
    try:
        if int(profile.get("effective_count", 0) or 0) < FACT_GOOD_MIN_COUNT:
            return False
        strength = str(profile.get("fact_profile_strength", "")).lower()
        return strength in {"good", "strong"}
    except Exception:
        return False


def _rival_score_between_labels(
    *,
    raw_x: np.ndarray,
    feature_names_list: List[str],
    rival_contrast_facts: Dict[str, object],
    label: str,
    rival: str,
) -> Tuple[float, List[str], List[str]]:
    """Return learned fact contrast score for label versus rival.

    Positive means the sample looks more like `label` than `rival` on features
    dynamically discovered from the training data. No filenames, public folder
    words, or hand-written source categories are consulted.
    """
    matched: List[str] = []
    failed: List[str] = []
    label_rivalries = rival_contrast_facts.get(label, {}) if isinstance(rival_contrast_facts, dict) else {}
    if not isinstance(label_rivalries, dict):
        return 0.0, matched, failed
    rival_info = label_rivalries.get(rival, {})
    if not isinstance(rival_info, dict):
        return 0.0, matched, failed
    score = score_sample_against_rival_contrast_facts(raw_x, rival_info, feature_names_list, matched, failed)
    return float(score), matched, failed


def rival_contrast_second_pass_choose_label(
    *,
    brain: dict,
    rows: List[Dict[str, float]],
    current_label: str,
    raw_x: np.ndarray,
    feature_names_list: List[str],
    fact_profiles: Dict[str, object],
    rival_contrast_facts: Dict[str, object],
) -> Tuple[str, Dict[str, object]]:
    """Cautious learned rival tie-breaker.

    The swap is allowed only inside the same learned top family and the same
    structure lane, only when scores are close, and only when the contrast facts
    strongly favor the challenger. This is the architecture-safe replacement for
    old category rescue logic: no filenames, no category-name branches, no
    one-off kick/tom/snare rules.
    """
    meta: Dict[str, object] = {
        "mode": "rival_contrast_second_pass",
        "swapped": False,
        "from_label": current_label,
        "to_label": current_label,
        "reason": "not_considered",
        "challenger_label": "",
        "challenger_contrast_score": 0.0,
        "current_reverse_contrast_score": 0.0,
        "score_gap": 0.0,
        "matched_key_facts": [],
        "failed_key_facts": [],
    }
    if not bool(brain.get("rival_second_pass_enabled", RIVAL_SECOND_PASS_ENABLED_DEFAULT)):
        meta["reason"] = "disabled"
        return current_label, meta
    if not rows or not isinstance(rival_contrast_facts, dict) or not rival_contrast_facts:
        meta["reason"] = "missing_rows_or_rival_facts"
        return current_label, meta

    row_by_label = {str(r.get("label", "")): r for r in rows}
    current_row = row_by_label.get(current_label)
    if not current_row:
        meta["reason"] = "current_row_missing"
        return current_label, meta

    top_by_label = brain.get("top_by_label", {}) if isinstance(brain.get("top_by_label"), dict) else {}
    structure_by_label = (
        brain.get("structure_by_label", {}) if isinstance(brain.get("structure_by_label"), dict) else {}
    )
    current_top = top_by_label.get(current_label, top_for_public_label(current_label))
    current_structure = structure_by_label.get(current_label, label_default_structure(current_label))
    current_profile = fact_profiles.get(current_label, {}) if isinstance(fact_profiles, dict) else {}
    if not _fact_profile_is_usable(current_profile):
        meta["reason"] = "current_fact_profile_not_usable"
        return current_label, meta

    current_score = float(current_row.get("current_score", 0.0))
    current_fact_score = float(current_row.get("fact_score", 0.0))
    best_candidate: Tuple[float, str, Dict[str, float], float, float, List[str], List[str], str] | None = None

    for challenger_row in sorted(rows, key=lambda r: (float(r.get("current_score", 0.0)), str(r.get("label", ""))))[
        : max(
            2,
            int(
                brain.get("rival_second_pass_max_candidates", RIVAL_SECOND_PASS_MAX_CANDIDATES)
                or RIVAL_SECOND_PASS_MAX_CANDIDATES
            ),
        )
    ]:
        challenger = str(challenger_row.get("label", ""))
        if not challenger or challenger == current_label:
            continue
        if top_by_label.get(challenger, top_for_public_label(challenger)) != current_top:
            continue
        if structure_by_label.get(challenger, label_default_structure(challenger)) != current_structure:
            continue
        challenger_profile = fact_profiles.get(challenger, {}) if isinstance(fact_profiles, dict) else {}
        if not _fact_profile_is_usable(challenger_profile):
            continue
        score_gap = float(challenger_row.get("current_score", 0.0)) - current_score
        if score_gap < -1e-9:
            # This can happen if an earlier ensemble head chose a label that was
            # not the best current-score row. Treat the challenger as close.
            score_gap = 0.0
        if score_gap > float(
            brain.get("rival_second_pass_max_score_gap", RIVAL_SECOND_PASS_MAX_SCORE_GAP)
            or RIVAL_SECOND_PASS_MAX_SCORE_GAP
        ):
            continue
        challenger_fact_score = float(challenger_row.get("fact_score", 0.0))
        if (
            challenger_fact_score
            + float(
                brain.get("rival_second_pass_max_fact_score_deficit", RIVAL_SECOND_PASS_MAX_FACT_SCORE_DEFICIT)
                or RIVAL_SECOND_PASS_MAX_FACT_SCORE_DEFICIT
            )
            < current_fact_score
        ):
            continue

        challenger_score, matched, failed = _rival_score_between_labels(
            raw_x=raw_x,
            feature_names_list=feature_names_list,
            rival_contrast_facts=rival_contrast_facts,
            label=challenger,
            rival=current_label,
        )
        reverse_score, _rev_matched, _rev_failed = _rival_score_between_labels(
            raw_x=raw_x,
            feature_names_list=feature_names_list,
            rival_contrast_facts=rival_contrast_facts,
            label=current_label,
            rival=challenger,
        )
        edge = float(challenger_score) - float(reverse_score)
        min_challenger = float(
            brain.get("rival_second_pass_min_challenger_score", RIVAL_SECOND_PASS_MIN_CHALLENGER_SCORE)
            or RIVAL_SECOND_PASS_MIN_CHALLENGER_SCORE
        )
        min_edge = float(
            brain.get("rival_second_pass_min_score_edge", RIVAL_SECOND_PASS_MIN_SCORE_EDGE)
            or RIVAL_SECOND_PASS_MIN_SCORE_EDGE
        )
        if challenger_score < min_challenger or edge < min_edge:
            continue
        # Higher priority = stronger contrast edge, then smaller production gap.
        priority = edge - (0.20 * max(0.0, score_gap))
        if best_candidate is None or priority > best_candidate[0]:
            best_candidate = (
                priority,
                challenger,
                challenger_row,
                challenger_score,
                reverse_score,
                matched[:8],
                failed[:8],
                f"contrast_edge={edge:.3f}; score_gap={score_gap:.3f}",
            )

    if best_candidate is None:
        meta["reason"] = "no_close_reliable_challenger_favored_by_contrast"
        return current_label, meta

    _priority, challenger, challenger_row, challenger_score, reverse_score, matched, failed, reason = best_candidate
    meta.update(
        {
            "swapped": True,
            "from_label": current_label,
            "to_label": challenger,
            "challenger_label": challenger,
            "challenger_contrast_score": float(challenger_score),
            "current_reverse_contrast_score": float(reverse_score),
            "score_gap": float(max(0.0, float(challenger_row.get("current_score", 0.0)) - current_score)),
            "matched_key_facts": matched,
            "failed_key_facts": failed,
            "reason": reason,
        }
    )
    return challenger, meta


def weak_profile_rival_override_choose_label(
    *,
    brain: dict,
    rows: List[Dict[str, float]],
    current_label: str,
    raw_x: np.ndarray,
    feature_names_list: List[str],
    fact_profiles: Dict[str, object],
) -> Tuple[str, Dict[str, object]]:
    """Dynamic fallback when a tiny/weak winner beats a reliable rival narrowly.

    This fixes a real v0.4.84 failure: a tiny weak-profile label can win raw
    distance against a much better-supported rival, then the structure gate
    preserves the wrong identity path.  This override does not inspect filenames
    and does not branch on category names.  It only uses learned support, learned
    fact-profile strength, top-family geometry, and measured sample facts.
    """
    meta: Dict[str, object] = {
        "mode": "weak_profile_rival_override",
        "swapped": False,
        "from_label": current_label,
        "to_label": current_label,
        "reason": "not_considered",
        "score_gap": 0.0,
        "challenger_fact_score": 0.0,
        "current_fact_score": 0.0,
    }
    if not rows or not isinstance(fact_profiles, dict):
        meta["reason"] = "missing_rows_or_fact_profiles"
        return current_label, meta
    row_by_label = {str(r.get("label", "")): r for r in rows}
    current_row = row_by_label.get(current_label)
    if not current_row:
        meta["reason"] = "current_row_missing"
        return current_label, meta
    reliability = (
        brain.get("label_reliability_by_label", {}) if isinstance(brain.get("label_reliability_by_label"), dict) else {}
    )
    current_rel = reliability.get(current_label, {}) if isinstance(reliability.get(current_label, {}), dict) else {}
    current_count = int(current_rel.get("training_count", current_rel.get("effective_training_count", 999)) or 999)
    current_tier = str(current_rel.get("support_tier", ""))
    current_profile = fact_profiles.get(current_label, {})
    current_profile_usable = _fact_profile_is_usable(current_profile if isinstance(current_profile, dict) else {})
    # Only override weak/tiny winners. Strong winners should be handled by the
    # normal rival-contrast second pass, not this safety valve.
    if (
        current_profile_usable
        and current_count > SMALL_LABEL_COUNT
        and current_tier not in {"tiny", "small", "provisional_tiny", "provisional_single_source"}
    ):
        meta["reason"] = "current_profile_and_support_are_not_weak"
        return current_label, meta
    current_score = float(current_row.get("current_score", 0.0))
    current_fact = float(current_row.get("fact_score", 0.0))
    top_by_label = brain.get("top_by_label", {}) if isinstance(brain.get("top_by_label"), dict) else {}
    current_top = str(top_by_label.get(current_label, top_for_public_label(current_label)))
    top_scores = top_prefilter_allowed_tops(brain, raw_x, top_n=4)[1]
    top_score_by_name = {str(k): float(v) for k, v in (top_scores or [])}
    current_top_score = float(top_score_by_name.get(current_top, 0.0))
    best: Tuple[float, str, Dict[str, float], float, float, str] | None = None
    for challenger_row in rows[1 : max(6, min(len(rows), 12))]:
        challenger = str(challenger_row.get("label", ""))
        if not challenger or challenger == current_label:
            continue
        profile = fact_profiles.get(challenger, {})
        if not isinstance(profile, dict) or not _fact_profile_is_usable(profile):
            continue
        challenger_rel = reliability.get(challenger, {}) if isinstance(reliability.get(challenger, {}), dict) else {}
        challenger_count = int(
            challenger_rel.get("training_count", challenger_rel.get("effective_training_count", 0)) or 0
        )
        if challenger_count < FACT_OK_MIN_COUNT:
            continue
        score_gap = float(challenger_row.get("current_score", 0.0)) - current_score
        if score_gap < 0.0:
            score_gap = 0.0
        if score_gap > 0.95:
            continue
        challenger_fact = float(challenger_row.get("fact_score", 0.0))
        if challenger_fact < max(0.18, current_fact + 0.12):
            continue
        challenger_top = str(top_by_label.get(challenger, top_for_public_label(challenger)))
        challenger_top_score = float(top_score_by_name.get(challenger_top, current_top_score))
        # Let a reliable rival from another top win only when the learned top
        # geometry does not strongly reject that top.  This is dynamic top-family
        # evidence, not a category-name rule.
        if challenger_top != current_top and (challenger_top_score - current_top_score) > 1.15:
            continue
        priority = (challenger_fact - current_fact) + (0.04 * min(challenger_count, 80)) - (0.25 * score_gap)
        reason = (
            f"weak_current_profile_or_support; current_count={current_count} current_tier={current_tier} "
            f"current_fact={current_fact:.3f} challenger_fact={challenger_fact:.3f} "
            f"score_gap={score_gap:.3f} challenger_count={challenger_count}"
        )
        if best is None or priority > best[0]:
            best = (priority, challenger, challenger_row, challenger_fact, score_gap, reason)
    if best is None:
        meta["reason"] = "no_reliable_fact_rival_beats_weak_winner"
        return current_label, meta
    _priority, challenger, _row, challenger_fact, gap, reason = best
    meta.update(
        {
            "swapped": True,
            "from_label": current_label,
            "to_label": challenger,
            "reason": reason,
            "score_gap": float(gap),
            "challenger_fact_score": float(challenger_fact),
            "current_fact_score": float(current_fact),
        }
    )
    return challenger, meta
