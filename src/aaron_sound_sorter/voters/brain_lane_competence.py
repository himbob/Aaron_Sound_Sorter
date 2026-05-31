# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Diagnostics for brain-lane competence and calibration.

This module does not choose final categories.  It makes each brain-lane vote
more auditable by exposing rank gaps, parent/top-family agreement, and a
bounded calibrated confidence that future measured panels can validate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from aaron_sound_sorter.domain.models import CategoryGuess
from aaron_sound_sorter.voters.brain_ensemble_policy import brain_lane_vote_weight, rank_support


@dataclass(frozen=True)
class BrainLaneCandidateDiagnostics:
    """Readable diagnostics for one label across the brain lanes."""

    label: str
    candidate_parent_path: str
    candidate_top_family: str
    lane_calibrated_confidence: float
    lane_authority_reason: str
    lane_exact_agreement_count: int
    lane_parent_agreement_count: int
    lane_top_family_agreement_count: int
    outlier_only_candidate: bool
    lanes: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["lane_calibrated_confidence"] = round(float(self.lane_calibrated_confidence), 6)
        return data


def brain_lane_candidate_diagnostics(
    *,
    label: str,
    present_lanes: list[str],
    lane_guesses: dict[str, dict[str, CategoryGuess]],
    weight_profile: str,
) -> BrainLaneCandidateDiagnostics:
    """Return diagnostics for a combined brain candidate.

    Scores are diagnostic only.  Product ranking still lives in
    ``combine_full_and_balanced_brain_votes`` so this module can be validated
    before it is trusted for routing behavior.
    """
    normalized = normalize_path(label)
    parent = parent_path(normalized)
    top = top_family(normalized)
    rows: list[dict[str, object]] = []
    calibrated_values: list[float] = []

    for lane in present_lanes:
        guess = lane_guesses.get(lane, {}).get(label)
        if guess is None:
            continue
        rank_gap = lane_rank_gap(lane_guesses.get(lane, {}), label)
        same_parent_gap = nearest_same_parent_gap(lane_guesses.get(lane, {}), label)
        same_top_gap = nearest_same_top_gap(lane_guesses.get(lane, {}), label)
        lane_weight = brain_lane_vote_weight(lane, weight_profile=weight_profile, label=label)
        calibrated = calibrated_lane_confidence(
            rank=guess.rank,
            raw_confidence=guess.confidence,
            rank_gap_to_next=rank_gap,
            lane_weight=lane_weight,
        )
        calibrated_values.append(calibrated)
        rows.append(
            {
                "lane_name": lane,
                "candidate_label": label,
                "candidate_score": round(float(guess.score), 6),
                "rank": int(guess.rank),
                "rank_gap_to_next": round(float(rank_gap), 6),
                "same_parent_gap": round(float(same_parent_gap), 6),
                "same_top_family_gap": round(float(same_top_gap), 6),
                "raw_distance_or_similarity": round(float(guess.score), 6),
                "raw_confidence": round(float(guess.confidence), 6),
                "calibrated_confidence": round(float(calibrated), 6),
                "calibration_reason": calibration_reason(lane, lane_weight, rank_gap),
            }
        )

    exact_count = len(rows)
    parent_count = count_lanes_matching_parent(label, parent, lane_guesses, present_lanes)
    top_count = count_lanes_matching_top(label, top, lane_guesses, present_lanes)
    outlier_only = present_lanes in (["outlier_baby"], ["harmonic_outlier_baby"])
    lane_confidence = max(calibrated_values) if calibrated_values else 0.0
    return BrainLaneCandidateDiagnostics(
        label=label,
        candidate_parent_path=parent,
        candidate_top_family=top,
        lane_calibrated_confidence=lane_confidence,
        lane_authority_reason=authority_reason(present_lanes, parent_count, top_count, outlier_only),
        lane_exact_agreement_count=exact_count,
        lane_parent_agreement_count=parent_count,
        lane_top_family_agreement_count=top_count,
        outlier_only_candidate=outlier_only,
        lanes=rows,
    )


def calibrated_lane_confidence(
    *,
    rank: int,
    raw_confidence: float,
    rank_gap_to_next: float,
    lane_weight: float,
) -> float:
    """Blend raw confidence, rank support, rank gap, and lane authority."""
    gap_signal = clamp01(float(rank_gap_to_next) / 3.0)
    base = 0.46 * clamp01(raw_confidence) + 0.34 * clamp01(rank_support(float(rank))) + 0.20 * gap_signal
    return clamp01(base * clamp01(lane_weight / 1.35))


def lane_rank_gap(lane_candidates: dict[str, CategoryGuess], label: str) -> float:
    """Return score gap from this label to the next-best lane candidate."""
    guess = lane_candidates.get(label)
    if guess is None:
        return 0.0
    ordered = sorted(lane_candidates.values(), key=lambda item: (float(item.score), int(item.rank), str(item.label)))
    for index, candidate in enumerate(ordered):
        if candidate.label != label:
            continue
        if index + 1 >= len(ordered):
            return 99.0
        return max(0.0, float(ordered[index + 1].score) - float(candidate.score))
    return 0.0


def nearest_same_parent_gap(lane_candidates: dict[str, CategoryGuess], label: str) -> float:
    return nearest_group_gap(lane_candidates, label, group_fn=lambda value: parent_path(normalize_path(value)))


def nearest_same_top_gap(lane_candidates: dict[str, CategoryGuess], label: str) -> float:
    return nearest_group_gap(lane_candidates, label, group_fn=lambda value: top_family(normalize_path(value)))


def nearest_group_gap(
    lane_candidates: dict[str, CategoryGuess],
    label: str,
    *,
    group_fn,
) -> float:
    guess = lane_candidates.get(label)
    if guess is None:
        return 0.0
    group = group_fn(label)
    best = 99.0
    for candidate in lane_candidates.values():
        if candidate.label == label or group_fn(candidate.label) != group:
            continue
        best = min(best, abs(float(candidate.score) - float(guess.score)))
    return best


def count_lanes_matching_parent(
    label: str,
    parent: str,
    lane_guesses: dict[str, dict[str, CategoryGuess]],
    present_lanes: list[str],
) -> int:
    """Count same-parent evidence only where this exact label is present.

    Sibling labels in other lanes are broad context, not authority for the
    evaluated label.  Counting those siblings inflated single-lane candidates
    into fake same-parent agreement.
    """
    return sum(
        1
        for lane in present_lanes
        if label in lane_guesses.get(lane, {})
        and parent_path(normalize_path(lane_guesses[lane][label].label)) == parent
    )


def count_lanes_matching_top(
    label: str,
    top: str,
    lane_guesses: dict[str, dict[str, CategoryGuess]],
    present_lanes: list[str],
) -> int:
    """Count top-family evidence only where this exact label is present."""
    return sum(
        1
        for lane in present_lanes
        if label in lane_guesses.get(lane, {}) and top_family(normalize_path(lane_guesses[lane][label].label)) == top
    )


def authority_reason(present_lanes: list[str], parent_count: int, top_count: int, outlier_only: bool) -> str:
    if outlier_only:
        return "outlier-only support is recall visibility, not terminal authority"
    if len(present_lanes) >= 2:
        return "exact label agreement across multiple brain lanes"
    if parent_count >= 2:
        return "single-label support has same-parent backing from another lane"
    if top_count >= 2:
        return "single-label support has top-family backing from another lane"
    return "single-lane support only; needs physics or arbiter corroboration"


def calibration_reason(lane: str, lane_weight: float, rank_gap: float) -> str:
    if lane in {"outlier_baby", "harmonic_outlier_baby"}:
        return "outlier lane is capped unless another lane or physics corroborates"
    if rank_gap <= 0.15:
        return "small rank gap marks this lane candidate as ambiguous"
    if lane_weight >= 1.0:
        return "lane has normal or elevated authority for this role profile"
    return "lane authority is reduced for this role profile"


def parent_path(path: str) -> str:
    parts = [part for part in normalize_path(path).split("/") if part]
    if not parts:
        return ""
    if parts[-1].lower() in {"loops", "one shots", "long fx"}:
        parts = parts[:-1]
    if len(parts) <= 1:
        return "/".join(parts)
    return "/".join(parts[:-1])


def top_family(path: str) -> str:
    return normalize_path(path).split("/", 1)[0]


def normalize_path(value: object) -> str:
    return str(value or "").replace("\\", "/").strip("/")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
