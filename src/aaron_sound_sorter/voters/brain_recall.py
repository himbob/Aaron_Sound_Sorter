# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Multi-lane brain recall voters.

Architecture contract:
  * Full brain is the stability lane trained on all usable examples.
  * Core baby brain is the precision lane trained on clean central anchors.
  * Spread baby brain is the balanced recall lane trained on diverse clean anchors.
  * Outlier baby brain is an edge-case recall lane.  It can keep a weird label
    alive for review/broad placement, but it must not force terminal placement
    by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aaron_sound_sorter.domain.models import CategoryGuess, VoterResult
from aaron_sound_sorter.voters.brain_ensemble_policy import (
    brain_ensemble_weight_profile,
    brain_lane_vote_weight,
    brain_role_fit_multiplier,
    choose_representative_brain_guess,
    is_reed_or_brass_label,
    rank_support,
)
from aaron_sound_sorter.voters.brain_lane_competence import brain_lane_candidate_diagnostics
from aaron_sound_sorter.voters.brain_voter import BrainVoter


@dataclass(frozen=True)
class BrainLaneBundle:
    """Container for full and optional baby brain vote results."""

    full: VoterResult
    babies: dict[str, VoterResult]
    combined: VoterResult


def make_disabled_balanced_result(reason: str, lane_name: str = "baby") -> VoterResult:
    """Return an empty baby-lane result with a readable diagnostic."""
    return VoterResult(
        voter_name=f"brain_{lane_name}",
        guesses=[],
        diagnostics={"enabled": False, "reason": str(reason), "lane_name": lane_name},
    )


class FullBrainVoter(BrainVoter):
    """Normal full-data brain lane."""

    voter_name = "brain_full"


class BalancedRecallBrainVoter(BrainVoter):
    """Generic baby brain lane with an explicit job name."""

    def __init__(self, policy=None, *, lane_name: str = "baby", purpose: str = "few_shot_label_recall") -> None:
        super().__init__(policy)
        self.lane_name = lane_name
        self.purpose = purpose
        self.voter_name = f"brain_{lane_name}"

    def vote(self, physics, facts, brain: dict[str, Any]) -> VoterResult:  # type: ignore[override]
        result = super().vote(physics, facts, brain)
        guesses: list[CategoryGuess] = []
        for guess in result.guesses:
            evidence = dict(guess.evidence)
            evidence.update(
                {
                    "brain_lane": self.lane_name,
                    "baby_brain_candidate": True,
                    "baby_brain_purpose": self.purpose,
                    "terminal_identity_is_hint_not_truth": self.lane_name != "core_baby",
                    "outlier_brain_never_final_truth": self.lane_name == "outlier_baby",
                }
            )
            guesses.append(
                CategoryGuess(
                    label=guess.label,
                    folder_path=guess.folder_path,
                    top_family=guess.top_family,
                    score=guess.score,
                    confidence=guess.confidence,
                    rank=guess.rank,
                    reason=f"{self.lane_name}_" + str(guess.reason),
                    evidence=evidence,
                )
            )
        diagnostics = dict(result.diagnostics)
        diagnostics.update({"enabled": True, "lane_name": self.lane_name, "purpose": self.purpose})
        return VoterResult(voter_name=self.voter_name, guesses=guesses, diagnostics=diagnostics)


def combine_full_and_balanced_brain_votes(
    *,
    full_result: VoterResult,
    balanced_result: VoterResult | None = None,
    baby_results: dict[str, VoterResult] | None = None,
    max_guesses: int = 120,
    facts: Any | None = None,
) -> VoterResult:
    """Combine full and baby brain lanes into one product BrainVoter result.

    This is the architecture-safe version of "sum of brains": all brain lanes
    remain evidence producers, and this combiner emits one ranked ``brain``
    result for the normal two-voter committee.  It does not create final
    placements.  It only merges rank/score evidence before PhysicsVoter and
    FamilyClaimArbiter see the candidate set.

    Lane intent:
      * core baby is the precision clean-center lane and gets the strongest vote.
      * spread baby is the balanced-diversity lane and gets strong recall weight.
      * full brain is the stability lane and still matters, but it no longer
        automatically buries clean baby evidence.
      * outlier baby is recall only.  It can keep rare labels visible but is
        down-weighted unless another lane agrees.
    """
    max_guesses = max(1, int(max_guesses or 120))
    lane_results: dict[str, VoterResult] = {"full": full_result}
    if balanced_result is not None:
        lane_results["spread_baby"] = balanced_result
    if baby_results:
        lane_results.update({k: v for k, v in baby_results.items() if v is not None})

    lane_evidence: dict[str, dict[str, Any]] = {}
    lane_guesses: dict[str, dict[str, CategoryGuess]] = {}

    for lane, result in lane_results.items():
        lane_guesses[lane] = {}
        for guess in result.guesses:
            label = str(guess.label)
            if not label:
                continue
            lane_guesses[lane][label] = guess
            evidence = lane_evidence.setdefault(label, {})
            evidence[f"{lane}_rank"] = int(guess.rank)
            evidence[f"{lane}_score"] = float(guess.score)
            evidence[f"{lane}_confidence"] = float(guess.confidence)
            evidence[f"{lane}_reason"] = str(guess.reason)

    combined_rows: list[tuple[float, float, str, CategoryGuess, dict[str, Any]]] = []
    for label, evidence in lane_evidence.items():
        present_lanes = [lane for lane in lane_results if f"{lane}_rank" in evidence]
        representative = choose_representative_brain_guess(label, present_lanes, lane_guesses)
        if representative is None:
            continue

        weight_profile, trust_reason = brain_ensemble_weight_profile(facts=facts, representative=representative)
        support = 0.0
        best_rank = 9999.0
        lane_count = 0
        lane_weight_report: dict[str, float] = {}
        for lane in present_lanes:
            rank = float(evidence.get(f"{lane}_rank", 9999.0) or 9999.0)
            best_rank = min(best_rank, rank)
            lane_count += 1
            guess = lane_guesses.get(lane, {}).get(label)
            lane_weight = brain_lane_vote_weight(lane, weight_profile=weight_profile, label=label)
            lane_weight_report[lane] = round(float(lane_weight), 4)
            support += (
                lane_weight
                * rank_support(rank)
                * brain_role_fit_multiplier(
                    guess,
                    weight_profile=weight_profile,
                    label=label,
                )
            )

        # Agreement across independent brain views is useful evidence, but it
        # must not undo the role-specific weighting profile.  In particular, the
        # outlier lane helps reed recall only when another lane keeps that same
        # source-family candidate alive.
        if lane_count >= 2:
            support *= 1.0 + min(0.20, 0.06 * (lane_count - 1))
        if present_lanes == ["outlier_baby"] or present_lanes == ["harmonic_outlier_baby"]:
            support *= 0.35
        if weight_profile == "generic_pitched_full_guarded" and is_reed_or_brass_label(label):
            # Generic pitched loops include keys, synths, strings, vocals, and
            # mixed melodies.  A reed-like candidate may stay visible, but it
            # should not win the product brain unless multiple stable lanes agree.
            stable_lane_count = sum(1 for lane in present_lanes if lane in {"full", "core_baby", "spread_baby"})
            if stable_lane_count < 2:
                # Outlier-only reed recall should stay visible but guarded.  The
                # arbiter still decides whether the broad Brass/Woodwind claim
                # is legal for the measured role.
                support *= 0.78

        confidence = max(0.0, min(1.0, support / 1.85))
        ensemble_score = 1.0 / max(0.001, support)
        merged_evidence = dict(representative.evidence)
        merged_evidence.update(evidence)
        lane_competence = brain_lane_candidate_diagnostics(
            label=label,
            present_lanes=present_lanes,
            lane_guesses=lane_guesses,
            weight_profile=weight_profile,
        ).to_dict()
        merged_evidence["brain_candidate_lanes"] = present_lanes
        merged_evidence["has_full_brain_support"] = "full" in present_lanes
        merged_evidence["has_core_baby_support"] = "core_baby" in present_lanes
        merged_evidence["has_spread_baby_support"] = "spread_baby" in present_lanes
        merged_evidence["has_outlier_baby_support"] = "outlier_baby" in present_lanes
        merged_evidence["has_harmonic_core_support"] = "harmonic_core_baby" in present_lanes
        merged_evidence["has_harmonic_spread_support"] = "harmonic_spread_baby" in present_lanes
        merged_evidence["has_harmonic_outlier_support"] = "harmonic_outlier_baby" in present_lanes
        merged_evidence["has_any_harmonic_core_recall_support"] = any(
            str(lane).startswith("harmonic_") for lane in present_lanes
        )
        merged_evidence["outlier_only_candidate"] = present_lanes in (["outlier_baby"], ["harmonic_outlier_baby"])
        merged_evidence["brain_ensemble_support"] = round(float(support), 6)
        merged_evidence["brain_ensemble_score"] = round(float(ensemble_score), 6)
        merged_evidence["brain_ensemble_weighting"] = "v31.89_category_competence_lane_weighting"
        merged_evidence["brain_ensemble_weight_profile"] = weight_profile
        merged_evidence["brain_ensemble_lane_trust_reason"] = trust_reason
        merged_evidence["brain_ensemble_lane_weight_report"] = lane_weight_report
        merged_evidence["brain_lane_competence"] = lane_competence
        merged_evidence["brain_lane_competence_profile"] = weight_profile
        merged_evidence["lane_calibrated_confidence"] = lane_competence["lane_calibrated_confidence"]
        merged_evidence["lane_authority_reason"] = lane_competence["lane_authority_reason"]
        merged_evidence["lane_exact_agreement_count"] = lane_competence["lane_exact_agreement_count"]
        merged_evidence["lane_parent_agreement_count"] = lane_competence["lane_parent_agreement_count"]
        merged_evidence["lane_top_family_agreement_count"] = lane_competence["lane_top_family_agreement_count"]

        combined_guess = CategoryGuess(
            label=representative.label,
            folder_path=representative.folder_path,
            top_family=representative.top_family,
            score=float(ensemble_score),
            confidence=confidence,
            rank=representative.rank,
            reason="brain_ensemble_weighted_lane_vote",
            evidence=merged_evidence,
        )
        combined_rows.append((-support, best_rank, str(label), combined_guess, merged_evidence))

    ordered = sorted(combined_rows, key=lambda row: (row[0], row[1], row[2]))[:max_guesses]
    reranked: list[CategoryGuess] = []
    for rank, (_neg_support, _best_rank, _label, guess, evidence) in enumerate(ordered, start=1):
        merged_evidence = dict(evidence)
        merged_evidence["combined_brain_rank"] = rank
        reranked.append(
            CategoryGuess(
                label=guess.label,
                folder_path=guess.folder_path,
                top_family=guess.top_family,
                score=guess.score,
                confidence=guess.confidence,
                rank=rank,
                reason=guess.reason,
                evidence=merged_evidence,
            )
        )

    return VoterResult(
        voter_name="brain",
        guesses=reranked,
        diagnostics={
            "architecture": "weighted_brain_ensemble"
            if set(lane_results) != {"full"}
            else "full_brain_only_product_vote",
            "ensemble_version": "v31.89_category_competence_lane_weighting",
            "lane_guess_counts": {lane: len(result.guesses) for lane, result in lane_results.items()},
            "lanes_affecting_product_vote": list(lane_results),
            "combined_guess_count": len(reranked),
            "lane_weights": {
                lane: brain_lane_vote_weight(lane, weight_profile="generic", label="") for lane in lane_results
            },
            "audit_note": "Individual brain lanes stay diagnostic evidence; this is the single product BrainVoter result passed to the arbiter.",
        },
    )
