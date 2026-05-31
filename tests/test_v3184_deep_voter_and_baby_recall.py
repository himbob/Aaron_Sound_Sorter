"""v31.84 deep scoring and baby-recall regression locks.

The synthetic names describe real failure classes from the FX run, but the
assertions use only internal category labels, measured roles, and voter outputs.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import CategoryGuess, SharedAudioFacts, VoterResult
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import EligibilityDecision
from aaron_sound_sorter.engine.family_claims import claim_from_folder_path
from aaron_sound_sorter.voters.scoring_tools import role_compatibility_adjustment


def guess(path: str, rank: int) -> CategoryGuess:
    return CategoryGuess(
        label=path,
        folder_path=path,
        top_family=path.split("/", 1)[0],
        score=float(rank),
        confidence=max(0.0, 1.0 - rank / 40.0),
        rank=rank,
        reason="synthetic",
        evidence={},
    )


def raw_claim(path: str, score: float = 4.0):
    return claim_from_folder_path(
        folder_path=path,
        source="strong_consensus",
        reason="synthetic raw",
        shared=[],
        raw_candidate_score=score,
        brain_rank=1,
        physics_rank=1,
        shared_winner=path,
        can_override=False,
        strength=0.80,
        is_real_candidate=True,
    )


def eligibility(role: str, broad: str, confidence: float = 1.0) -> EligibilityDecision:
    return EligibilityDecision(
        role_name=role,
        confidence=confidence,
        allowed_top_families=(broad.split("/", 1)[0], "_TO_REVIEW"),
        blocked_path_fragments=(),
        broad_folder_path=broad,
        reason="synthetic eligibility",
    )


def facts(
    shape: str,
    confidence: float,
    roles: dict[str, float],
    baby_rows=None,
    role_evidence: dict[str, float] | None = None,
) -> SharedAudioFacts:
    measured_roles = dict(roles)
    if role_evidence is not None:
        measured_roles["evidence"] = role_evidence
    evidence = {
        "shape_vote": {"primary_shape": shape, "confidence": confidence},
        "shape_confidence": confidence,
        "measured_roles": measured_roles,
        "direct_body_view": {"available": True, "measured_roles": measured_roles},
    }
    if baby_rows is not None:
        evidence["core_baby_vote_result"] = {
            "diagnostics": {"enabled": True, "lane_name": "core_baby"},
            "top_guesses": baby_rows,
        }
        evidence["spread_baby_vote_result"] = {
            "diagnostics": {"enabled": True, "lane_name": "spread_baby"},
            "top_guesses": [],
        }
        evidence["outlier_baby_vote_result"] = {
            "diagnostics": {"enabled": True, "lane_name": "outlier_baby"},
            "top_guesses": [],
        }
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence=evidence,
    )


def baby_row(path: str, rank: int) -> dict:
    return {
        "rank": rank,
        "label": path,
        "folder_path": path,
        "top_family": path.split("/", 1)[0],
        "score": float(rank),
        "confidence": max(0.0, 1.0 - rank / 40.0),
        "reason": "synthetic baby recall",
        "evidence": {},
    }


def test_role_evidence_is_diagnostic_only_for_incompatible_fx_candidate() -> None:
    measured = facts("pitched_phrase", 0.90, {"pitched_music_loop": 1.0})
    delta, evidence = role_compatibility_adjustment(
        "FX/Designed Noise FX/Alarm/Long FX",
        {"labels": ["FX/Designed Noise FX/Alarm/Long FX"]},
        measured,
        voter_name="physics",
    )

    assert delta == 0.0
    assert evidence["role_adjustment_applied"] == 0.0
    assert evidence["role_adjustment_mode"] == "diagnostic_only"
    assert evidence["role_recommended_adjustment"] > 0.50
    assert evidence["family_compatibility"]["is_family_compatible"] is False


def test_baby_recall_alone_cannot_rescue_sax_from_strong_fx_alarm() -> None:
    measured = facts(
        "pitched_phrase",
        1.0,
        {"pitched_music_loop": 1.0, "pitched_music_phrase": 1.0, "vocal_music_phrase": 0.0},
        baby_rows=[baby_row("Instruments/Woodwinds/Saxophone/One Shots", 1)],
    )
    raw = raw_claim("FX/Designed Noise FX/Siren/Long FX", 3.0)
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("pitched_music_loop", "Instruments/Instrument Loops/Loops", 1.0),
        measured,
        brain_result=VoterResult(voter_name="brain_full", guesses=[]),
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "FX/Designed Noise FX/Siren/Long FX"
    assert final.consensus_status == "strong_consensus"


def test_supported_kick_candidate_beats_generic_percussion_broadening() -> None:
    measured = facts(
        "hit_with_tail",
        0.79,
        {"percussive_one_shot": 1.0},
        role_evidence={
            "duration_sec": 0.25,
            "event_count_estimate": 1.0,
            "high_total": 0.01,
            "low_pitched_hit_raw": 1.0,
            "low_total": 0.99,
        },
    )
    raw = raw_claim("Drums/Percussion/Generic Percussion/One Shots", 4.0)
    brain = VoterResult(
        voter_name="brain_full",
        guesses=[guess("Drums/Kick Drums/Sub Kick/One Shots", 1)],
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("percussive_one_shot", "Drums/Percussion/One Shots", 1.0),
        measured,
        brain_result=brain,
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Drums/Kick Drums/Generic Kick/One Shots"
    assert final.consensus_status == "profile_candidate_kick_claim"


def test_low_kick_profile_candidate_can_rescue_fx_sub_hit() -> None:
    measured = facts(
        "hit_with_tail",
        0.79,
        {"percussive_one_shot": 1.0},
        role_evidence={
            "duration_sec": 0.20,
            "event_count_estimate": 1.0,
            "high_total": 0.01,
            "low_pitched_hit_raw": 1.0,
            "low_total": 0.99,
        },
    )
    raw = raw_claim("FX/Impacts and Hits/Sub Hit/One Shots", 5.0)
    brain = VoterResult(
        voter_name="brain_full",
        guesses=[guess("Drums/Kick Drums/Sub Kick/One Shots", 1)],
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("percussive_one_shot", "Drums/Percussion/One Shots", 1.0),
        measured,
        brain_result=brain,
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert final.folder_path == "Drums/Kick Drums/Generic Kick/One Shots"
    assert final.consensus_status == "profile_candidate_kick_claim"


def test_baby_kick_recall_does_not_steal_bright_snare_or_clap_hit() -> None:
    measured = facts(
        "single_hit",
        0.83,
        {"percussive_one_shot": 1.0},
        baby_rows=[baby_row("Drums/Kick Drums/Sub Kick/One Shots", 1)],
        role_evidence={
            "duration_sec": 0.25,
            "event_count_estimate": 1.0,
            "high_total": 0.50,
            "low_pitched_hit_raw": 0.43,
            "low_total": 0.47,
        },
    )
    raw = raw_claim("Drums/Snares/Acoustic Snare/One Shots", 5.0)
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("percussive_one_shot", "Drums/Percussion/One Shots", 1.0),
        measured,
        brain_result=VoterResult(voter_name="brain_full", guesses=[]),
        physics_result=None,
    )
    final = core.arbiter.adjudicate(raw_claim=raw, consensus_claims=[], eligibility_claims=claims, facts=measured)

    assert all(claim.source != "baby_recall_kick_claim" for claim in claims)
    assert final.folder_path == "Drums/Snares/Acoustic Snare/One Shots"


def test_bass_profile_claim_requires_measured_bass_loop_not_just_bass_shape() -> None:
    measured = facts("bass_phrase", 0.94, {"pitched_music_loop": 0.95, "bass_loop": 0.0})
    raw = raw_claim("Instruments/Instrument Loops/Loops", 4.0)
    brain = VoterResult(
        voter_name="brain_full",
        guesses=[guess("Instruments/Bass/Electric Bass/One Shots", 1)],
    )
    core = DecisionCoreV2()

    claims = core.gather_eligibility_claims(
        raw,
        eligibility("pitched_music_loop", "Instruments/Instrument Loops/Loops", 0.95),
        measured,
        brain_result=brain,
        physics_result=None,
    )

    assert all(claim.source != "profile_candidate_bass_claim" for claim in claims)
