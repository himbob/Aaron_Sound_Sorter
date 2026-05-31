"""Regression tests for golden FX-smoke failures found after resolver patches.

These are intentionally decision-core tests, not filename tests.  They reproduce
observed failure *patterns* with synthetic voter candidates and measured roles:

* real bass loops were flattened to Instrument Loops or stolen by FX/Risers;
* normal pitched musical loops were stolen by FX/Risers/Alarm;
* real transition/bell FX still needs an FX path when transition evidence is
  actually strong.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import ConsensusDecision, SharedAudioFacts
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import EligibilityDecision


def raw_decision(path: str, score: float = 10.0, candidates: list[dict] | None = None) -> ConsensusDecision:
    """Build a raw decision with candidate rows."""
    top = path.split("/", 1)[0]
    return ConsensusDecision(
        final_label=path,
        final_top=top,
        folder_path=path,
        consensus_status="strong_consensus",
        reason="synthetic raw decision",
        combined_rank_score=score,
        shared_candidates=candidates or [],
    )


def facts(shape: str = "", confidence: float = 0.0) -> SharedAudioFacts:
    """Build minimal shared facts carrying shape evidence."""
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={"shape_vote": {"primary_shape": shape, "confidence": confidence}},
    )


def eligibility(role: str, confidence: float, broad: str) -> EligibilityDecision:
    """Build a simple eligibility object."""
    return EligibilityDecision(
        role_name=role,
        confidence=confidence,
        allowed_top_families=(broad.split("/", 1)[0], "_TO_REVIEW"),
        blocked_path_fragments=(),
        broad_folder_path=broad,
        reason="synthetic measured role",
    )


def test_real_bass_loop_with_bass_candidate_must_not_flatten_to_instrument_loops_or_riser() -> None:
    """03_bass-like failures must land in the Bass Loops broad bucket."""
    raw = raw_decision(
        "FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX",
        score=10,
        candidates=[
            {
                "folder_path": "FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX",
                "top_family": "FX",
                "combined_rank_score": 10,
            },
            {
                "folder_path": "Instruments/Bass/Synth Bass/One Shots",
                "top_family": "Instruments",
                "combined_rank_score": 11,
            },
            {
                "folder_path": "Instruments/Instrument Loops/Loops",
                "top_family": "Instruments",
                "combined_rank_score": 16,
            },
        ],
    )
    final = DecisionCoreV2().apply_eligibility(
        raw,
        eligibility("bass_loop", 1.0, "Instruments/Bass/Bass Loops"),
        facts("bass_phrase", 1.0),
    )
    assert final.folder_path == "Instruments/Bass/Bass Loops"


def test_false_bass_role_without_bass_candidate_must_not_force_synth_lead_to_bass_loops() -> None:
    """The bass rescue requires Bass candidate support, not just a role score."""
    raw = raw_decision(
        "Instruments/Instrument Loops/Loops",
        score=10,
        candidates=[
            {
                "folder_path": "Instruments/Synths/Synth Lead/One Shots",
                "top_family": "Instruments",
                "combined_rank_score": 7,
            },
            {
                "folder_path": "FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/Long FX",
                "top_family": "FX",
                "combined_rank_score": 11,
            },
        ],
    )
    final = DecisionCoreV2().apply_eligibility(
        raw,
        eligibility("bass_loop", 1.0, "Instruments/Bass/Bass Loops"),
        facts("bass_phrase", 1.0),
    )
    assert final.folder_path != "Instruments/Bass/Bass Loops"
    assert final.final_top == "Instruments"


def test_plain_piano_guitar_string_loop_must_not_be_stolen_by_close_fx_riser_candidate() -> None:
    """Pitched musical loops need transition evidence before FX/Riser can steal them."""
    raw = raw_decision(
        "Instruments/Instrument Loops/Loops",
        score=10,
        candidates=[
            {
                "folder_path": "Instruments/Instrument Loops/Loops",
                "top_family": "Instruments",
                "combined_rank_score": 10,
            },
            {
                "folder_path": "FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX",
                "top_family": "FX",
                "combined_rank_score": 9,
            },
        ],
    )
    final = DecisionCoreV2().apply_eligibility(
        raw,
        eligibility("pitched_music_loop", 0.96, "Instruments/Instrument Loops/Loops"),
        facts("pitched_phrase", 0.93),
    )
    assert final.folder_path == "Instruments/Instrument Loops/Loops"


def test_tonal_alert_role_without_transition_shape_must_not_steal_synth_lead_to_alarm() -> None:
    """Synth leads/pitched phrases should not become Alarm/Riser without transition shape."""
    raw = raw_decision(
        "Instruments/Instrument Loops/Loops",
        score=11,
        candidates=[
            {
                "folder_path": "Instruments/Synths/Synth Lead/One Shots",
                "top_family": "Instruments",
                "combined_rank_score": 7,
            },
            {"folder_path": "FX/Designed Noise FX/Alarm/Long FX", "top_family": "FX", "combined_rank_score": 8},
        ],
    )
    final = DecisionCoreV2().apply_eligibility(
        raw,
        eligibility("fx_tonal_alert_or_siren", 0.78, "FX/Designed Noise FX/Alarm/Long FX"),
        facts("vocal_phrase", 0.95),
    )
    assert final.final_top == "Instruments"
    assert "Alarm" not in final.folder_path


def test_real_transition_bell_fx_can_still_route_to_fx_when_transition_shape_is_strong() -> None:
    """The riser guard must not block actual transition/growing bell FX."""
    raw = raw_decision(
        "Instruments/Instrument Loops/Loops",
        score=11,
        candidates=[
            {
                "folder_path": "Instruments/Instrument Loops/Loops",
                "top_family": "Instruments",
                "combined_rank_score": 11,
            },
            {
                "folder_path": "FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX",
                "top_family": "FX",
                "combined_rank_score": 7,
            },
        ],
    )
    final = DecisionCoreV2().apply_eligibility(
        raw,
        eligibility("pitched_music_loop", 0.92, "Instruments/Instrument Loops/Loops"),
        facts("transition_riser", 0.91),
    )
    assert final.final_top == "FX"
    assert "Riser" in final.folder_path or "Risers" in final.folder_path
