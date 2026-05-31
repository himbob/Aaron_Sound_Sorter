"""Hard regressions from the curated FX_Aaron2 smoke matrix.

These tests intentionally reproduce the actual post-sort failure classes without
using source filenames in production code:

* Bass loop raw/role evidence must not be flattened to generic Instrument Loops.
* Drum loop raw evidence must not be demoted to generic Instrument Loops.
* Sax/reed-like pitched musical loops must not become Human/Voice FX because
  shape is vocal-like.
* True vocal evidence must still be allowed to stay in Human/Voice FX.

Every future sorter patch must keep these green and must also run the full FX
smoke manifest audit.  Unit tests alone are not enough.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import ConsensusDecision, SharedAudioFacts
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import EligibilityDecision


def raw_decision(path: str, score: float = 10.0, candidates: list[dict] | None = None) -> ConsensusDecision:
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


def eligibility(role: str, confidence: float, broad: str) -> EligibilityDecision:
    top = broad.split("/", 1)[0]
    return EligibilityDecision(
        role_name=role,
        confidence=confidence,
        allowed_top_families=(top, "_TO_REVIEW"),
        blocked_path_fragments=(),
        broad_folder_path=broad,
        reason="synthetic measured role",
    )


def facts(shape: str, confidence: float, measured_role: str) -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {"primary_shape": shape, "confidence": confidence},
            "measured_roles": {"detected_parent_role": measured_role, measured_role: 1.0},
        },
    )


def test_raw_bass_loop_bucket_is_not_flattened_to_generic_instrument_loop() -> None:
    raw = raw_decision(
        "Instruments/Bass/Bass Loops",
        candidates=[
            {"folder_path": "Instruments/Bass/Bass Loops", "top_family": "Instruments", "combined_rank_score": 8},
            {
                "folder_path": "Instruments/Instrument Loops/Loops",
                "top_family": "Instruments",
                "combined_rank_score": 10,
            },
        ],
    )
    final = DecisionCoreV2().apply_eligibility(
        raw,
        eligibility("pitched_music_loop", 0.95, "Instruments/Instrument Loops/Loops"),
        facts("bass_phrase", 0.97, "bass_loop"),
    )
    assert final.folder_path == "Instruments/Bass/Bass Loops"


def test_generic_instrument_bass_loop_with_bass_candidate_is_promoted_to_bass_loops() -> None:
    raw = raw_decision(
        "Instruments/Instrument Loops/Loops",
        candidates=[
            {
                "folder_path": "Instruments/Instrument Loops/Loops",
                "top_family": "Instruments",
                "combined_rank_score": 10,
            },
            {
                "folder_path": "Instruments/Bass/Synth Bass/One Shots",
                "top_family": "Instruments",
                "combined_rank_score": 11,
            },
        ],
    )
    final = DecisionCoreV2().apply_eligibility(
        raw,
        eligibility("bass_loop", 1.0, "Instruments/Bass/Bass Loops"),
        facts("bass_phrase", 1.0, "bass_loop"),
    )
    assert final.folder_path == "Instruments/Bass/Bass Loops"


def test_raw_drum_loop_bucket_is_not_flattened_to_generic_instrument_loop() -> None:
    raw = raw_decision(
        "Drums/Drum Loops/Loops",
        candidates=[
            {"folder_path": "Drums/Drum Loops/Loops", "top_family": "Drums", "combined_rank_score": 6},
            {
                "folder_path": "Instruments/Instrument Loops/Loops",
                "top_family": "Instruments",
                "combined_rank_score": 12,
            },
        ],
    )
    final = DecisionCoreV2().apply_eligibility(
        raw,
        eligibility("bass_loop", 0.92, "Instruments/Bass/Bass Loops"),
        facts("sustained_pad", 0.91, "bass_loop"),
    )
    assert final.folder_path == "Drums/Drum Loops/Loops"


def test_sax_like_pitched_music_with_vocal_shape_does_not_become_human_voice_fx() -> None:
    raw = raw_decision(
        "Instruments/Instrument Loops/Loops",
        candidates=[
            {
                "folder_path": "Instruments/Instrument Loops/Loops",
                "top_family": "Instruments",
                "combined_rank_score": 9,
            },
            {"folder_path": "FX/Human and Voice FX", "top_family": "FX", "combined_rank_score": 12},
        ],
    )
    final = DecisionCoreV2().apply_eligibility(
        raw,
        eligibility("vocal_phrase", 0.82, "FX/Human and Voice FX"),
        facts("vocal_phrase", 0.95, "pitched_music_loop"),
    )
    assert final.folder_path == "Instruments/Instrument Loops/Loops"


def test_human_voice_raw_winner_can_stay_voice_when_measured_role_is_vocal() -> None:
    raw = raw_decision(
        "FX/Human and Voice FX",
        candidates=[
            {"folder_path": "FX/Human and Voice FX", "top_family": "FX", "combined_rank_score": 7},
            {
                "folder_path": "Instruments/Instrument Loops/Loops",
                "top_family": "Instruments",
                "combined_rank_score": 15,
            },
        ],
    )
    final = DecisionCoreV2().apply_eligibility(
        raw,
        eligibility("vocal_phrase", 0.88, "FX/Human and Voice FX"),
        facts("vocal_phrase", 0.96, "vocal_phrase"),
    )
    assert final.folder_path == "Instruments/Voice/Phrase/One Shots"


def test_raw_human_voice_false_positive_for_pitched_music_routes_to_instruments() -> None:
    raw = raw_decision(
        "FX/Human and Voice FX",
        candidates=[
            {"folder_path": "FX/Human and Voice FX", "top_family": "FX", "combined_rank_score": 8},
            {
                "folder_path": "Instruments/Instrument Loops/Loops",
                "top_family": "Instruments",
                "combined_rank_score": 12,
            },
        ],
    )
    final = DecisionCoreV2().apply_eligibility(
        raw,
        eligibility("pitched_music_loop", 0.93, "Instruments/Instrument Loops/Loops"),
        facts("vocal_phrase", 0.95, "pitched_music_loop"),
    )
    assert final.folder_path == "Instruments/Instrument Loops/Loops"


from aaron_sound_sorter.domain.models import CategoryGuess, VoterResult
from aaron_sound_sorter.engine.consensus import ConsensusRunner
from aaron_sound_sorter.engine.family_claim_arbiter import FamilyClaimArbiter


def guess(label: str, rank: int, score: float = 1.0) -> CategoryGuess:
    return CategoryGuess(
        label=label,
        folder_path=label,
        top_family=label.split("/", 1)[0],
        score=score,
        confidence=1.0,
        rank=rank,
        reason="synthetic",
        evidence={},
    )


def test_consensus_bass_role_with_brain_bass_support_stops_at_bass_loop_bucket() -> None:
    """Full consensus seam must not choose generic Instrument Loops for clear bass role."""
    brain = VoterResult(
        voter_name="brain",
        guesses=[
            guess("Instruments/Bass/Synth Bass/One Shots", 1),
            guess("Instruments/Instrument Loops/Loops", 4),
        ],
    )
    physics = VoterResult(
        voter_name="physics",
        guesses=[
            guess("Instruments/Instrument Loops/Loops", 1),
            guess("Instruments/Bass/Synth Bass/One Shots", 30),
        ],
    )
    f = facts("bass_phrase", 1.0, "bass_loop")
    f.evidence["measured_roles"] = {"bass_loop": 1.0}
    raw_claim, consensus_claims = ConsensusRunner().choose(brain, physics, f)
    final = FamilyClaimArbiter().adjudicate(
        raw_claim=raw_claim,
        consensus_claims=consensus_claims,
        eligibility_claims=[],
        facts=f,
    )
    assert final.folder_path == "Instruments/Bass/Bass Loops"


def test_voice_shout_raw_human_voice_is_not_stolen_by_instrument_candidate() -> None:
    raw = raw_decision(
        "FX/Human and Voice FX",
        candidates=[
            {"folder_path": "FX/Human and Voice FX", "top_family": "FX", "combined_rank_score": 10},
            {
                "folder_path": "Instruments/Synths/Synth Chord/One Shots",
                "top_family": "Instruments",
                "combined_rank_score": 6,
            },
        ],
    )
    final = DecisionCoreV2().apply_eligibility(
        raw,
        eligibility("voiced_one_shot", 0.86, "FX/Human and Voice FX"),
        facts("hit_with_tail", 0.70, "voiced_one_shot"),
    )
    assert final.folder_path == "Instruments/Voice/Phrase/One Shots"
