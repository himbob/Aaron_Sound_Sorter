"""Focused v31.76 follow-up tests for family-claim arbitration.

These tests lock the shared architecture rule behind the Mac failures from
v31.75: measured role and direct/body evidence may broaden to a safe family
only when the family has a positive claim.  Voice, bass, and drum-loop claims
must block each other through the same contract instead of one-off filename
rules.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import ConsensusDecision, SharedAudioFacts
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import EligibilityDecision


def _raw_decision(path: str, *, score: float = 8.0) -> ConsensusDecision:
    top = path.split("/", 1)[0]
    return ConsensusDecision(
        final_label=path,
        final_top=top,
        folder_path=path,
        consensus_status="test_raw",
        reason="synthetic raw decision",
        shared_winner=path,
        brain_rank=1,
        physics_rank=2,
        combined_rank_score=score,
        shared_candidates=[],
    )


def test_bass_loop_claim_blocks_weak_human_voice_sink() -> None:
    facts = SharedAudioFacts(
        False,
        True,
        False,
        False,
        True,
        evidence={
            "measured_roles": {"bass_loop": 0.90, "voiced_one_shot": 0.74},
            "direct_body_view": {"measured_roles": {"bass_loop": 0.93}},
            "shape_vote": {"primary_shape": "bass_phrase", "confidence": 0.91},
        },
    )
    raw = _raw_decision("Instruments/Voice/Phrase/One Shots")
    eligibility = EligibilityDecision(
        role_name="bass_loop",
        confidence=0.90,
        allowed_top_families=("Instruments",),
        broad_folder_path="Instruments/Bass/Bass Loops",
    )

    decision = DecisionCoreV2().apply_eligibility(raw, eligibility, facts)

    assert decision.folder_path == "Instruments/Bass/Bass Loops"


def test_strong_measured_vocal_shape_can_rescue_drum_hit_false_positive() -> None:
    facts = SharedAudioFacts(
        False,
        False,
        True,
        True,
        False,
        evidence={
            "measured_roles": {"voiced_one_shot": 0.90},
            "direct_body_view": {"measured_roles": {"voiced_one_shot": 0.92}},
            "shape_vote": {"primary_shape": "vocal_phrase", "confidence": 0.91, "f0_voiced_ratio": 0.92},
            "feature_values_by_name": {
                "formant_light_voice_identity": 0.90,
                "voiced_one_shot": 0.90,
            },
        },
        feature_values_by_name={
            "formant_light_voice_identity": 0.90,
            "voiced_one_shot": 0.90,
        },
    )
    raw = _raw_decision("Drums/Percussion/Generic Percussion/One Shots")
    eligibility = EligibilityDecision(
        role_name="voiced_one_shot",
        confidence=0.90,
        allowed_top_families=("FX", "Instruments"),
        broad_folder_path="Instruments/Voice/Phrase/One Shots",
    )

    decision = DecisionCoreV2().apply_eligibility(raw, eligibility, facts)

    assert decision.folder_path == "Instruments/Voice/Phrase/One Shots"


def test_measured_drum_loop_blocks_bass_one_shot_sink() -> None:
    facts = SharedAudioFacts(
        False,
        True,
        False,
        False,
        True,
        evidence={
            "measured_roles": {"low_rhythmic_drum_loop": 0.84},
            "shape_vote": {"primary_shape": "beat_loop", "confidence": 0.90},
        },
    )
    raw = _raw_decision(
        "Instruments/Bass/Synth Bass/One Shots",
        score=8.0,
    )
    object.__setattr__(
        raw,
        "shared_candidates",
        [
            {
                "label": "Drums/Drum Loops/Loops",
                "folder_path": "Drums/Drum Loops/Loops",
                "top_family": "Drums",
                "brain_rank": 6,
                "physics_rank": 5,
                "combined_rank_score": 11.0,
                "evidence": {"candidate_role_signature": {"low_rhythmic_drum_loop": 0.92}},
            }
        ],
    )
    eligibility = EligibilityDecision(
        role_name="low_rhythmic_drum_loop",
        confidence=0.84,
        allowed_top_families=("Drums",),
        broad_folder_path="Drums/Drum Loops/Loops",
    )

    decision = DecisionCoreV2().apply_eligibility(raw, eligibility, facts)

    assert decision.folder_path == "Drums/Drum Loops/Loops"
