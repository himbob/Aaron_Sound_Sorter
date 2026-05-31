"""Synthetic regressions for direct/body voice and low drum-loop policy."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import ConsensusDecision, SharedAudioFacts
from aaron_sound_sorter.domain.roles import measured_roles_from_features
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import EligibilityDecision


def candidate(path: str, score: float) -> dict:
    """Build a synthetic shared candidate row."""
    return {
        "folder_path": path,
        "label": path,
        "top_family": path.split("/", 1)[0],
        "combined_rank_score": score,
    }


def test_low_tonal_kick_loop_measures_as_drum_loop_not_bass_loop() -> None:
    """Repeated low pitched transient pulses should not default to bass-loop role."""
    values = {
        "sub_bass_ratio_lt_150hz": 0.78,
        "bass_ratio_150_500hz": 0.18,
        "presence_ratio_2000_8000hz": 0.01,
        "air_ratio_gt_8000hz": 0.00,
        "pitch_confidence": 0.95,
        "f0_voiced_ratio": 0.18,
        "log_crest": 1.70,
        "attack_rise_time_norm": 0.004,
        "temporal_centroid_ratio": 0.48,
        "tail_energy_ratio": 0.62,
        "log_transient_count": 3.80,
        "loop_pitched_event_ratio": 0.92,
        "loop_percussive_event_ratio": 0.05,
        "loop_drumlike_frame_ratio": 0.04,
        "loop_tonal_to_percussive_balance": 0.92,
        "loop_sustained_tonal_frame_ratio": 0.94,
        "loop_non_event_tonal_ratio": 0.95,
    }

    roles = measured_roles_from_features(
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        feature_values=values,
    )

    assert roles.low_rhythmic_drum_loop >= 0.70
    assert roles.bass_loop < roles.low_rhythmic_drum_loop


def test_direct_body_voiced_one_shot_can_rescue_reverbed_vocal_shot_from_drum() -> None:
    """A tail-heavy vocal shot should use direct/body voice evidence before drum fallback."""
    raw = ConsensusDecision(
        final_label="Drums/Percussion/Generic Percussion/One Shots",
        final_top="Drums",
        folder_path="Drums/Percussion/Generic Percussion/One Shots",
        consensus_status="candidate_true_bucket_rescue",
        reason="synthetic wrong drum rescue",
        combined_rank_score=10.0,
        shared_candidates=[
            candidate("Drums/Percussion/Generic Percussion/One Shots", 10.0),
            candidate("FX/Human and Voice FX/Crowd/One Shots", 28.0),
            candidate("FX/Human and Voice FX/Spoken Voice/One Shots", 31.0),
        ],
    )
    eligibility = EligibilityDecision(
        role_name="percussive_one_shot",
        confidence=0.82,
        allowed_top_families=("Drums", "FX", "_TO_REVIEW"),
        broad_folder_path="Drums/Percussion/Generic Percussion/One Shots",
        reason="synthetic full-file percussive read",
    )
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=True,
        is_long=False,
        evidence={
            "shape_vote": {
                "primary_shape": "hit_with_tail",
                "confidence": 0.75,
                "f0_voiced_ratio": 0.92,
                "onset_count": 1.0,
                "tail_ratio": 0.08,
                "percussive_event_ratio": 0.05,
                "drumlike_frame_ratio": 0.04,
            },
            "measured_roles": {"voiced_one_shot": 0.90, "percussive_one_shot": 0.20},
            "feature_values_by_name": {
                "formant_light_voice_identity": 0.90,
                "voiced_one_shot": 0.90,
            },
            "direct_body_view": {
                "available": True,
                "measured_roles": {"voiced_one_shot": 0.91, "percussive_one_shot": 0.20},
            },
        },
        feature_values_by_name={
            "formant_light_voice_identity": 0.90,
            "voiced_one_shot": 0.90,
        },
    )

    final = DecisionCoreV2().apply_eligibility(raw, eligibility, facts)

    assert final.folder_path == "Instruments/Voice/Phrase/One Shots"


def test_strong_vocal_role_and_close_voice_candidate_beats_abstract_alarm_candidate() -> None:
    """Strong vocal role should not review merely because alarm/synth is two rank points closer."""
    raw = ConsensusDecision(
        final_label="Instruments/Instrument Loops/Loops",
        final_top="Instruments",
        folder_path="Instruments/Instrument Loops/Loops",
        consensus_status="weak_consensus",
        reason="synthetic generic loop",
        combined_rank_score=15.0,
        shared_candidates=[
            candidate("FX/Designed Noise FX/Alarm/One Shots", 15.0),
            candidate("Instruments/Synths/Synth Chord/One Shots", 15.0),
            candidate("FX/Human and Voice FX/Crowd/One Shots", 17.0),
            candidate("FX/Human and Voice FX/Spoken Voice/One Shots", 19.0),
        ],
    )
    eligibility = EligibilityDecision(
        role_name="vocal_phrase",
        confidence=0.82,
        allowed_top_families=("FX", "Instruments", "_TO_REVIEW"),
        broad_folder_path="Instruments/Voice/Phrase/One Shots",
        reason="synthetic vocal phrase role",
    )
    facts = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=False,
        is_single_event_like=True,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {"primary_shape": "vocal_phrase", "confidence": 0.80},
            "measured_roles": {"voiced_one_shot": 0.82, "pitched_music_phrase": 0.86},
            "direct_body_view": {
                "available": True,
                "measured_roles": {"pitched_music_phrase": 0.86},
            },
        },
    )

    final = DecisionCoreV2().apply_eligibility(raw, eligibility, facts)

    assert final.folder_path == "Instruments/Voice/Phrase/One Shots"
