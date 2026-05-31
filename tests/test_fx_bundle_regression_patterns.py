"""Synthetic regressions from full FX_Aaron2 after-test inspection.

These tests encode measured-audio failure patterns found by reading the sample
pack labels after sorting. The sorter code must not read source filenames;
these tests use synthetic candidate/fact payloads to lock the general policy.
"""

from __future__ import annotations

from aaron_sound_sorter.domain.models import ConsensusDecision, SharedAudioFacts
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.engine.eligibility import EligibilityDecision, infer_parent_eligibility


def candidate(path: str, score: float, *, role_strengths: dict[str, float] | None = None) -> dict:
    """Build a shared candidate like consensus output."""
    signature = role_strengths or {}
    return {
        "folder_path": path,
        "label": path,
        "top_family": path.split("/", 1)[0],
        "combined_rank_score": score,
        "candidate_role_signature": signature,
        "brain_evidence": {"candidate_role_signature": signature},
        "physics_evidence": {"candidate_role_signature": signature},
    }


def raw(path: str, score: float, candidates: list[dict]) -> ConsensusDecision:
    """Build a raw consensus decision."""
    return ConsensusDecision(
        final_label=path,
        final_top=path.split("/", 1)[0],
        folder_path=path,
        consensus_status="synthetic_raw",
        reason="synthetic raw decision",
        combined_rank_score=score,
        shared_candidates=candidates,
    )


def facts(shape: str, confidence: float, roles: dict[str, float]) -> SharedAudioFacts:
    """Build shared facts with shape and measured roles."""
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "shape_vote": {"primary_shape": shape, "confidence": confidence},
            "measured_roles": roles,
        },
    )


def test_drum_loop_role_not_blocked_by_weak_breath_candidate() -> None:
    """A weak Human/Voice candidate must not veto a measured drum loop."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Instrument Loops/Loops",
            18.0,
            [
                candidate("FX/Impacts and Hits/Generic Impact/Long FX", 14.0),
                candidate("FX/Human and Voice FX/Breath/Long FX", 19.0),
                candidate("Drums/Drum Loops/Loops", 28.0),
            ],
        ),
        EligibilityDecision(
            role_name="drum_loop",
            confidence=0.78,
            allowed_top_families=("Drums", "_TO_REVIEW"),
            broad_folder_path="Drums/Drum Loops/Loops",
            reason="synthetic measured drum loop",
        ),
        facts(
            "bass_phrase",
            0.96,
            {
                "low_rhythmic_drum_loop": 0.74,
                "drum_loop": 0.78,
                "pitched_music_loop": 1.0,
                "vocal_music_phrase": 0.0,
                "voiced_one_shot": 0.0,
            },
        ),
    )

    assert final.folder_path == "Drums/Drum Loops/Loops"


def test_decisive_human_voice_candidate_blocks_pitched_percussion_drum_rescue() -> None:
    """A Human/Voice phrase candidate should beat false pitched-percussion loop rescue."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Instrument Loops/Loops",
            26.0,
            [
                candidate(
                    "FX/Human and Voice FX/Spoken Voice/Long FX", 6.0, role_strengths={"pitched_music_phrase": 0.86}
                ),
                candidate(
                    "FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/Long FX", 31.0
                ),
                candidate("Drums/Drum Loops/Loops", 42.0),
                candidate("Instruments/Instrument Loops/Loops", 26.0),
            ],
        ),
        EligibilityDecision(
            role_name="pitched_percussion_loop",
            confidence=0.78,
            allowed_top_families=("Drums", "_TO_REVIEW"),
            broad_folder_path="Drums/Drum Loops/Loops",
            reason="synthetic false pitched percussion role",
        ),
        facts(
            "bass_phrase",
            0.76,
            {
                "pitched_music_loop": 0.80,
                "low_rhythmic_drum_loop": 0.12,
                "vocal_music_phrase": 0.0,
                "voiced_one_shot": 0.0,
            },
        ),
    )

    assert final.folder_path == "Instruments/Voice/Phrase/One Shots"


def test_sax_like_reed_loop_eligibility_uses_woodwind_sax_loop_bucket() -> None:
    """Sustained mid/high pitched reed loops should not stop at generic Instrument Loops."""
    facts_obj = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "duration_sec": 12.0,
            "event_count_estimate": 33.0,
            "event_rate_hz": 2.77,
            "onset_span_ratio": 0.90,
            "attack_rise_time_norm": 0.73,
            "temporal_centroid_ratio": 0.52,
            "shape_vote": {"primary_shape": "pitched_phrase", "confidence": 0.86},
            "measured_roles": {
                "pitched_music_loop": 1.0,
                "pitched_music_phrase": 0.87,
                "vocal_music_phrase": 0.0,
                "voiced_one_shot": 0.0,
            },
            "feature_values_by_name": {
                "pitch_confidence": 0.97,
                "f0_voiced_ratio": 0.94,
                "loop_pitched_event_ratio": 1.0,
                "loop_sustained_tonal_frame_ratio": 0.86,
                "loop_non_event_tonal_ratio": 0.94,
                "loop_percussive_event_ratio": 0.0,
                "loop_drumlike_frame_ratio": 0.04,
                "sub_bass_ratio_lt_150hz": 0.02,
                "bass_ratio_150_500hz": 0.05,
                "mid_ratio_500_2000hz": 0.58,
                "presence_ratio_2000_8000hz": 0.35,
                "air_ratio_gt_8000hz": 0.01,
                "spectral_flatness_mean": 0.35,
                "formant_like_peak_spacing": 0.0,
            },
        },
        feature_values_by_name={
            "pitch_confidence": 0.97,
            "f0_voiced_ratio": 0.94,
            "loop_pitched_event_ratio": 1.0,
            "loop_sustained_tonal_frame_ratio": 0.86,
            "loop_non_event_tonal_ratio": 0.94,
            "loop_percussive_event_ratio": 0.0,
            "loop_drumlike_frame_ratio": 0.04,
            "sub_bass_ratio_lt_150hz": 0.02,
            "bass_ratio_150_500hz": 0.05,
            "mid_ratio_500_2000hz": 0.58,
            "presence_ratio_2000_8000hz": 0.35,
            "air_ratio_gt_8000hz": 0.01,
            "spectral_flatness_mean": 0.35,
            "formant_like_peak_spacing": 0.0,
        },
    )

    eligibility = infer_parent_eligibility(facts_obj)

    assert eligibility.role_name == "pitched_reed_or_instrument_loop"
    assert eligibility.broad_folder_path == "Instruments/Woodwinds/Saxophone/Loops"


def test_low_mid_sax_loop_goes_to_sax_bucket_not_pitched_percussion() -> None:
    """Tenor/low-mid sax loops are reed loops, not pitched percussion loops."""
    facts_obj = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "duration_sec": 11.89,
            "event_count_estimate": 38.0,
            "event_rate_hz": 3.20,
            "onset_span_ratio": 0.96,
            "attack_rise_time_norm": 0.065,
            "temporal_centroid_ratio": 0.40,
            "shape_vote": {"primary_shape": "pitched_phrase", "confidence": 0.92},
            "measured_roles": {
                "pitched_music_loop": 0.99,
                "pitched_music_phrase": 0.97,
                "low_rhythmic_drum_loop": 0.03,
                "vocal_music_phrase": 0.0,
                "voiced_one_shot": 0.0,
            },
            "feature_values_by_name": {
                "pitch_confidence": 0.97,
                "f0_voiced_ratio": 0.93,
                "loop_pitched_event_ratio": 1.0,
                "loop_sustained_tonal_frame_ratio": 0.92,
                "loop_non_event_tonal_ratio": 0.88,
                "loop_percussive_event_ratio": 0.0,
                "loop_drumlike_frame_ratio": 0.08,
                "sub_bass_ratio_lt_150hz": 0.001,
                "bass_ratio_150_500hz": 0.46,
                "mid_ratio_500_2000hz": 0.48,
                "presence_ratio_2000_8000hz": 0.052,
                "air_ratio_gt_8000hz": 0.0,
                "spectral_flatness_mean": 0.23,
                "formant_like_peak_spacing": 0.80,
            },
        },
        feature_values_by_name={
            "pitch_confidence": 0.97,
            "f0_voiced_ratio": 0.93,
            "loop_pitched_event_ratio": 1.0,
            "loop_sustained_tonal_frame_ratio": 0.92,
            "loop_non_event_tonal_ratio": 0.88,
            "loop_percussive_event_ratio": 0.0,
            "loop_drumlike_frame_ratio": 0.08,
            "sub_bass_ratio_lt_150hz": 0.001,
            "bass_ratio_150_500hz": 0.46,
            "mid_ratio_500_2000hz": 0.48,
            "presence_ratio_2000_8000hz": 0.052,
            "air_ratio_gt_8000hz": 0.0,
            "spectral_flatness_mean": 0.23,
            "formant_like_peak_spacing": 0.80,
        },
    )

    eligibility = infer_parent_eligibility(facts_obj)

    assert eligibility.role_name == "pitched_reed_or_instrument_loop"
    assert eligibility.broad_folder_path == "Instruments/Woodwinds/Saxophone/Loops"


def test_noisy_low_mid_sax_loop_still_goes_to_sax_bucket() -> None:
    """A lower-confidence noisy sax loop is still not generic percussion or voice."""
    facts_obj = SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={
            "duration_sec": 12.0,
            "event_count_estimate": 33.0,
            "event_rate_hz": 1.37,
            "onset_span_ratio": 0.95,
            "attack_rise_time_norm": 0.08,
            "temporal_centroid_ratio": 0.48,
            "shape_vote": {"primary_shape": "sustained_pad", "confidence": 0.75},
            "measured_roles": {"pitched_music_loop": 0.80, "voiced_one_shot": 0.0, "vocal_music_phrase": 0.0},
            "feature_values_by_name": {
                "pitch_confidence": 0.36,
                "f0_voiced_ratio": 0.82,
                "loop_pitched_event_ratio": 1.0,
                "loop_sustained_tonal_frame_ratio": 0.79,
                "loop_non_event_tonal_ratio": 0.76,
                "loop_percussive_event_ratio": 0.0,
                "loop_drumlike_frame_ratio": 0.145,
                "sub_bass_ratio_lt_150hz": 0.0003,
                "bass_ratio_150_500hz": 0.46,
                "mid_ratio_500_2000hz": 0.49,
                "presence_ratio_2000_8000hz": 0.056,
                "air_ratio_gt_8000hz": 0.0,
                "spectral_flatness_mean": 0.337,
                "formant_like_peak_spacing": 0.0,
            },
        },
        feature_values_by_name={
            "pitch_confidence": 0.36,
            "f0_voiced_ratio": 0.82,
            "loop_pitched_event_ratio": 1.0,
            "loop_sustained_tonal_frame_ratio": 0.79,
            "loop_non_event_tonal_ratio": 0.76,
            "loop_percussive_event_ratio": 0.0,
            "loop_drumlike_frame_ratio": 0.145,
            "sub_bass_ratio_lt_150hz": 0.0003,
            "bass_ratio_150_500hz": 0.46,
            "mid_ratio_500_2000hz": 0.49,
            "presence_ratio_2000_8000hz": 0.056,
            "air_ratio_gt_8000hz": 0.0,
            "spectral_flatness_mean": 0.337,
            "formant_like_peak_spacing": 0.0,
        },
    )

    eligibility = infer_parent_eligibility(facts_obj)

    assert eligibility.role_name == "pitched_reed_or_instrument_loop"
    assert eligibility.broad_folder_path == "Instruments/Woodwinds/Saxophone/Loops"


def test_sax_claim_without_actual_sax_candidate_broadens_human_voice_false_positive() -> None:
    """A measured reed role without a real sax candidate lands in safe broad Instruments."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "FX/Human and Voice FX/Spoken Voice/Long FX",
            5.0,
            [
                candidate("FX/Human and Voice FX/Spoken Voice/Long FX", 5.0),
                candidate("Instruments/Instrument Loops/Loops", 12.0),
            ],
        ),
        EligibilityDecision(
            role_name="pitched_reed_or_instrument_loop",
            confidence=0.78,
            allowed_top_families=("Instruments", "_TO_REVIEW"),
            blocked_path_fragments=("FX", "Human", "Voice"),
            broad_folder_path="Instruments/Woodwinds/Saxophone/Loops",
            reason="synthetic measured sax loop",
        ),
        facts("sustained_pad", 0.75, {"pitched_music_loop": 0.80, "voiced_one_shot": 0.0, "vocal_music_phrase": 0.0}),
    )

    assert final.folder_path == "Instruments/Instrument Loops/Loops"


def test_piano_like_vocal_shape_needs_positive_voice_claim() -> None:
    """Stable piano/key loops can look vocal-shaped but must stay Instruments."""
    final = DecisionCoreV2().apply_eligibility(
        raw(
            "Instruments/Instrument Loops/Loops",
            9.0,
            [
                candidate("FX/Designed Noise FX/Siren/Long FX", 9.0, role_strengths={"pitched_music_phrase": 0.98}),
                candidate("Instruments/Instrument Loops/Loops", 9.0, role_strengths={"pitched_music_loop": 0.95}),
                candidate(
                    "FX/Human and Voice FX/Spoken Voice/Long FX", 22.0, role_strengths={"pitched_music_phrase": 0.86}
                ),
            ],
        ),
        EligibilityDecision(
            role_name="pitched_music_loop",
            confidence=1.0,
            allowed_top_families=("Instruments", "_TO_REVIEW"),
            broad_folder_path="Instruments/Instrument Loops/Loops",
            reason="synthetic pitched/key loop",
        ),
        facts(
            "vocal_phrase",
            1.0,
            {
                "pitched_music_loop": 1.0,
                "pitched_music_phrase": 1.0,
                "vocal_music_phrase": 0.86,
                "voiced_one_shot": 0.0,
                "direct_body_view": {"measured_roles": {"vocal_music_phrase": 0.86}},
            },
        ),
    )

    assert final.folder_path.startswith("Instruments/")
    assert "Human and Voice" not in final.folder_path
