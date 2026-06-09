from __future__ import annotations

from pathlib import Path

import numpy as np

from aaron_sound_sorter.core import FEATURE_NAMES, FP_SIZE
from aaron_sound_sorter.domain.facts import build_shared_audio_facts
from aaron_sound_sorter.domain.models import AudioPhysics, CategoryGuess, SharedAudioFacts, VoterResult
from aaron_sound_sorter.engine.decision_core_v2 import DecisionCoreV2
from aaron_sound_sorter.voters.shape_voter import ShapeVoter, shape_compatible_tops


def _fp(**values: float) -> np.ndarray:
    fp = np.zeros((FP_SIZE,), dtype=np.float32)
    for name, value in values.items():
        fp[FEATURE_NAMES.index(name)] = float(value)
    return fp


def _physics(duration: float = 2.4, **values: float) -> AudioPhysics:
    return AudioPhysics(Path("shape.wav"), _fp(**values), duration, "ok")


def _guess(label: str, rank: int) -> CategoryGuess:
    return CategoryGuess(
        label=label,
        folder_path=label,
        top_family=label.split("/", 1)[0],
        score=float(rank),
        confidence=1.0 / (1.0 + rank),
        rank=rank,
        reason="test",
    )


def _finalize_consensus(brain: VoterResult, physics: VoterResult, facts: SharedAudioFacts):
    """Run the production DecisionCore path, not a dead arbiter shortcut."""
    return DecisionCoreV2().choose(brain, physics, facts)


def test_shape_voter_detects_beat_loop_without_category_destination() -> None:
    physics = _physics(
        duration=3.0,
        log_transient_count=np.log1p(12.0),
        event_rate_hz=2.2,
        onset_span_ratio=0.82,
        onset_interval_regularity=0.18,
        temporal_centroid_ratio=0.52,
        attack_rise_time_norm=0.10,
        tail_energy_ratio=0.18,
        loop_percussive_event_ratio=0.82,
        loop_drumlike_frame_ratio=0.78,
        loop_mean_event_low_ratio=0.42,
        loop_mean_event_high_ratio=0.38,
        loop_sustained_tonal_frame_ratio=0.06,
        loop_non_event_tonal_ratio=0.05,
    )
    facts = build_shared_audio_facts(physics)

    result = ShapeVoter().vote(physics, facts, {})

    shape = result.diagnostics["shape_vote"]
    assert result.voter_name == "shape"
    assert result.guesses[0].top_family == "_SHAPE_DIAGNOSTIC"
    assert result.guesses[0].evidence["voter_is_category_router"] is False
    assert shape["primary_shape"] == "beat_loop"
    assert shape["confidence"] >= 0.68
    assert shape_compatible_tops("beat_loop") == ["Drums"]


def test_shape_voter_promotes_repeated_percussive_phrase_to_drum_loop_shape() -> None:
    physics = _physics(
        duration=8.0,
        log_transient_count=np.log1p(24.0),
        event_rate_hz=2.1,
        onset_span_ratio=0.94,
        onset_interval_regularity=0.32,
        temporal_centroid_ratio=0.50,
        attack_rise_time_norm=0.08,
        tail_energy_ratio=0.40,
        loop_percussive_event_ratio=0.45,
        loop_drumlike_frame_ratio=0.20,
        loop_pitched_event_ratio=0.54,
        loop_sustained_tonal_frame_ratio=0.70,
        loop_non_event_tonal_ratio=0.78,
        sub_bass_ratio_lt_150hz=0.70,
        bass_ratio_150_500hz=0.16,
        mid_ratio_500_2000hz=0.06,
        presence_ratio_2000_8000hz=0.05,
        spectral_flatness_mean=0.35,
        spectral_entropy_mean=0.62,
    )
    facts = build_shared_audio_facts(physics)

    shape = ShapeVoter().vote(physics, facts, {}).diagnostics["shape_vote"]

    assert shape["primary_shape"] == "beat_loop"
    assert "Drums" in shape_compatible_tops(shape["primary_shape"])
    assert dict(shape["shape_scores"])["repeated_phrase_loop"] >= 0.70


def test_shape_voter_does_not_promote_ambiguous_musical_phrase_to_drum_loop() -> None:
    physics = _physics(
        duration=11.0,
        log_transient_count=np.log1p(16.0),
        event_rate_hz=1.4,
        onset_span_ratio=0.94,
        onset_interval_regularity=0.18,
        temporal_centroid_ratio=0.49,
        attack_rise_time_norm=0.01,
        tail_energy_ratio=0.65,
        loop_percussive_event_ratio=0.44,
        loop_drumlike_frame_ratio=0.0,
        loop_pitched_event_ratio=0.56,
        loop_sustained_tonal_frame_ratio=0.395833,
        loop_non_event_tonal_ratio=0.333333,
        loop_mean_event_low_ratio=0.307448,
        loop_mean_event_high_ratio=0.274097,
        loop_mean_event_noise_ratio=0.498078,
        loop_event_timbre_diversity=0.862049,
        sub_bass_ratio_lt_150hz=0.11,
        bass_ratio_150_500hz=0.12,
        mid_ratio_500_2000hz=0.45,
        presence_ratio_2000_8000hz=0.31,
        air_ratio_gt_8000hz=0.01,
        pitch_confidence=0.326893,
        f0_voiced_ratio=0.304979,
        harmonic_energy_ratio=0.428344,
        harmonic_to_noise_ratio=0.559217,
        spectral_flatness_mean=0.13,
        spectral_entropy_mean=0.29,
        stereo_width=0.115158,
    )
    facts = build_shared_audio_facts(physics)

    shape = ShapeVoter().vote(physics, facts, {}).diagnostics["shape_vote"]

    assert shape["primary_shape"] == "repeated_phrase_loop"
    assert "Drums" not in shape_compatible_tops(shape["primary_shape"])


def test_shape_voter_detects_pitched_phrase_not_percussion_hit() -> None:
    physics = _physics(
        duration=2.2,
        log_transient_count=np.log1p(3.0),
        event_rate_hz=0.55,
        onset_span_ratio=0.52,
        temporal_centroid_ratio=0.46,
        attack_rise_time_norm=0.18,
        tail_energy_ratio=0.45,
        pitch_confidence=0.86,
        f0_voiced_ratio=0.82,
        loop_pitched_event_ratio=0.86,
        loop_sustained_tonal_frame_ratio=0.72,
        loop_non_event_tonal_ratio=0.66,
        loop_percussive_event_ratio=0.08,
        loop_drumlike_frame_ratio=0.04,
        spectral_flatness_mean=0.05,
    )
    facts = build_shared_audio_facts(physics)

    shape = ShapeVoter().vote(physics, facts, {}).diagnostics["shape_vote"]

    assert shape["primary_shape"] in {"pitched_phrase", "bass_phrase"}
    assert shape["confidence"] >= 0.55
    assert "Drums" not in shape_compatible_tops(shape["primary_shape"])
    assert "shape_scores" in shape
    assert "solo_isolation_score" in shape


def test_shape_voter_v2_detects_mixed_compound_musical_loop() -> None:
    physics = _physics(
        duration=4.0,
        log_transient_count=np.log1p(24.0),
        event_rate_hz=2.3,
        onset_span_ratio=0.86,
        onset_interval_regularity=0.25,
        temporal_centroid_ratio=0.50,
        attack_rise_time_norm=0.12,
        tail_energy_ratio=0.58,
        pitch_confidence=0.78,
        f0_voiced_ratio=0.72,
        harmonic_energy_ratio=0.52,
        harmonic_to_noise_ratio=0.35,
        spectral_peak_stability=0.18,
        stereo_width=0.76,
        spectral_flatness_mean=0.27,
        spectral_entropy_mean=0.58,
        body_noise_ratio=0.34,
        tail_noise_ratio=0.42,
        body_flatness=0.26,
        tail_flatness=0.35,
        loop_pitched_event_ratio=0.86,
        loop_percussive_event_ratio=0.28,
        loop_drumlike_frame_ratio=0.18,
        loop_sustained_tonal_frame_ratio=0.70,
        loop_non_event_tonal_ratio=0.62,
        loop_noisy_event_ratio=0.36,
        loop_event_timbre_diversity=0.52,
        loop_mean_event_noise_ratio=0.33,
        loop_mean_event_low_ratio=0.30,
        loop_mean_event_high_ratio=0.18,
        sub_bass_ratio_lt_150hz=0.18,
        bass_ratio_150_500hz=0.20,
        mid_ratio_500_2000hz=0.42,
        presence_ratio_2000_8000hz=0.13,
        air_ratio_gt_8000hz=0.06,
    )
    facts = build_shared_audio_facts(physics)

    shape = ShapeVoter().vote(physics, facts, {}).diagnostics["shape_vote"]
    top_shapes = [name for name, _score in shape["shape_scores"][:3]]

    assert shape["primary_shape"] in {"mixed_instrument_loop", "compound_musical_loop", "instrument_plus_fx_loop"}
    assert "compound_musical_loop" in top_shapes
    assert shape["layered_loop_score"] >= 0.70
    assert shape["instrument_plus_fx_loop_score"] >= 0.68
    assert shape["solo_isolation_score"] < 0.45
    assert "Instruments" in shape_compatible_tops(shape["primary_shape"])


def test_shape_voter_v2_keeps_solo_phrase_isolated() -> None:
    physics = _physics(
        duration=3.0,
        log_transient_count=np.log1p(7.0),
        event_rate_hz=0.85,
        onset_span_ratio=0.58,
        onset_interval_regularity=0.42,
        temporal_centroid_ratio=0.48,
        attack_rise_time_norm=0.17,
        tail_energy_ratio=0.48,
        pitch_confidence=0.90,
        f0_voiced_ratio=0.88,
        harmonic_energy_ratio=0.70,
        harmonic_to_noise_ratio=0.62,
        spectral_peak_stability=0.42,
        stereo_width=0.18,
        spectral_flatness_mean=0.12,
        spectral_entropy_mean=0.34,
        body_noise_ratio=0.10,
        tail_noise_ratio=0.18,
        loop_pitched_event_ratio=0.96,
        loop_percussive_event_ratio=0.04,
        loop_drumlike_frame_ratio=0.04,
        loop_sustained_tonal_frame_ratio=0.86,
        loop_non_event_tonal_ratio=0.82,
        loop_event_timbre_diversity=0.12,
        loop_mean_event_low_ratio=0.12,
        loop_mean_event_high_ratio=0.12,
        sub_bass_ratio_lt_150hz=0.04,
        bass_ratio_150_500hz=0.16,
        mid_ratio_500_2000hz=0.55,
        presence_ratio_2000_8000hz=0.10,
        air_ratio_gt_8000hz=0.04,
    )
    facts = build_shared_audio_facts(physics)

    shape = ShapeVoter().vote(physics, facts, {}).diagnostics["shape_vote"]

    assert shape["primary_shape"] in {"pitched_phrase", "solo_phrase", "pitched_phrase_shape"}
    assert shape["solo_isolation_score"] >= 0.78
    assert shape["layered_loop_score"] < 0.45
    assert "solo_phrase" in [name for name, _score in shape["shape_scores"][:4]]


def test_shape_voter_v2_distinguishes_echo_tail_hit_from_true_loop() -> None:
    physics = _physics(
        duration=1.4,
        log_transient_count=np.log1p(1.0),
        event_rate_hz=0.7,
        onset_span_ratio=0.04,
        onset_interval_regularity=0.90,
        temporal_centroid_ratio=0.20,
        attack_rise_time_norm=0.02,
        tail_energy_ratio=0.72,
        pitch_confidence=0.30,
        f0_voiced_ratio=0.10,
        spectral_flatness_mean=0.22,
        spectral_entropy_mean=0.44,
        loop_pitched_event_ratio=0.10,
        loop_percussive_event_ratio=0.18,
        loop_drumlike_frame_ratio=0.15,
        loop_sustained_tonal_frame_ratio=0.12,
        loop_non_event_tonal_ratio=0.08,
    )
    facts = build_shared_audio_facts(physics)

    shape = ShapeVoter().vote(physics, facts, {}).diagnostics["shape_vote"]

    assert shape["primary_shape"] in {"hit_with_tail", "echo_tail_hit", "single_hit"}
    assert shape["echo_tail_score"] >= 0.80
    assert shape["true_repetition_score"] < 0.35


def test_shape_voter_v2_detects_fx_motion_shapes_without_routing() -> None:
    physics = _physics(
        duration=1.8,
        log_transient_count=np.log1p(2.0),
        event_rate_hz=0.45,
        onset_span_ratio=0.42,
        temporal_centroid_ratio=0.56,
        attack_rise_time_norm=0.30,
        tail_energy_ratio=0.58,
        centroid_slope_norm=0.24,
        spectral_flatness_mean=0.44,
        spectral_entropy_mean=0.74,
        stereo_width=0.68,
        body_noise_ratio=0.44,
        tail_noise_ratio=0.50,
        presence_ratio_2000_8000hz=0.24,
        air_ratio_gt_8000hz=0.08,
    )
    facts = build_shared_audio_facts(physics)

    shape = ShapeVoter().vote(physics, facts, {}).diagnostics["shape_vote"]

    assert shape["primary_shape"] in {"transition_riser", "whoosh_sweep", "hybrid_fx_motion"}
    assert "FX" in shape_compatible_tops(shape["primary_shape"])


def test_shape_voter_v2_detects_glitch_and_static_boundary() -> None:
    glitch_physics = _physics(
        duration=1.0,
        log_transient_count=np.log1p(16.0),
        event_rate_hz=7.0,
        onset_span_ratio=0.72,
        onset_interval_regularity=0.18,
        attack_rise_time_norm=0.04,
        tail_energy_ratio=0.16,
        spectral_flatness_mean=0.50,
        spectral_entropy_mean=0.78,
        spectral_flux_mean=0.34,
        spectral_flux_variance=0.06,
        body_noise_ratio=0.52,
    )
    static_physics = _physics(
        duration=4.0,
        log_transient_count=np.log1p(1.0),
        event_rate_hz=0.12,
        onset_span_ratio=0.08,
        attack_rise_time_norm=0.28,
        tail_energy_ratio=0.54,
        centroid_slope_norm=0.01,
        spectral_flatness_mean=0.62,
        spectral_entropy_mean=0.86,
        zcr_mean=0.34,
        body_noise_ratio=0.64,
        tail_noise_ratio=0.62,
    )

    glitch_shape = (
        ShapeVoter().vote(glitch_physics, build_shared_audio_facts(glitch_physics), {}).diagnostics["shape_vote"]
    )
    static_shape = (
        ShapeVoter().vote(static_physics, build_shared_audio_facts(static_physics), {}).diagnostics["shape_vote"]
    )

    assert glitch_shape["primary_shape"] == "glitch_stutter"
    assert shape_compatible_tops(glitch_shape["primary_shape"]) == ["FX"]
    assert static_shape["primary_shape"] in {"static_bed", "noise_texture"}
    assert shape_compatible_tops(static_shape["primary_shape"]) == ["Textures", "FX"]


def test_single_low_pitched_hit_does_not_become_bass_phrase_shape() -> None:
    physics = _physics(
        duration=0.25,
        log_transient_count=np.log1p(1.0),
        event_rate_hz=4.8,
        onset_span_ratio=0.0,
        temporal_centroid_ratio=0.20,
        attack_rise_time_norm=0.002,
        tail_energy_ratio=0.25,
        pitch_confidence=0.96,
        f0_voiced_ratio=0.0,
        loop_pitched_event_ratio=1.0,
        loop_sustained_tonal_frame_ratio=1.0,
        loop_non_event_tonal_ratio=1.0,
        loop_percussive_event_ratio=0.0,
        loop_drumlike_frame_ratio=0.0,
        sub_bass_ratio_lt_150hz=0.88,
        bass_ratio_150_500hz=0.12,
        presence_ratio_2000_8000hz=0.0,
        air_ratio_gt_8000hz=0.0,
    )
    facts = build_shared_audio_facts(physics)

    shape = ShapeVoter().vote(physics, facts, {}).diagnostics["shape_vote"]

    assert shape["primary_shape"] in {"single_hit", "hit_with_tail"}
    assert shape["primary_shape"] != "bass_phrase"


def test_final_shape_invariant_blocks_beat_loop_from_bird_fx() -> None:
    facts = SharedAudioFacts(
        False,
        True,
        False,
        False,
        False,
        evidence={"shape_vote": {"primary_shape": "beat_loop", "confidence": 0.86}},
    )
    fx = "FX/Biological Foley/Animals/Bird Call/Loops"
    drum = "Drums/Drum Loops/Loops"

    decision = _finalize_consensus(
        VoterResult("brain", [_guess(fx, 1), _guess(drum, 6)]),
        VoterResult("physics", [_guess(fx, 1), _guess(drum, 8)]),
        facts,
    )

    assert decision.consensus_status == "final_shape_review_broad_drum_loop_invariant"
    assert decision.final_top == "Drums"
    assert decision.final_label == drum


def test_final_shape_invariant_uses_broad_drum_loop_when_fx_is_incompatible() -> None:
    facts = SharedAudioFacts(
        False,
        True,
        False,
        False,
        False,
        evidence={"shape_vote": {"primary_shape": "beat_loop", "confidence": 0.88}},
    )
    fx = "FX/Biological Foley/Animals/Bird Call/Loops"

    decision = _finalize_consensus(
        VoterResult("brain", [_guess(fx, 1)]),
        VoterResult("physics", [_guess(fx, 1)]),
        facts,
    )

    assert decision.consensus_status == "final_shape_review_broad_drum_loop_invariant"
    assert decision.final_label == "Drums/Drum Loops/Loops"


def _guess_with_evidence(label: str, rank: int, evidence: dict) -> CategoryGuess:
    return CategoryGuess(
        label=label,
        folder_path=label,
        top_family=label.split("/", 1)[0],
        score=float(rank),
        confidence=1.0 / (1.0 + rank),
        rank=rank,
        reason="test",
        evidence=evidence,
    )


def test_pitched_music_loop_role_cannot_rescue_into_drums() -> None:
    facts = SharedAudioFacts(
        False,
        True,
        False,
        False,
        True,
        evidence={
            "measured_roles": {"pitched_music_loop": 0.98, "percussive_drum_loop": 0.0},
            "shape_vote": {"primary_shape": "pitched_phrase", "confidence": 0.91},
        },
    )
    fx = "FX/Human and Voice FX/Spoken Voice/Long FX"
    drum = "Drums/Percussion/Generic Percussion/Loops"
    instrument = "Instruments/Instrument Loops/Loops"
    evidence = {
        "candidate_role_signature": {"pitched_music_loop": 1.0},
        "profile_effective_count": 4,
        "profile_fact_profile_strength": "tentative",
    }

    decision = _finalize_consensus(
        VoterResult(
            "brain",
            [
                _guess_with_evidence(fx, 1, evidence),
                _guess_with_evidence(drum, 2, evidence),
                _guess_with_evidence(instrument, 5, evidence),
            ],
        ),
        VoterResult(
            "physics",
            [
                _guess_with_evidence(fx, 1, evidence),
                _guess_with_evidence(drum, 3, evidence),
                _guess_with_evidence(instrument, 8, evidence),
            ],
        ),
        facts,
    )

    assert decision.final_top == "Instruments"
    assert decision.final_label == instrument
    assert decision.final_label != drum


def test_depth_decider_keeps_clear_terminal_identity_even_when_profile_is_tiny() -> None:
    facts = SharedAudioFacts(
        False,
        True,
        False,
        False,
        True,
        evidence={"measured_roles": {"pitched_music_loop": 0.96}},
    )
    terminal = "Instruments/Woodwinds/Saxophone/Loops"
    broad = "Instruments/Instrument Loops/Loops"
    tiny = {
        "candidate_role_signature": {"pitched_music_loop": 1.0},
        "profile_effective_count": 4,
        "profile_fact_profile_strength": "tentative",
    }
    strong = {
        "candidate_role_signature": {"pitched_music_loop": 0.92},
        "profile_effective_count": 80,
        "profile_fact_profile_strength": "strong",
    }

    decision = _finalize_consensus(
        VoterResult(
            "brain",
            [
                _guess_with_evidence(terminal, 1, tiny),
                _guess_with_evidence(broad, 4, strong),
            ],
        ),
        VoterResult(
            "physics",
            [
                _guess_with_evidence(terminal, 1, tiny),
                _guess_with_evidence(broad, 13, strong),
            ],
        ),
        facts,
    )

    assert decision.consensus_status == "strong_consensus"
    assert decision.final_label == terminal


def test_family_rescue_stops_at_broad_bucket_when_terminal_identity_is_not_proven() -> None:
    facts = SharedAudioFacts(
        False,
        True,
        False,
        False,
        True,
        evidence={
            "measured_roles": {"pitched_music_loop": 0.96},
            "dynamic_role_gate": {
                "selected_top_families": ["Instruments"],
                "top_scores": [
                    {"prefix": "Instruments", "score": 1.0},
                    {"prefix": "FX", "score": 80.0},
                ],
            },
        },
    )
    raw_fx = "FX/Designed Noise FX/Siren/Long FX"
    guitar = "Instruments/Guitar/Electric Guitar/Loops"
    broad = "Instruments/Instrument Loops/Loops"
    evidence = {
        "candidate_role_signature": {"pitched_music_loop": 0.80},
        "profile_effective_count": 80,
        "profile_fact_profile_strength": "strong",
    }

    decision = _finalize_consensus(
        VoterResult(
            "brain",
            [
                _guess_with_evidence(raw_fx, 1, evidence),
                _guess_with_evidence(guitar, 4, evidence),
                _guess_with_evidence(broad, 10, evidence),
            ],
        ),
        VoterResult(
            "physics",
            [
                _guess_with_evidence(raw_fx, 1, evidence),
                _guess_with_evidence(guitar, 14, evidence),
                _guess_with_evidence(broad, 11, evidence),
            ],
        ),
        facts,
    )

    assert decision.consensus_status == "strong_consensus"
    assert decision.final_label == raw_fx


def test_decisive_top_family_gate_needs_shape_or_role_support() -> None:
    facts = SharedAudioFacts(
        False,
        True,
        False,
        False,
        True,
        evidence={
            "shape_vote": {"primary_shape": "vocal_phrase", "confidence": 0.95},
            "dynamic_role_gate": {
                "selected_top_families": ["Instruments"],
                "top_scores": [
                    {"prefix": "Instruments", "score": 1.0},
                    {"prefix": "FX", "score": 80.0},
                ],
            },
        },
    )
    fx_voice = "FX/Human and Voice FX/Spoken Voice/Long FX"
    inst_loop = "Instruments/Instrument Loops/Loops"

    decision = _finalize_consensus(
        VoterResult("brain", [_guess(fx_voice, 1), _guess(inst_loop, 4)]),
        VoterResult("physics", [_guess(fx_voice, 1), _guess(inst_loop, 5)]),
        facts,
    )

    assert decision.final_label == "Instruments/Voice/Phrase/One Shots"
    assert decision.consensus_status == "final_shape_review_voice_phrase_invariant"


def test_shape_voter_routes_fast_pitched_repetition_away_from_drums() -> None:
    """Repeated pitched attacks are musical phrase structure, not drum proof."""
    physics = _physics(
        duration=4.0,
        log_transient_count=np.log1p(14.0),
        event_rate_hz=2.0,
        onset_span_ratio=0.88,
        onset_interval_regularity=0.20,
        temporal_centroid_ratio=0.44,
        attack_rise_time_norm=0.05,
        tail_energy_ratio=0.35,
        pitch_confidence=0.78,
        f0_voiced_ratio=0.70,
        harmonic_energy_ratio=0.70,
        harmonic_to_noise_ratio=0.62,
        loop_pitched_event_ratio=0.82,
        loop_sustained_tonal_frame_ratio=0.70,
        loop_non_event_tonal_ratio=0.66,
        loop_tonal_to_percussive_balance=0.72,
        loop_percussive_event_ratio=0.18,
        loop_drumlike_frame_ratio=0.08,
        loop_mean_event_low_ratio=0.18,
        loop_mean_event_high_ratio=0.20,
        loop_mean_event_noise_ratio=0.12,
        spectral_flatness_mean=0.07,
        spectral_entropy_mean=0.28,
    )
    facts = build_shared_audio_facts(physics)
    facts.evidence["onset_pitched_onset_score"] = 0.72
    facts.evidence["onset_percussive_onset_score"] = 0.28
    facts.evidence["plucked_string_authority_score"] = 0.62
    facts.evidence["synth_tonal_source_score"] = 0.58

    shape = ShapeVoter().vote(physics, facts, {}).diagnostics["shape_vote"]
    scores = dict(shape["shape_scores"])

    assert shape["primary_shape"] == "pitched_repetition_phrase"
    assert scores["pitched_repetition_phrase"] > scores["beat_loop"]
    assert "Instruments" in shape_compatible_tops(shape["primary_shape"])
    assert "Drums" not in shape_compatible_tops(shape["primary_shape"])
