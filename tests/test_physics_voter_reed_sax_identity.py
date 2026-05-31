"""PhysicsVoter reed/sax identity witness regressions.

These tests lock the v31.102 architecture change: wet sax-like physics should
be visible inside the PhysicsVoter itself, not only as a late arbiter rescue.
The evidence is source-name blind and uses the stored named feature map.
"""

from __future__ import annotations

import numpy as np

from aaron_sound_sorter.core import FEATURE_NAMES, FP_SIZE
from aaron_sound_sorter.domain.facts import build_shared_audio_facts
from aaron_sound_sorter.domain.models import AudioPhysics
from aaron_sound_sorter.domain.policies import PhysicsVoterPolicy
from aaron_sound_sorter.voters.physics_voter import PhysicsVoter


def fingerprint_from_named(values: dict[str, float]) -> np.ndarray:
    """Build a finite fingerprint using stable feature names."""
    fp = np.zeros((FP_SIZE,), dtype=np.float32)
    for name, value in values.items():
        fp[FEATURE_NAMES.index(name)] = float(value)
    return fp


def physics_from_named(values: dict[str, float], duration: float = 8.0) -> AudioPhysics:
    """Create AudioPhysics without any filename evidence."""
    return AudioPhysics(
        source_path=__import__("pathlib").Path("anonymous.wav"),
        fingerprint=fingerprint_from_named(values),
        duration_sec=duration,
        read_status="ok",
    )


def fact_profile(values: dict[str, float]) -> dict:
    """Tiny neutral fact profile sufficient for PhysicsVoter ranking tests."""
    return {
        "effective_count": 12,
        "fact_profile_strength": "ok",
        "feature_stats": {
            name: {
                "median": value,
                "iqr": 0.10,
                "mad": 0.05,
                "p10": value - 0.10,
                "p90": value + 0.10,
                "valid_count": 8,
            }
            for name, value in values.items()
        },
    }


WET_SAX_LIKE_VALUES = {
    "pitch_confidence": 0.88,
    "f0_voiced_ratio": 0.99,
    "loop_pitched_event_ratio": 1.0,
    "loop_sustained_tonal_frame_ratio": 1.0,
    "loop_percussive_event_ratio": 0.0,
    "harmonic_energy_ratio": 0.46,
    "inharmonicity": 0.18,
    "fundamental_dominance_ratio": 0.93,
    "body_flatness": 0.14,
    "tail_flatness": 0.29,
    "spectral_flatness_mean": 0.14,
    "body_noise_ratio": 0.14,
    "tail_noise_ratio": 0.22,
    "body_entropy": 0.18,
    "tail_entropy": 0.34,
    "bass_ratio_150_500hz": 0.86,
    "mid_ratio_500_2000hz": 0.09,
    "presence_ratio_2000_8000hz": 0.035,
    "air_ratio_gt_8000hz": 0.0001,
    "tail_energy_ratio": 0.44,
    "spectral_peak_stability": 0.27,
    "formant_like_peak_spacing": 0.0,
}


CLEAN_SYNTH_VALUES = {
    "pitch_confidence": 0.72,
    "f0_voiced_ratio": 1.0,
    "loop_pitched_event_ratio": 1.0,
    "loop_sustained_tonal_frame_ratio": 1.0,
    "loop_percussive_event_ratio": 0.0,
    "harmonic_energy_ratio": 0.24,
    "inharmonicity": 0.22,
    "fundamental_dominance_ratio": 0.99,
    "body_flatness": 0.002,
    "tail_flatness": 0.002,
    "spectral_flatness_mean": 0.002,
    "body_noise_ratio": 0.002,
    "tail_noise_ratio": 0.002,
    "body_entropy": 0.36,
    "tail_entropy": 0.36,
    "bass_ratio_150_500hz": 0.20,
    "mid_ratio_500_2000hz": 0.79,
    "presence_ratio_2000_8000hz": 0.0,
    "air_ratio_gt_8000hz": 0.0,
    "tail_energy_ratio": 0.73,
    "spectral_peak_stability": 0.22,
    "formant_like_peak_spacing": 1.25,
}


CLEAN_KEYS_VALUES = {
    "pitch_confidence": 0.88,
    "f0_voiced_ratio": 1.0,
    "loop_pitched_event_ratio": 1.0,
    "loop_sustained_tonal_frame_ratio": 1.0,
    "loop_percussive_event_ratio": 0.0,
    "harmonic_energy_ratio": 0.62,
    "inharmonicity": 0.13,
    "fundamental_dominance_ratio": 0.001,
    "body_flatness": 0.035,
    "tail_flatness": 0.034,
    "spectral_flatness_mean": 0.034,
    "body_noise_ratio": 0.034,
    "tail_noise_ratio": 0.034,
    "body_entropy": 0.31,
    "tail_entropy": 0.31,
    "bass_ratio_150_500hz": 0.33,
    "mid_ratio_500_2000hz": 0.66,
    "presence_ratio_2000_8000hz": 0.005,
    "air_ratio_gt_8000hz": 0.0,
    "tail_energy_ratio": 0.64,
    "spectral_peak_stability": 0.36,
    "formant_like_peak_spacing": 0.0,
}


CLEAN_LOW_MID_PIANO_LOOP_VALUES = {
    "pitch_confidence": 0.59,
    "f0_voiced_ratio": 0.95,
    "loop_pitched_event_ratio": 1.0,
    "loop_sustained_tonal_frame_ratio": 1.0,
    "loop_percussive_event_ratio": 0.0,
    "harmonic_energy_ratio": 0.42,
    "inharmonicity": 0.13,
    "fundamental_dominance_ratio": 0.24,
    "body_flatness": 0.047,
    "tail_flatness": 0.052,
    "spectral_flatness_mean": 0.047,
    "body_noise_ratio": 0.047,
    "tail_noise_ratio": 0.052,
    "body_entropy": 0.34,
    "tail_entropy": 0.37,
    "bass_ratio_150_500hz": 0.35,
    "mid_ratio_500_2000hz": 0.435,
    "presence_ratio_2000_8000hz": 0.015,
    "air_ratio_gt_8000hz": 0.002,
    "tail_energy_ratio": 0.38,
    "spectral_peak_stability": 0.20,
    "formant_like_peak_spacing": 0.0,
}


CLEAN_SYNTH_PAD_LOOP_VALUES = {
    "pitch_confidence": 0.80,
    "f0_voiced_ratio": 1.0,
    "loop_pitched_event_ratio": 1.0,
    "loop_sustained_tonal_frame_ratio": 1.0,
    "loop_percussive_event_ratio": 0.0,
    "harmonic_energy_ratio": 0.53,
    "inharmonicity": 0.16,
    "fundamental_dominance_ratio": 0.0,
    "body_flatness": 0.0005,
    "tail_flatness": 0.001,
    "spectral_flatness_mean": 0.0005,
    "body_noise_ratio": 0.18,
    "tail_noise_ratio": 0.12,
    "body_entropy": 0.30,
    "tail_entropy": 0.31,
    "bass_ratio_150_500hz": 0.66,
    "mid_ratio_500_2000hz": 0.34,
    "presence_ratio_2000_8000hz": 0.0,
    "air_ratio_gt_8000hz": 0.0,
    "tail_energy_ratio": 0.68,
    "spectral_peak_stability": 0.18,
    "formant_like_peak_spacing": 0.0,
    "loop_mean_event_low_ratio": 0.64,
    "loop_mean_event_high_ratio": 0.0,
}


BOWED_STRING_VALUES = {
    "pitch_confidence": 0.92,
    "f0_voiced_ratio": 1.0,
    "loop_pitched_event_ratio": 1.0,
    "loop_sustained_tonal_frame_ratio": 1.0,
    "loop_percussive_event_ratio": 0.0,
    "harmonic_energy_ratio": 0.44,
    "inharmonicity": 0.20,
    "fundamental_dominance_ratio": 0.05,
    "body_flatness": 0.32,
    "tail_flatness": 0.32,
    "spectral_flatness_mean": 0.32,
    "body_noise_ratio": 0.30,
    "tail_noise_ratio": 0.36,
    "body_entropy": 0.50,
    "tail_entropy": 0.53,
    "bass_ratio_150_500hz": 0.26,
    "mid_ratio_500_2000hz": 0.49,
    "presence_ratio_2000_8000hz": 0.25,
    "air_ratio_gt_8000hz": 0.002,
    "tail_energy_ratio": 0.62,
    "spectral_peak_stability": 0.10,
    "formant_like_peak_spacing": 4.0,
}


def test_reed_sax_identity_witness_flags_wet_sustained_sax_physics() -> None:
    facts = build_shared_audio_facts(physics_from_named(WET_SAX_LIKE_VALUES))

    evidence = PhysicsVoter.reed_sax_identity_evidence(facts)

    assert evidence["reed_sax_identity_eligible"] is True
    assert evidence["reed_sax_identity_score"] >= 0.80
    assert evidence["reed_sax_air_noise_score"] >= 0.90
    assert evidence["reed_sax_non_percussive_score"] >= 0.90


def test_reed_sax_identity_witness_rejects_clean_synth_clean_keys_and_bowed_strings() -> None:
    synth_facts = build_shared_audio_facts(physics_from_named(CLEAN_SYNTH_VALUES))
    keys_facts = build_shared_audio_facts(physics_from_named(CLEAN_KEYS_VALUES))
    string_facts = build_shared_audio_facts(physics_from_named(BOWED_STRING_VALUES))

    synth = PhysicsVoter.reed_sax_identity_evidence(synth_facts)
    keys = PhysicsVoter.reed_sax_identity_evidence(keys_facts)
    strings = PhysicsVoter.reed_sax_identity_evidence(string_facts)

    assert synth["reed_sax_identity_eligible"] is False
    assert synth["reed_sax_clean_synth_penalty"] > 0.0
    assert keys["reed_sax_identity_eligible"] is False
    assert keys["reed_sax_clean_keys_penalty"] > 0.0
    assert strings["reed_sax_identity_eligible"] is False
    assert strings["reed_sax_bowed_string_penalty"] > 0.0


def test_reed_sax_identity_witness_rejects_clean_low_mid_piano_loop_decoy() -> None:
    facts = build_shared_audio_facts(physics_from_named(CLEAN_LOW_MID_PIANO_LOOP_VALUES))
    facts.evidence["shape_vote"] = {
        "primary_shape": "pitched_phrase",
        "low_event_ratio": 0.406,
        "mid_event_ratio": 0.573,
        "high_event_ratio": 0.021,
    }

    evidence = PhysicsVoter.reed_sax_identity_evidence(facts)

    assert evidence["reed_sax_identity_eligible"] is False
    assert evidence["reed_sax_first_arrival_eligible"] is False
    assert evidence["reed_sax_clean_low_mid_keys_loop"] is True
    assert evidence["reed_sax_clean_keys_penalty"] >= 0.54


def test_reed_sax_identity_witness_rejects_clean_synth_pad_loop_decoy() -> None:
    facts = build_shared_audio_facts(physics_from_named(CLEAN_SYNTH_PAD_LOOP_VALUES))
    facts.evidence["shape_vote"] = {
        "primary_shape": "vocal_phrase",
        "low_event_ratio": 0.64,
        "mid_event_ratio": 0.36,
        "high_event_ratio": 0.0,
    }

    evidence = PhysicsVoter.reed_sax_identity_evidence(facts, FIRST_ARRIVAL_WET_SAX_TELEMETRY)

    assert evidence["reed_sax_identity_eligible"] is False
    assert evidence["reed_sax_first_arrival_eligible"] is False
    assert evidence["reed_sax_clean_synth_pad_loop"] is True
    assert evidence["reed_sax_clean_synth_penalty"] >= 0.50


def test_reed_sax_identity_obeys_strong_non_woodwind_instrument_branch() -> None:
    """A strong Mallet/Bell panel prevents sax lift from first-arrival glare."""
    facts = build_shared_audio_facts(physics_from_named(WET_SAX_LIKE_VALUES))
    facts.evidence["shape_vote"] = {
        "primary_shape": "vocal_phrase",
        "low_event_ratio": 0.14,
        "mid_event_ratio": 0.73,
        "high_event_ratio": 0.12,
    }
    facts.evidence["physics_layer_decision"] = {
        "physics_layer_branch": "MalletBell",
        "physics_layer_branch_confidence": 0.79,
        "instrument_branch_selected": "MalletBell",
        "instrument_branch_selected_confidence": 0.79,
        "instrument_branch_Woodwinds": 0.77,
        "instrument_woodwind_source_signal": False,
        "instrument_MalletBell_subpanel_selected": "SteelPanHandpan",
        "instrument_MalletBell_subpanel_confidence": 0.82,
        "instrument_MalletBell_subpanel_margin": 0.15,
    }

    evidence = PhysicsVoter.reed_sax_identity_evidence(facts, FIRST_ARRIVAL_WET_SAX_TELEMETRY)

    assert evidence["reed_sax_branch_identity_conflict"] is True
    assert evidence["reed_sax_identity_eligible"] is False
    assert evidence["reed_sax_first_arrival_eligible"] is False


def test_physics_voter_can_rank_sax_identity_before_generic_loop_when_reed_evidence_is_strong() -> None:
    sax = "Instruments/Woodwinds/Saxophone/One Shots"
    generic_loop = "Instruments/Instrument Loops/Loops"
    synth = "Instruments/Synths/Synth Lead/One Shots"
    brain = {
        "labels": [generic_loop, synth, sax],
        "top_by_label": {generic_loop: "Instruments", synth: "Instruments", sax: "Instruments"},
        "structure_by_label": {generic_loop: "loop", synth: "one_shot", sax: "one_shot"},
        "category_fact_profiles": {
            generic_loop: fact_profile({"pitch_confidence": 0.80, "f0_voiced_ratio": 0.95}),
            synth: fact_profile({"pitch_confidence": 0.80, "body_flatness": 0.01}),
            sax: fact_profile({"pitch_confidence": 0.80, "f0_voiced_ratio": 0.95}),
        },
    }
    physics = physics_from_named(WET_SAX_LIKE_VALUES)
    facts = build_shared_audio_facts(physics)

    result = PhysicsVoter(PhysicsVoterPolicy(top_n=3)).vote(physics, facts, brain)

    assert result.guesses[0].label == sax
    assert result.guesses[0].evidence["reed_sax_physics_check"] == "structure_neutral_identity_lift"
    assert (
        result.guesses[0].evidence["reed_sax_score_after_adjustment"]
        < result.guesses[0].evidence["reed_sax_score_before_adjustment"]
    )


FIRST_ARRIVAL_WET_SAX_TELEMETRY = {
    "status": "ok",
    "cepstral_pitch_period_coherence": 0.72,
    "first_arrival_conical_balance": 0.58,
    "first_arrival_presence_contrast_db": 18.5,
    "pre_onset_diffuse_ratio": 0.18,
    "tail_diffusion_ratio": 0.16,
    "stochastic_modulation_coherence": 0.62,
    "formant_stability_score": 0.56,
    "selected_event_count": 3,
}


WEAK_GLOBAL_WET_SAX_VALUES = {
    "pitch_confidence": 0.86,
    "f0_voiced_ratio": 0.44,
    "loop_pitched_event_ratio": 1.0,
    "loop_sustained_tonal_frame_ratio": 1.0,
    "loop_percussive_event_ratio": 0.0,
    "harmonic_energy_ratio": 0.06,
    "inharmonicity": 0.18,
    "fundamental_dominance_ratio": 0.55,
    "body_flatness": 0.12,
    "tail_flatness": 0.27,
    "spectral_flatness_mean": 0.12,
    "body_noise_ratio": 0.13,
    "tail_noise_ratio": 0.26,
    "body_entropy": 0.20,
    "tail_entropy": 0.38,
    "bass_ratio_150_500hz": 0.80,
    "mid_ratio_500_2000hz": 0.11,
    "presence_ratio_2000_8000hz": 0.006,
    "air_ratio_gt_8000hz": 0.0001,
    "tail_energy_ratio": 0.69,
    "spectral_peak_stability": 0.20,
    "formant_like_peak_spacing": 0.0,
}


DENSE_VOCAL_PHRASE_VALUES = {
    "pitch_confidence": 0.82,
    "f0_voiced_ratio": 0.95,
    "loop_pitched_event_ratio": 1.0,
    "loop_sustained_tonal_frame_ratio": 0.92,
    "loop_percussive_event_ratio": 0.0,
    "harmonic_energy_ratio": 0.34,
    "inharmonicity": 0.20,
    "fundamental_dominance_ratio": 0.45,
    "body_flatness": 0.31,
    "tail_flatness": 0.34,
    "spectral_flatness_mean": 0.31,
    "body_noise_ratio": 0.30,
    "tail_noise_ratio": 0.32,
    "body_entropy": 0.42,
    "tail_entropy": 0.46,
    "bass_ratio_150_500hz": 0.18,
    "mid_ratio_500_2000hz": 0.56,
    "presence_ratio_2000_8000hz": 0.12,
    "air_ratio_gt_8000hz": 0.08,
    "tail_energy_ratio": 0.46,
    "spectral_peak_stability": 0.25,
    "formant_like_peak_spacing": 0.0,
    "log_transient_count": float(np.log1p(82.0)),
}


def test_first_arrival_telemetry_can_lift_reverb_masked_sax_when_global_harmonics_are_weak() -> None:
    facts = build_shared_audio_facts(physics_from_named(WEAK_GLOBAL_WET_SAX_VALUES))

    without_gate = PhysicsVoter.reed_sax_identity_evidence(facts)
    with_gate = PhysicsVoter.reed_sax_identity_evidence(facts, FIRST_ARRIVAL_WET_SAX_TELEMETRY)

    assert without_gate["reed_sax_identity_eligible"] is False
    assert with_gate["reed_sax_identity_eligible"] is True
    assert with_gate["reed_sax_first_arrival_eligible"] is True
    assert with_gate["reed_sax_first_arrival_harmonic_identity_score"] >= 0.40


def test_dense_vocal_phrase_penalty_blocks_first_arrival_sax_false_positive() -> None:
    facts = build_shared_audio_facts(physics_from_named(DENSE_VOCAL_PHRASE_VALUES))

    evidence = PhysicsVoter.reed_sax_identity_evidence(facts, FIRST_ARRIVAL_WET_SAX_TELEMETRY)

    assert evidence["reed_sax_identity_eligible"] is False
    assert evidence["reed_sax_dense_vocal_phrase_penalty"] > 0.0
    assert evidence["reed_sax_first_arrival_eligible"] is False


def test_reed_sax_identity_rejects_repeated_low_rhythmic_loop_decoy() -> None:
    values = {
        "pitch_confidence": 0.93,
        "f0_voiced_ratio": 0.47,
        "loop_pitched_event_ratio": 0.88,
        "loop_sustained_tonal_frame_ratio": 0.77,
        "loop_percussive_event_ratio": 0.13,
        "harmonic_energy_ratio": 0.46,
        "inharmonicity": 0.18,
        "fundamental_dominance_ratio": 0.55,
        "body_flatness": 0.34,
        "tail_flatness": 0.36,
        "spectral_flatness_mean": 0.34,
        "body_noise_ratio": 0.34,
        "tail_noise_ratio": 0.36,
        "body_entropy": 0.38,
        "tail_entropy": 0.46,
        "bass_ratio_150_500hz": 0.74,
        "mid_ratio_500_2000hz": 0.17,
        "presence_ratio_2000_8000hz": 0.08,
        "air_ratio_gt_8000hz": 0.01,
        "tail_energy_ratio": 0.66,
        "spectral_peak_stability": 0.22,
        "formant_like_peak_spacing": 0.0,
        "log_transient_count": float(np.log1p(53.0)),
        "loop_mean_event_low_ratio": 0.74,
        "loop_mean_event_high_ratio": 0.09,
    }
    facts = build_shared_audio_facts(physics_from_named(values))
    facts.evidence["shape_vote"] = {
        "primary_shape": "bass_phrase",
        "true_repetition_score": 0.85,
        "low_event_ratio": 0.74,
        "mid_event_ratio": 0.17,
        "high_event_ratio": 0.09,
    }

    evidence = PhysicsVoter.reed_sax_identity_evidence(facts, FIRST_ARRIVAL_WET_SAX_TELEMETRY)

    assert evidence["reed_sax_identity_eligible"] is False
    assert evidence["reed_sax_rhythmic_low_loop_decoy"] is True
    assert evidence["reed_sax_first_arrival_eligible"] is False


def test_reed_sax_identity_rejects_clean_highless_pitched_loop_decoy() -> None:
    values = {
        "pitch_confidence": 0.55,
        "f0_voiced_ratio": 0.70,
        "loop_pitched_event_ratio": 1.0,
        "loop_sustained_tonal_frame_ratio": 1.0,
        "loop_percussive_event_ratio": 0.0,
        "harmonic_energy_ratio": 0.48,
        "inharmonicity": 0.14,
        "fundamental_dominance_ratio": 0.20,
        "body_flatness": 0.014,
        "tail_flatness": 0.018,
        "spectral_flatness_mean": 0.014,
        "body_noise_ratio": 0.014,
        "tail_noise_ratio": 0.018,
        "body_entropy": 0.42,
        "tail_entropy": 0.42,
        "bass_ratio_150_500hz": 0.62,
        "mid_ratio_500_2000hz": 0.38,
        "presence_ratio_2000_8000hz": 0.0005,
        "air_ratio_gt_8000hz": 0.0,
        "tail_energy_ratio": 0.66,
        "spectral_peak_stability": 0.20,
        "formant_like_peak_spacing": 0.0,
        "log_transient_count": float(np.log1p(23.0)),
        "loop_mean_event_low_ratio": 0.62,
        "loop_mean_event_high_ratio": 0.0005,
    }
    facts = build_shared_audio_facts(physics_from_named(values))
    facts.evidence["shape_vote"] = {
        "primary_shape": "bass_phrase",
        "true_repetition_score": 0.82,
        "low_event_ratio": 0.62,
        "mid_event_ratio": 0.38,
        "high_event_ratio": 0.0005,
    }

    evidence = PhysicsVoter.reed_sax_identity_evidence(facts, FIRST_ARRIVAL_WET_SAX_TELEMETRY)

    assert evidence["reed_sax_identity_eligible"] is False
    assert evidence["reed_sax_clean_highless_pitched_loop"] is True
    assert evidence["reed_sax_first_arrival_eligible"] is False


def test_reed_sax_identity_rejects_clean_mid_vocal_phrase_loop_decoy() -> None:
    values = {
        "pitch_confidence": 0.88,
        "f0_voiced_ratio": 0.94,
        "loop_pitched_event_ratio": 1.0,
        "loop_sustained_tonal_frame_ratio": 1.0,
        "loop_percussive_event_ratio": 0.0,
        "harmonic_energy_ratio": 0.37,
        "inharmonicity": 0.18,
        "fundamental_dominance_ratio": 0.42,
        "body_flatness": 0.087,
        "tail_flatness": 0.095,
        "spectral_flatness_mean": 0.087,
        "body_noise_ratio": 0.12,
        "tail_noise_ratio": 0.13,
        "body_entropy": 0.35,
        "tail_entropy": 0.39,
        "bass_ratio_150_500hz": 0.34,
        "mid_ratio_500_2000hz": 0.65,
        "presence_ratio_2000_8000hz": 0.008,
        "air_ratio_gt_8000hz": 0.0,
        "tail_energy_ratio": 0.65,
        "spectral_peak_stability": 0.25,
        "formant_like_peak_spacing": 0.0,
        "log_transient_count": float(np.log1p(46.0)),
        "loop_mean_event_low_ratio": 0.34,
        "loop_mean_event_high_ratio": 0.008,
    }
    facts = build_shared_audio_facts(physics_from_named(values))
    facts.evidence["shape_vote"] = {
        "primary_shape": "vocal_phrase",
        "true_repetition_score": 0.86,
        "low_event_ratio": 0.34,
        "mid_event_ratio": 0.65,
        "high_event_ratio": 0.008,
    }
    telemetry = dict(FIRST_ARRIVAL_WET_SAX_TELEMETRY)
    telemetry["first_arrival_conical_balance"] = 0.0

    evidence = PhysicsVoter.reed_sax_identity_evidence(facts, telemetry)

    assert evidence["reed_sax_identity_eligible"] is False
    assert evidence["reed_sax_clean_mid_vocal_phrase_loop"] is True
    assert evidence["reed_sax_vocal_penalty_blocks_sax"] is True
