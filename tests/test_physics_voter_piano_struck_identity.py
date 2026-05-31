"""PhysicsVoter struck-resonator/piano identity witness regressions."""

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


def physics_from_named(values: dict[str, float], duration: float = 10.0) -> AudioPhysics:
    """Create AudioPhysics without filename evidence."""
    return AudioPhysics(
        source_path=__import__("pathlib").Path("anonymous.wav"),
        fingerprint=fingerprint_from_named(values),
        duration_sec=duration,
        read_status="ok",
    )


def fact_profile(values: dict[str, float]) -> dict:
    """Tiny fact profile sufficient for PhysicsVoter ranking tests."""
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


PIANO_LOOP_VALUES = {
    "pitch_confidence": 0.97,
    "f0_voiced_ratio": 1.0,
    "loop_pitched_event_ratio": 1.0,
    "loop_sustained_tonal_frame_ratio": 1.0,
    "loop_percussive_event_ratio": 0.0,
    "loop_drumlike_frame_ratio": 0.0,
    "loop_mean_event_high_ratio": 0.003,
    "loop_mean_event_low_ratio": 0.066,
    "mid_ratio_500_2000hz": 0.93,
    "bass_ratio_150_500hz": 0.066,
    "spectral_flatness_mean": 0.035,
    "spectral_entropy_mean": 0.323,
    "harmonic_energy_ratio": 0.50,
    "fundamental_dominance_ratio": 0.02,
    "log_transient_count": float(np.log1p(36.0)),
}

FIRST_ARRIVAL_PIANO_TELEMETRY = {
    "status": "ok",
    "strike_inharmonic_energy_ratio": 0.65,
    "settle_inharmonic_energy_ratio": 0.18,
    "strike_to_settle_inharmonic_drop": 0.47,
    "strike_flatness": 0.36,
    "settle_flatness": 0.057,
    "strike_to_settle_flatness_drop": 0.303,
    "strike_high_ratio": 0.20,
    "settle_high_ratio": 0.0011,
    "spectral_darkening_db": 22.5,
    "cepstral_pitch_period_coherence": 0.52,
    "selected_event_count": 8,
}

SYNTH_PLUCK_VALUES = {
    **PIANO_LOOP_VALUES,
    "spectral_flatness_mean": 0.006,
    "harmonic_energy_ratio": 0.18,
    "fundamental_dominance_ratio": 0.80,
    "loop_mean_event_high_ratio": 0.002,
}

WEAK_SYNTH_TELEMETRY = {
    "status": "ok",
    "strike_inharmonic_energy_ratio": 0.04,
    "settle_inharmonic_energy_ratio": 0.03,
    "strike_to_settle_inharmonic_drop": 0.01,
    "strike_flatness": 0.03,
    "settle_flatness": 0.025,
    "strike_to_settle_flatness_drop": 0.005,
    "strike_high_ratio": 0.01,
    "settle_high_ratio": 0.009,
    "spectral_darkening_db": 0.3,
    "cepstral_pitch_period_coherence": 0.08,
    "selected_event_count": 6,
}


class TelemetryPhysicsVoter(PhysicsVoter):
    """PhysicsVoter test double with deterministic first-arrival telemetry."""

    def __init__(self, telemetry: dict[str, float]) -> None:
        super().__init__(PhysicsVoterPolicy(top_n=3))
        self._telemetry = telemetry

    def extract_first_arrival_telemetry(self, physics: AudioPhysics, facts):  # type: ignore[override]
        return self._telemetry


def test_piano_struck_witness_requires_first_arrival_strike_physics() -> None:
    facts = build_shared_audio_facts(physics_from_named(PIANO_LOOP_VALUES))

    with_gate = PhysicsVoter.piano_struck_identity_evidence(facts, FIRST_ARRIVAL_PIANO_TELEMETRY)
    without_gate = PhysicsVoter.piano_struck_identity_evidence(facts, {"status": "ok"})

    assert with_gate["piano_struck_identity_eligible"] is True
    assert with_gate["piano_struck_identity_score"] >= 0.85
    assert with_gate["piano_hammer_shock_score"] >= 0.70
    assert with_gate["piano_damping_score"] >= 0.70
    assert without_gate["piano_struck_identity_eligible"] is False


def test_piano_struck_witness_rejects_clean_synth_like_pluck_without_shockwave() -> None:
    facts = build_shared_audio_facts(physics_from_named(SYNTH_PLUCK_VALUES))

    evidence = PhysicsVoter.piano_struck_identity_evidence(facts, WEAK_SYNTH_TELEMETRY)

    assert evidence["piano_struck_identity_eligible"] is False
    assert evidence["piano_synth_penalty"] > 0.0


def test_physics_voter_can_rank_keys_candidate_first_when_piano_strike_evidence_is_strong() -> None:
    keys = "Instruments/Keys/Rhodes/One Shots"
    synth = "Instruments/Synths/Synth Lead/One Shots"
    generic = "Instruments/Instrument Loops/Loops"
    brain = {
        "labels": [generic, synth, keys],
        "top_by_label": {generic: "Instruments", synth: "Instruments", keys: "Instruments"},
        "structure_by_label": {generic: "loop", synth: "one_shot", keys: "one_shot"},
        "category_fact_profiles": {
            generic: fact_profile({"pitch_confidence": 0.92, "f0_voiced_ratio": 0.95}),
            synth: fact_profile({"pitch_confidence": 0.95, "spectral_flatness_mean": 0.01}),
            keys: fact_profile({"pitch_confidence": 0.92, "f0_voiced_ratio": 0.95}),
        },
    }
    physics = physics_from_named(PIANO_LOOP_VALUES)
    facts = build_shared_audio_facts(physics)

    result = TelemetryPhysicsVoter(FIRST_ARRIVAL_PIANO_TELEMETRY).vote(physics, facts, brain)

    assert result.guesses[0].label == keys
    assert result.guesses[0].evidence["piano_struck_physics_check"] == "first_arrival_struck_resonator_identity_lift"
    assert (
        result.guesses[0].evidence["piano_struck_score_after_adjustment"]
        < result.guesses[0].evidence["piano_struck_score_before_adjustment"]
    )
