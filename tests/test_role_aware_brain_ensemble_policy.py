"""Regression tests for role-aware brain ensemble weighting."""

from __future__ import annotations

from aaron_sound_sorter.domain.models import CategoryGuess, SharedAudioFacts, VoterResult
from aaron_sound_sorter.voters.brain_recall import brain_lane_vote_weight, combine_full_and_balanced_brain_votes


def guess(label: str, rank: int, *, role: str = "") -> CategoryGuess:
    evidence = {
        "measured_role_strengths": {
            "pitched_music_loop": 0.0,
            "pitched_music_phrase": 0.0,
            "low_rhythmic_drum_loop": 0.0,
            "percussive_drum_loop": 0.0,
            "bright_drum_loop": 0.0,
            "bass_loop": 0.0,
            "vocal_music_phrase": 0.0,
            "voiced_one_shot": 0.0,
        },
        "family_compatibility": {
            "detected_parent_role": role,
            "candidate_top_family": label.split("/", 1)[0],
            "is_family_compatible": True,
        },
        "detected_parent_role": role,
        "measured_parent_role_strength": 1.0,
        "candidate_parent_role_strength": 1.0,
    }
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


def voter(name: str, labels: list[str], *, role: str = "") -> VoterResult:
    return VoterResult(
        voter_name=name,
        guesses=[guess(label, index, role=role) for index, label in enumerate(labels, start=1)],
        diagnostics={"enabled": True, "lane_name": name.replace("brain_", "")},
    )


def facts_with_roles(**roles: float) -> SharedAudioFacts:
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=True,
        is_single_event_like=False,
        is_short_hit_like=False,
        is_long=True,
        evidence={"measured_roles": roles},
    )


def test_drum_loop_profile_keeps_full_brain_above_baby_synths() -> None:
    full = voter("brain_full", ["Drums/Drum Loops/Loops"], role="low_rhythmic_drum_loop")
    core = voter("brain_core_baby", ["Instruments/Synths/Synth Pad/One Shots"], role="low_rhythmic_drum_loop")
    spread = voter("brain_spread_baby", ["Instruments/Synths/Synth Pad/One Shots"], role="low_rhythmic_drum_loop")

    result = combine_full_and_balanced_brain_votes(
        full_result=full,
        baby_results={"core_baby": core, "spread_baby": spread},
        facts=facts_with_roles(low_rhythmic_drum_loop=1.0),
    )

    assert result.guesses[0].label == "Drums/Drum Loops/Loops"
    assert result.guesses[0].evidence["brain_ensemble_weight_profile"] == "drum_loop_full_priority"


def test_bass_profile_lets_core_and_spread_baby_beat_full_generic_loop() -> None:
    full = voter("brain_full", ["Instruments/Instrument Loops/Loops"], role="bass_loop")
    core = voter("brain_core_baby", ["Instruments/Bass/Synth Bass/One Shots"], role="bass_loop")
    spread = voter("brain_spread_baby", ["Instruments/Bass/Electric Bass/One Shots"], role="bass_loop")

    result = combine_full_and_balanced_brain_votes(
        full_result=full,
        baby_results={"core_baby": core, "spread_baby": spread},
        facts=facts_with_roles(bass_loop=1.0),
    )

    assert "Bass" in result.guesses[0].label
    assert result.guesses[0].evidence["brain_ensemble_weight_profile"] == "bass_loop_baby_priority"


def test_generic_pitched_profile_keeps_outlier_reed_recall_visible_but_not_general_owner() -> None:
    full = voter("brain_full", ["Instruments/Instrument Loops/Loops"], role="pitched_music_loop")
    core = voter("brain_core_baby", ["Instruments/Guitar/Nylon Guitar/One Shots"], role="pitched_music_loop")
    spread = voter("brain_spread_baby", ["Instruments/Synths/Synth Lead/One Shots"], role="pitched_music_loop")
    outlier = voter("brain_outlier_baby", ["Instruments/Woodwinds/Saxophone/One Shots"], role="pitched_music_loop")

    result = combine_full_and_balanced_brain_votes(
        full_result=full,
        baby_results={"core_baby": core, "spread_baby": spread, "outlier_baby": outlier},
        facts=facts_with_roles(pitched_music_loop=1.0),
    )

    labels = [guess.label for guess in result.guesses[:4]]
    assert "Instruments/Woodwinds/Saxophone/One Shots" in labels
    reed_guess = next(guess for guess in result.guesses if "Saxophone" in guess.label)
    assert reed_guess.evidence["brain_ensemble_weight_profile"] == "generic_pitched_full_guarded"
    assert reed_guess.evidence["brain_ensemble_lane_weight_report"]["outlier_baby"] == 0.58


def test_drum_loop_profile_keeps_baby_lanes_demoted_by_category_competence() -> None:
    full = voter("brain_full", ["Drums/Drum Loops/Loops"], role="low_rhythmic_drum_loop")
    core = voter("brain_core_baby", ["Instruments/Synths/Synth Pad/One Shots"], role="low_rhythmic_drum_loop")
    spread = voter("brain_spread_baby", ["Instruments/Synths/Synth Pad/One Shots"], role="low_rhythmic_drum_loop")

    result = combine_full_and_balanced_brain_votes(
        full_result=full,
        baby_results={"core_baby": core, "spread_baby": spread},
        facts=facts_with_roles(low_rhythmic_drum_loop=1.0),
    )

    assert result.guesses[0].label == "Drums/Drum Loops/Loops"
    full_weight = brain_lane_vote_weight("full", weight_profile="drum_loop_full_priority")
    core_weight = brain_lane_vote_weight("core_baby", weight_profile="drum_loop_full_priority")
    spread_weight = brain_lane_vote_weight("spread_baby", weight_profile="drum_loop_full_priority")

    assert full_weight > core_weight
    assert full_weight > spread_weight


def test_generic_pitched_profile_does_not_boost_fx_synth_riser_as_instrument_synth() -> None:
    """Instrument-context synth boosts must not apply to FX/Synth Riser leaves."""
    from aaron_sound_sorter.voters.brain_ensemble_policy import brain_role_fit_multiplier

    fx_guess = CategoryGuess(
        label="FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX",
        folder_path="FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX",
        top_family="FX",
        score=8.0,
        confidence=0.2,
        rank=8,
        reason="synthetic",
        evidence={
            "family_compatibility": {"is_family_compatible": False},
            "measured_parent_role_strength": 0.72,
            "candidate_parent_role_strength": 0.0,
        },
    )
    synth_guess = CategoryGuess(
        label="Instruments/Synths/Synth Lead/Loops",
        folder_path="Instruments/Synths/Synth Lead/Loops",
        top_family="Instruments",
        score=8.0,
        confidence=0.2,
        rank=8,
        reason="synthetic",
        evidence={
            "family_compatibility": {"is_family_compatible": True},
            "measured_parent_role_strength": 0.72,
            "candidate_parent_role_strength": 0.70,
        },
    )

    assert (
        brain_role_fit_multiplier(
            fx_guess,
            weight_profile="generic_pitched_full_guarded",
        )
        == 0.32
    )
    assert (
        brain_role_fit_multiplier(
            synth_guess,
            weight_profile="generic_pitched_full_guarded",
        )
        > 1.0
    )
