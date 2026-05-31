from __future__ import annotations

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.voters.physics_layers import LayeredPhysicsScorer


def facts(
    shape: dict,
    roles: dict | None = None,
    values: dict | None = None,
    *,
    fa: dict | None = None,
    is_loop_like: bool = True,
    is_single_event_like: bool = False,
    is_short_hit_like: bool = False,
    is_long: bool = True,
) -> SharedAudioFacts:
    evidence = {"shape_vote": shape, "measured_roles": roles or {}}
    if fa is not None:
        evidence["first_arrival_telemetry"] = fa
    return SharedAudioFacts(
        is_broken_or_tiny=False,
        is_loop_like=is_loop_like,
        is_single_event_like=is_single_event_like,
        is_short_hit_like=is_short_hit_like,
        is_long=is_long,
        evidence=evidence,
        feature_values_by_name=values or {},
    )


BASE_PHRASE = {
    "primary_shape": "pitched_phrase",
    "confidence": 0.95,
    "onset_count": 8.0,
    "onset_span_ratio": 0.68,
    "pitch_confidence": 0.90,
    "pitched_event_ratio": 1.0,
    "percussive_event_ratio": 0.0,
    "drumlike_frame_ratio": 0.0,
    "sustained_tonal_frame_ratio": 0.90,
    "non_event_tonal_ratio": 0.90,
}


def test_organ_panel_can_beat_synth_and_strings_without_name_help() -> None:
    decision = LayeredPhysicsScorer().analyze(
        facts(
            {**BASE_PHRASE, "primary_shape": "sustained_pad", "onset_count": 3.0},
            roles={"pitched_music_loop": 0.90},
            values={
                "pitch_confidence": 0.90,
                "f0_voiced_ratio": 0.95,
                "loop_pitched_event_ratio": 1.0,
                "loop_sustained_tonal_frame_ratio": 0.95,
                "loop_non_event_tonal_ratio": 0.96,
                "harmonic_energy_ratio": 0.80,
                "fundamental_dominance_ratio": 0.72,
                "spectral_flatness_mean": 0.04,
                "spectral_entropy_mean": 0.30,
                "attack_rise_time_norm": 0.22,
                "mid_ratio_500_2000hz": 0.62,
                "presence_ratio_2000_8000hz": 0.04,
                "air_ratio_gt_8000hz": 0.005,
                "loop_percussive_event_ratio": 0.0,
                "loop_drumlike_frame_ratio": 0.0,
                "stereo_width": 0.55,
            },
            fa={"status": "ok"},
        )
    )

    assert decision.top_family == "Instruments"
    assert decision.branch == "KeysPiano"
    assert decision.evidence["instrument_KeysPiano_subpanel_selected"] == "Organ"
    assert decision.evidence["instrument_branch_KeysPiano"] > decision.evidence["instrument_branch_Synth"]


def test_synth_pad_panel_blocks_string_drone_steal_for_clean_synth_body() -> None:
    decision = LayeredPhysicsScorer().analyze(
        facts(
            {**BASE_PHRASE, "primary_shape": "sustained_pad", "onset_count": 1.0},
            roles={"pitched_music_loop": 0.90},
            values={
                "pitch_confidence": 0.90,
                "f0_voiced_ratio": 0.95,
                "loop_pitched_event_ratio": 0.95,
                "loop_sustained_tonal_frame_ratio": 0.98,
                "loop_non_event_tonal_ratio": 0.98,
                "harmonic_energy_ratio": 0.75,
                "fundamental_dominance_ratio": 0.80,
                "spectral_flatness_mean": 0.02,
                "spectral_entropy_mean": 0.28,
                "attack_rise_time_norm": 0.25,
                "mid_ratio_500_2000hz": 0.55,
                "presence_ratio_2000_8000hz": 0.03,
                "air_ratio_gt_8000hz": 0.01,
                "loop_percussive_event_ratio": 0.0,
                "loop_drumlike_frame_ratio": 0.0,
                "stereo_width": 0.62,
            },
            fa={"status": "ok"},
        )
    )

    assert decision.branch == "Synth"
    assert decision.evidence["instrument_Synth_subpanel_selected"] in {"SynthPad", "SynthDrone"}
    assert decision.evidence["instrument_branch_Synth"] > decision.evidence["instrument_branch_Strings"]


def test_bass_subpanel_reports_specific_low_end_identity() -> None:
    decision = LayeredPhysicsScorer().analyze(
        facts(
            {
                **BASE_PHRASE,
                "primary_shape": "bass_phrase",
                "low_event_ratio": 0.88,
                "mid_event_ratio": 0.10,
                "high_event_ratio": 0.02,
            },
            roles={"bass_loop": 0.75, "pitched_music_loop": 0.90},
            values={
                "pitch_confidence": 0.90,
                "f0_voiced_ratio": 0.80,
                "f0_median_hz": 55.0,
                "low_peak_frequency_hz": 55.0,
                "loop_pitched_event_ratio": 0.95,
                "loop_sustained_tonal_frame_ratio": 0.90,
                "harmonic_energy_ratio": 0.70,
                "fundamental_dominance_ratio": 0.80,
                "sub_bass_ratio_lt_150hz": 0.72,
                "bass_ratio_150_500hz": 0.20,
                "mid_ratio_500_2000hz": 0.06,
                "presence_ratio_2000_8000hz": 0.01,
                "air_ratio_gt_8000hz": 0.0,
                "spectral_flatness_mean": 0.03,
                "spectral_entropy_mean": 0.25,
                "loop_mean_event_low_ratio": 0.88,
                "loop_percussive_event_ratio": 0.0,
                "loop_drumlike_frame_ratio": 0.0,
                "tail_energy_ratio": 0.55,
            },
            fa={"status": "ok"},
        )
    )

    assert decision.branch == "Bass"
    assert decision.evidence["instrument_Bass_subpanel_selected"] in {"808Sub", "SynthBass"}
    assert decision.evidence["instrument_branch_Bass"] >= 0.90


def test_brass_panel_blocks_string_or_guitar_steal_for_bright_harmonic_horn() -> None:
    decision = LayeredPhysicsScorer().analyze(
        facts(
            {**BASE_PHRASE, "primary_shape": "solo_phrase", "onset_count": 4.0},
            roles={"pitched_music_loop": 0.85},
            values={
                "pitch_confidence": 0.88,
                "f0_voiced_ratio": 0.90,
                "f0_median_hz": 450.0,
                "loop_pitched_event_ratio": 1.0,
                "loop_sustained_tonal_frame_ratio": 0.80,
                "harmonic_energy_ratio": 0.85,
                "fundamental_dominance_ratio": 0.50,
                "spectral_flatness_mean": 0.08,
                "spectral_entropy_mean": 0.32,
                "attack_rise_time_norm": 0.04,
                "mid_ratio_500_2000hz": 0.42,
                "presence_ratio_2000_8000hz": 0.30,
                "air_ratio_gt_8000hz": 0.04,
                "loop_percussive_event_ratio": 0.0,
                "loop_drumlike_frame_ratio": 0.0,
            },
            fa={"status": "ok", "first_arrival_conical_balance": 0.70, "first_arrival_presence_contrast_db": 30.0},
        )
    )

    assert decision.branch == "Brass"
    assert decision.evidence["instrument_Brass_subpanel_selected"] in {"TrumpetHorn", "BrassStab", "BrassSustainLoop"}
    assert decision.evidence["instrument_branch_Brass"] > decision.evidence["instrument_branch_Strings"]
    assert decision.evidence["instrument_branch_Brass"] > decision.evidence["instrument_branch_PluckedString"]


def test_flute_like_airy_woodwind_can_block_strings() -> None:
    decision = LayeredPhysicsScorer().analyze(
        facts(
            {**BASE_PHRASE, "onset_count": 5.0},
            roles={"pitched_music_loop": 0.85},
            values={
                "pitch_confidence": 0.86,
                "f0_voiced_ratio": 0.88,
                "f0_median_hz": 850.0,
                "loop_pitched_event_ratio": 1.0,
                "loop_sustained_tonal_frame_ratio": 0.85,
                "harmonic_energy_ratio": 0.55,
                "spectral_flatness_mean": 0.18,
                "spectral_entropy_mean": 0.42,
                "mid_ratio_500_2000hz": 0.24,
                "presence_ratio_2000_8000hz": 0.18,
                "air_ratio_gt_8000hz": 0.17,
                "sub_bass_ratio_lt_150hz": 0.01,
                "bass_ratio_150_500hz": 0.05,
                "loop_percussive_event_ratio": 0.0,
                "loop_drumlike_frame_ratio": 0.0,
                "body_noise_ratio": 0.28,
            },
            fa={"status": "ok", "stochastic_modulation_coherence": 0.35, "first_arrival_presence_contrast_db": 25.0},
        )
    )

    assert decision.branch == "Woodwinds"
    assert decision.evidence["instrument_Woodwinds_subpanel_selected"] in {"Flute", "AiryWoodwind"}
    assert decision.evidence["instrument_branch_Woodwinds"] > decision.evidence["instrument_branch_Strings"]


def test_bowed_string_panel_is_reported_without_promoting_voice() -> None:
    decision = LayeredPhysicsScorer().analyze(
        facts(
            {**BASE_PHRASE, "primary_shape": "sustained_pad", "onset_count": 2.0},
            roles={"pitched_music_loop": 0.90},
            values={
                "pitch_confidence": 0.90,
                "f0_voiced_ratio": 0.96,
                "loop_pitched_event_ratio": 0.95,
                "loop_sustained_tonal_frame_ratio": 0.98,
                "loop_non_event_tonal_ratio": 0.96,
                "harmonic_energy_ratio": 0.70,
                "spectral_flatness_mean": 0.12,
                "spectral_entropy_mean": 0.38,
                "attack_rise_time_norm": 0.18,
                "tail_energy_ratio": 0.80,
                "mid_ratio_500_2000hz": 0.42,
                "presence_ratio_2000_8000hz": 0.14,
                "air_ratio_gt_8000hz": 0.03,
                "body_noise_ratio": 0.22,
                "loop_percussive_event_ratio": 0.0,
                "loop_drumlike_frame_ratio": 0.0,
            },
            fa={"status": "ok", "cepstral_pitch_period_coherence": 0.60, "stochastic_modulation_coherence": 0.24},
        )
    )

    assert decision.top_family == "Instruments"
    assert decision.evidence["instrument_Strings_subpanel_selected"] in {"BowedSustain", "StringDrone"}
    assert decision.evidence["instrument_branch_Voice"] < 0.60


def test_mallet_bell_panels_report_metallic_pitched_hits() -> None:
    decision = LayeredPhysicsScorer().analyze(
        facts(
            {**BASE_PHRASE, "primary_shape": "solo_phrase", "onset_count": 2.0},
            roles={"pitched_music_loop": 0.50},
            values={
                "pitch_confidence": 0.75,
                "f0_voiced_ratio": 0.60,
                "loop_pitched_event_ratio": 0.70,
                "harmonic_energy_ratio": 0.45,
                "inharmonicity": 0.60,
                "spectral_flatness_mean": 0.06,
                "spectral_entropy_mean": 0.36,
                "attack_rise_time_norm": 0.02,
                "tail_energy_ratio": 0.70,
                "mid_ratio_500_2000hz": 0.24,
                "presence_ratio_2000_8000hz": 0.34,
                "air_ratio_gt_8000hz": 0.12,
                "loop_mean_event_high_ratio": 0.40,
                "loop_percussive_event_ratio": 0.0,
                "loop_drumlike_frame_ratio": 0.0,
            },
            fa={
                "status": "ok",
                "strike_inharmonic_energy_ratio": 0.70,
                "settle_inharmonic_energy_ratio": 0.65,
                "first_arrival_conical_balance": 0.50,
            },
        )
    )

    assert decision.branch == "MalletBell"
    assert decision.evidence["instrument_MalletBell_subpanel_selected"] in {
        "BellChime",
        "MalletKeys",
        "SteelPanHandpan",
        "KalimbaMbira",
        "SingingBowlGong",
    }
    assert decision.evidence["instrument_branch_MalletBell"] > decision.evidence["instrument_branch_KeysPiano"]


def test_plucked_string_requires_real_branch_authority_before_stealing() -> None:
    decision = LayeredPhysicsScorer().analyze(
        facts(
            {**BASE_PHRASE, "primary_shape": "solo_phrase", "onset_count": 5.0},
            roles={"pitched_music_loop": 0.82},
            values={
                "pitch_confidence": 0.86,
                "f0_voiced_ratio": 0.80,
                "loop_pitched_event_ratio": 0.90,
                "harmonic_energy_ratio": 0.70,
                "spectral_flatness_mean": 0.08,
                "spectral_entropy_mean": 0.35,
                "attack_rise_time_norm": 0.025,
                "tail_energy_ratio": 0.28,
                "mid_ratio_500_2000hz": 0.42,
                "presence_ratio_2000_8000hz": 0.23,
                "air_ratio_gt_8000hz": 0.05,
                "loop_percussive_event_ratio": 0.02,
                "loop_drumlike_frame_ratio": 0.02,
                "strike_inharmonic_energy_ratio": 0.45,
            },
            fa={"status": "ok", "strike_inharmonic_energy_ratio": 0.45},
        )
    )

    assert decision.evidence["instrument_PluckedString_subpanel_selected"] in {
        "AcousticGuitar",
        "ElectricGuitar",
        "WorldPluck",
        "NylonOrSoftPluck",
    }
    assert "instrument_plucked_branch_steal_guard_open" in decision.evidence
    assert (
        decision.evidence["instrument_branch_PluckedString"]
        <= decision.evidence["instrument_branch_MixedInstrument"] + 0.10
    )
