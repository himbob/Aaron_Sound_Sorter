from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from aaron_sound_sorter import api as mod


def fp(**vals):
    x = [0.0] * mod.FP_SIZE
    for name, value in vals.items():
        x[mod.FEATURE_NAMES.index(name)] = value
    return x


def brain():
    labels = [
        "Drums/Drum Loops/Loops",
        "Drums/Kick Drums/Generic Kick/Loops",
        "Drums/Percussion/Shakers and Tambourines/Loops",
        "Drums/Claps Snaps Slaps/Generic Clap/Loops",
        "FX/Textures/Mechanical Texture/Loops",
        "Instruments/Instrument Loops/Loops",
        "Instruments/Bass/Electric Bass/Loops",
    ]
    return {
        "labels": labels,
        "top_by_label": {
            labels[0]: "Drums",
            labels[1]: "Drums",
            labels[2]: "Drums",
            labels[3]: "Drums",
            labels[4]: "FX",
            labels[5]: "Instruments",
            labels[6]: "Instruments",
        },
        "structure_by_label": {label: "loop" for label in labels},
        "category_fact_profiles": {},
        "membership_gate_enabled": False,
        "model_ensemble_enabled": False,
        "committee_agreement_enabled": False,
        "adaptive_depth_placement_enabled": False,
    }


def test_ambiguous_drum_loop_leaf_collapses_to_broad_learned_drum_loop():
    b = brain()
    x = fp(
        log_transient_count=mod.math.log1p(32.0),
        temporal_centroid_ratio=0.46,
        onset_interval_regularity=0.15,
        onset_span_ratio=0.95,
        event_rate_hz=4.7,
        attack_rise_time_norm=0.01,
        tail_energy_ratio=0.22,
        presence_ratio_2000_8000hz=0.72,
        air_ratio_gt_8000hz=0.25,
        spectral_flatness_mean=0.58,
        spectral_entropy_mean=0.84,
        pitch_confidence=0.33,
    )
    final_top, final_label, status, reason = mod.apply_learned_conflict_gates(
        b,
        x,
        "Drums/Percussion/Shakers and Tambourines/Loops",
        "Drums",
        "Drums",
        "Drums/_Ambiguous Leaf/Claps Snaps Slaps vs Percussion/Loops",
        "auto_place",
        "AUDIT strong_pick",
        similarity=0.535,
        margin=2.59,
        duration_sec=6.85,
        top5=[
            ("Drums/Percussion/Shakers and Tambourines/Loops", 0.1),
            ("Drums/Claps Snaps Slaps/Generic Clap/Loops", 0.2),
            ("Drums/Drum Loops/Loops", 0.3),
        ],
    )
    assert status == "auto_place"
    assert final_top == "Drums"
    assert final_label == "Drums/Drum Loops/Loops"
    assert "generic_loop_family_placement" in reason


def test_non_percussive_fx_loop_candidate_cannot_survive_as_drum_loop():
    b = brain()
    x = fp(
        log_transient_count=mod.math.log1p(12.0),
        temporal_centroid_ratio=0.50,
        onset_span_ratio=0.84,
        event_rate_hz=1.5,
        attack_rise_time_norm=0.04,
        tail_energy_ratio=0.55,
        sub_bass_ratio_lt_150hz=0.02,
        bass_ratio_150_500hz=0.04,
        mid_ratio_500_2000hz=0.60,
        presence_ratio_2000_8000hz=0.07,
        air_ratio_gt_8000hz=0.02,
        spectral_flatness_mean=0.14,
        spectral_entropy_mean=0.31,
        pitch_confidence=0.20,
    )
    label, top, action, reason = mod.physical_family_guard_decision(
        b,
        x,
        "Drums/Drum Loops/Loops",
        "Drums",
        "Drums",
        [
            ("Drums/Drum Loops/Loops", 0.1),
            ("FX/Textures/Mechanical Texture/Loops", 0.2),
        ],
        8.0,
    )
    assert action == "switch"
    assert top == "FX"
    assert "unsupported_drum_loop" in reason


def test_mixed_drum_beat_physics_is_allowed_to_remain_drum_loop():
    b = brain()
    x = fp(
        log_transient_count=mod.math.log1p(17.0),
        temporal_centroid_ratio=0.47,
        onset_span_ratio=0.90,
        event_rate_hz=1.75,
        attack_rise_time_norm=0.00,
        tail_energy_ratio=0.28,
        sub_bass_ratio_lt_150hz=0.55,
        bass_ratio_150_500hz=0.38,
        mid_ratio_500_2000hz=0.02,
        presence_ratio_2000_8000hz=0.03,
        air_ratio_gt_8000hz=0.01,
        spectral_flatness_mean=0.46,
        spectral_entropy_mean=0.56,
        pitch_confidence=0.95,
    )
    label, top, action, reason = mod.physical_family_guard_decision(
        b,
        x,
        "Drums/Drum Loops/Loops",
        "Drums",
        "Drums",
        [("Drums/Drum Loops/Loops", 0.1)],
        9.8,
    )
    assert action == "pass"


def test_sub_heavy_kick_loop_raw_drum_is_not_laundered_to_bass_loop():
    b = brain()
    x = fp(
        log_transient_count=mod.math.log1p(12.0),
        temporal_centroid_ratio=0.49,
        onset_span_ratio=0.90,
        event_rate_hz=2.05,
        attack_rise_time_norm=0.00,
        tail_energy_ratio=0.72,
        sub_bass_ratio_lt_150hz=0.88,
        bass_ratio_150_500hz=0.10,
        mid_ratio_500_2000hz=0.01,
        presence_ratio_2000_8000hz=0.00,
        air_ratio_gt_8000hz=0.00,
        spectral_flatness_mean=0.02,
        spectral_entropy_mean=0.14,
        pitch_confidence=0.92,
    )
    label, top, action, reason = mod.physical_family_guard_decision(
        b,
        x,
        "Instruments/Bass/Electric Bass/Loops",
        "Instruments",
        "Drums",
        [
            ("Instruments/Bass/Electric Bass/Loops", 0.1),
            ("Instruments/Instrument Loops/Loops", 0.2),
        ],
        5.9,
    )
    assert action == "switch"
    assert top == "Drums"
    assert label == "Drums/Drum Loops/Loops"
    assert "supported_low_pulse_loop_back_to_drums" in reason


def test_sub_heavy_instrument_loop_without_raw_drum_family_stays_instruments():
    b = brain()
    x = fp(
        log_transient_count=mod.math.log1p(12.0),
        temporal_centroid_ratio=0.49,
        onset_span_ratio=0.90,
        event_rate_hz=2.05,
        attack_rise_time_norm=0.00,
        tail_energy_ratio=0.72,
        sub_bass_ratio_lt_150hz=0.88,
        bass_ratio_150_500hz=0.10,
        mid_ratio_500_2000hz=0.01,
        presence_ratio_2000_8000hz=0.00,
        air_ratio_gt_8000hz=0.00,
        spectral_flatness_mean=0.02,
        spectral_entropy_mean=0.14,
        pitch_confidence=0.92,
    )
    label, top, action, reason = mod.physical_family_guard_decision(
        b,
        x,
        "Instruments/Bass/Electric Bass/Loops",
        "Instruments",
        "Instruments",
        [
            ("Instruments/Bass/Electric Bass/Loops", 0.1),
            ("Drums/Kick Drums/Generic Kick/Loops", 0.2),
        ],
        5.9,
    )
    assert action == "pass"
    assert top == ""
    assert label == ""
    assert "no_action" in reason


def test_voiced_rap_loop_still_reviews_instead_of_drum_loop():
    b = brain()
    x = fp(
        log_transient_count=mod.math.log1p(33.0),
        temporal_centroid_ratio=0.52,
        onset_span_ratio=0.98,
        event_rate_hz=4.86,
        attack_rise_time_norm=0.03,
        tail_energy_ratio=0.20,
        sub_bass_ratio_lt_150hz=0.08,
        bass_ratio_150_500hz=0.21,
        mid_ratio_500_2000hz=0.55,
        presence_ratio_2000_8000hz=0.12,
        air_ratio_gt_8000hz=0.04,
        spectral_flatness_mean=0.42,
        spectral_entropy_mean=0.56,
        pitch_confidence=0.65,
        f0_voiced_ratio=0.66,
        harmonic_energy_ratio=0.17,
        attack_pitch_confidence=0.60,
        body_pitch_confidence=0.56,
    )
    label, top, action, reason = mod.physical_family_guard_decision(
        b,
        x,
        "Drums/Drum Loops/Loops",
        "Drums",
        "Drums",
        [("Drums/Drum Loops/Loops", 0.1)],
        6.85,
    )
    assert action == "review"
    assert top == "_TO_REVIEW"
    assert "voiced_formant_non_drum" in reason
