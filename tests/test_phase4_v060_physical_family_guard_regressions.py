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
        "Drums/Cymbals/Generic Cymbal/One Shots",
        "Drums/Kick Drums/Generic Kick/One Shots",
        "Drums/Drum Loops/Loops",
        "Instruments/Synths/Synth Chord/One Shots",
        "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/One Shots",
    ]
    return {
        "labels": labels,
        "top_by_label": {
            labels[0]: "Drums",
            labels[1]: "Drums",
            labels[2]: "Drums",
            labels[3]: "Instruments",
            labels[4]: "FX",
        },
        "category_fact_profiles": {},
        "membership_gate_enabled": True,
    }


def test_tonal_musical_one_shot_cannot_auto_place_as_cymbal():
    b = brain()
    x = fp(
        spectral_flatness_mean=0.05,
        spectral_entropy_mean=0.50,
        pitch_confidence=0.61,
        sub_bass_ratio_lt_150hz=0.0001,
        presence_ratio_2000_8000hz=0.03,
        air_ratio_gt_8000hz=0.01,
        temporal_centroid_ratio=0.15,
        attack_rise_time_norm=0.023,
        tail_energy_ratio=0.05,
        event_rate_hz=1.3,
        log_transient_count=mod.math.log1p(1.0),
    )
    label, top, action, reason = mod.physical_family_guard_decision(
        b,
        x,
        "Drums/Cymbals/Generic Cymbal/One Shots",
        "Drums",
        "Drums",
        [
            ("Instruments/Synths/Synth Chord/One Shots", 0.1),
            ("Drums/Cymbals/Generic Cymbal/One Shots", 0.2),
        ],
        0.80,
    )
    assert action == "switch"
    assert top == "Instruments"
    assert "tonal_non_drum" in reason


def test_tonal_musical_arp_without_instrument_candidate_goes_review_not_cymbal():
    b = brain()
    x = fp(
        spectral_flatness_mean=0.10,
        spectral_entropy_mean=0.51,
        pitch_confidence=0.49,
        sub_bass_ratio_lt_150hz=0.001,
        presence_ratio_2000_8000hz=0.50,
        air_ratio_gt_8000hz=0.08,
        attack_rise_time_norm=0.19,
        tail_energy_ratio=0.12,
        event_rate_hz=1.3,
        onset_span_ratio=0.19,
        log_transient_count=mod.math.log1p(4.0),
    )
    label, top, action, reason = mod.physical_family_guard_decision(
        b,
        x,
        "Drums/Cymbals/Generic Cymbal/One Shots",
        "Drums",
        "Drums",
        [("Drums/Cymbals/Generic Cymbal/One Shots", 0.1)],
        3.10,
    )
    assert action == "review"
    assert "tonal_non_drum" in reason


def test_drum_roll_or_fill_cannot_auto_place_as_riser():
    b = brain()
    x = fp(
        spectral_flatness_mean=0.44,
        spectral_entropy_mean=0.63,
        pitch_confidence=0.57,
        centroid_slope_norm=-0.07,
        temporal_centroid_ratio=0.09,
        attack_rise_time_norm=0.07,
        tail_energy_ratio=0.003,
        onset_span_ratio=0.13,
        event_rate_hz=1.06,
        log_transient_count=mod.math.log1p(10.0),
    )
    label, top, action, reason = mod.physical_family_guard_decision(
        b,
        x,
        "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/One Shots",
        "FX",
        "FX",
        [
            ("FX/Structural and Transitional FX/Risers and Builds/Generic Riser/One Shots", 0.1),
            ("Drums/Drum Loops/Loops", 0.2),
        ],
        9.46,
    )
    assert action == "switch"
    assert top == "Drums"
    assert "rhythmic_fx_to_drums" in reason


def test_short_low_kick_cannot_auto_place_as_downlifter():
    b = brain()
    x = fp(
        spectral_flatness_mean=0.08,
        spectral_entropy_mean=0.12,
        pitch_confidence=0.95,
        sub_bass_ratio_lt_150hz=0.99,
        temporal_centroid_ratio=0.07,
        attack_rise_time_norm=0.003,
        tail_energy_ratio=0.0002,
        event_rate_hz=1.3,
        log_transient_count=mod.math.log1p(1.0),
    )
    label, top, action, reason = mod.physical_family_guard_decision(
        b,
        x,
        "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/One Shots",
        "FX",
        "FX",
        [
            ("FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/One Shots", 0.1),
            ("Drums/Kick Drums/Generic Kick/One Shots", 0.2),
        ],
        0.84,
    )
    assert action == "switch"
    assert top == "Drums"
    assert ("short_hit_to_drums" in reason) or ("low_body_pitched_ring_to_drums" in reason)
