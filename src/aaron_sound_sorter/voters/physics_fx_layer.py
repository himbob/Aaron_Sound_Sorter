# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Measured designed-FX physics layer."""

from __future__ import annotations

from typing import Any

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.voters.physics_layer_utils import *


class PhysicsFXRoleLayer:
    """Measured role layer for designed FX behavior.

    FX is not treated as "anything weird."  This layer looks for measured
    action roles: transitions, motion, impacts, chopped digital events,
    synthetic blips, formant effects, radio/electrical activity, and designed
    noise hybrids.  It also reports conflicts when the same sample is better
    explained by Drums, Instruments, Textures, or a compound musical loop.
    """

    BRANCHES = (
        "RiserBuild",
        "DropDownlifter",
        "WhooshSweep",
        "ReverseSwell",
        "ImpactHit",
        "GlitchStutter",
        "BlipBeep",
        "SirenAlarm",
        "FormantFX",
        "RadioElectrical",
        "TextureAmbience",
        "MachineMechanical",
        "FoleyMaterial",
        "SmallObjectCluster",
        "HumanCreatureFX",
        "DesignedNoiseHybrid",
    )

    def decide(
        self,
        facts: SharedAudioFacts,
        *,
        drum_anchor: float = 0.0,
        drum_branch_confidence: float = 0.0,
        instrument_anchor: float = 0.0,
        instrument_branch_confidence: float = 0.0,
        compound_strength: float = 0.0,
        compound_prefer_broad_loop: bool = False,
    ) -> tuple[str, float, dict[str, Any]]:
        evidence = self.evidence(
            facts,
            drum_anchor=drum_anchor,
            drum_branch_confidence=drum_branch_confidence,
            instrument_anchor=instrument_anchor,
            instrument_branch_confidence=instrument_branch_confidence,
            compound_strength=compound_strength,
            compound_prefer_broad_loop=compound_prefer_broad_loop,
        )
        branch_scores = {name: float(evidence[f"fx_branch_{name}"]) for name in self.BRANCHES}
        branch, branch_strength = max(branch_scores.items(), key=lambda item: item[1])
        evidence["fx_branch_selected"] = branch
        evidence["fx_branch_selected_confidence"] = round(float(branch_strength), 6)
        for name, value in branch_scores.items():
            evidence[f"fx_branch_score_{name}"] = round(float(value), 6)
        return branch, safe_float(evidence.get("fx_role_strength", branch_strength), 0.0), evidence

    def evidence(
        self,
        facts: SharedAudioFacts,
        *,
        drum_anchor: float = 0.0,
        drum_branch_confidence: float = 0.0,
        instrument_anchor: float = 0.0,
        instrument_branch_confidence: float = 0.0,
        compound_strength: float = 0.0,
        compound_prefer_broad_loop: bool = False,
    ) -> dict[str, Any]:
        values = facts.feature_values_by_name or {}
        shape = fx_shape_vote(facts)
        measured_roles(facts)
        subpanel_flat = physics_subpanel_flat(facts)
        fa = first_arrival(facts)

        shape_name = str(shape.get("primary_shape", ""))
        shape_confidence = number(shape, "confidence", 0.0)
        shape_scores = shape.get("shape_scores", [])
        top_shape_scores = {
            str(item[0]): safe_float(item[1], 0.0)
            for item in shape_scores
            if isinstance(item, (list, tuple)) and len(item) >= 2
        }

        is_single = bool(getattr(facts, "is_single_event_like", False))
        is_short = bool(getattr(facts, "is_short_hit_like", False))
        is_loop = bool(getattr(facts, "is_loop_like", False))
        is_long = bool(getattr(facts, "is_long", False))

        slope = number(values, "centroid_slope_norm", number(shape, "centroid_slope_norm", 0.0))
        attack = number(values, "attack_rise_time_norm", number(shape, "attack_rise_time_norm", 1.0))
        temporal = number(values, "temporal_centroid_ratio", number(shape, "temporal_centroid_ratio", 0.50))
        tail = number(values, "tail_energy_ratio", number(shape, "tail_ratio", 0.0))
        flatness = number(values, "spectral_flatness_mean", number(shape, "spectral_flatness_mean", 0.0))
        entropy = number(values, "spectral_entropy_mean", number(shape, "spectral_entropy_mean", 0.0))
        flux = number(values, "spectral_flux_mean", 0.0)
        flux_var = number(values, "spectral_flux_variance", 0.0)
        zcr = number(values, "zcr_mean", 0.0)
        stereo = number(values, "stereo_width", 0.0)
        log_crest = number(values, "log_crest", 0.0)
        event_count = max(
            expm1_value(number(values, "log_transient_count", 0.0)),
            number(shape, "onset_count", 0.0),
        )
        onset_span = number(values, "onset_span_ratio", number(shape, "onset_span_ratio", 0.0))
        pulse_regularity = number(values, "onset_interval_regularity", number(shape, "pulse_regularity", 0.0))
        event_rate = number(values, "event_rate_hz", number(shape, "onset_density_hz", 0.0))
        pitch_conf = number(values, "pitch_confidence", number(shape, "pitch_confidence", 0.0))
        f0_voiced = number(values, "f0_voiced_ratio", number(shape, "f0_voiced_ratio", 0.0))
        harmonic = number(values, "harmonic_energy_ratio", 0.0)
        inharmonicity = number(values, "inharmonicity", 0.0)
        peak_stability = number(values, "spectral_peak_stability", 0.0)
        low_peak_hz = number(values, "low_peak_frequency_hz", 0.0)
        formant_spacing = number(values, "formant_like_peak_spacing", 0.0)
        sub = number(values, "sub_bass_ratio_lt_150hz", 0.0)
        bass = number(values, "bass_ratio_150_500hz", 0.0)
        mid = number(values, "mid_ratio_500_2000hz", 0.0)
        presence = number(values, "presence_ratio_2000_8000hz", 0.0)
        air = number(values, "air_ratio_gt_8000hz", 0.0)
        low_total = sub + bass
        high = presence + air
        body_noise = number(values, "body_noise_ratio", flatness)
        tail_noise = number(values, "tail_noise_ratio", body_noise)
        attack_noise = number(values, "attack_noise_ratio", flatness)
        attack_high = number(values, "attack_high_ratio", high)
        body_high = number(values, "body_high_ratio", high)
        tail_high = number(values, "tail_high_ratio", high)
        number(values, "attack_low_ratio", low_total)
        tail_low = number(values, "tail_low_ratio", low_total)
        loop_pitched = number(values, "loop_pitched_event_ratio", number(shape, "pitched_event_ratio", 0.0))
        loop_percussive = number(values, "loop_percussive_event_ratio", number(shape, "percussive_event_ratio", 0.0))
        loop_drumlike = number(values, "loop_drumlike_frame_ratio", number(shape, "drumlike_frame_ratio", 0.0))
        loop_noisy = number(values, "loop_noisy_event_ratio", 0.0)
        event_low = number(values, "loop_mean_event_low_ratio", number(shape, "low_event_ratio", 0.0))
        number(values, "loop_mean_event_high_ratio", number(shape, "high_event_ratio", 0.0))
        sustained_tonal = number(
            values, "loop_sustained_tonal_frame_ratio", number(shape, "sustained_tonal_frame_ratio", 0.0)
        )
        non_event_tonal = number(values, "loop_non_event_tonal_ratio", number(shape, "non_event_tonal_ratio", 0.0))
        true_repetition = number(values, "loop_true_repetition_score", number(shape, "true_repetition_score", 0.0))

        fa_stochastic = safe_float(fa.get("stochastic_modulation_coherence", 0.0), 0.0)
        fa_presence = safe_float(fa.get("first_arrival_presence_contrast_db", 0.0), 0.0)
        fa_formant_stability = safe_float(fa.get("formant_stability_score", 0.0), 0.0)
        fa_formant_std = safe_float(fa.get("formant_center_std_hz", 0.0), 0.0)

        long_or_motion = max(1.0 if is_long else 0.0, 1.0 if is_loop else 0.0, ramp(onset_span, 0.30, 0.82))
        single_or_short = max(1.0 if is_single else 0.0, 1.0 if is_short else 0.0, inverse_ramp(event_count, 1.0, 4.0))
        noisy_body = clamp01(
            0.30 * ramp(flatness, 0.22, 0.64)
            + 0.25 * ramp(entropy, 0.48, 0.86)
            + 0.20 * ramp(max(body_noise, tail_noise, attack_noise, loop_noisy), 0.20, 0.70)
            + 0.15 * ramp(zcr, 0.08, 0.34)
            + 0.10 * ramp(high, 0.08, 0.36)
        )
        low_peak_kick_band = inverse_ramp(low_peak_hz, 78.0, 180.0) if low_peak_hz > 0.0 else 0.0
        low_centered_tonal_hit = clamp01(
            0.30 * ramp(low_total, 0.70, 0.95)
            + 0.20 * inverse_ramp(high, 0.02, 0.14)
            + 0.18 * inverse_ramp(noisy_body, 0.05, 0.28)
            + 0.14 * single_or_short
            + 0.10 * inverse_ramp(stereo, 0.04, 0.24)
            + 0.08 * low_peak_kick_band
        )
        spectral_motion = clamp01(
            0.38 * ramp(abs(slope), 0.04, 0.22)
            + 0.22 * ramp(flux, 0.10, 0.46)
            + 0.18 * ramp(flux_var, 0.006, 0.055)
            + 0.12 * ramp(stereo, 0.24, 0.74)
            + 0.10 * ramp(max(attack_high, body_high, tail_high), 0.10, 0.44)
        )
        reverse_envelope = clamp01(
            0.34 * ramp(attack, 0.14, 0.62)
            + 0.24 * ramp(temporal, 0.44, 0.78)
            + 0.18 * ramp(tail, 0.22, 0.74)
            + 0.14 * inverse_ramp(event_count, 1.0, 7.0)
            + 0.10 * long_or_motion
        )
        impact_envelope = clamp01(
            0.28 * single_or_short
            + 0.22 * inverse_ramp(attack, 0.018, 0.16)
            + 0.18 * inverse_ramp(temporal, 0.12, 0.38)
            + 0.16 * ramp(tail, 0.20, 0.72)
            + 0.10 * ramp(log_crest, 1.15, 2.35)
            + 0.06 * ramp(stereo, 0.20, 0.72)
        )
        chopped_events = clamp01(
            0.26 * ramp(event_count, 4.0, 22.0)
            + 0.20 * ramp(event_rate, 1.8, 8.0)
            + 0.18 * inverse_ramp(pulse_regularity, 0.08, 0.46)
            + 0.15 * ramp(flux_var, 0.010, 0.075)
            + 0.12 * ramp(onset_span, 0.12, 0.74)
            + 0.09 * ramp(noisy_body, 0.30, 0.78)
        )
        synthetic_tone = clamp01(
            0.28 * ramp(max(pitch_conf, peak_stability), 0.45, 0.92)
            + 0.20 * inverse_ramp(flatness, 0.02, 0.22)
            + 0.18 * inverse_ramp(tail, 0.02, 0.36)
            + 0.16 * single_or_short
            + 0.10 * ramp(f0_voiced, 0.20, 0.80)
            + 0.08 * inverse_ramp(event_count, 1.0, 3.0)
        )
        formant_motion = clamp01(
            0.24 * ramp(formant_spacing, 0.34, 0.88)
            + 0.20 * ramp(f0_voiced, 0.35, 0.90)
            + 0.18 * ramp(max(fa_formant_stability, fa_stochastic), 0.28, 0.78)
            + 0.14 * ramp(fa_formant_std, 240.0, 900.0)
            + 0.12 * ramp(noisy_body, 0.24, 0.62)
            + 0.12 * inverse_ramp(harmonic, 0.10, 0.58)
        )
        band_limited_noise = clamp01(
            0.26 * ramp(noisy_body, 0.28, 0.74)
            + 0.20 * ramp(max(mid, presence), 0.28, 0.76)
            + 0.18 * inverse_ramp(low_total, 0.02, 0.28)
            + 0.16 * inverse_ramp(air, 0.01, 0.20)
            + 0.12 * ramp(zcr, 0.08, 0.32)
            + 0.08 * ramp(fa_stochastic, 0.24, 0.76)
        )
        raw_texture_bed = clamp01(
            0.30 * ramp(noisy_body, 0.24, 0.70)
            + 0.20 * (1.0 if is_long else 0.0)
            + 0.16 * ramp(tail, 0.24, 0.80)
            + 0.14 * inverse_ramp(abs(slope), 0.02, 0.16)
            + 0.12 * inverse_ramp(event_count, 1.0, 9.0)
            + 0.08 * ramp(stereo, 0.18, 0.72)
        )
        small_object_cluster = clamp01(
            0.24 * ramp(event_count, 2.0, 12.0)
            + 0.20 * ramp(max(inharmonicity, high, attack_high, body_high), 0.18, 0.62)
            + 0.18 * inverse_ramp(low_total, 0.08, 0.40)
            + 0.14 * inverse_ramp(tail, 0.02, 0.42)
            + 0.12 * ramp(max(chopped_events, flux_var * 8.0), 0.18, 0.70)
            + 0.12 * inverse_ramp(max(loop_drumlike, loop_percussive), 0.18, 0.52)
        )
        foley_material_motion = clamp01(
            0.22 * max(impact_envelope, chopped_events)
            + 0.18 * noisy_body
            + 0.16 * single_or_short
            + 0.14 * ramp(max(inharmonicity, spectral_motion), 0.14, 0.58)
            + 0.12 * ramp(max(mid, high), 0.22, 0.74)
            + 0.10 * inverse_ramp(max(pitch_conf, harmonic), 0.20, 0.72)
            + 0.08 * ramp(onset_span, 0.02, 0.42)
        )
        machine_mechanical_motion = clamp01(
            0.24 * band_limited_noise
            + 0.18 * ramp(event_rate, 0.45, 4.0)
            + 0.16 * ramp(max(fa_stochastic, spectral_motion), 0.18, 0.66)
            + 0.14 * ramp(max(mid, bass), 0.24, 0.76)
            + 0.12 * long_or_motion
            + 0.10 * inverse_ramp(max(f0_voiced, harmonic), 0.24, 0.78)
            + 0.06 * ramp(noisy_body, 0.24, 0.70)
        )
        siren_alarm_motion = clamp01(
            0.26 * synthetic_tone
            + 0.22 * long_or_motion
            + 0.18 * spectral_motion
            + 0.14 * ramp(max(peak_stability, pitch_conf), 0.40, 0.90)
            + 0.12 * ramp(max(f0_voiced, harmonic), 0.18, 0.76)
            + 0.08 * inverse_ramp(low_centered_tonal_hit, 0.28, 0.70)
        )
        human_creature_fx = clamp01(
            0.30 * formant_motion
            + 0.18 * ramp(max(f0_voiced, formant_spacing), 0.28, 0.86)
            + 0.16 * ramp(noisy_body, 0.18, 0.58)
            + 0.14 * long_or_motion
            + 0.12 * ramp(max(fa_stochastic, fa_formant_stability), 0.22, 0.70)
            + 0.10 * inverse_ramp(harmonic, 0.18, 0.70)
        )

        riser_shape = max(
            top_shape_scores.get("transition_riser", 0.0),
            shape_confidence if shape_name == "transition_riser" else 0.0,
        )
        drop_shape = max(
            top_shape_scores.get("transition_drop", 0.0),
            top_shape_scores.get("transition_downlifter", 0.0),
            shape_confidence if shape_name in {"transition_drop", "transition_downlifter"} else 0.0,
        )
        impact_shape = max(
            top_shape_scores.get("impact_with_tail", 0.0),
            top_shape_scores.get("hit_with_tail", 0.0),
            shape_confidence if shape_name in {"impact_with_tail", "hit_with_tail"} else 0.0,
        )
        glitch_shape = max(
            top_shape_scores.get("glitch_stutter", 0.0),
            shape_confidence if shape_name == "glitch_stutter" else 0.0,
        )
        blip_shape = max(
            top_shape_scores.get("ui_blip", 0.0),
            shape_confidence if shape_name == "ui_blip" else 0.0,
        )
        siren_alarm_shape = max(
            top_shape_scores.get("siren_alarm_tone", 0.0),
            shape_confidence if shape_name == "siren_alarm_tone" else 0.0,
        )
        texture_shape = max(
            top_shape_scores.get("texture_bed", 0.0),
            top_shape_scores.get("noise_texture", 0.0),
            top_shape_scores.get("static_bed", 0.0),
            shape_confidence if shape_name in {"texture_bed", "noise_texture", "static_bed"} else 0.0,
        )
        foley_shape = max(
            top_shape_scores.get("foley_action", 0.0),
            shape_confidence if shape_name == "foley_action" else 0.0,
        )
        mechanical_shape = max(
            top_shape_scores.get("mechanical_motion", 0.0),
            shape_confidence if shape_name == "mechanical_motion" else 0.0,
        )
        reverse_shape = max(
            top_shape_scores.get("reverse_swell", 0.0),
            shape_confidence if shape_name == "reverse_swell" else 0.0,
        )
        whoosh_shape = max(
            top_shape_scores.get("whoosh_sweep", 0.0),
            shape_confidence if shape_name == "whoosh_sweep" else 0.0,
        )
        measured_transition_motion = bool(
            shape_name
            in {"transition_riser", "transition_drop", "transition_downlifter", "reverse_swell", "whoosh_sweep"}
            or abs(slope) >= 0.045
            or spectral_motion >= 0.18
        )
        if not measured_transition_motion:
            riser_shape = min(riser_shape, 0.25)
            drop_shape = min(drop_shape, 0.25)
            reverse_shape = min(reverse_shape, 0.25)
            whoosh_shape = min(whoosh_shape, 0.25)
        blip_band_support = clamp01(
            0.38 * ramp(mid + high, 0.06, 0.38)
            + 0.32 * inverse_ramp(low_total, 0.50, 0.88)
            + 0.20 * (ramp(low_peak_hz, 180.0, 1200.0) if low_peak_hz > 0.0 else 0.0)
            + 0.10 * ramp(high, 0.04, 0.18)
        )
        if low_centered_tonal_hit >= 0.55 and shape_name != "ui_blip":
            blip_shape = min(blip_shape, 0.18)
        fx_action_shape = max(
            riser_shape, drop_shape, impact_shape, glitch_shape, blip_shape, reverse_shape, whoosh_shape
        )
        transition_shape_support = max(riser_shape, drop_shape, reverse_shape, whoosh_shape)
        repeated_rhythmic_loop = clamp01(
            0.24 * (1.0 if is_loop else 0.0)
            + 0.22 * ramp(true_repetition, 0.54, 0.88)
            + 0.18 * ramp(event_count, 8.0, 28.0)
            + 0.14 * ramp(onset_span, 0.50, 0.92)
            + 0.12 * ramp(max(loop_percussive, loop_drumlike), 0.12, 0.52)
            + 0.10 * ramp(drum_anchor, 0.22, 0.62)
        )
        transition_loop_decoy_guard = bool(
            repeated_rhythmic_loop >= 0.58
            and max(drum_anchor, loop_percussive, loop_drumlike) >= 0.20
            and event_count >= 8.0
        )
        sub_transition_authority = safe_float(subpanel_flat.get("fx_transition_authority_score", 0.0), 0.0)
        sub_riser_authority = safe_float(subpanel_flat.get("fx_riser_build_score", 0.0), 0.0)
        strong_directed_transition_shape = bool(
            shape_name
            in {"transition_riser", "transition_drop", "transition_downlifter", "reverse_swell", "whoosh_sweep"}
            and shape_confidence >= 0.86
            and transition_shape_support >= 0.74
            and (abs(slope) >= 0.085 or spectral_motion >= 0.28 or sub_transition_authority >= 0.58)
            and pulse_regularity <= 0.42
            and max(sustained_tonal, non_event_tonal, loop_pitched) <= 0.52
        )
        designed_motion_loop_exception = bool(
            (
                shape_name == "hybrid_fx_motion"
                and transition_shape_support >= 0.70
                and spectral_motion >= 0.42
                and drum_anchor < 0.42
                and instrument_anchor < 0.58
            )
            or strong_directed_transition_shape
        )
        if designed_motion_loop_exception:
            transition_loop_decoy_guard = False

        branch_scores = {
            "RiserBuild": clamp01(
                0.30 * ramp(slope, 0.055, 0.22)
                + 0.20 * long_or_motion
                + 0.18 * spectral_motion
                + 0.14 * ramp(max(noisy_body, loop_pitched), 0.28, 0.76)
                + 0.10 * ramp(tail, 0.18, 0.62)
                + 0.18 * riser_shape
            ),
            "DropDownlifter": clamp01(
                0.30 * ramp(-slope, 0.055, 0.22)
                + 0.19 * long_or_motion
                + 0.16 * spectral_motion
                + 0.14 * ramp(max(tail_low, low_total, event_low), 0.22, 0.72)
                + 0.11 * ramp(tail, 0.18, 0.68)
                + 0.18 * drop_shape
            ),
            "WhooshSweep": clamp01(
                0.25 * noisy_body
                + 0.23 * spectral_motion
                + 0.16 * ramp(high, 0.08, 0.36)
                + 0.14 * ramp(stereo, 0.24, 0.78)
                + 0.12 * long_or_motion
                + 0.10 * inverse_ramp(max(loop_pitched, sustained_tonal), 0.10, 0.54)
                + 0.12 * whoosh_shape
            ),
            "ReverseSwell": max(reverse_envelope, clamp01(0.82 * reverse_shape + 0.18 * reverse_envelope)),
            "ImpactHit": max(
                impact_envelope,
                clamp01(
                    0.56 * impact_envelope
                    + 0.22 * noisy_body
                    + 0.12 * ramp(low_total, 0.14, 0.60)
                    + 0.10 * impact_shape
                ),
            ),
            "GlitchStutter": max(
                chopped_events,
                clamp01(0.62 * chopped_events + 0.22 * glitch_shape + 0.16 * noisy_body),
            ),
            "BlipBeep": max(
                clamp01(synthetic_tone * max(0.12, blip_band_support)),
                clamp01(
                    0.68 * synthetic_tone * max(0.12, blip_band_support)
                    + 0.20 * blip_shape
                    + 0.12 * inverse_ramp(noisy_body, 0.08, 0.42)
                ),
            ),
            "SirenAlarm": max(
                siren_alarm_motion,
                clamp01(0.62 * siren_alarm_motion + 0.26 * siren_alarm_shape + 0.12 * band_limited_noise),
            ),
            "FormantFX": formant_motion,
            "RadioElectrical": band_limited_noise,
            "TextureAmbience": max(
                raw_texture_bed,
                clamp01(0.64 * raw_texture_bed + 0.26 * texture_shape + 0.10 * noisy_body),
            ),
            "MachineMechanical": max(
                machine_mechanical_motion,
                clamp01(0.68 * machine_mechanical_motion + 0.20 * mechanical_shape + 0.12 * band_limited_noise),
            ),
            "FoleyMaterial": max(
                foley_material_motion,
                clamp01(0.62 * foley_material_motion + 0.22 * foley_shape + 0.16 * impact_envelope),
            ),
            "SmallObjectCluster": small_object_cluster,
            "HumanCreatureFX": human_creature_fx,
            "DesignedNoiseHybrid": clamp01(
                0.26 * noisy_body
                + 0.22 * spectral_motion
                + 0.16 * ramp(max(stereo, fa_presence / 36.0), 0.20, 0.82)
                + 0.14 * ramp(max(inharmonicity, body_noise, tail_noise), 0.18, 0.64)
                + 0.12 * ramp(max(chopped_events, reverse_envelope, formant_motion, raw_texture_bed), 0.26, 0.78)
                + 0.10 * inverse_ramp(max(loop_pitched, harmonic), 0.20, 0.74)
            ),
        }
        if (
            shape_name == "transition_riser"
            and shape_confidence >= 0.86
            and (slope >= 0.055 or sub_riser_authority >= 0.60)
        ):
            strongest_transition_neighbor = max(
                branch_scores["DropDownlifter"],
                branch_scores["WhooshSweep"],
                branch_scores["ReverseSwell"],
                branch_scores["ImpactHit"],
            )
            branch_scores["RiserBuild"] = max(
                branch_scores["RiserBuild"],
                min(0.97, strongest_transition_neighbor + 0.026),
            )

        branch_scores = apply_source_panel_lifts(
            branch_scores,
            {
                "RiserBuild": subpanel_flat.get("fx_riser_build_score", 0.0),
                "DropDownlifter": subpanel_flat.get("fx_drop_downlifter_score", 0.0),
                "WhooshSweep": subpanel_flat.get("fx_whoosh_sweep_score", 0.0),
                "ReverseSwell": subpanel_flat.get("fx_reverse_score", 0.0),
                "ImpactHit": subpanel_flat.get("fx_impact_score", 0.0),
                "GlitchStutter": subpanel_flat.get("fx_glitch_stutter_score", 0.0),
                "BlipBeep": subpanel_flat.get("fx_blip_beep_score", 0.0),
                "SirenAlarm": subpanel_flat.get("tonal_alert_siren_score", 0.0),
                "FormantFX": subpanel_flat.get("fx_formant_score", 0.0),
                "RadioElectrical": subpanel_flat.get("fx_radio_electrical_score", 0.0),
                "MachineMechanical": subpanel_flat.get("fx_machine_mechanical_score", 0.0),
                "FoleyMaterial": subpanel_flat.get("fx_foley_material_score", 0.0),
                "SmallObjectCluster": subpanel_flat.get("fx_small_object_score", 0.0),
                "TextureAmbience": max(
                    safe_float(subpanel_flat.get("texture_water_ocean_score", 0.0), 0.0),
                    safe_float(subpanel_flat.get("texture_rain_score", 0.0), 0.0),
                    safe_float(subpanel_flat.get("texture_wind_score", 0.0), 0.0),
                    safe_float(subpanel_flat.get("texture_fire_score", 0.0), 0.0),
                    safe_float(subpanel_flat.get("texture_noise_static_score", 0.0), 0.0),
                ),
                "HumanCreatureFX": max(
                    safe_float(subpanel_flat.get("human_breath_mouth_score", 0.0), 0.0),
                    safe_float(subpanel_flat.get("human_spoken_voice_score", 0.0), 0.0),
                    safe_float(subpanel_flat.get("human_scream_score", 0.0), 0.0),
                    safe_float(subpanel_flat.get("human_applause_crowd_score", 0.0), 0.0),
                    safe_float(subpanel_flat.get("animal_bird_score", 0.0), 0.0),
                    safe_float(subpanel_flat.get("animal_voice_score", 0.0), 0.0),
                    safe_float(subpanel_flat.get("animal_cricket_insect_score", 0.0), 0.0),
                ),
                "DesignedNoiseHybrid": max(
                    safe_float(subpanel_flat.get("fx_motion_score", 0.0), 0.0),
                    safe_float(subpanel_flat.get("texture_noise_static_score", 0.0), 0.0),
                ),
            },
            scale=0.94,
            floor=0.42,
        )
        if (
            shape_name == "transition_riser"
            and shape_confidence >= 0.86
            and (slope >= 0.055 or sub_riser_authority >= 0.60)
        ):
            strongest_transition_neighbor = max(
                branch_scores["DropDownlifter"],
                branch_scores["WhooshSweep"],
                branch_scores["ReverseSwell"],
                branch_scores["ImpactHit"],
            )
            branch_scores["RiserBuild"] = max(
                branch_scores["RiserBuild"],
                min(0.98, strongest_transition_neighbor + 0.026),
            )

        if transition_loop_decoy_guard:
            transition_cap = 0.50 + 0.12 * inverse_ramp(repeated_rhythmic_loop, 0.58, 0.86)
            for branch_name in ("RiserBuild", "DropDownlifter", "WhooshSweep", "ReverseSwell"):
                branch_scores[branch_name] = min(branch_scores[branch_name], transition_cap)
        clean_mid_blip_tone = clamp01(
            0.36 * ramp(synthetic_tone, 0.70, 0.94)
            + 0.28 * ramp(blip_band_support, 0.48, 0.78)
            + 0.18 * inverse_ramp(tail, 0.02, 0.22)
            + 0.10 * inverse_ramp(event_count, 1.0, 3.0)
            + 0.08 * inverse_ramp(low_centered_tonal_hit, 0.34, 0.72)
        )
        if clean_mid_blip_tone >= 0.58:
            branch_scores["BlipBeep"] = max(
                branch_scores["BlipBeep"],
                clamp01(0.68 + 0.28 * ramp(clean_mid_blip_tone, 0.58, 0.90)),
            )
            if branch_scores["ImpactHit"] >= branch_scores["BlipBeep"] and noisy_body < 0.16 and tail < 0.24:
                branch_scores["ImpactHit"] = max(0.0, branch_scores["BlipBeep"] - 0.03)
        strongest = max(branch_scores.values()) if branch_scores else 0.0
        branch = max(branch_scores.items(), key=lambda item: item[1])[0]
        transition_loop_guard_blocks_fx = bool(
            transition_loop_decoy_guard
            and branch in {"RiserBuild", "DropDownlifter", "WhooshSweep", "ReverseSwell"}
            and not designed_motion_loop_exception
        )
        action_strength = max(
            branch_scores["RiserBuild"],
            branch_scores["DropDownlifter"],
            branch_scores["WhooshSweep"],
            branch_scores["ReverseSwell"],
            branch_scores["ImpactHit"],
            branch_scores["GlitchStutter"],
            branch_scores["BlipBeep"],
            branch_scores["SirenAlarm"],
        )
        texture_bed_strength = max(raw_texture_bed, branch_scores["TextureAmbience"])
        drum_conflict = clamp01(
            0.46 * ramp(drum_anchor, 0.58, 0.90)
            + 0.22 * ramp(drum_branch_confidence, 0.52, 0.86)
            + 0.12 * ramp(max(loop_percussive, loop_drumlike), 0.30, 0.80)
            + 0.34 * low_centered_tonal_hit
            + 0.44 * repeated_rhythmic_loop
            - 0.30 * ramp(max(branch_scores["ImpactHit"], branch_scores["FoleyMaterial"]), 0.78, 0.96)
        )
        instrument_conflict = clamp01(
            0.54 * ramp(instrument_anchor, 0.62, 0.92)
            + 0.24 * ramp(instrument_branch_confidence, 0.54, 0.86)
            + 0.22 * ramp(max(loop_pitched, sustained_tonal, non_event_tonal), 0.58, 0.98)
            + 0.14 * repeated_rhythmic_loop * ramp(max(loop_pitched, sustained_tonal), 0.48, 0.90)
            - 0.34 * ramp(max(action_strength, texture_bed_strength, branch_scores["HumanCreatureFX"]), 0.72, 0.94)
        )
        mixed_loop_conflict = clamp01(
            0.48 * ramp(compound_strength, 0.54, 0.86)
            + 0.28 * (1.0 if compound_prefer_broad_loop else 0.0)
            + 0.16 * ramp(max(loop_pitched, sustained_tonal), 0.62, 0.98)
            + 0.08 * ramp(onset_span, 0.50, 0.90)
            + 0.22 * repeated_rhythmic_loop
            - 0.30 * ramp(max(action_strength, texture_bed_strength), 0.74, 0.96)
        )
        texture_conflict = clamp01(
            0.42 * ramp(texture_bed_strength, 0.56, 0.88)
            - 0.48 * ramp(branch_scores["TextureAmbience"], 0.58, 0.86)
            - 0.30 * ramp(action_strength, 0.66, 0.92)
        )
        conflict = max(drum_conflict, instrument_conflict, mixed_loop_conflict, texture_conflict)
        role_raw = clamp01(0.72 * strongest + 0.18 * action_strength + 0.10 * spectral_motion)
        role_raw = max(
            role_raw,
            clamp01(0.82 * fx_action_shape + 0.18 * action_strength),
            clamp01(0.74 * texture_bed_strength + 0.16 * texture_shape + 0.10 * noisy_body),
            clamp01(
                0.70 * max(machine_mechanical_motion, foley_material_motion, human_creature_fx)
                + 0.16 * noisy_body
                + 0.14 * spectral_motion
            ),
        )
        role_strength = clamp01(role_raw - 0.24 * conflict)
        non_action_fx_strength = max(
            branch_scores["TextureAmbience"],
            branch_scores["MachineMechanical"],
            branch_scores["FoleyMaterial"],
            branch_scores["SmallObjectCluster"],
            branch_scores["HumanCreatureFX"],
            branch_scores["SirenAlarm"],
        )
        non_action_fx_branch = branch in {
            "TextureAmbience",
            "MachineMechanical",
            "FoleyMaterial",
            "SmallObjectCluster",
            "HumanCreatureFX",
            "SirenAlarm",
        }
        strong_transition_action = bool(
            branch in {"RiserBuild", "DropDownlifter", "WhooshSweep", "ReverseSwell"}
            and action_strength >= 0.78
            and strongest >= 0.78
            and role_strength >= 0.66
            and conflict < 0.60
            and max(transition_shape_support, spectral_motion, sub_transition_authority) >= 0.38
            and not transition_loop_guard_blocks_fx
        )
        allow_fx = bool(
            (
                role_strength >= 0.70
                or strong_transition_action
                or (transition_shape_support >= 0.68 and role_strength >= 0.60 and conflict < 0.38)
                or (branch == "ImpactHit" and role_strength >= 0.64 and impact_envelope >= 0.70 and conflict < 0.50)
                or (
                    non_action_fx_branch
                    and non_action_fx_strength >= 0.66
                    and role_strength >= 0.58
                    and conflict < 0.56
                )
            )
            and conflict < 0.64
            and (
                action_strength >= 0.68
                or transition_shape_support >= 0.68
                or impact_envelope >= 0.70
                or non_action_fx_strength >= 0.66
                or branch in {"FormantFX", "RadioElectrical", "DesignedNoiseHybrid"}
            )
            and not (
                branch in {"RiserBuild", "DropDownlifter", "WhooshSweep", "ReverseSwell"}
                and transition_loop_guard_blocks_fx
            )
        )

        out = {
            "fx_role_strength_raw": round(float(role_raw), 6),
            "fx_role_strength": round(float(role_strength), 6),
            "fx_action_strength": round(float(action_strength), 6),
            "fx_spectral_motion": round(float(spectral_motion), 6),
            "fx_noisy_body": round(float(noisy_body), 6),
            "fx_texture_bed_strength": round(float(texture_bed_strength), 6),
            "fx_non_action_strength": round(float(non_action_fx_strength), 6),
            "fx_raw_texture_bed": round(float(raw_texture_bed), 6),
            "fx_low_centered_tonal_hit": round(float(low_centered_tonal_hit), 6),
            "fx_blip_band_support": round(float(blip_band_support), 6),
            "fx_clean_mid_blip_tone": round(float(clean_mid_blip_tone), 6),
            "fx_siren_alarm_motion": round(float(siren_alarm_motion), 6),
            "fx_machine_mechanical_motion": round(float(machine_mechanical_motion), 6),
            "fx_foley_material_motion": round(float(foley_material_motion), 6),
            "fx_small_object_cluster": round(float(small_object_cluster), 6),
            "fx_human_creature_motion": round(float(human_creature_fx), 6),
            "fx_measured_transition_motion": bool(measured_transition_motion),
            "fx_action_shape_support": round(float(fx_action_shape), 6),
            "fx_transition_shape_support": round(float(transition_shape_support), 6),
            "fx_repeated_rhythmic_loop_conflict": round(float(repeated_rhythmic_loop), 6),
            "fx_transition_loop_decoy_guard": bool(transition_loop_guard_blocks_fx),
            "fx_transition_loop_decoy_guard_raw": bool(transition_loop_decoy_guard),
            "fx_designed_motion_loop_exception": bool(designed_motion_loop_exception),
            "fx_vs_drum_conflict": round(float(drum_conflict), 6),
            "fx_vs_instrument_conflict": round(float(instrument_conflict), 6),
            "fx_vs_mixed_loop_conflict": round(float(mixed_loop_conflict), 6),
            "fx_vs_texture_conflict": round(float(texture_conflict), 6),
            "fx_role_conflict_strength": round(float(conflict), 6),
            "fx_role_allows_fx": allow_fx,
            "fx_reverse_envelope": round(float(reverse_envelope), 6),
            "fx_impact_envelope": round(float(impact_envelope), 6),
            "fx_chopped_events": round(float(chopped_events), 6),
            "fx_synthetic_tone": round(float(synthetic_tone), 6),
            "fx_formant_motion": round(float(formant_motion), 6),
            "fx_band_limited_noise": round(float(band_limited_noise), 6),
            "fx_shape_primary": shape_name,
            "fx_shape_confidence": round(float(shape_confidence), 6),
        }
        for name, value in branch_scores.items():
            out[f"fx_branch_{name}"] = round(float(value), 6)
        return out
