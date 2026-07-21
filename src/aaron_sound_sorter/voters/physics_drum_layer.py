# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Measured drum/percussion physics layer."""

from __future__ import annotations

from typing import Any

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.voters.physics_layer_utils import *


class PhysicsDrumLayer:
    """Measured drum/percussion layer for short struck or scraped events.

    This layer deliberately does not know filenames or source labels.  It uses
    attack timing, temporal centroid, band energy, spectral flatness/entropy,
    onset counts, tail shape, and first-arrival strike evidence to separate
    broad drum branches before the learned profile residuals decide the leaf.
    """

    BRANCHES = (
        "DrumLoop",
        "Kick",
        "TomOrConga",
        "Snare",
        "Clap",
        "Hat",
        "Cymbal",
        "RimOrStick",
        "ShakerTambourine",
        "ScrapeGuiro",
        "MetallicPercussion",
    )

    @staticmethod
    def _measured_short_event_like(facts: SharedAudioFacts, event_count: float) -> bool:
        return bool(
            getattr(facts, "is_short_hit_like", False)
            or getattr(facts, "is_single_event_like", False)
            or event_count <= 2.25
        )

    @staticmethod
    def _measured_not_vocal_phrase(vocal_role: float, vocal_phrase_role: float, formant_voice: float) -> bool:
        return bool(vocal_phrase_role < 0.35 and formant_voice < 0.45 and vocal_role < 0.90)

    def decide(self, facts: SharedAudioFacts) -> tuple[str, float, dict[str, Any]]:
        evidence = self.evidence(facts)
        branch_scores = {name: float(evidence[f"drum_branch_{name}"]) for name in self.BRANCHES}
        branch, branch_strength = max(branch_scores.items(), key=lambda item: item[1])
        drum_loop_strength = branch_scores.get("DrumLoop", 0.0)
        shape_name = str(evidence.get("drum_anchor_shape_name", ""))
        event_count = safe_float(evidence.get("drum_event_count", 0.0), 0.0)
        if (
            drum_loop_strength >= 0.62
            and shape_name in {"beat_loop", "top_loop", "repeated_phrase_loop"}
            and event_count >= 4.0
            and (drum_loop_strength >= branch_strength - 0.12 or drum_loop_strength >= 0.82)
        ):
            branch = "DrumLoop"
            branch_strength = drum_loop_strength
        evidence["drum_branch_selected"] = branch
        evidence["drum_branch_selected_confidence"] = round(float(branch_strength), 6)
        for name, value in branch_scores.items():
            evidence[f"drum_branch_score_{name}"] = round(float(value), 6)
        return branch, branch_strength, evidence

    def evidence(self, facts: SharedAudioFacts) -> dict[str, Any]:
        values = facts.feature_values_by_name or {}
        shape = shape_vote(facts)
        roles = measured_roles(facts)
        subpanel_flat = physics_subpanel_flat(facts)
        sub_drum_loop_source = safe_float(subpanel_flat.get("drum_loop_source_score", 0.0), 0.0)
        sub_rhythmic_break_loop = safe_float(subpanel_flat.get("rhythmic_break_loop_score", 0.0), 0.0)
        sub_onset_pitched = safe_float(subpanel_flat.get("onset_pitched_onset_score", 0.0), 0.0)
        sub_onset_percussive = safe_float(subpanel_flat.get("onset_percussive_onset_score", 0.0), 0.0)
        sub_synth_tonal = safe_float(subpanel_flat.get("synth_tonal_source_score", 0.0), 0.0)
        sub_voice = safe_float(subpanel_flat.get("voice_score", 0.0), 0.0)
        sub_human_spoken_voice = safe_float(subpanel_flat.get("human_spoken_voice_score", 0.0), 0.0)
        sub_voice_choir = safe_float(subpanel_flat.get("voice_choir_score", 0.0), 0.0)
        sub_formant_fx = safe_float(subpanel_flat.get("fx_formant_score", 0.0), 0.0)
        sub_hand_drum_membrane = safe_float(subpanel_flat.get("hand_drum_membrane_score", 0.0), 0.0)
        sub_pitched_metal_percussion = safe_float(subpanel_flat.get("pitched_metal_percussion_score", 0.0), 0.0)
        sub_struck_wood = safe_float(subpanel_flat.get("struck_wood_score", 0.0), 0.0)
        sub_compact_struck_percussion = safe_float(subpanel_flat.get("compact_struck_tonal_percussion_score", 0.0), 0.0)
        sub_struck_percussion_exception = bool(subpanel_flat.get("struck_percussion_guard_exception"))
        sub_hand_drum_material = bool(subpanel_flat.get("hand_drum_material_evidence"))
        sub_pitched_metal_material = bool(subpanel_flat.get("pitched_metal_material_evidence"))
        sub_struck_wood_material = bool(subpanel_flat.get("struck_wood_material_evidence"))
        fa = first_arrival(facts)
        shape_name = str(shape.get("primary_shape", ""))
        shape_confidence = number(shape, "confidence", 0.0)

        safe_float(getattr(facts, "duration_sec", 0.0), 0.0)
        # duration_sec is carried on AudioPhysics, not SharedAudioFacts.  The
        # analyzer writes duration-like facts into feature vectors inconsistently,
        # so prefer structure booleans and event timing below.
        is_single = bool(getattr(facts, "is_single_event_like", False))
        is_short = bool(getattr(facts, "is_short_hit_like", False))
        is_loop = bool(getattr(facts, "is_loop_like", False))
        is_long = bool(getattr(facts, "is_long", False))

        attack = number(values, "attack_rise_time_norm", number(shape, "attack_rise_time_norm", 1.0))
        temporal = number(values, "temporal_centroid_ratio", number(shape, "temporal_centroid_ratio", 0.50))
        tail = number(values, "tail_energy_ratio", number(shape, "tail_ratio", 0.0))
        flatness = number(values, "spectral_flatness_mean", number(shape, "spectral_flatness_mean", 0.0))
        entropy = number(values, "spectral_entropy_mean", number(shape, "spectral_entropy_mean", 0.0))
        log_crest = number(values, "log_crest", 0.0)
        event_count = expm1_value(number(values, "log_transient_count", number(shape, "onset_count", 0.0)))
        shape_events = number(shape, "onset_count", event_count)
        event_count = max(event_count, shape_events)
        onset_span = number(values, "onset_span_ratio", number(shape, "onset_span_ratio", 0.0))
        pitch_conf = number(values, "pitch_confidence", number(shape, "pitch_confidence", 0.0))
        f0_voiced = number(values, "f0_voiced_ratio", number(shape, "f0_voiced_ratio", 0.0))
        harmonic = number(values, "harmonic_energy_ratio", 0.0)
        inharmonicity = number(values, "inharmonicity", 0.0)
        low_peak_hz = number(values, "low_peak_frequency_hz", 0.0)
        kick_drop = abs(number(values, "kick_pitch_drop_cents", 0.0))
        sub_decay_ms = number(values, "sub_decay_time_ms", 0.0)
        noise_ms = number(values, "noise_burst_duration_ms", 0.0)

        sub = number(values, "sub_bass_ratio_lt_150hz", 0.0)
        bass = number(values, "bass_ratio_150_500hz", 0.0)
        mid = number(values, "mid_ratio_500_2000hz", 0.0)
        presence = number(values, "presence_ratio_2000_8000hz", 0.0)
        air = number(values, "air_ratio_gt_8000hz", 0.0)
        high = presence + air
        attack_high = number(values, "attack_high_ratio", 0.0)
        body_high = number(values, "body_high_ratio", 0.0)
        tail_high = number(values, "tail_high_ratio", 0.0)
        number(values, "attack_low_ratio", 0.0)
        number(values, "body_low_ratio", 0.0)
        number(values, "tail_low_ratio", 0.0)
        attack_noise = number(values, "attack_noise_ratio", 0.0)
        body_noise = number(values, "body_noise_ratio", 0.0)
        loop_percussive = number(values, "loop_percussive_event_ratio", number(shape, "percussive_event_ratio", 0.0))
        loop_drumlike = number(values, "loop_drumlike_frame_ratio", number(shape, "drumlike_frame_ratio", 0.0))
        loop_pitched = number(values, "loop_pitched_event_ratio", number(shape, "pitched_event_ratio", 0.0))
        loop_noisy = number(values, "loop_noisy_event_ratio", 0.0)
        event_low = number(values, "loop_mean_event_low_ratio", number(shape, "low_event_ratio", 0.0))
        event_high = number(values, "loop_mean_event_high_ratio", number(shape, "high_event_ratio", 0.0))

        fa_status = str(fa.get("status", "missing"))
        safe_float(fa.get("strike_high_ratio", 0.0), 0.0)
        fa_settle_high = safe_float(fa.get("settle_high_ratio", 0.0), 0.0)
        fa_strike_flatness = safe_float(fa.get("strike_flatness", 0.0), 0.0)
        safe_float(fa.get("settle_flatness", 0.0), 0.0)
        safe_float(fa.get("strike_entropy", 0.0), 0.0)
        safe_float(fa.get("settle_entropy", 0.0), 0.0)
        fa_darkening_db = safe_float(fa.get("spectral_darkening_db", 0.0), 0.0)
        fa_inharmonic = safe_float(fa.get("strike_inharmonic_energy_ratio", 0.0), 0.0)
        fa_events = safe_float(fa.get("selected_event_count", 0.0), 0.0)

        fast_attack = inverse_ramp(attack, 0.018, 0.145)
        very_fast_attack = inverse_ramp(attack, 0.008, 0.055)
        early_energy = inverse_ramp(temporal, 0.16, 0.48)
        early_struck = inverse_ramp(temporal, 0.11, 0.32)
        transient = clamp01(0.55 * ramp(log_crest, 1.20, 2.35) + 0.45 * fast_attack)
        single_or_few = max(
            1.0 if is_single else 0.0,
            inverse_ramp(event_count, 1.0, 5.0),
            inverse_ramp(onset_span, 0.18, 0.62),
        )
        short_hit = max(1.0 if is_short else 0.0, inverse_ramp(event_count, 1.0, 7.0))
        not_loop = 0.0 if is_loop else inverse_ramp(onset_span, 0.35, 0.88)
        not_long_texture = 0.0 if is_long else 1.0
        percussive_role = max(
            role_value(roles, "percussive_one_shot"),
            role_value(roles, "bright_drum_loop"),
            role_value(roles, "percussive_drum_loop"),
            role_value(roles, "low_rhythmic_drum_loop"),
        )
        voiced_role = max(
            role_value(roles, "voiced_one_shot"),
            role_value(roles, "vocal_music_phrase"),
        )
        event_evidence = max(loop_percussive, loop_drumlike, ramp(event_count, 1.0, 3.0), 1.0 if is_single else 0.0)
        sustained_tonal = number(
            values,
            "loop_sustained_tonal_frame_ratio",
            number(shape, "sustained_tonal_frame_ratio", 0.0),
        )
        non_event_tonal = number(
            values,
            "loop_non_event_tonal_ratio",
            number(shape, "non_event_tonal_ratio", 0.0),
        )
        pitched_phrase_penalty = (
            0.22 * ramp(max(loop_pitched, pitch_conf), 0.86, 1.0) * inverse_ramp(event_high, 0.03, 0.22)
        )
        sustained_phrase_penalty = 0.18 * ramp(sustained_tonal, 0.80, 1.0) * inverse_ramp(transient, 0.35, 0.80)
        true_vocal_phrase_penalty = (
            0.52
            * (1.0 if shape_name == "vocal_phrase" else 0.0)
            * ramp(shape_confidence, 0.78, 0.96)
            * ramp(max(pitch_conf, f0_voiced), 0.58, 0.92)
            * ramp(max(sustained_tonal, non_event_tonal), 0.72, 0.98)
            * inverse_ramp(max(loop_percussive, loop_drumlike), 0.04, 0.30)
        )
        tonal_non_drum_hit_penalty = (
            0.46
            * ramp(max(loop_pitched, pitch_conf, f0_voiced), 0.60, 0.94)
            * ramp(non_event_tonal, 0.70, 0.98)
            * inverse_ramp(max(loop_percussive, loop_drumlike), 0.04, 0.30)
            * inverse_ramp(event_low, 0.28, 0.68)
            * inverse_ramp(event_high, 0.40, 0.72)
        )
        low_sub_kick_exception = bool(sub >= 0.55 and 0.0 < low_peak_hz <= 135.0 and f0_voiced <= 0.20)
        clean_tonal_solo_phrase = bool(
            shape_name in {"solo_phrase", "pitched_phrase", "vocal_phrase", "sustained_pad"}
            and shape_confidence >= 0.82
            and max(loop_pitched, pitch_conf) >= 0.80
            and max(sustained_tonal, non_event_tonal) >= 0.82
            and max(loop_percussive, loop_drumlike) <= 0.14
            and (f0_voiced >= 0.70 or harmonic >= 0.52)
            and not low_sub_kick_exception
        )
        clean_tonal_solo_penalty = (
            0.48
            * (1.0 if clean_tonal_solo_phrase else 0.0)
            * ramp(max(f0_voiced, harmonic), 0.52, 0.94)
            * inverse_ramp(max(loop_percussive, loop_drumlike), 0.02, 0.18)
        )
        drum_anchor = clamp01(
            0.20 * short_hit
            + 0.18 * fast_attack
            + 0.16 * early_energy
            + 0.14 * transient
            + 0.12 * not_loop
            + 0.10 * event_evidence
            + 0.10 * percussive_role
            - pitched_phrase_penalty
            - sustained_phrase_penalty
            - true_vocal_phrase_penalty
            - tonal_non_drum_hit_penalty
            - clean_tonal_solo_penalty
        )
        if shape_name == "vocal_phrase" and shape_confidence >= 0.86 and max(loop_percussive, loop_drumlike) <= 0.18:
            drum_anchor = min(drum_anchor, 0.54)
        if clean_tonal_solo_phrase:
            drum_anchor = min(drum_anchor, 0.52)
        if not_long_texture <= 0.0 and event_count <= 2.0:
            drum_anchor *= 0.55
        designed_fx_slow_tail_stand_down = bool(
            shape_name in {"designed_low_fx", "designed_motion_fx_loop", "hybrid_fx_motion"}
            and shape_confidence >= 0.78
            and is_long
            and event_count <= 2.25
            and attack >= 0.09
            and temporal >= 0.34
            and tail >= 0.68
            and not low_sub_kick_exception
        )
        if designed_fx_slow_tail_stand_down:
            drum_anchor = min(drum_anchor, 0.50)

        low_body = sub + bass
        max(high, event_high, attack_high, body_high, tail_high)
        high_noise = max(presence + air, attack_high, body_high, event_high)
        bright_tail = max(tail_high, fa_settle_high)
        noisy = max(flatness, loop_noisy, attack_noise, body_noise, fa_strike_flatness)
        metallic_ring = clamp01(
            0.32 * max(ramp(inharmonicity, 0.10, 0.34), ramp(fa_inharmonic, 0.18, 0.55))
            + 0.26 * ramp(harmonic, 0.26, 0.72)
            + 0.22 * ramp(high_noise, 0.18, 0.75)
            + 0.20 * ramp(max(tail, bright_tail), 0.05, 0.55)
        )
        broadband_noise = clamp01(
            0.38 * ramp(flatness, 0.24, 0.62)
            + 0.28 * ramp(entropy, 0.52, 0.82)
            + 0.18 * ramp(noisy, 0.24, 0.68)
            + 0.16 * ramp(high_noise, 0.10, 0.60)
        )
        low_kick_core = clamp01(
            0.34 * ramp(sub, 0.45, 0.88)
            + 0.20 * inverse_ramp(low_peak_hz, 55.0, 180.0)
            + 0.18 * inverse_ramp(temporal, 0.17, 0.36)
            + 0.13 * inverse_ramp(high, 0.02, 0.28)
            + 0.10 * ramp(sub_decay_ms, 20.0, 120.0)
            + 0.05 * inverse_ramp(kick_drop, 0.0, 900.0)
        )
        tom_conga_core = clamp01(
            0.22 * ramp(bass, 0.28, 0.82)
            + 0.18 * ramp(low_body, 0.45, 0.94)
            + 0.18 * ramp(harmonic, 0.12, 0.82)
            + 0.14 * ramp(pitch_conf, 0.35, 0.88)
            + 0.11 * inverse_ramp(high, 0.02, 0.30)
            + 0.09 * ramp(low_peak_hz, 140.0, 420.0)
            + 0.08 * inverse_ramp(max(flatness, entropy), 0.20, 0.62)
        )
        snare_noise_core = clamp01(
            0.24 * broadband_noise
            + 0.22 * ramp(mid + presence, 0.28, 0.86)
            + 0.18 * fast_attack
            + 0.16 * inverse_ramp(tail, 0.03, 0.28)
            + 0.12 * ramp(noise_ms, 45.0, 220.0)
            + 0.08 * inverse_ramp(low_body, 0.18, 0.70)
        )
        # Low-tuned snares and short snare-wire hits can have far less high
        # crack than a textbook snare, but they still differ from tom/conga:
        # they are short, struck early, moderately noisy/raspy, and often sit
        # around low-mid drum body rather than true sub-kick territory.
        low_mid_peak = ramp(low_peak_hz, 145.0, 230.0) * inverse_ramp(low_peak_hz, 360.0, 620.0)
        low_tuned_snare_core = clamp01(
            0.22 * ramp(low_body, 0.48, 0.82)
            + 0.18 * low_mid_peak
            + 0.17 * ramp(broadband_noise, 0.18, 0.48)
            + 0.15 * inverse_ramp(tail, 0.08, 0.30)
            + 0.12 * early_struck
            + 0.10 * single_or_few
            + 0.06 * inverse_ramp(sub, 0.18, 0.52)
        )
        snare_core = max(snare_noise_core, low_tuned_snare_core)
        clap_core = clamp01(
            0.28 * broadband_noise
            + 0.22 * ramp(mid, 0.36, 0.92)
            + 0.18 * ramp(loop_percussive, 0.40, 1.0)
            + 0.16 * inverse_ramp(harmonic, 0.04, 0.42)
            + 0.10 * inverse_ramp(tail, 0.02, 0.22)
            + 0.06 * inverse_ramp(low_body, 0.08, 0.40)
        )
        rim_core = clamp01(
            0.26 * early_struck
            + 0.23 * very_fast_attack
            + 0.18 * ramp(mid + presence, 0.30, 0.92)
            + 0.14 * inverse_ramp(tail, 0.02, 0.20)
            + 0.11 * inverse_ramp(low_body, 0.05, 0.46)
            + 0.08 * ramp(pitch_conf, 0.20, 0.62)
        )
        hat_core = clamp01(
            0.30 * ramp(high_noise, 0.32, 0.88)
            + 0.22 * very_fast_attack
            + 0.18 * inverse_ramp(tail, 0.015, 0.18)
            + 0.13 * ramp(broadband_noise, 0.28, 0.72)
            + 0.10 * inverse_ramp(low_body, 0.01, 0.22)
            + 0.07 * inverse_ramp(noise_ms, 30.0, 190.0)
        )
        cymbal_core = clamp01(
            0.24 * ramp(high_noise, 0.24, 0.86)
            + 0.19 * ramp(max(tail, bright_tail), 0.07, 0.58)
            + 0.20 * metallic_ring
            + 0.16 * ramp(noise_ms, 80.0, 390.0)
            + 0.09 * inverse_ramp(low_body, 0.02, 0.34)
            + 0.07 * inverse_ramp(fa_darkening_db, -8.0, 8.0)
            + 0.05 * ramp(event_count, 1.0, 4.0)
        )
        shaker_core = clamp01(
            0.28 * ramp(high_noise, 0.22, 0.82)
            + 0.22 * ramp(event_count, 2.0, 8.0)
            + 0.18 * ramp(onset_span, 0.20, 0.82)
            + 0.16 * ramp(broadband_noise, 0.26, 0.78)
            + 0.10 * inverse_ramp(low_body, 0.04, 0.35)
            + 0.06 * inverse_ramp(harmonic, 0.04, 0.42)
        )
        scrape_core = clamp01(
            0.25 * ramp(event_count, 2.0, 9.0)
            + 0.22 * ramp(onset_span, 0.28, 0.88)
            + 0.20 * ramp(noisy, 0.20, 0.66)
            + 0.14 * ramp(entropy, 0.45, 0.82)
            + 0.11 * inverse_ramp(harmonic, 0.03, 0.38)
            + 0.08 * ramp(mid + presence, 0.12, 0.70)
        )
        metallic_core = clamp01(
            0.34 * metallic_ring
            + 0.18 * ramp(high_noise, 0.16, 0.72)
            + 0.18 * ramp(pitch_conf, 0.32, 0.78)
            + 0.14 * ramp(harmonic, 0.22, 0.78)
            + 0.10 * fast_attack
            + 0.06 * inverse_ramp(low_body, 0.04, 0.45)
        )

        # Branch scores are gated by the drum anchor but can still express a
        # strong measured branch when old top-family logic called the shape a
        # bass or pitched phrase.  This is the fix for low pitched drum hits.
        branch_gate = clamp01(0.40 + 0.60 * drum_anchor)
        branch_scores = {
            "DrumLoop": clamp01(
                0.35 * sub_drum_loop_source
                + 0.45 * sub_rhythmic_break_loop
                + 0.10 * ramp(event_count, 4.0, 24.0)
                + 0.10 * ramp(onset_span, 0.34, 0.90)
            ),
            "Kick": clamp01(branch_gate * low_kick_core + 0.10 * ramp(sub, 0.72, 0.96)),
            "TomOrConga": clamp01(branch_gate * tom_conga_core),
            "Snare": clamp01(branch_gate * snare_core),
            "Clap": clamp01(branch_gate * clap_core),
            "Hat": clamp01(branch_gate * hat_core),
            "Cymbal": clamp01(branch_gate * cymbal_core),
            "RimOrStick": clamp01(branch_gate * rim_core),
            "ShakerTambourine": clamp01(branch_gate * shaker_core),
            "ScrapeGuiro": clamp01(branch_gate * scrape_core),
            "MetallicPercussion": clamp01(branch_gate * metallic_core),
        }
        branch_scores = apply_source_panel_lifts(
            branch_scores,
            {
                "Kick": subpanel_flat.get("drum_kick_source_score", 0.0),
                "TomOrConga": subpanel_flat.get("drum_tom_conga_source_score", 0.0),
                "Snare": subpanel_flat.get("drum_snare_source_score", 0.0),
                "Clap": subpanel_flat.get("drum_clap_source_score", 0.0),
                "Hat": subpanel_flat.get("drum_closed_hat_source_score", 0.0),
                "Cymbal": subpanel_flat.get("drum_cymbal_source_score", 0.0),
                "RimOrStick": subpanel_flat.get("drum_rim_stick_source_score", 0.0),
                "ShakerTambourine": subpanel_flat.get("drum_shaker_tambourine_source_score", 0.0),
                "ScrapeGuiro": subpanel_flat.get("drum_guiro_scrape_source_score", 0.0),
                "MetallicPercussion": subpanel_flat.get("drum_metallic_percussion_source_score", 0.0),
            },
            scale=0.96,
            floor=0.34,
        )
        if sub_hand_drum_material and sub_hand_drum_membrane >= 0.52 and sub_compact_struck_percussion >= 0.42:
            branch_scores["TomOrConga"] = max(
                branch_scores.get("TomOrConga", 0.0),
                min(0.88, 0.40 + 0.54 * sub_hand_drum_membrane),
            )
        if sub_struck_wood_material and sub_struck_wood >= 0.54 and sub_compact_struck_percussion >= 0.42:
            branch_scores["RimOrStick"] = max(
                branch_scores.get("RimOrStick", 0.0),
                min(0.88, 0.40 + 0.54 * sub_struck_wood),
            )
        if (
            sub_pitched_metal_material
            and sub_pitched_metal_percussion >= 0.52
            and sub_compact_struck_percussion >= 0.38
        ):
            branch_scores["MetallicPercussion"] = max(
                branch_scores.get("MetallicPercussion", 0.0),
                min(0.90, 0.40 + 0.56 * sub_pitched_metal_percussion),
            )
        if sub_rhythmic_break_loop >= 0.62 and sub_drum_loop_source >= 0.30:
            branch_scores["DrumLoop"] = max(
                branch_scores.get("DrumLoop", 0.0),
                min(0.94, 0.58 + 0.30 * sub_rhythmic_break_loop),
            )

        tonal_synth_or_arp_loop_decoy = bool(
            shape_name in {"beat_loop", "repeated_phrase_loop", "bass_phrase", "pitched_phrase", "pitched_phrase_shape"}
            and shape_confidence >= 0.78
            and max(loop_pitched, pitch_conf, f0_voiced) >= 0.78
            and max(sustained_tonal, non_event_tonal) >= 0.82
            and max(loop_percussive, loop_drumlike) <= 0.08
            and sub_onset_pitched >= sub_onset_percussive + 0.20
            and (
                sub_synth_tonal >= 0.58
                or role_value(roles, "pitched_music_phrase") >= 0.78
                or role_value(roles, "pitched_music_loop") >= 0.78
            )
            and not low_sub_kick_exception
        )
        if tonal_synth_or_arp_loop_decoy:
            branch_scores["DrumLoop"] = min(branch_scores["DrumLoop"], 0.34)
            drum_anchor = min(drum_anchor, 0.46)

        clap_source = safe_float(subpanel_flat.get("drum_clap_source_score", 0.0), 0.0)
        snare_source = safe_float(subpanel_flat.get("drum_snare_source_score", 0.0), 0.0)
        rim_source = safe_float(subpanel_flat.get("drum_rim_stick_source_score", 0.0), 0.0)
        hand_clap_burst_signal = bool(
            clap_source >= 0.78
            and clap_source >= snare_source - 0.16
            and clap_source >= rim_source - 0.04
            and safe_float(subpanel_flat.get("role_one_shot_score", 0.0), 0.0) >= 0.80
            and safe_float(subpanel_flat.get("physics_subpanel_noisy_air", subpanel_flat.get("noisy_air", 0.0)), 0.0)
            >= 0.70
            and safe_float(subpanel_flat.get("physics_subpanel_clean_tone", subpanel_flat.get("clean_tone", 0.0)), 0.0)
            <= 0.32
        )
        if hand_clap_burst_signal:
            branch_scores["Clap"] = max(
                branch_scores.get("Clap", 0.0),
                branch_scores.get("Snare", 0.0) + 0.035,
                branch_scores.get("RimOrStick", 0.0) + 0.045,
                clap_source,
            )
            branch_scores["Snare"] = min(branch_scores.get("Snare", 0.0), branch_scores["Clap"] - 0.025)
            branch_scores["RimOrStick"] = min(branch_scores.get("RimOrStick", 0.0), branch_scores["Clap"] - 0.035)
        if (
            snare_source >= 0.74
            and snare_source >= safe_float(subpanel_flat.get("drum_metallic_percussion_source_score", 0.0), 0.0) + 0.08
        ):
            branch_scores["Snare"] = max(
                branch_scores.get("Snare", 0.0),
                branch_scores.get("MetallicPercussion", 0.0) + 0.025,
                min(0.92, 0.42 + 0.52 * snare_source),
            )

        # Calibration note: this decoy guard should be retuned against clean
        # tom/conga and guitar-loop validation sets before changing thresholds.
        # It prevents broad tonal pluck/phrase material from lifting tom/conga
        # or rim/stick branches merely because the attack is fast and low-mid-heavy.
        plucked_phrase_decoy = bool(
            safe_float(subpanel_flat.get("plucked_string_authority_score", 0.0), 0.0) >= 0.40
            and safe_float(subpanel_flat.get("role_phrase_score", 0.0), 0.0)
            >= safe_float(subpanel_flat.get("role_one_shot_score", 0.0), 0.0) + 0.03
            and safe_float(subpanel_flat.get("drum_hit_score", 0.0), 0.0) <= 0.48
            and max(loop_pitched, pitch_conf, f0_voiced) >= 0.52
            and max(loop_percussive, loop_drumlike) <= 0.28
        )
        if plucked_phrase_decoy:
            for branch_name in ("TomOrConga", "Snare", "RimOrStick", "Clap"):
                branch_scores[branch_name] = min(branch_scores[branch_name], 0.46)
            drum_anchor = min(drum_anchor, 0.56)

        # A voiced/formant hit can have a fast attack and enough mid-band noise
        # to look like snare math.  It is not a drum branch when the measured
        # loop frames contain no drumlike/percussive evidence and the shape
        # voter reads a voiced phrase or pitched phrase.  Keep this low-level
        # and source-name blind so the final arbiter does not need to rescue
        # human/synth/reed stabs after the drum layer has already lied.
        voiced_non_drum_phrase_decoy = bool(
            (
                shape_name
                in {"vocal_phrase", "solo_phrase", "pitched_phrase", "sustained_pad", "hit_with_tail", "single_hit"}
                or voiced_role >= 0.55
            )
            and shape_confidence >= 0.60
            and max(pitch_conf, f0_voiced, harmonic) >= 0.55
            and max(loop_percussive, loop_drumlike) <= 0.14
            and max(
                safe_float(subpanel_flat.get("voice_score", 0.0), 0.0),
                safe_float(subpanel_flat.get("human_spoken_voice_score", 0.0), 0.0),
                safe_float(subpanel_flat.get("human_breath_mouth_score", 0.0), 0.0),
                safe_float(subpanel_flat.get("fx_formant_score", 0.0), 0.0),
                safe_float(subpanel_flat.get("synth_tonal_source_score", 0.0), 0.0),
                safe_float(subpanel_flat.get("woodwind_sax_score", 0.0), 0.0),
                1.0 if bool(subpanel_flat.get("tonal_voiced_non_drum_hit_guard")) else 0.0,
            )
            >= 0.50
            and not low_sub_kick_exception
            and not sub_struck_percussion_exception
        )
        if voiced_non_drum_phrase_decoy:
            for branch_name in (
                "TomOrConga",
                "Snare",
                "Clap",
                "Hat",
                "Cymbal",
                "RimOrStick",
                "ShakerTambourine",
                "ScrapeGuiro",
                "MetallicPercussion",
            ):
                branch_scores[branch_name] = min(branch_scores[branch_name], 0.38)
            drum_anchor = min(drum_anchor, 0.42)
        voice_formant_identity = max(sub_voice, sub_human_spoken_voice, sub_voice_choir, sub_formant_fx)
        voice_phrase_or_loop_shape = shape_name in {
            "vocal_phrase",
            "pitched_repetition_phrase",
            "pitched_phrase",
            "pitched_phrase_shape",
            "mixed_instrument_loop",
            "instrument_plus_fx_loop",
            "layered_phrase",
            "solo_phrase",
            "echo_tail_hit",
        }
        voiced_phrase_loop_stand_down = bool(
            voice_phrase_or_loop_shape
            and shape_confidence >= 0.58
            and voice_formant_identity >= 0.66
            and max(sub_voice, sub_human_spoken_voice) >= 0.52
            and max(sub_drum_loop_source, sub_rhythmic_break_loop) <= 0.42
            and max(loop_percussive, loop_drumlike) <= 0.32
            and max(
                safe_float(subpanel_flat.get("role_phrase_score", 0.0), 0.0),
                safe_float(subpanel_flat.get("role_loop_score", 0.0), 0.0),
                voiced_role,
            )
            >= safe_float(subpanel_flat.get("role_one_shot_score", 0.0), 0.0) - 0.06
            and not (percussive_role >= 0.78 and sub_drum_loop_source >= 0.52)
            and not low_sub_kick_exception
            and not sub_struck_percussion_exception
        )
        if voiced_phrase_loop_stand_down:
            for branch_name in (
                "DrumLoop",
                "TomOrConga",
                "Snare",
                "Clap",
                "Hat",
                "Cymbal",
                "RimOrStick",
                "ShakerTambourine",
                "ScrapeGuiro",
                "MetallicPercussion",
            ):
                branch_scores[branch_name] = min(branch_scores[branch_name], 0.34)
            drum_anchor = min(drum_anchor, 0.38)
        if designed_fx_slow_tail_stand_down:
            for branch_name in (
                "Snare",
                "Clap",
                "Hat",
                "Cymbal",
                "RimOrStick",
                "ShakerTambourine",
                "ScrapeGuiro",
                "MetallicPercussion",
            ):
                branch_scores[branch_name] = min(branch_scores[branch_name], 0.52)
            drum_anchor = min(drum_anchor, 0.50)

        # Bias the family anchor upward when any concrete drum branch is very
        # strong.  This prevents single low pitched drum hits from being treated
        # as bass instruments just because they have stable F0.
        strongest_branch = max(branch_scores.values()) if branch_scores else 0.0
        if not designed_fx_slow_tail_stand_down:
            drum_anchor = max(drum_anchor, clamp01(0.80 * strongest_branch + 0.10 * short_hit + 0.10 * fast_attack))
        if fa_status == "ok" and fa_events >= 1.0 and not designed_fx_slow_tail_stand_down:
            drum_anchor = max(drum_anchor, clamp01(0.88 * drum_anchor + 0.12 * early_struck))

        out = {
            "drum_anchor_strength": round(float(drum_anchor), 6),
            "drum_anchor_short_hit": round(float(short_hit), 6),
            "drum_anchor_fast_attack": round(float(fast_attack), 6),
            "drum_anchor_early_energy": round(float(early_energy), 6),
            "drum_anchor_transient": round(float(transient), 6),
            "drum_anchor_not_loop": round(float(not_loop), 6),
            "drum_anchor_percussive_role": round(float(percussive_role), 6),
            "drum_anchor_voiced_role": round(float(voiced_role), 6),
            "drum_anchor_true_vocal_phrase_penalty": round(float(true_vocal_phrase_penalty), 6),
            "drum_anchor_tonal_non_drum_hit_penalty": round(float(tonal_non_drum_hit_penalty), 6),
            "drum_anchor_clean_tonal_solo_penalty": round(float(clean_tonal_solo_penalty), 6),
            "drum_anchor_clean_tonal_solo_phrase": bool(clean_tonal_solo_phrase),
            "drum_anchor_low_sub_kick_exception": bool(low_sub_kick_exception),
            "drum_anchor_tonal_synth_or_arp_loop_decoy": bool(tonal_synth_or_arp_loop_decoy),
            "drum_anchor_designed_fx_slow_tail_stand_down": bool(designed_fx_slow_tail_stand_down),
            "drum_anchor_shape_name": shape_name,
            "drum_anchor_shape_confidence": round(float(shape_confidence), 6),
            "drum_voiced_phrase_loop_stand_down": bool(voiced_phrase_loop_stand_down),
            "drum_voice_formant_identity": round(float(voice_formant_identity), 6),
            "drum_low_body": round(float(low_body), 6),
            "drum_high_noise": round(float(high_noise), 6),
            "drum_broadband_noise": round(float(broadband_noise), 6),
            "drum_snare_noise_core": round(float(snare_noise_core), 6),
            "drum_low_tuned_snare_core": round(float(low_tuned_snare_core), 6),
            "drum_metallic_ring": round(float(metallic_ring), 6),
            "drum_event_count": round(float(event_count), 6),
            "drum_onset_span": round(float(onset_span), 6),
            "drum_low_peak_frequency_hz": round(float(low_peak_hz), 6),
            "drum_first_arrival_status": fa_status,
            "drum_plucked_phrase_decoy_guard": bool(plucked_phrase_decoy),
            "drum_voiced_non_drum_phrase_decoy_guard": bool(voiced_non_drum_phrase_decoy),
            "drum_source_panel_kick": round(
                float(safe_float(subpanel_flat.get("drum_kick_source_score", 0.0), 0.0)), 6
            ),
            "drum_source_panel_snare": round(
                float(safe_float(subpanel_flat.get("drum_snare_source_score", 0.0), 0.0)), 6
            ),
            "drum_source_panel_clap": round(
                float(safe_float(subpanel_flat.get("drum_clap_source_score", 0.0), 0.0)), 6
            ),
            "drum_source_panel_hat": round(
                float(safe_float(subpanel_flat.get("drum_closed_hat_source_score", 0.0), 0.0)), 6
            ),
            "drum_source_panel_cymbal": round(
                float(safe_float(subpanel_flat.get("drum_cymbal_source_score", 0.0), 0.0)), 6
            ),
            "drum_source_panel_tom_conga": round(
                float(safe_float(subpanel_flat.get("drum_tom_conga_source_score", 0.0), 0.0)), 6
            ),
            "drum_source_panel_rim_stick": round(
                float(safe_float(subpanel_flat.get("drum_rim_stick_source_score", 0.0), 0.0)), 6
            ),
            "drum_source_panel_shaker_tambourine": round(
                float(safe_float(subpanel_flat.get("drum_shaker_tambourine_source_score", 0.0), 0.0)), 6
            ),
            "drum_source_panel_guiro_scrape": round(
                float(safe_float(subpanel_flat.get("drum_guiro_scrape_source_score", 0.0), 0.0)), 6
            ),
            "drum_source_panel_metallic_percussion": round(
                float(safe_float(subpanel_flat.get("drum_metallic_percussion_source_score", 0.0), 0.0)), 6
            ),
            "drum_source_panel_compact_struck_tonal_percussion": round(float(sub_compact_struck_percussion), 6),
            "drum_source_panel_hand_drum_membrane": round(float(sub_hand_drum_membrane), 6),
            "drum_source_panel_pitched_metal_percussion": round(float(sub_pitched_metal_percussion), 6),
            "drum_source_panel_struck_wood": round(float(sub_struck_wood), 6),
            "drum_source_panel_drum_loop": round(float(sub_drum_loop_source), 6),
            "drum_source_panel_rhythmic_break_loop": round(float(sub_rhythmic_break_loop), 6),
        }
        for name, value in branch_scores.items():
            out[f"drum_branch_{name}"] = round(float(value), 6)
        return out
