# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Measured pitched-instrument physics layer."""

from __future__ import annotations

from typing import Any

from aaron_sound_sorter.domain.models import SharedAudioFacts
from aaron_sound_sorter.voters.physics_compound_music_layer import PhysicsCompoundMusicLayer
from aaron_sound_sorter.voters.physics_layer_utils import *


class PhysicsInstrumentLayer:
    """Measured pitched-source layer for musical instruments.

    This layer is source-name blind.  It reads stable pitch, harmonicity,
    spectral flatness/entropy, band balance, first-arrival formant/strike
    evidence, and measured role facts.  Its job is broad source family and
    branch evidence, not exact file naming.
    """

    BRANCHES = (
        "Bass",
        "KeysPiano",
        "PluckedString",
        "Woodwinds",
        "Brass",
        "Voice",
        "Synth",
        "Strings",
        "MalletBell",
        "MixedInstrument",
    )

    def __init__(self, compound_layer: PhysicsCompoundMusicLayer | None = None) -> None:
        self.compound_layer = compound_layer or PhysicsCompoundMusicLayer()

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
        branch_scores = {name: float(evidence[f"instrument_branch_{name}"]) for name in self.BRANCHES}
        branch, branch_strength = max(branch_scores.items(), key=lambda item: item[1])
        evidence["instrument_branch_selected"] = branch
        evidence["instrument_branch_selected_confidence"] = round(float(branch_strength), 6)
        for name, value in branch_scores.items():
            evidence[f"instrument_branch_score_{name}"] = round(float(value), 6)
        return branch, branch_strength, evidence

    def evidence(self, facts: SharedAudioFacts) -> dict[str, Any]:
        values = facts.feature_values_by_name or {}
        shape = shape_vote(facts)
        roles = measured_roles(facts)
        role_ev = roles.get("evidence", {}) if isinstance(roles.get("evidence", {}), dict) else {}
        fa = first_arrival(facts)

        is_loop = bool(getattr(facts, "is_loop_like", False))
        is_single = bool(getattr(facts, "is_single_event_like", False))
        is_short = bool(getattr(facts, "is_short_hit_like", False))
        is_long = bool(getattr(facts, "is_long", False))

        pitch_conf = number(values, "pitch_confidence", number(shape, "pitch_confidence", 0.0))
        f0_voiced = number(values, "f0_voiced_ratio", number(shape, "f0_voiced_ratio", 0.0))
        f0_hz = number(values, "f0_median_hz", number(shape, "f0_median_hz", 0.0))
        low_peak_hz = number(values, "low_peak_frequency_hz", 0.0)
        loop_pitched = number(values, "loop_pitched_event_ratio", number(shape, "pitched_event_ratio", 0.0))
        sustained_tonal = number(
            values, "loop_sustained_tonal_frame_ratio", number(shape, "sustained_tonal_frame_ratio", 0.0)
        )
        non_event_tonal = number(values, "loop_non_event_tonal_ratio", number(shape, "non_event_tonal_ratio", 0.0))
        harmonic = number(values, "harmonic_energy_ratio", number(shape, "harmonic_energy_ratio", 0.0))
        inharmonicity = number(values, "inharmonicity", number(shape, "inharmonicity", 0.0))
        fundamental = number(values, "fundamental_dominance_ratio", number(shape, "fundamental_dominance_ratio", 0.0))
        flatness = number(values, "spectral_flatness_mean", number(shape, "spectral_flatness_mean", 0.0))
        entropy = number(values, "spectral_entropy_mean", number(shape, "spectral_entropy_mean", 0.0))
        peak_stability = number(values, "spectral_peak_stability", 0.0)
        formant_spacing = number(values, "formant_like_peak_spacing", 0.0)
        number(values, "overtone_slope", 0.0)

        attack = number(values, "attack_rise_time_norm", number(shape, "attack_rise_time_norm", 1.0))
        number(values, "temporal_centroid_ratio", number(shape, "temporal_centroid_ratio", 0.5))
        tail = number(values, "tail_energy_ratio", number(shape, "tail_ratio", 0.0))
        event_count = max(
            expm1_value(number(values, "log_transient_count", 0.0)),
            number(shape, "onset_count", 0.0),
        )
        onset_span = number(values, "onset_span_ratio", number(shape, "onset_span_ratio", 0.0))
        zcr = number(values, "zcr_mean", number(shape, "zcr_mean", 0.0))

        sub = number(values, "sub_bass_ratio_lt_150hz", 0.0)
        bass = number(values, "bass_ratio_150_500hz", 0.0)
        mid = number(values, "mid_ratio_500_2000hz", 0.0)
        presence = number(values, "presence_ratio_2000_8000hz", 0.0)
        air = number(values, "air_ratio_gt_8000hz", 0.0)
        high = presence + air
        low_total = sub + bass
        event_low = number(values, "loop_mean_event_low_ratio", number(shape, "low_event_ratio", 0.0))
        event_mid = number(values, "loop_mean_event_mid_ratio", number(shape, "mid_event_ratio", mid))
        event_high = number(values, "loop_mean_event_high_ratio", number(shape, "high_event_ratio", 0.0))
        percussive_loop = number(values, "loop_percussive_event_ratio", number(shape, "percussive_event_ratio", 0.0))
        drumlike_loop = number(values, "loop_drumlike_frame_ratio", number(shape, "drumlike_frame_ratio", 0.0))
        body_flatness = number(values, "body_flatness", flatness)
        number(values, "body_entropy", entropy)
        body_noise = number(values, "body_noise_ratio", flatness)
        tail_noise = number(values, "tail_noise_ratio", body_noise)
        subpanel_block = facts.evidence.get("physics_subpanels", {}) if isinstance(facts.evidence, dict) else {}
        subpanel_flat = subpanel_block.get("flat", {}) if isinstance(subpanel_block, dict) else {}
        if not isinstance(subpanel_flat, dict):
            subpanel_flat = {}
        sub_role_loop = safe_float(subpanel_flat.get("role_loop_score", 0.0), 0.0)
        sub_role_one_shot = safe_float(subpanel_flat.get("role_one_shot_score", 0.0), 0.0)
        sub_role_phrase = safe_float(subpanel_flat.get("role_phrase_score", 0.0), 0.0)
        sub_plucked = safe_float(subpanel_flat.get("plucked_string_score", 0.0), 0.0)
        sub_reed = safe_float(subpanel_flat.get("reed_wind_score", 0.0), 0.0)
        sub_keys = safe_float(subpanel_flat.get("struck_keys_score", 0.0), 0.0)
        sub_bowed = safe_float(subpanel_flat.get("bowed_string_score", 0.0), 0.0)
        sub_voice = safe_float(subpanel_flat.get("voice_score", 0.0), 0.0)
        sub_human_spoken = safe_float(subpanel_flat.get("human_spoken_voice_score", 0.0), 0.0)
        sub_human_breath = safe_float(subpanel_flat.get("human_breath_mouth_score", 0.0), 0.0)
        sub_fx_formant = safe_float(subpanel_flat.get("fx_formant_score", 0.0), 0.0)
        sub_drum_hit = safe_float(subpanel_flat.get("drum_hit_score", 0.0), 0.0)
        sub_metallic = safe_float(subpanel_flat.get("metallic_noise_score", 0.0), 0.0)
        sub_scrape = safe_float(subpanel_flat.get("scrape_rasp_score", 0.0), 0.0)
        sub_fx_motion = safe_float(subpanel_flat.get("fx_motion_score", 0.0), 0.0)
        sub_texture_bed = safe_float(subpanel_flat.get("texture_bed_score", 0.0), 0.0)
        sub_low_end = safe_float(subpanel_flat.get("low_end_source_score", 0.0), 0.0)
        sub_plucked_authority = safe_float(
            subpanel_flat.get("plucked_string_authority_score", sub_plucked), sub_plucked
        )
        sub_reed_authority = safe_float(subpanel_flat.get("reed_wind_authority_score", sub_reed), sub_reed)
        sub_keys_authority = safe_float(subpanel_flat.get("struck_keys_authority_score", sub_keys), sub_keys)
        sub_synth_tonal = safe_float(subpanel_flat.get("synth_tonal_source_score", 0.0), 0.0)
        sub_drum_loop_source = safe_float(subpanel_flat.get("drum_loop_source_score", 0.0), 0.0)
        sub_rhythmic_break_loop = safe_float(subpanel_flat.get("rhythmic_break_loop_score", 0.0), 0.0)
        sub_fx_transition_authority = safe_float(
            subpanel_flat.get("fx_transition_authority_score", sub_fx_motion), sub_fx_motion
        )

        pitch_role = max(role_value(roles, "pitched_music_phrase"), role_value(roles, "pitched_music_loop"))
        bass_role = role_value(roles, "bass_loop")
        vocal_role = max(role_value(roles, "vocal_music_phrase"), role_value(roles, "voiced_one_shot"))
        drum_role = max(
            role_value(roles, "percussive_one_shot"),
            role_value(roles, "bright_drum_loop"),
            role_value(roles, "percussive_drum_loop"),
            role_value(roles, "low_rhythmic_drum_loop"),
        )
        formant_voice = safe_float(role_ev.get("formant_light_voice_identity", 0.0), 0.0)
        shape_name = str(shape.get("primary_shape", ""))
        shape_confidence = number(shape, "confidence", 0.0)
        short_bright_percussion_hit = bool(
            (shape_name in {"single_hit", "hit_with_tail"} or not shape_name)
            and (shape_confidence >= 0.58 or not shape_name)
            and self._measured_short_event_like(facts, event_count)
            and role_value(roles, "vocal_music_phrase") < 0.35
            and (
                (drum_role >= 0.22 and attack <= 0.055) or (event_high >= 0.82 and event_mid <= 0.18 and attack <= 0.16)
            )
        )

        fa_status = str(fa.get("status", "missing"))
        fa_cep = safe_float(fa.get("cepstral_pitch_period_coherence", 0.0), 0.0)
        fa_conical = safe_float(fa.get("first_arrival_conical_balance", 0.0), 0.0)
        fa_presence = safe_float(fa.get("first_arrival_presence_contrast_db", 0.0), 0.0)
        fa_stochastic = safe_float(fa.get("stochastic_modulation_coherence", 0.0), 0.0)
        fa_formant_center = safe_float(fa.get("formant_center_mean_hz", 0.0), 0.0)
        fa_formant_std = safe_float(fa.get("formant_center_std_hz", 9999.0), 9999.0)
        fa_formant_stability = safe_float(fa.get("formant_stability_score", 0.0), 0.0)
        strike_flatness = safe_float(fa.get("strike_flatness", 0.0), 0.0)
        settle_flatness = safe_float(fa.get("settle_flatness", 0.0), 0.0)
        safe_float(fa.get("strike_high_ratio", 0.0), 0.0)
        safe_float(fa.get("settle_high_ratio", 0.0), 0.0)
        darkening_db = safe_float(fa.get("spectral_darkening_db", 0.0), 0.0)
        strike_inharmonic = safe_float(fa.get("strike_inharmonic_energy_ratio", 0.0), 0.0)
        settle_inharmonic = safe_float(fa.get("settle_inharmonic_energy_ratio", 0.0), 0.0)

        pitch_strength = max(
            pitch_role,
            0.35 * ramp(pitch_conf, 0.35, 0.88)
            + 0.25 * ramp(f0_voiced, 0.35, 0.92)
            + 0.20 * ramp(loop_pitched, 0.55, 1.0)
            + 0.20 * ramp(max(sustained_tonal, non_event_tonal), 0.48, 1.0),
        )
        tonal_clean = clamp01(
            0.28 * inverse_ramp(flatness, 0.035, 0.42)
            + 0.22 * inverse_ramp(entropy, 0.28, 0.72)
            + 0.20 * ramp(max(harmonic, fundamental, fa_cep), 0.18, 0.86)
            + 0.18 * ramp(max(sustained_tonal, non_event_tonal), 0.55, 1.0)
            + 0.12 * ramp(peak_stability, 0.08, 0.48)
        )
        phrase_structure = max(
            1.0 if is_loop or is_long else 0.0,
            ramp(event_count, 2.0, 12.0),
            ramp(onset_span, 0.32, 0.82),
            sub_role_loop,
            0.82 * sub_role_phrase,
        )
        non_drum_phrase = clamp01(inverse_ramp(max(percussive_loop, drumlike_loop, drum_role), 0.04, 0.42))
        not_short_one_shot = 0.0 if is_short else max(0.45 if is_single else 0.0, 1.0 if is_loop or is_long else 0.0)
        instrument_anchor = clamp01(
            0.30 * pitch_strength
            + 0.24 * tonal_clean
            + 0.18 * phrase_structure
            + 0.14 * non_drum_phrase
            + 0.08 * ramp(max(fa_cep, harmonic, fundamental), 0.20, 0.75)
            + 0.06 * not_short_one_shot
        )
        # Short vocal shots are still instruments in this library when voice
        # evidence is strong, but do not let random short percussive attacks use
        # this layer as an instrument steal.
        if vocal_role >= 0.60:
            instrument_anchor = max(
                instrument_anchor,
                clamp01(
                    0.76 * vocal_role
                    + 0.14 * ramp(max(flatness, body_flatness), 0.24, 0.48)
                    + 0.10 * ramp(high, 0.08, 0.34)
                ),
            )
        if is_short and vocal_role < 0.58:
            instrument_anchor = min(instrument_anchor, max(0.45, pitch_strength * 0.65))
        if short_bright_percussion_hit:
            instrument_anchor = min(instrument_anchor, 0.38)

        low_bass_energy = clamp01(
            0.50 * ramp(sub, 0.18, 0.72) + 0.30 * ramp(event_low, 0.55, 0.96) + 0.20 * inverse_ramp(high, 0.002, 0.10)
        )
        bass_core = clamp01(
            0.30 * bass_role
            + 0.24 * low_bass_energy
            + 0.18 * ramp(pitch_conf, 0.50, 0.93)
            + 0.12 * inverse_ramp(f0_hz, 35.0, 210.0)
            + 0.10 * ramp(fundamental, 0.35, 0.90)
            + 0.06 * inverse_ramp(max(percussive_loop, drumlike_loop), 0.02, 0.28)
        )
        piano_hammer = clamp01(
            0.22 * ramp(max(strike_flatness - settle_flatness, darkening_db / 10.0), 0.08, 0.55)
            + 0.22 * ramp(max(strike_inharmonic, fa_conical), 0.32, 0.82)
            + 0.18 * inverse_ramp(flatness, 0.004, 0.12)
            + 0.16 * ramp(mid, 0.35, 0.90)
            + 0.14 * ramp(pitch_conf, 0.50, 0.95)
            + 0.08 * inverse_ramp(high, 0.005, 0.16)
        )
        keys_core = clamp01(
            0.32 * piano_hammer
            + 0.24 * ramp(pitch_strength, 0.55, 1.0)
            + 0.18 * inverse_ramp(flatness, 0.006, 0.16)
            + 0.14 * ramp(mid, 0.30, 0.92)
            + 0.08 * ramp(event_count, 3.0, 18.0)
            + 0.04 * inverse_ramp(max(percussive_loop, drumlike_loop), 0.02, 0.28)
        )
        reed_air = clamp01(
            0.22 * ramp(max(fa_stochastic, body_noise, tail_noise), 0.20, 0.72)
            + 0.20 * ramp(fa_presence, 18.0, 34.0)
            + 0.18 * ramp(max(fa_formant_stability, formant_spacing), 0.30, 0.90)
            + 0.15 * ramp(max(presence, high), 0.025, 0.22)
            + 0.15 * ramp(pitch_strength, 0.58, 1.0)
            + 0.10 * inverse_ramp(low_total, 0.22, 0.92)
        )
        reed_core = clamp01(
            0.46 * reed_air
            + 0.20 * ramp(max(fa_cep, harmonic), 0.24, 0.82)
            + 0.16 * ramp(max(settle_flatness, flatness), 0.12, 0.36)
            + 0.10 * ramp(max(mid, bass), 0.22, 0.72)
            + 0.08 * inverse_ramp(max(percussive_loop, drumlike_loop), 0.03, 0.30)
        )
        clean_tonal_reed_solo_signal = bool(
            shape_name == "solo_phrase"
            and shape_confidence >= 0.84
            and 1.0 <= event_count <= 8.0
            and pitch_conf >= 0.50
            and f0_voiced >= 0.84
            and loop_pitched >= 0.90
            and max(sustained_tonal, non_event_tonal) >= 0.84
            and 0.42 <= event_mid <= 0.96
            and 0.035 <= event_high <= 0.24
            and event_low <= 0.32
            and flatness <= 0.09
            and entropy <= 0.46
            and fa_status == "ok"
            and fa_presence >= 16.0
            and 0.0 < fa_formant_center <= 1900.0
            and fa_formant_std <= 190.0
            and max(percussive_loop, drumlike_loop) <= 0.08
        )
        dark_low_mid_reed_loop_signal = bool(
            shape_name in {"bass_phrase", "pitched_phrase", "solo_phrase", "sustained_pad"}
            and shape_confidence >= 0.82
            and 4.0 <= event_count <= 12.0
            and pitch_conf >= 0.72
            and f0_voiced >= 0.84
            and loop_pitched >= 0.90
            and max(sustained_tonal, non_event_tonal) >= 0.84
            and 0.42 <= event_low <= 0.82
            and 0.18 <= event_mid <= 0.58
            and event_high <= 0.040
            and flatness <= 0.045
            and entropy <= 0.42
            and fa_status == "ok"
            and fa_presence >= 28.0
            and 760.0 <= fa_formant_center <= 1500.0
            and 0.0 < fa_formant_std <= 140.0
            and fa_stochastic >= 0.42
            and max(percussive_loop, drumlike_loop) <= 0.08
        )
        if clean_tonal_reed_solo_signal:
            reed_core = max(
                reed_core,
                clamp01(
                    0.30 * ramp(fa_presence, 16.0, 30.0)
                    + 0.24 * ramp(f0_voiced, 0.84, 1.0)
                    + 0.18 * ramp(loop_pitched, 0.90, 1.0)
                    + 0.14 * inverse_ramp(flatness, 0.02, 0.09)
                    + 0.14 * inverse_ramp(fa_formant_std, 40.0, 190.0)
                ),
            )
        if dark_low_mid_reed_loop_signal:
            reed_core = max(
                reed_core,
                clamp01(
                    0.22 * ramp(fa_presence, 28.0, 36.0)
                    + 0.20 * ramp(fa_stochastic, 0.42, 0.72)
                    + 0.18 * inverse_ramp(fa_formant_std, 50.0, 140.0)
                    + 0.16 * ramp(f0_voiced, 0.84, 1.0)
                    + 0.14 * inverse_ramp(event_high, 0.0, 0.040)
                    + 0.10 * inverse_ramp(flatness, 0.0, 0.045)
                ),
            )
        if vocal_role >= 0.58 and max(flatness, body_flatness) >= 0.34:
            reed_core = min(reed_core, 0.50 + 0.12 * inverse_ramp(vocal_role, 0.58, 0.96))
        brass_core = clamp01(
            0.26 * ramp(pitch_strength, 0.58, 1.0)
            + 0.20 * ramp(max(harmonic, fa_conical), 0.28, 0.86)
            + 0.18 * ramp(mid + presence, 0.35, 0.96)
            + 0.14 * ramp(fa_presence, 20.0, 36.0)
            + 0.12 * inverse_ramp(flatness, 0.04, 0.30)
            + 0.10 * ramp(f0_hz, 180.0, 760.0)
        )
        voice_core = clamp01(
            0.42 * vocal_role
            + 0.18 * formant_voice
            + 0.15 * ramp(max(flatness, body_flatness), 0.24, 0.48)
            + 0.12 * ramp(max(high, event_high), 0.10, 0.34)
            + 0.08 * ramp(max(zcr, body_noise), 0.12, 0.34)
            + 0.05 * ramp(f0_voiced, 0.45, 0.94)
        )
        processed_vocal_shot_signal = bool(
            shape_name == "vocal_phrase"
            and shape_confidence >= 0.82
            and formant_voice >= 0.62
            and f0_voiced >= 0.68
            and max(flatness, body_flatness, body_noise) >= 0.20
            and max(percussive_loop, drumlike_loop) <= 0.14
        )
        if processed_vocal_shot_signal:
            voice_core = max(
                voice_core,
                clamp01(
                    0.38 * ramp(formant_voice, 0.62, 0.95)
                    + 0.24 * ramp(f0_voiced, 0.68, 1.0)
                    + 0.20 * ramp(max(flatness, body_flatness, body_noise), 0.20, 0.46)
                    + 0.18 * ramp(max(high, event_high), 0.08, 0.30)
                ),
            )
            reed_core = min(reed_core, max(0.42, voice_core - 0.08))
            brass_core = min(brass_core, max(0.42, voice_core - 0.10))
        if clean_tonal_reed_solo_signal or dark_low_mid_reed_loop_signal:
            voice_core = min(voice_core, max(0.42, reed_core - 0.18))
        rap_voice_texture = clamp01(
            0.22 * ramp(f0_voiced, 0.70, 0.94)
            + 0.20 * ramp(max(flatness, body_flatness), 0.24, 0.50)
            + 0.18 * ramp(fa_formant_std, 220.0, 900.0)
            + 0.15 * ramp(fa_stochastic, 0.36, 0.72)
            + 0.14 * ramp(fa_presence, 18.0, 34.0)
            + 0.11 * ramp(event_count, 12.0, 44.0)
        )
        if (
            pitch_strength >= 0.62
            and max(percussive_loop, drumlike_loop) <= 0.18
            and shape_name in {"bass_phrase", "vocal_phrase", "pitched_phrase", "sustained_pad"}
        ):
            voice_core = max(voice_core, rap_voice_texture)
        if short_bright_percussion_hit:
            voice_core = min(voice_core, 0.20)
            reed_core = min(reed_core, 0.24)
            brass_core = min(brass_core, 0.26)
        synth_core = clamp01(
            0.24 * ramp(pitch_strength, 0.55, 1.0)
            + 0.22 * inverse_ramp(flatness, 0.002, 0.08)
            + 0.16 * ramp(fundamental, 0.45, 0.96)
            + 0.14 * ramp(max(sustained_tonal, non_event_tonal), 0.70, 1.0)
            + 0.12 * inverse_ramp(max(presence, air), 0.0, 0.08)
            + 0.12 * ramp(event_count, 4.0, 32.0)
        )
        clean_synth_signature = clamp01(
            0.28 * ramp(pitch_strength, 0.70, 1.0)
            + 0.22 * inverse_ramp(flatness, 0.004, 0.16)
            + 0.16 * inverse_ramp(low_total, 0.02, 0.40)
            + 0.14 * ramp(mid, 0.42, 0.95)
            + 0.10 * inverse_ramp(body_noise, 0.04, 0.30)
            + 0.06 * ramp(f0_hz, 260.0, 900.0)
            + 0.04 * inverse_ramp(max(formant_voice, vocal_role), 0.02, 0.28)
        )
        synth_core = max(synth_core, clean_synth_signature)
        if clean_synth_signature >= 0.62 and vocal_role < 0.25:
            reed_core = min(reed_core, max(0.46, clean_synth_signature - 0.08))
            brass_core = min(brass_core, max(0.50, clean_synth_signature - 0.04))
        strings_core = clamp01(
            0.24 * ramp(pitch_strength, 0.58, 1.0)
            + 0.18 * ramp(max(high, presence), 0.10, 0.32)
            + 0.18 * ramp(max(harmonic, fa_cep), 0.28, 0.82)
            + 0.15 * ramp(max(tail, sustained_tonal), 0.55, 1.0)
            + 0.13 * ramp(body_noise, 0.18, 0.42)
            + 0.12 * inverse_ramp(max(formant_voice, percussive_loop, drumlike_loop), 0.04, 0.34)
        )
        plucked_core = clamp01(
            0.22 * ramp(pitch_strength, 0.55, 1.0)
            + 0.20 * inverse_ramp(attack, 0.01, 0.18)
            + 0.18 * inverse_ramp(tail, 0.10, 0.72)
            + 0.14 * ramp(max(mid, presence), 0.22, 0.80)
            + 0.13 * ramp(strike_inharmonic, 0.32, 0.86)
            + 0.13 * inverse_ramp(max(percussive_loop, drumlike_loop), 0.02, 0.32)
        )
        mallet_bell_core = clamp01(
            0.26 * ramp(max(inharmonicity, strike_inharmonic), 0.22, 0.82)
            + 0.22 * ramp(max(high, presence, event_high), 0.12, 0.45)
            + 0.18 * ramp(pitch_strength, 0.45, 0.95)
            + 0.14 * inverse_ramp(tail, 0.10, 0.65)
            + 0.12 * ramp(fa_conical, 0.20, 0.82)
            + 0.08 * inverse_ramp(low_total, 0.04, 0.42)
        )
        # Low-level shared subpanels are broad witnesses, not final routers.
        # Authority scores add separation between similar pitched panels before
        # any final arbiter logic sees the candidates.  This keeps the heavy
        # lifting in PhysicsVoter while staying source-name blind.
        subpanel_non_drum_phrase = bool(max(percussive_loop, drumlike_loop, sub_drum_hit, sub_drum_loop_source) <= 0.48)
        subpanel_pitched_phrase = bool(
            pitch_strength >= 0.42 or max(loop_pitched, sustained_tonal, non_event_tonal) >= 0.60
        )
        if sub_low_end >= 0.64 and low_total >= 0.30 and high <= 0.20 and sub_plucked_authority < 0.48:
            bass_core = max(bass_core, min(0.94, 0.56 + 0.40 * sub_low_end))
        if (
            sub_keys_authority >= 0.48
            and subpanel_pitched_phrase
            and sub_reed_authority < sub_keys_authority + 0.10
            and sub_plucked_authority < sub_keys_authority + 0.12
        ):
            keys_core = max(keys_core, min(0.94, 0.50 + 0.48 * sub_keys_authority))
        if (
            sub_reed_authority >= 0.50
            and subpanel_pitched_phrase
            and sub_plucked_authority < sub_reed_authority + 0.08
            and sub_voice < sub_reed + 0.10
        ):
            reed_core = max(reed_core, min(0.95, 0.48 + 0.48 * sub_reed_authority))
        if (
            sub_plucked_authority >= 0.36
            and subpanel_pitched_phrase
            and subpanel_non_drum_phrase
            and sub_reed_authority <= sub_plucked_authority + 0.11
            and sub_keys_authority <= sub_plucked_authority + 0.16
            and sub_metallic < sub_plucked + 0.12
        ):
            plucked_core = max(plucked_core, min(0.96, 0.55 + 0.48 * sub_plucked_authority))
        if sub_synth_tonal >= 0.58 and sub_plucked_authority < 0.42 and sub_reed_authority < 0.42:
            synth_core = max(synth_core, min(0.94, 0.50 + 0.42 * sub_synth_tonal))
        if (
            sub_bowed >= 0.62
            and subpanel_pitched_phrase
            and sub_plucked < sub_bowed + 0.12
            and sub_reed < sub_bowed + 0.10
        ):
            strings_core = max(strings_core, min(0.92, 0.48 + 0.42 * sub_bowed))
        human_voice_texture = clamp01(
            max(
                sub_voice,
                sub_human_spoken,
                0.72 * sub_human_spoken + 0.28 * max(sub_fx_formant, sub_human_breath),
                min(sub_human_spoken, max(sub_fx_formant, sub_human_breath) + 0.12),
            )
        )
        non_voice_tonal_loop_voice_decoy = bool(
            shape_name
            in {
                "vocal_phrase",
                "pitched_phrase",
                "mixed_instrument_loop",
                "compound_musical_loop",
                "repeated_phrase_loop",
                "sustained_pad",
                "bass_phrase",
                "",
            }
            and event_count >= 6.0
            and max(loop_pitched, sustained_tonal, non_event_tonal) >= 0.72
            and max(percussive_loop, drumlike_loop, drum_role) <= 0.30
            and (
                sub_synth_tonal >= 0.58
                or sub_metallic >= 0.55
                or sub_keys_authority >= 0.48
                or sub_reed_authority >= 0.58
            )
            and not (
                vocal_role >= 0.86
                and sub_voice >= 0.76
                and sub_human_spoken >= 0.76
                and sub_metallic < 0.52
                and sub_synth_tonal < 0.54
            )
        )
        human_voice_phrase_signal = bool(
            human_voice_texture >= 0.66
            and pitch_strength >= 0.55
            and f0_voiced >= 0.50
            and phrase_structure >= 0.62
            and max(percussive_loop, drumlike_loop, drum_role) <= 0.26
            and sub_reed_authority <= human_voice_texture - 0.08
            and sub_plucked_authority <= human_voice_texture + 0.04
            and not non_voice_tonal_loop_voice_decoy
            and shape_name
            in {
                "vocal_phrase",
                "pitched_phrase",
                "mixed_instrument_loop",
                "compound_musical_loop",
                "repeated_phrase_loop",
                "sustained_pad",
                "bass_phrase",
                "",
            }
        )
        if sub_voice >= 0.66 and vocal_role >= 0.42 and sub_reed < sub_voice + 0.08:
            voice_core = max(voice_core, min(0.94, 0.50 + 0.42 * sub_voice))
        if human_voice_phrase_signal:
            voice_core = max(voice_core, min(0.94, 0.50 + 0.42 * human_voice_texture))
            reed_core = min(reed_core, max(0.42, voice_core - 0.06))
            brass_core = min(brass_core, max(0.42, voice_core - 0.08))
        if non_voice_tonal_loop_voice_decoy:
            decoy_floor = max(synth_core, mallet_bell_core, keys_core, strings_core, brass_core, reed_core)
            voice_core = min(voice_core, max(0.42, decoy_floor - 0.035))
            if sub_synth_tonal >= 0.56:
                synth_core = max(synth_core, min(0.92, 0.50 + 0.42 * sub_synth_tonal))
            if sub_metallic >= 0.50:
                mallet_bell_core = max(mallet_bell_core, min(0.92, 0.46 + 0.42 * sub_metallic))
        if sub_metallic >= 0.68 and high >= 0.14:
            mallet_bell_core = max(mallet_bell_core, min(0.92, 0.48 + 0.40 * sub_metallic))
        # Mixed/compound musical loops: strong instrument anchor but no single
        # identity branch dominates, or several branches are plausible.
        shape_name = str(shape.get("primary_shape", ""))
        # Bass identity must be a measured low-end source, not merely a low-heavy
        # midrange instrument. Guitar loops often carry strong low-band body or
        # autocorrelation octave errors around 180-260 Hz; requiring either a
        # true sub/fundamental region or measured bass-role support keeps those
        # guitar phrases from becoming Bass/808.
        low_pitch_bass_identity = bool(
            (0.0 < f0_hz <= 155.0 and low_total >= 0.36)
            or (155.0 < f0_hz <= 260.0 and low_total >= 0.55 and mid <= 0.14 and bass_role >= 0.30)
            or (sub >= 0.45 and 0.0 < low_peak_hz <= 135.0 and f0_voiced <= 0.12 and mid <= 0.24 and high <= 0.08)
        )
        high_register_low_band_conflict = bool(f0_hz >= 420.0 and f0_voiced <= 0.36 and event_count >= 6.0)
        clean_bass_phrase = (
            shape_name == "bass_phrase"
            and pitch_conf >= 0.70
            and loop_pitched >= 0.75
            and event_low >= 0.70
            and max(percussive_loop, drumlike_loop) <= 0.20
            and low_pitch_bass_identity
            and not high_register_low_band_conflict
        )
        wet_woodwind_signal = (
            shape_name in {"vocal_phrase", "pitched_phrase", "sustained_pad"}
            and event_count <= 32.0
            and pitch_conf >= 0.74
            and loop_pitched >= 0.86
            and harmonic >= 0.45
            and presence >= 0.28
            and flatness <= 0.34
            and max(percussive_loop, drumlike_loop) <= 0.20
        )
        woodwind_source_signal = bool(
            wet_woodwind_signal or clean_tonal_reed_solo_signal or dark_low_mid_reed_loop_signal
        )

        # Instrument source panels. These are conservative measured sub-branch
        # witnesses, not final routers. They give the broad branch scorer more
        # structure so a new category repair does not steal unrelated samples.
        stereo_width = number(values, "stereo_width", number(shape, "stereo_width", 0.0))
        slow_attack = ramp(attack, 0.08, 0.36)
        fast_attack = inverse_ramp(attack, 0.008, 0.12)
        event_repetition = max(ramp(event_count, 4.0, 18.0), ramp(onset_span, 0.35, 0.86))
        loop_role_score = safe_float(subpanel_flat.get("role_loop_score", 0.0), 0.0)
        phrase_role_score = safe_float(subpanel_flat.get("role_phrase_score", 0.0), 0.0)
        loop_periodicity = max(
            safe_float(subpanel_flat.get("loop_onset_periodicity", 0.0), 0.0),
            safe_float(subpanel_flat.get("loop_pulse_clarity", 0.0), 0.0),
            safe_float(subpanel_flat.get("onset_true_repetition_likelihood", 0.0), 0.0),
        )
        arp_pad_or_bass_decoy = bool(
            shape_name in {"bass_phrase", "sustained_pad"}
            and phrase_role_score >= loop_role_score + 0.08
            and max(sustained_tonal, non_event_tonal) >= 0.86
            and max(percussive_loop, drumlike_loop) <= 0.12
            and (low_total >= 0.64 or event_high <= 0.045)
        )
        true_arp_event_motion = min(
            event_repetition,
            max(loop_periodicity, ramp(event_count, 6.0, 18.0)),
            max(
                ramp(event_mid + event_high, 0.28, 0.86),
                inverse_ramp(max(sustained_tonal, non_event_tonal), 0.52, 0.94),
            ),
        )
        low_register = inverse_ramp(f0_hz, 70.0, 260.0) if f0_hz > 0.0 else ramp(low_total, 0.40, 0.86)
        mid_register = ramp(f0_hz, 120.0, 520.0) if f0_hz > 0.0 else ramp(mid, 0.24, 0.82)
        high_register = ramp(f0_hz, 420.0, 1100.0) if f0_hz > 0.0 else ramp(high, 0.12, 0.46)
        stable_sustain = clamp01(
            0.42 * ramp(max(sustained_tonal, non_event_tonal), 0.68, 1.0)
            + 0.24 * ramp(f0_voiced, 0.62, 1.0)
            + 0.18 * ramp(max(harmonic, fa_cep), 0.24, 0.88)
            + 0.16 * inverse_ramp(max(percussive_loop, drumlike_loop), 0.03, 0.28)
        )
        resonant_strike = clamp01(
            0.32 * fast_attack
            + 0.26 * ramp(max(strike_inharmonic, fa_conical, inharmonicity), 0.22, 0.78)
            + 0.22 * ramp(pitch_strength, 0.45, 0.95)
            + 0.20 * inverse_ramp(max(percussive_loop, drumlike_loop), 0.04, 0.36)
        )
        noisy_breath_edge = clamp01(
            0.32 * ramp(max(fa_stochastic, body_noise, tail_noise), 0.20, 0.70)
            + 0.22 * ramp(high, 0.030, 0.34)
            + 0.20 * ramp(fa_presence, 14.0, 34.0)
            + 0.14 * inverse_ramp(flatness, 0.02, 0.42)
            + 0.12 * ramp(pitch_strength, 0.52, 1.0)
        )

        bass_panels = {
            "808Sub": clamp01(
                0.34 * ramp(sub, 0.45, 0.90)
                + 0.22 * inverse_ramp(f0_hz, 35.0, 115.0)
                + 0.18 * inverse_ramp(high, 0.002, 0.10)
                + 0.14 * ramp(max(tail, fundamental), 0.30, 0.95)
                + 0.12 * inverse_ramp(max(body_noise, flatness), 0.02, 0.24)
            ),
            "SynthBass": clamp01(
                0.28 * ramp(low_total, 0.45, 0.92)
                + 0.22 * inverse_ramp(flatness, 0.004, 0.16)
                + 0.20 * ramp(max(fundamental, harmonic), 0.32, 0.94)
                + 0.18 * inverse_ramp(high, 0.004, 0.16)
                + 0.12 * ramp(max(sustained_tonal, loop_pitched), 0.58, 1.0)
            ),
            "ElectricBass": clamp01(
                0.24 * ramp(low_total, 0.34, 0.86)
                + 0.20 * fast_attack
                + 0.18 * ramp(mid, 0.10, 0.46)
                + 0.16 * ramp(max(harmonic, fa_cep), 0.30, 0.88)
                + 0.12 * ramp(body_noise, 0.04, 0.28)
                + 0.10 * inverse_ramp(high, 0.02, 0.24)
            ),
            "UprightBass": clamp01(
                0.25 * ramp(low_total, 0.34, 0.86)
                + 0.22 * slow_attack
                + 0.18 * ramp(max(harmonic, fa_cep), 0.24, 0.82)
                + 0.16 * ramp(body_noise, 0.06, 0.32)
                + 0.11 * inverse_ramp(high, 0.02, 0.20)
                + 0.08 * inverse_ramp(flatness, 0.04, 0.28)
            ),
            "BassLoop": clamp01(0.62 * bass_core + 0.22 * event_repetition + 0.16 * ramp(loop_pitched, 0.70, 1.0)),
        }
        keys_panels = {
            "AcousticPiano": clamp01(0.72 * piano_hammer + 0.16 * resonant_strike + 0.12 * ramp(mid, 0.32, 0.82)),
            "ElectricPiano": clamp01(
                0.30 * keys_core
                + 0.20 * inverse_ramp(flatness, 0.006, 0.12)
                + 0.18 * ramp(mid, 0.34, 0.88)
                + 0.14 * ramp(max(sustained_tonal, loop_pitched), 0.62, 1.0)
                + 0.10 * inverse_ramp(high, 0.004, 0.14)
                + 0.08 * event_repetition
            ),
            "Organ": clamp01(
                0.32 * stable_sustain
                + 0.22 * slow_attack
                + 0.18 * inverse_ramp(max(strike_inharmonic, inharmonicity), 0.04, 0.30)
                + 0.14 * ramp(max(harmonic, fundamental), 0.34, 0.92)
                + 0.08 * ramp(event_repetition, 0.18, 0.86)
                + 0.06 * ramp(stereo_width, 0.20, 0.70)
            ),
            "ClavinetHarpsichord": clamp01(
                0.28 * fast_attack
                + 0.22 * ramp(max(strike_inharmonic, fa_conical), 0.20, 0.72)
                + 0.18 * ramp(max(mid, presence), 0.30, 0.86)
                + 0.16 * ramp(pitch_strength, 0.48, 1.0)
                + 0.16 * inverse_ramp(tail, 0.08, 0.48)
            ),
        }
        plucked_panels = {
            "AcousticGuitar": clamp01(
                0.26 * fast_attack
                + 0.22 * ramp(max(mid, presence), 0.24, 0.78)
                + 0.18 * ramp(pitch_strength, 0.50, 1.0)
                + 0.14 * inverse_ramp(flatness, 0.02, 0.22)
                + 0.12 * inverse_ramp(low_total, 0.04, 0.54)
                + 0.08 * inverse_ramp(tail, 0.12, 0.74)
            ),
            "ElectricGuitar": clamp01(
                0.22 * ramp(max(mid, presence), 0.32, 0.94)
                + 0.20 * ramp(pitch_strength, 0.48, 1.0)
                + 0.18 * ramp(max(body_noise, flatness), 0.10, 0.44)
                + 0.16 * ramp(max(tail, sustained_tonal), 0.26, 0.86)
                + 0.14 * fast_attack
                + 0.10 * inverse_ramp(low_total, 0.04, 0.54)
            ),
            "NylonOrSoftPluck": clamp01(
                0.28 * fast_attack
                + 0.20 * ramp(mid, 0.24, 0.74)
                + 0.18 * inverse_ramp(high, 0.02, 0.24)
                + 0.16 * ramp(pitch_strength, 0.48, 1.0)
                + 0.10 * inverse_ramp(max(body_noise, flatness), 0.04, 0.30)
                + 0.08 * inverse_ramp(tail, 0.10, 0.68)
            ),
            "WorldPluck": clamp01(
                0.26 * resonant_strike
                + 0.20 * ramp(high, 0.08, 0.42)
                + 0.18 * ramp(pitch_strength, 0.42, 0.96)
                + 0.16 * inverse_ramp(tail, 0.08, 0.58)
                + 0.12 * ramp(max(inharmonicity, strike_inharmonic), 0.12, 0.52)
                + 0.08 * inverse_ramp(low_total, 0.04, 0.44)
            ),
        }
        string_panels = {
            "BowedSustain": clamp01(
                0.30 * stable_sustain
                + 0.22 * slow_attack
                + 0.18 * ramp(max(harmonic, fa_cep), 0.28, 0.88)
                + 0.14 * ramp(max(body_noise, fa_stochastic), 0.10, 0.38)
                + 0.10 * ramp(max(mid, presence), 0.26, 0.82)
                + 0.06 * inverse_ramp(formant_voice, 0.04, 0.42)
            ),
            "StringPluck": clamp01(
                0.56 * plucked_core
                + 0.20 * resonant_strike
                + 0.14 * ramp(harmonic, 0.32, 0.90)
                + 0.10 * inverse_ramp(high, 0.02, 0.30)
            ),
            "StringDrone": clamp01(
                0.38 * stable_sustain
                + 0.24 * inverse_ramp(event_count, 1.0, 6.0)
                + 0.18 * ramp(max(tail, sustained_tonal), 0.70, 1.0)
                + 0.12 * ramp(stereo_width, 0.18, 0.70)
                + 0.08 * inverse_ramp(max(percussive_loop, drumlike_loop), 0.02, 0.18)
            ),
            "StringLoop": clamp01(0.60 * strings_core + 0.22 * event_repetition + 0.18 * ramp(loop_pitched, 0.68, 1.0)),
        }
        brass_panels = {
            "TrumpetHorn": clamp01(
                0.26 * ramp(max(presence, high), 0.12, 0.46)
                + 0.24 * ramp(max(harmonic, fa_conical), 0.34, 0.92)
                + 0.20 * ramp(pitch_strength, 0.56, 1.0)
                + 0.14 * ramp(fa_presence, 18.0, 36.0)
                + 0.10 * mid_register
                + 0.06 * inverse_ramp(flatness, 0.04, 0.34)
            ),
            "LowBrass": clamp01(
                0.25 * low_register
                + 0.22 * ramp(max(harmonic, fa_conical), 0.30, 0.90)
                + 0.18 * ramp(mid + presence, 0.32, 0.90)
                + 0.14 * ramp(pitch_strength, 0.54, 1.0)
                + 0.12 * inverse_ramp(high, 0.02, 0.24)
                + 0.09 * ramp(fa_presence, 14.0, 34.0)
            ),
            "BrassStab": clamp01(
                0.26 * fast_attack
                + 0.24 * ramp(max(harmonic, fa_conical), 0.30, 0.90)
                + 0.20 * ramp(max(presence, high), 0.10, 0.46)
                + 0.16 * inverse_ramp(tail, 0.10, 0.70)
                + 0.14 * ramp(pitch_strength, 0.46, 0.96)
            ),
            "BrassSustainLoop": clamp01(0.56 * brass_core + 0.24 * stable_sustain + 0.20 * event_repetition),
        }
        woodwind_panels = {
            "Sax": clamp01(
                0.52 * reed_core
                + 0.18 * ramp(fa_presence, 18.0, 36.0)
                + 0.14 * ramp(max(fa_stochastic, body_noise), 0.20, 0.64)
                + 0.10 * ramp(mid + bass, 0.28, 0.86)
                + 0.06 * inverse_ramp(event_high, 0.00, 0.22)
            ),
            "Flute": clamp01(
                0.28 * ramp(max(air, high), 0.08, 0.44)
                + 0.22 * ramp(pitch_strength, 0.54, 1.0)
                + 0.18 * ramp(max(fa_stochastic, body_noise), 0.18, 0.58)
                + 0.14 * inverse_ramp(low_total, 0.03, 0.40)
                + 0.10 * high_register
                + 0.08 * inverse_ramp(max(fa_conical, strike_inharmonic), 0.04, 0.36)
            ),
            "Clarinet": clamp01(
                0.28 * reed_core
                + 0.22 * ramp(max(fa_formant_stability, formant_spacing), 0.32, 0.90)
                + 0.18 * ramp(mid, 0.34, 0.86)
                + 0.14 * inverse_ramp(high, 0.02, 0.26)
                + 0.10 * ramp(pitch_strength, 0.54, 1.0)
                + 0.08 * inverse_ramp(fa_formant_std, 40.0, 220.0)
            ),
            "Bassoon": clamp01(
                0.30 * reed_core
                + 0.22 * low_register
                + 0.18 * ramp(low_total, 0.28, 0.78)
                + 0.14 * inverse_ramp(high, 0.02, 0.22)
                + 0.10 * ramp(pitch_strength, 0.50, 0.96)
                + 0.06 * ramp(max(fa_stochastic, body_noise), 0.12, 0.48)
            ),
            "AiryWoodwind": clamp01(
                0.44 * noisy_breath_edge
                + 0.24 * ramp(pitch_strength, 0.50, 1.0)
                + 0.18 * inverse_ramp(low_total, 0.04, 0.48)
                + 0.14 * stable_sustain
            ),
        }
        synth_panels = {
            "SynthLead": clamp01(
                0.28 * clean_synth_signature
                + 0.22 * ramp(pitch_strength, 0.68, 1.0)
                + 0.18 * event_repetition
                + 0.14 * mid_register
                + 0.10 * inverse_ramp(max(body_noise, flatness), 0.02, 0.22)
                + 0.08 * inverse_ramp(low_total, 0.04, 0.48)
            ),
            "SynthPad": clamp01(
                0.32 * stable_sustain
                + 0.24 * slow_attack
                + 0.16 * ramp(stereo_width, 0.18, 0.74)
                + 0.14 * inverse_ramp(flatness, 0.004, 0.16)
                + 0.08 * inverse_ramp(event_count, 1.0, 8.0)
                + 0.06 * ramp(max(sustained_tonal, non_event_tonal), 0.76, 1.0)
            ),
            "SynthPluck": clamp01(
                0.28 * fast_attack
                + 0.22 * inverse_ramp(flatness, 0.004, 0.12)
                + 0.18 * ramp(fundamental, 0.36, 0.92)
                + 0.16 * inverse_ramp(tail, 0.08, 0.58)
                + 0.10 * ramp(pitch_strength, 0.50, 1.0)
                + 0.06 * inverse_ramp(max(body_noise, fa_stochastic), 0.02, 0.26)
            ),
            "SynthArp": clamp01(
                0.36 * true_arp_event_motion
                + 0.22 * ramp(loop_pitched, 0.72, 1.0)
                + 0.16 * clean_synth_signature
                + 0.10 * inverse_ramp(flatness, 0.004, 0.16)
                + 0.08 * ramp(event_count, 6.0, 24.0)
                + 0.08 * loop_periodicity
                - 0.32 * (1.0 if arp_pad_or_bass_decoy else 0.0)
            ),
            "SynthDrone": clamp01(
                0.42 * stable_sustain
                + 0.22 * inverse_ramp(event_count, 1.0, 7.0)
                + 0.18 * inverse_ramp(flatness, 0.004, 0.16)
                + 0.10 * ramp(stereo_width, 0.18, 0.72)
                + 0.08 * ramp(max(tail, sustained_tonal), 0.72, 1.0)
            ),
        }
        mallet_panels = {
            "BellChime": clamp01(
                0.30 * ramp(max(inharmonicity, strike_inharmonic, settle_inharmonic), 0.22, 0.86)
                + 0.24 * ramp(max(high, presence, event_high), 0.14, 0.58)
                + 0.18 * ramp(max(tail, settle_inharmonic), 0.18, 0.86)
                + 0.16 * ramp(pitch_strength, 0.40, 0.92)
                + 0.12 * inverse_ramp(low_total, 0.04, 0.40)
            ),
            "MalletKeys": clamp01(
                0.30 * resonant_strike
                + 0.22 * ramp(pitch_strength, 0.48, 0.96)
                + 0.18 * ramp(mid + high, 0.34, 0.92)
                + 0.16 * inverse_ramp(tail, 0.10, 0.66)
                + 0.14 * ramp(max(inharmonicity, strike_inharmonic), 0.10, 0.50)
            ),
            "SteelPanHandpan": clamp01(
                0.26 * ramp(max(inharmonicity, strike_inharmonic), 0.18, 0.70)
                + 0.24 * ramp(max(harmonic, fa_cep), 0.24, 0.82)
                + 0.18 * ramp(mid + high, 0.40, 0.96)
                + 0.16 * ramp(max(tail, sustained_tonal), 0.22, 0.86)
                + 0.16 * ramp(pitch_strength, 0.44, 0.96)
            ),
            "KalimbaMbira": clamp01(
                0.30 * fast_attack
                + 0.24 * ramp(max(high, presence), 0.10, 0.48)
                + 0.18 * ramp(max(inharmonicity, strike_inharmonic), 0.12, 0.58)
                + 0.16 * ramp(pitch_strength, 0.42, 0.94)
                + 0.12 * inverse_ramp(tail, 0.08, 0.56)
            ),
            "SingingBowlGong": clamp01(
                0.32 * ramp(max(tail, sustained_tonal), 0.70, 1.0)
                + 0.24 * ramp(max(inharmonicity, settle_inharmonic), 0.18, 0.80)
                + 0.16 * inverse_ramp(event_count, 1.0, 5.0)
                + 0.14 * ramp(pitch_strength, 0.34, 0.86)
                + 0.14 * ramp(stereo_width, 0.18, 0.72)
            ),
        }

        def _selected_panel(panel_scores: dict[str, float]) -> tuple[str, float, float]:
            if not panel_scores:
                return "", 0.0, 0.0
            ordered = sorted(panel_scores.items(), key=lambda item: item[1], reverse=True)
            best_name, best_score = ordered[0]
            second = ordered[1][1] if len(ordered) > 1 else 0.0
            return best_name, float(best_score), float(best_score - second)

        panel_groups = {
            "Bass": bass_panels,
            "KeysPiano": keys_panels,
            "PluckedString": plucked_panels,
            "Strings": string_panels,
            "Brass": brass_panels,
            "Woodwinds": woodwind_panels,
            "Synth": synth_panels,
            "MalletBell": mallet_panels,
        }
        panel_summary = {branch_name: _selected_panel(scores) for branch_name, scores in panel_groups.items()}
        keys_sub, keys_sub_score, keys_sub_margin = panel_summary["KeysPiano"]
        wood_sub, wood_sub_score, wood_sub_margin = panel_summary["Woodwinds"]
        pluck_name, pluck_score, pluck_margin = panel_summary["PluckedString"]
        synth_sub, synth_sub_score, synth_sub_margin = panel_summary["Synth"]
        pluck_family = {"AcousticGuitar", "ElectricGuitar", "NylonOrSoftPluck", "WorldPluck"}
        electric_keys_panel_source = bool(
            keys_sub == "ElectricPiano"
            and keys_sub_score >= 0.78
            and low_total <= 0.20
            and mid >= 0.65
            and high <= 0.16
            and flatness <= 0.075
            and fa_formant_std >= 240.0
            and max(percussive_loop, drumlike_loop) <= 0.12
        )
        wide_formant_voice_decoy = bool(
            fa_formant_std >= 600.0
            and max(sub_voice, voice_core, rap_voice_texture) >= 0.56
            and sub_plucked_authority < 0.48
            and sub_synth_tonal < 0.66
            and shape_name in {"vocal_phrase", "pitched_phrase", "repeated_phrase_loop", "sustained_pad", "bass_phrase"}
            and shape_confidence >= 0.72
            and pitch_strength >= 0.50
            and high <= 0.22
            and event_high <= 0.28
            and max(percussive_loop, drumlike_loop) <= 0.18
        )
        keys_reed_decoy_body = bool(
            keys_sub in {"ElectricPiano", "AcousticPiano", "Organ"}
            and keys_sub_score >= 0.82
            and keys_sub_score >= wood_sub_score + 0.08
            and sub_keys_authority >= sub_reed_authority + 0.08
            and event_mid >= 0.58
            and event_high <= 0.08
            and flatness <= 0.12
        )
        synth_lead_air_decoy = bool(
            (
                synth_sub in {"SynthLead", "SynthArp"}
                and synth_sub_score >= 0.82
                and synth_sub_score >= wood_sub_score + 0.015
                and sub_synth_tonal >= max(sub_reed_authority, sub_keys_authority) + 0.06
                and 0.07 <= high <= 0.26
                and event_count >= 10.0
                and max(percussive_loop, drumlike_loop) <= 0.14
            )
            or (
                synth_sub in {"SynthLead", "SynthArp"}
                and synth_sub_score >= 0.84
                and synth_sub_score >= wood_sub_score - 0.025
                and sub_synth_tonal >= sub_reed_authority + 0.10
                and event_count >= 36.0
                and high >= 0.18
                and event_high >= 0.18
                and max(percussive_loop, drumlike_loop) <= 0.14
            )
        )
        bright_articulated_pluck_source = bool(
            pluck_name in pluck_family
            and pluck_score >= 0.66
            and sub_plucked_authority >= max(sub_reed_authority, sub_keys_authority) - 0.02
            and event_high >= 0.22
            and high >= 0.24
            and fa_formant_std >= 300.0
            and shape_name in {"pitched_phrase", "pitched_phrase_shape", "repeated_phrase_loop", "solo_phrase"}
            and shape_confidence >= 0.70
            and pitch_strength >= 0.42
            and max(percussive_loop, drumlike_loop) <= 0.18
            and not (low_total >= 0.92 and mid <= 0.08 and high <= 0.030)
            and not (
                max(inharmonicity, strike_inharmonic, settle_inharmonic) >= 0.55
                and max(high, event_high) >= 0.32
                and panel_summary["MalletBell"][1] >= pluck_score - 0.04
            )
        )
        synth_reed_decoy_body = bool(
            sub_synth_tonal >= sub_reed_authority + 0.02
            and wood_sub in {"Sax", "AiryWoodwind"}
            and wood_sub_margin < 0.055
            and event_count >= 44.0
            and 0.14 <= high <= 0.32
            and event_high >= 0.12
            and 0.08 <= flatness <= 0.24
            and max(percussive_loop, drumlike_loop) <= 0.14
        )
        strong_synth_identity_source = bool(
            synth_sub in {"SynthLead", "SynthArp", "SynthPad", "SynthDrone"}
            and synth_sub_score >= 0.68
            and synth_sub_score >= wood_sub_score - 0.030
            and sub_synth_tonal >= sub_reed_authority - 0.020
            and shape_name in {"pitched_phrase", "pitched_phrase_shape", "repeated_phrase_loop", "sustained_pad"}
            and shape_confidence >= 0.70
            and loop_pitched >= 0.88
            and max(sustained_tonal, non_event_tonal) >= 0.80
            and max(percussive_loop, drumlike_loop) <= 0.12
            and (
                (low_total <= 0.12 and mid >= 0.58 and high <= 0.30 and flatness <= 0.26)
                or (event_high <= 0.045 and flatness <= 0.090 and mid >= 0.34)
            )
        )
        wet_airy_woodwind_over_voice_source = bool(
            wood_sub in {"Sax", "AiryWoodwind"}
            and wood_sub_score >= 0.86
            and wood_sub_score >= max(sub_voice, voice_core, human_voice_texture, rap_voice_texture) + 0.015
            and shape_name in {"pitched_phrase", "pitched_phrase_shape", "repeated_phrase_loop", "solo_phrase"}
            and shape_confidence >= 0.74
            and pitch_strength >= 0.54
            and max(percussive_loop, drumlike_loop) <= 0.18
            and not wide_formant_voice_decoy
            and not bright_articulated_pluck_source
            and not synth_reed_decoy_body
            and not strong_synth_identity_source
            and not keys_reed_decoy_body
            and not synth_lead_air_decoy
        )
        sax_low_high_plucklike_body = bool(high < 0.035 and flatness > 0.090 and fa_presence < 32.0)
        sax_unfocused_decoy_body = bool(fa_formant_std >= 320.0 and pitch_conf < 0.72 and wood_sub_score < 0.82)
        sax_panel_source = bool(
            (
                wood_sub == "Sax"
                and wood_sub_score >= 0.74
                and (wood_sub_margin >= 0.065 or wood_sub_score >= 0.82 or fa_presence >= 32.0)
                and not sax_low_high_plucklike_body
                and not sax_unfocused_decoy_body
                and not wide_formant_voice_decoy
                and not bright_articulated_pluck_source
                and not synth_reed_decoy_body
                and not keys_reed_decoy_body
                and not synth_lead_air_decoy
            )
            or wet_airy_woodwind_over_voice_source
        )
        airy_woodwind_source = bool(
            (
                wood_sub == "AiryWoodwind"
                and mid >= 0.34
                and not bright_articulated_pluck_source
                and not synth_reed_decoy_body
                and not keys_reed_decoy_body
                and not synth_lead_air_decoy
                and (
                    (
                        wood_sub_score >= 0.80
                        and (wood_sub_margin >= 0.075 or fa_presence >= 29.0 or high >= 0.16)
                        and high >= 0.035
                    )
                    or (
                        wood_sub_score >= 0.76
                        and low_total <= 0.14
                        and mid >= 0.60
                        and high >= 0.035
                        and high <= 0.28
                        and fa_presence >= 30.0
                    )
                )
            )
            or wet_airy_woodwind_over_voice_source
        )
        other_reed_source = bool(
            wood_sub in {"Clarinet", "Bassoon", "Flute"}
            and wood_sub_score >= 0.80
            and mid >= 0.30
            and (wood_sub_margin >= 0.075 or fa_presence >= 28.0)
            and not wide_formant_voice_decoy
            and not bright_articulated_pluck_source
            and not synth_reed_decoy_body
            and not keys_reed_decoy_body
            and not synth_lead_air_decoy
        )
        dark_reed_body_source = bool(
            wood_sub in {"Sax", "Clarinet"}
            and wood_sub_score >= 0.74
            and 0.30 <= low_total <= 0.70
            and mid >= 0.35
            and high <= 0.045
            and flatness <= 0.070
            and fa_presence >= 29.0
            and event_count <= 18.0
            and max(percussive_loop, drumlike_loop) <= 0.12
            and not wide_formant_voice_decoy
            and not bright_articulated_pluck_source
            and not keys_reed_decoy_body
            and not synth_lead_air_decoy
        )
        breathy_low_mid_sax_source = bool(
            wood_sub == "Sax"
            and wood_sub_score >= 0.72
            and pitch_conf >= 0.72
            and f0_voiced >= 0.75
            and 0.25 <= low_total <= 0.75
            and 0.25 <= mid <= 0.70
            and high <= 0.085
            and 0.12 <= flatness <= 0.34
            and event_count <= 28.0
            and max(percussive_loop, drumlike_loop) <= 0.12
            and not wide_formant_voice_decoy
            and not bright_articulated_pluck_source
            and not keys_reed_decoy_body
            and not synth_lead_air_decoy
        )
        focused_reed_phrase_source = bool(
            wood_sub in {"Sax", "Clarinet"}
            and wood_sub_score >= 0.74
            and pitch_conf >= 0.78
            and f0_voiced >= 0.75
            and mid >= 0.35
            and high <= 0.075
            and event_high >= 0.040
            and 0.030 <= flatness <= 0.140
            and fa_presence >= 30.0
            and fa_formant_std <= 190.0
            and event_count <= 72.0
            and max(percussive_loop, drumlike_loop) <= 0.12
            and not wide_formant_voice_decoy
            and not bright_articulated_pluck_source
            and not synth_reed_decoy_body
            and not keys_reed_decoy_body
            and not synth_lead_air_decoy
        )
        low_heavy_bass_body = bool(low_total >= 0.92 and mid <= 0.08 and high <= 0.030)
        airy_flute_like_body = bool(high >= 0.30 and low_total <= 0.14 and wood_sub in {"Flute", "AiryWoodwind"})
        metallic_mallet_like_body = bool(
            max(inharmonicity, strike_inharmonic, settle_inharmonic) >= 0.55
            and max(high, event_high) >= 0.32
            and panel_summary["MalletBell"][1] >= pluck_score - 0.04
        )
        articulated_plucked_source_candidate = bool(
            pluck_name in pluck_family
            and pluck_score >= 0.60
            and (
                pluck_margin >= 0.045
                or sub_plucked >= max(sub_reed, sub_keys) - 0.065
                or (sub_plucked >= 0.54 and pluck_score >= 0.64)
            )
            and shape_name
            in {
                "pitched_phrase",
                "pitched_phrase_shape",
                "vocal_phrase",
                "solo_phrase",
                "repeated_phrase_loop",
                "bass_phrase",
                "sustained_pad",
            }
            and shape_confidence >= 0.64
            and pitch_strength >= 0.40
            and (attack <= 0.18 or (pluck_margin >= 0.12 and event_count >= 6.0))
            and not (shape_name == "sustained_pad" and attack >= 0.12 and event_count <= 4.0)
            and max(percussive_loop, drumlike_loop) <= 0.30
            and not low_heavy_bass_body
            and not airy_flute_like_body
            and not metallic_mallet_like_body
        )
        strong_woodwind_over_pluck = bool(
            (
                wood_sub in {"Sax", "AiryWoodwind", "Clarinet", "Bassoon", "Flute"}
                and wood_sub_score >= 0.86
                and wood_sub_score >= pluck_score + 0.10
                and sub_reed >= sub_plucked + 0.08
                and (mid >= 0.34 or fa_presence >= 32.0)
                and not articulated_plucked_source_candidate
            )
            or (dark_reed_body_source and not articulated_plucked_source_candidate)
            or (breathy_low_mid_sax_source and sub_reed >= sub_plucked + 0.06)
            or (focused_reed_phrase_source and sub_reed >= sub_plucked + 0.06)
        )
        non_plucked_tonal_decoy = bool(
            (
                max(sub_voice, vocal_role, rap_voice_texture) >= 0.50
                and max(sub_voice, vocal_role, rap_voice_texture) >= sub_plucked_authority - 0.02
            )
            or (
                max(sub_reed, sub_reed_authority, wood_sub_score) >= sub_plucked_authority - 0.04
                and wood_sub_score >= pluck_score - 0.06
                and max(sub_reed, wood_sub_score) >= 0.48
            )
            or (
                sub_synth_tonal >= sub_plucked_authority + 0.08
                and sub_plucked_authority < 0.52
                and shape_name in {"bass_phrase", "sustained_pad", "repeated_phrase_loop", "vocal_phrase"}
                and max(sustained_tonal, non_event_tonal, loop_pitched) >= 0.78
            )
            or (
                sub_keys_authority >= sub_plucked_authority + 0.05
                and sub_synth_tonal >= 0.55
                and sub_plucked_authority < 0.54
            )
        )
        plucked_string_source_signal = bool(
            (
                articulated_plucked_source_candidate
                and not strong_woodwind_over_pluck
                and (not non_plucked_tonal_decoy or bright_articulated_pluck_source)
            )
            or bright_articulated_pluck_source
        )
        reed_woodwind_source_signal = bool(
            (
                sax_panel_source
                or airy_woodwind_source
                or dark_reed_body_source
                or breathy_low_mid_sax_source
                or focused_reed_phrase_source
                or other_reed_source
            )
            and shape_name
            in {"pitched_phrase", "vocal_phrase", "solo_phrase", "repeated_phrase_loop", "bass_phrase", "sustained_pad"}
            and shape_confidence >= 0.68
            and pitch_strength >= 0.44
            and (mid + high) >= 0.24
            and high <= 0.48
            and 0.012 <= flatness <= 0.38
            and max(percussive_loop, drumlike_loop) <= 0.22
            and not electric_keys_panel_source
            and not (
                plucked_string_source_signal
                and (pluck_score >= wood_sub_score - 0.08 or sub_plucked >= sub_reed - 0.04 or pluck_margin >= 0.075)
                and not (
                    wood_sub == "Sax"
                    and wood_sub_score >= 0.89
                    and wood_sub_margin >= 0.095
                    and sub_reed >= sub_plucked + 0.09
                )
            )
        )
        low_mid_wet_sax_signal = bool(
            wood_sub == "Sax"
            and wood_sub_score >= 0.72
            and shape_name in {"bass_phrase", "pitched_phrase", "solo_phrase", "sustained_pad"}
            and shape_confidence >= 0.82
            and 8.0 <= event_count <= 24.0
            and pitch_conf >= 0.70
            and f0_voiced >= 0.84
            and loop_pitched >= 0.90
            and max(sustained_tonal, non_event_tonal) >= 0.84
            and 260.0 <= f0_hz <= 760.0
            and event_low >= 0.62
            and event_mid <= 0.22
            and event_high <= 0.065
            and 0.10 <= flatness <= 0.32
            and entropy <= 0.44
            and fa_status == "ok"
            and fa_presence >= 20.0
            and 700.0 <= fa_formant_center <= 1650.0
            and 0.0 < fa_formant_std <= 180.0
            and fa_stochastic >= 0.38
            and max(percussive_loop, drumlike_loop) <= 0.10
            and formant_voice <= 0.22
        )
        if low_mid_wet_sax_signal:
            woodwind_source_signal = True
            reed_woodwind_source_signal = True
            plucked_string_source_signal = False
        if wet_airy_woodwind_over_voice_source:
            woodwind_source_signal = True
            reed_woodwind_source_signal = True
            plucked_string_source_signal = False
        if clean_tonal_reed_solo_signal or dark_low_mid_reed_loop_signal:
            woodwind_source_signal = True
            reed_woodwind_source_signal = True
            plucked_string_source_signal = False
        if reed_woodwind_source_signal:
            woodwind_source_signal = True
        if bright_articulated_pluck_source:
            reed_woodwind_source_signal = False
            woodwind_source_signal = False
        if wide_formant_voice_decoy:
            reed_woodwind_source_signal = False
            woodwind_source_signal = False
        if strong_synth_identity_source:
            reed_woodwind_source_signal = False
            woodwind_source_signal = False

        bass_core = max(bass_core, 0.88 * panel_summary["Bass"][1])
        keys_core = max(keys_core, 0.88 * panel_summary["KeysPiano"][1])
        plucked_core = max(plucked_core, 0.84 * panel_summary["PluckedString"][1])
        strings_core = max(strings_core, 0.86 * panel_summary["Strings"][1])
        brass_core = max(brass_core, 0.86 * panel_summary["Brass"][1])
        reed_core = max(reed_core, 0.88 * panel_summary["Woodwinds"][1])
        if low_mid_wet_sax_signal:
            reed_core = max(reed_core, 0.96)
            bass_core = min(bass_core, 0.72)
            keys_core = min(keys_core, 0.74)
        if wet_airy_woodwind_over_voice_source:
            reed_core = max(reed_core, 0.95)
            brass_core = min(brass_core, 0.80)
            keys_core = min(keys_core, 0.76)
            synth_core = min(synth_core, 0.76)
        synth_core = max(synth_core, 0.86 * panel_summary["Synth"][1])
        mallet_bell_core = max(mallet_bell_core, 0.86 * panel_summary["MalletBell"][1])
        if low_mid_wet_sax_signal:
            synth_core = min(synth_core, 0.72)

        # Prevent the broad PluckedString panel from stealing clean reed/keys/synth
        # loops unless it is both high-confidence and clearly separated.
        nearest_non_pluck = max(keys_core, reed_core, synth_core, strings_core, brass_core)
        pluck_can_steal = bool(
            (pluck_score >= 0.74 and pluck_margin >= 0.10 and plucked_core >= nearest_non_pluck + 0.055)
            or (
                plucked_string_source_signal
                and not reed_woodwind_source_signal
                and pluck_score >= 0.60
                and (plucked_core >= nearest_non_pluck - 0.18 or sub_plucked >= max(sub_reed, sub_keys) - 0.06)
            )
        )
        if not pluck_can_steal and nearest_non_pluck >= 0.60:
            plucked_core = min(plucked_core, nearest_non_pluck + 0.02)
        if plucked_string_source_signal and not reed_woodwind_source_signal:
            plucked_core = max(plucked_core, min(0.95, max(0.76, nearest_non_pluck + 0.045)))
            reed_core = min(reed_core, plucked_core - 0.055)
            mallet_bell_core = min(mallet_bell_core, plucked_core - 0.030)
            strings_core = min(strings_core, plucked_core - 0.025)
            bass_core = min(bass_core, plucked_core - 0.025)
        if reed_woodwind_source_signal:
            reed_core = max(reed_core, min(0.96, max(0.78, 0.92 * wood_sub_score)))
            if not plucked_string_source_signal:
                plucked_core = min(plucked_core, reed_core - 0.050)
            mallet_bell_core = min(mallet_bell_core, reed_core + 0.015)
            strings_core = min(strings_core, reed_core + 0.010)

        raw_scores = {
            "Bass": bass_core,
            "KeysPiano": keys_core,
            "PluckedString": plucked_core,
            "Woodwinds": reed_core,
            "Brass": brass_core,
            "Voice": voice_core,
            "Synth": synth_core,
            "Strings": strings_core,
            "MalletBell": mallet_bell_core,
        }
        if strong_synth_identity_source:
            nearest_non_synth = max(
                raw_scores[b] for b in ("Woodwinds", "KeysPiano", "PluckedString", "Strings", "Brass", "MalletBell")
            )
            raw_scores["Synth"] = max(raw_scores["Synth"], min(0.96, nearest_non_synth + 0.030))
            raw_scores["Woodwinds"] = min(raw_scores["Woodwinds"], raw_scores["Synth"] - 0.055)
        if clean_bass_phrase:
            raw_scores["Bass"] = max(raw_scores["Bass"], 1.0)
        if woodwind_source_signal:
            raw_scores["Woodwinds"] = max(raw_scores["Woodwinds"], 0.98)
            raw_scores["Voice"] = min(raw_scores["Voice"], 0.74)
            if dark_low_mid_reed_loop_signal:
                raw_scores["PluckedString"] = min(raw_scores["PluckedString"], 0.66)
                raw_scores["Synth"] = min(raw_scores["Synth"], 0.76)
        if plucked_string_source_signal and not reed_woodwind_source_signal:
            raw_scores["PluckedString"] = max(
                raw_scores["PluckedString"],
                min(
                    0.96,
                    max(raw_scores["Woodwinds"], raw_scores["MalletBell"], raw_scores["Strings"], raw_scores["Bass"])
                    + 0.035,
                ),
            )
            raw_scores["Woodwinds"] = min(raw_scores["Woodwinds"], raw_scores["PluckedString"] - 0.055)
            raw_scores["MalletBell"] = min(raw_scores["MalletBell"], raw_scores["PluckedString"] - 0.030)
            raw_scores["Strings"] = min(raw_scores["Strings"], raw_scores["PluckedString"] - 0.025)
            raw_scores["Bass"] = min(raw_scores["Bass"], raw_scores["PluckedString"] - 0.025)
        if reed_woodwind_source_signal:
            raw_scores["Woodwinds"] = max(
                raw_scores["Woodwinds"],
                min(
                    0.97,
                    max(
                        raw_scores["MalletBell"],
                        raw_scores["Strings"],
                        raw_scores["Brass"],
                        raw_scores["KeysPiano"],
                        raw_scores["Synth"],
                    )
                    + 0.020,
                ),
            )
            if not plucked_string_source_signal:
                raw_scores["PluckedString"] = min(raw_scores["PluckedString"], raw_scores["Woodwinds"] - 0.050)
            raw_scores["Voice"] = min(raw_scores["Voice"], raw_scores["Woodwinds"] - 0.040)
        if non_voice_tonal_loop_voice_decoy:
            non_voice_best = max(
                raw_scores[b]
                for b in ("Synth", "MalletBell", "KeysPiano", "Strings", "Brass", "Woodwinds", "MixedInstrument")
                if b in raw_scores
            )
            raw_scores["Voice"] = min(raw_scores["Voice"], max(0.38, non_voice_best - 0.050))
            if sub_synth_tonal >= 0.56:
                raw_scores["Synth"] = max(raw_scores["Synth"], min(0.94, non_voice_best + 0.018))
            elif sub_metallic >= 0.50:
                raw_scores["MalletBell"] = max(raw_scores["MalletBell"], min(0.94, non_voice_best + 0.018))
        compound = self.compound_layer.decide(
            raw_scores=raw_scores,
            instrument_anchor=instrument_anchor,
            pitch_strength=pitch_strength,
            tonal_clean=tonal_clean,
            low_total=low_total,
            mid=mid,
            high=high,
            event_low=event_low,
            event_mid=event_mid,
            event_high=event_high,
            event_count=event_count,
            onset_span=onset_span,
            flatness=flatness,
            entropy=entropy,
            body_noise=body_noise,
            tail_noise=tail_noise,
            tail=tail,
            loop_pitched=loop_pitched,
            sustained_tonal=sustained_tonal,
            non_event_tonal=non_event_tonal,
            percussive_loop=percussive_loop,
            drumlike_loop=drumlike_loop,
            is_loop=is_loop,
            is_long=is_long,
            clean_bass_phrase=clean_bass_phrase,
            wet_woodwind_signal=woodwind_source_signal,
        )
        plausible_count = int(compound["compound_music_plausible_branch_count"])
        strongest = float(compound["compound_music_strongest_branch_score"])

        branch_gate = clamp01(0.35 + 0.65 * instrument_anchor)
        branch_scores = {name: clamp01(branch_gate * value) for name, value in raw_scores.items()}
        if clean_bass_phrase:
            branch_scores["Bass"] = 1.0
        if vocal_role >= 0.70 and not woodwind_source_signal:
            branch_scores["Voice"] = max(
                branch_scores["Voice"],
                clamp01(branch_gate * (0.58 + 0.36 * ramp(vocal_role, 0.70, 0.96))),
            )
            branch_scores["Strings"] = min(branch_scores["Strings"], branch_scores["Voice"] - 0.03)
            branch_scores["Woodwinds"] = min(branch_scores["Woodwinds"], branch_scores["Voice"] - 0.04)
        if rap_voice_texture >= 0.58 and not woodwind_source_signal:
            branch_scores["Voice"] = max(
                branch_scores["Voice"],
                clamp01(branch_gate * (0.62 + 0.30 * ramp(rap_voice_texture, 0.58, 0.78))),
            )
            branch_scores["Woodwinds"] = min(branch_scores["Woodwinds"], branch_scores["Voice"] - 0.04)
        if processed_vocal_shot_signal and not woodwind_source_signal:
            branch_scores["Voice"] = max(branch_scores["Voice"], 0.74, branch_gate * max(voice_core, 0.82))
            branch_scores["Woodwinds"] = min(branch_scores["Woodwinds"], branch_scores["Voice"] - 0.06)
            branch_scores["Brass"] = min(branch_scores["Brass"], branch_scores["Voice"] - 0.08)
        if human_voice_phrase_signal and not woodwind_source_signal:
            branch_scores["Voice"] = max(branch_scores["Voice"], branch_gate * max(voice_core, 0.78))
            branch_scores["Woodwinds"] = min(branch_scores["Woodwinds"], branch_scores["Voice"] - 0.05)
            branch_scores["Brass"] = min(branch_scores["Brass"], branch_scores["Voice"] - 0.06)
        if wide_formant_voice_decoy and not woodwind_source_signal and branch_scores["Voice"] >= 0.46:
            branch_scores["Voice"] = max(branch_scores["Voice"], min(0.92, branch_gate * max(voice_core, 0.74)))
            branch_scores["Woodwinds"] = min(branch_scores["Woodwinds"], branch_scores["Voice"] - 0.055)
            branch_scores["Brass"] = min(branch_scores["Brass"], branch_scores["Voice"] - 0.060)
        if woodwind_source_signal:
            branch_scores["Woodwinds"] = max(branch_scores["Woodwinds"], 0.94)
            branch_scores["Voice"] = min(branch_scores["Voice"], branch_scores["Woodwinds"] - 0.04)
        if (
            branch_scores["Strings"] >= 0.72
            and branch_scores["Woodwinds"] - branch_scores["Strings"] <= 0.03
            and high >= 0.18
            and fa_formant_center >= 1500.0
            and branch_scores["Voice"] < 0.55
        ):
            branch_scores["Woodwinds"] = min(branch_scores["Woodwinds"], branch_scores["Strings"] - 0.02)
        branch_scores["MixedInstrument"] = clamp01(
            max(
                branch_gate * float(compound["compound_music_mixed_branch_score"]),
                0.54 if instrument_anchor >= 0.82 and strongest < 0.52 else 0.0,
            )
        )
        if bool(compound["compound_music_prefer_broad_loop"]):
            branch_scores["MixedInstrument"] = max(
                branch_scores["MixedInstrument"], float(compound["compound_music_mixed_branch_score"])
            )
            for specific_branch in (
                "Woodwinds",
                "Brass",
                "Voice",
                "Synth",
                "Strings",
                "PluckedString",
                "KeysPiano",
                "MalletBell",
            ):
                branch_scores[specific_branch] = min(
                    branch_scores[specific_branch], branch_scores["MixedInstrument"] - 0.03
                )
        # Panel authority is intentionally narrow: it may lift a measured
        # sub-branch only when the panel itself is strong. This avoids the old
        # failure mode where a broad branch such as PluckedString stole sax,
        # keys, or synth loops merely because it was plausible.
        synth_sub, synth_sub_score, synth_sub_margin = panel_summary["Synth"]
        brass_sub, brass_sub_score, brass_sub_margin = panel_summary["Brass"]
        strings_sub, strings_sub_score, strings_sub_margin = panel_summary["Strings"]
        mallet_sub, mallet_sub_score, mallet_sub_margin = panel_summary["MalletBell"]

        organ_panel_authority = bool(
            keys_sub == "Organ" and keys_sub_score >= 0.76 and keys_sub_margin >= 0.10 and not woodwind_source_signal
        )
        strong_electric_keys_panel_source = bool(
            keys_sub == "ElectricPiano"
            and keys_sub_score >= 0.86
            and keys_sub_margin >= 0.08
            and shape_name
            in {"pitched_phrase", "pitched_phrase_shape", "solo_phrase", "repeated_phrase_loop", "sustained_pad"}
            and shape_confidence >= 0.80
            and loop_pitched >= 0.82
            and event_mid >= 0.55
            and event_high <= 0.24
            and flatness <= 0.16
            and max(percussive_loop, drumlike_loop) <= 0.12
            and not woodwind_source_signal
        )
        clean_electric_keys_loop_source = bool(
            electric_keys_panel_source
            or strong_electric_keys_panel_source
            or (
                keys_sub == "ElectricPiano"
                and keys_sub_score >= 0.74
                and shape_name in {"pitched_phrase", "solo_phrase", "repeated_phrase_loop", "sustained_pad"}
                and shape_confidence >= 0.82
                and loop_pitched >= 0.88
                and max(sustained_tonal, non_event_tonal) >= 0.80
                and event_mid >= 0.62
                and event_high <= 0.22
                and flatness <= 0.085
                and entropy <= 0.46
                and fa_formant_std >= 220.0
                and max(percussive_loop, drumlike_loop) <= 0.10
                and not woodwind_source_signal
            )
        )
        if organ_panel_authority or clean_electric_keys_loop_source:
            nearest = max(
                branch_scores[b] for b in ("Synth", "Strings", "Brass", "Woodwinds", "PluckedString", "MalletBell")
            )
            branch_scores["KeysPiano"] = max(branch_scores["KeysPiano"], min(0.96, nearest + 0.025))
            branch_scores["Synth"] = min(branch_scores["Synth"], branch_scores["KeysPiano"] - 0.02)
            branch_scores["Strings"] = min(branch_scores["Strings"], branch_scores["KeysPiano"] - 0.045)
            if clean_electric_keys_loop_source:
                branch_scores["Brass"] = min(branch_scores["Brass"], branch_scores["KeysPiano"] - 0.030)
                branch_scores["Woodwinds"] = min(branch_scores["Woodwinds"], branch_scores["KeysPiano"] - 0.035)
                branch_scores["MalletBell"] = min(branch_scores["MalletBell"], branch_scores["KeysPiano"] - 0.030)
        clean_synth_pad_source = bool(
            synth_sub in {"SynthPad", "SynthDrone"}
            and synth_sub_score >= 0.82
            and synth_sub_margin >= 0.075
            and flatness <= 0.085
            and max(body_noise, fa_stochastic) <= 0.20
        )
        if clean_synth_pad_source:
            nearest = max(branch_scores[b] for b in ("KeysPiano", "Strings", "Woodwinds", "Brass", "PluckedString"))
            branch_scores["Synth"] = max(branch_scores["Synth"], min(0.96, nearest + 0.025))
            branch_scores["Strings"] = min(branch_scores["Strings"], branch_scores["Synth"] - 0.03)
            branch_scores["KeysPiano"] = min(branch_scores["KeysPiano"], branch_scores["Synth"] - 0.02)
        brass_panel_authority = bool(brass_sub_score >= 0.80 and brass_sub_margin >= 0.045)
        if brass_panel_authority and not clean_electric_keys_loop_source:
            nearest = max(branch_scores[b] for b in ("Strings", "PluckedString", "Woodwinds", "Synth", "KeysPiano"))
            branch_scores["Brass"] = max(branch_scores["Brass"], min(0.95, nearest + 0.02))
            branch_scores["Strings"] = min(branch_scores["Strings"], branch_scores["Brass"] - 0.02)
            branch_scores["PluckedString"] = min(branch_scores["PluckedString"], branch_scores["Brass"] - 0.02)
            if brass_sub_score >= wood_sub_score + 0.03 and not woodwind_source_signal:
                branch_scores["Woodwinds"] = min(branch_scores["Woodwinds"], branch_scores["Brass"] - 0.025)
        if (
            wood_sub in {"Flute", "Clarinet", "Bassoon", "AiryWoodwind", "Sax"}
            and wood_sub_score >= 0.72
            and (wood_sub_margin >= 0.055 or high >= 0.18 or woodwind_source_signal)
            and (
                woodwind_source_signal
                or reed_woodwind_source_signal
                or (wood_sub in {"Flute", "AiryWoodwind"} and high >= 0.30 and low_total <= 0.14)
            )
            and not clean_electric_keys_loop_source
            and not (brass_panel_authority and brass_sub_score >= wood_sub_score - 0.05 and not woodwind_source_signal)
        ):
            nearest = max(branch_scores[b] for b in ("Strings", "Brass", "Synth", "KeysPiano", "PluckedString"))
            branch_scores["Woodwinds"] = max(branch_scores["Woodwinds"], min(0.95, nearest + 0.018))
            branch_scores["Strings"] = min(branch_scores["Strings"], branch_scores["Woodwinds"] - 0.018)
        if (
            strings_sub in {"BowedSustain", "StringDrone"}
            and strings_sub_score >= 0.78
            and strings_sub_margin >= 0.05
            and not clean_synth_pad_source
            and not organ_panel_authority
            and not woodwind_source_signal
        ):
            nearest = max(branch_scores[b] for b in ("Woodwinds", "Brass", "Synth", "KeysPiano"))
            branch_scores["Strings"] = max(branch_scores["Strings"], min(0.95, nearest + 0.016))
        clear_metallic_pitched_hit = bool(
            mallet_sub_score >= 0.70
            and max(inharmonicity, strike_inharmonic, settle_inharmonic) >= 0.55
            and pitch_strength >= 0.55
            and event_count <= 4.0
            and max(percussive_loop, drumlike_loop) <= 0.12
        )
        if (
            (mallet_sub_score >= 0.78 or clear_metallic_pitched_hit)
            and max(inharmonicity, strike_inharmonic, settle_inharmonic) >= 0.25
            and not clean_electric_keys_loop_source
        ):
            nearest = max(branch_scores[b] for b in ("KeysPiano", "PluckedString", "Strings", "Synth", "Woodwinds"))
            lift_margin = 0.032 if clear_metallic_pitched_hit else 0.018
            branch_scores["MalletBell"] = max(branch_scores["MalletBell"], min(0.94, nearest + lift_margin))
            if clear_metallic_pitched_hit and not bool(compound["compound_music_prefer_broad_loop"]):
                branch_scores["MixedInstrument"] = min(
                    branch_scores["MixedInstrument"], branch_scores["MalletBell"] - 0.030
                )
        if plucked_string_source_signal and not reed_woodwind_source_signal:
            nearest = max(
                branch_scores[b] for b in ("KeysPiano", "Strings", "Synth", "Woodwinds", "MalletBell", "Bass")
            )
            branch_scores["PluckedString"] = max(branch_scores["PluckedString"], min(0.96, nearest + 0.035))
            branch_scores["Woodwinds"] = min(branch_scores["Woodwinds"], branch_scores["PluckedString"] - 0.055)
            branch_scores["MalletBell"] = min(branch_scores["MalletBell"], branch_scores["PluckedString"] - 0.030)
            branch_scores["Strings"] = min(branch_scores["Strings"], branch_scores["PluckedString"] - 0.025)
            branch_scores["Bass"] = min(branch_scores["Bass"], branch_scores["PluckedString"] - 0.025)
        if (
            not reed_woodwind_source_signal
            and not woodwind_source_signal
            and not (wood_sub in {"Flute", "AiryWoodwind"} and high >= 0.30 and low_total <= 0.14)
        ):
            nearest_non_woodwind = max(
                branch_scores[b] for b in ("PluckedString", "KeysPiano", "Synth", "Strings", "Brass", "MalletBell")
            )
            branch_scores["Woodwinds"] = min(branch_scores["Woodwinds"], max(0.58, nearest_non_woodwind - 0.025))
        if reed_woodwind_source_signal:
            nearest = max(branch_scores[b] for b in ("KeysPiano", "Strings", "Synth", "Brass", "MalletBell"))
            branch_scores["Woodwinds"] = max(branch_scores["Woodwinds"], min(0.96, nearest + 0.024))
            if not plucked_string_source_signal:
                branch_scores["PluckedString"] = min(branch_scores["PluckedString"], branch_scores["Woodwinds"] - 0.050)
            branch_scores["Voice"] = min(branch_scores["Voice"], branch_scores["Woodwinds"] - 0.035)

        # Do not let broad mixed-loop safety swallow a measured articulated
        # voice texture. MixedInstrument is a safety bucket, not a source
        # identity, and voice must remain the selected branch when its measured
        # vocal texture is the clearest identity.
        if (
            (rap_voice_texture >= 0.58 or human_voice_phrase_signal)
            and not woodwind_source_signal
            and branch_scores["Voice"] >= 0.50
        ):
            branch_scores["MixedInstrument"] = min(branch_scores["MixedInstrument"], branch_scores["Voice"] - 0.02)

        if non_plucked_tonal_decoy:
            strongest_non_plucked = max(
                branch_scores["Voice"],
                branch_scores["Synth"],
                branch_scores["Woodwinds"],
                branch_scores["Brass"],
                branch_scores["KeysPiano"],
                branch_scores["MixedInstrument"],
            )
            branch_scores["PluckedString"] = min(
                branch_scores["PluckedString"],
                max(0.0, strongest_non_plucked - 0.030),
            )

        if high_register_low_band_conflict and not low_pitch_bass_identity:
            strongest_non_bass = max(
                branch_scores["Synth"],
                branch_scores["Woodwinds"],
                branch_scores["Brass"],
                branch_scores["KeysPiano"],
                branch_scores["PluckedString"],
                branch_scores["MixedInstrument"],
            )
            branch_scores["Bass"] = min(
                branch_scores["Bass"],
                max(0.0, strongest_non_bass - 0.020),
            )

        out = {
            "instrument_anchor_strength": round(float(instrument_anchor), 6),
            "instrument_pitch_strength": round(float(pitch_strength), 6),
            "instrument_tonal_clean": round(float(tonal_clean), 6),
            "instrument_phrase_structure": round(float(phrase_structure), 6),
            "instrument_non_drum_phrase": round(float(non_drum_phrase), 6),
            "instrument_pitch_role": round(float(pitch_role), 6),
            "instrument_vocal_role": round(float(vocal_role), 6),
            "instrument_bass_role": round(float(bass_role), 6),
            "instrument_drum_role": round(float(drum_role), 6),
            "instrument_f0_median_hz": round(float(f0_hz), 6),
            "instrument_low_total": round(float(low_total), 6),
            "instrument_mid_ratio": round(float(mid), 6),
            "instrument_high_ratio": round(float(high), 6),
            "instrument_flatness": round(float(flatness), 6),
            "instrument_entropy": round(float(entropy), 6),
            "instrument_first_arrival_status": fa_status,
            "instrument_first_arrival_formant_center_hz": round(float(fa_formant_center), 6),
            "instrument_first_arrival_formant_std_hz": round(float(fa_formant_std), 6),
            "instrument_first_arrival_stochastic": round(float(fa_stochastic), 6),
            "instrument_first_arrival_presence_db": round(float(fa_presence), 6),
            "instrument_branch_plausible_count": int(plausible_count),
            "instrument_branch_strongest_pre_gate": round(float(strongest), 6),
            "instrument_shape_primary": shape_name,
            "instrument_event_count": round(float(event_count), 6),
            "instrument_event_mid_ratio": round(float(event_mid), 6),
            "instrument_event_high_ratio": round(float(event_high), 6),
            "instrument_subpanel_role_loop_score": round(float(sub_role_loop), 6),
            "instrument_subpanel_role_one_shot_score": round(float(sub_role_one_shot), 6),
            "instrument_subpanel_role_phrase_score": round(float(sub_role_phrase), 6),
            "instrument_subpanel_plucked_string_score": round(float(sub_plucked), 6),
            "instrument_subpanel_plucked_string_authority_score": round(float(sub_plucked_authority), 6),
            "instrument_subpanel_reed_wind_score": round(float(sub_reed), 6),
            "instrument_subpanel_reed_wind_authority_score": round(float(sub_reed_authority), 6),
            "instrument_subpanel_struck_keys_score": round(float(sub_keys), 6),
            "instrument_subpanel_struck_keys_authority_score": round(float(sub_keys_authority), 6),
            "instrument_subpanel_synth_tonal_source_score": round(float(sub_synth_tonal), 6),
            "instrument_subpanel_drum_loop_source_score": round(float(sub_drum_loop_source), 6),
            "instrument_subpanel_rhythmic_break_loop_score": round(float(sub_rhythmic_break_loop), 6),
            "instrument_subpanel_fx_transition_authority_score": round(float(sub_fx_transition_authority), 6),
            "instrument_subpanel_bowed_string_score": round(float(sub_bowed), 6),
            "instrument_subpanel_voice_score": round(float(sub_voice), 6),
            "instrument_subpanel_human_spoken_voice_score": round(float(sub_human_spoken), 6),
            "instrument_subpanel_human_breath_mouth_score": round(float(sub_human_breath), 6),
            "instrument_subpanel_fx_formant_score": round(float(sub_fx_formant), 6),
            "instrument_subpanel_drum_hit_score": round(float(sub_drum_hit), 6),
            "instrument_subpanel_metallic_noise_score": round(float(sub_metallic), 6),
            "instrument_subpanel_scrape_rasp_score": round(float(sub_scrape), 6),
            "instrument_subpanel_fx_motion_score": round(float(sub_fx_motion), 6),
            "instrument_subpanel_texture_bed_score": round(float(sub_texture_bed), 6),
            "instrument_rap_voice_texture": round(float(rap_voice_texture), 6),
            "instrument_human_voice_texture": round(float(human_voice_texture), 6),
            "instrument_human_voice_phrase_signal": bool(human_voice_phrase_signal),
            "instrument_wide_formant_voice_decoy": bool(wide_formant_voice_decoy),
            "instrument_keys_reed_decoy_body": bool(keys_reed_decoy_body),
            "instrument_non_voice_tonal_loop_voice_decoy": bool(non_voice_tonal_loop_voice_decoy),
            "instrument_processed_vocal_shot_signal": bool(processed_vocal_shot_signal),
            "instrument_clean_bass_phrase": bool(clean_bass_phrase),
            "instrument_low_pitch_bass_identity": bool(low_pitch_bass_identity),
            "instrument_high_register_low_band_conflict": bool(high_register_low_band_conflict),
            "instrument_wet_woodwind_signal": bool(wet_woodwind_signal),
            "instrument_clean_tonal_reed_solo_signal": bool(clean_tonal_reed_solo_signal),
            "instrument_dark_low_mid_reed_loop_signal": bool(dark_low_mid_reed_loop_signal),
            "instrument_low_mid_wet_sax_signal": bool(low_mid_wet_sax_signal),
            "instrument_dark_reed_body_source": bool(dark_reed_body_source),
            "instrument_breathy_low_mid_sax_source": bool(breathy_low_mid_sax_source),
            "instrument_focused_reed_phrase_source": bool(focused_reed_phrase_source),
            "instrument_reed_woodwind_source_signal": bool(reed_woodwind_source_signal),
            "instrument_woodwind_source_signal": bool(woodwind_source_signal),
            "instrument_articulated_plucked_source_candidate": bool(articulated_plucked_source_candidate),
            "instrument_bright_articulated_pluck_source": bool(bright_articulated_pluck_source),
            "instrument_synth_reed_decoy_body": bool(synth_reed_decoy_body),
            "instrument_strong_synth_identity_source": bool(strong_synth_identity_source),
            "instrument_synth_lead_air_decoy": bool(synth_lead_air_decoy),
            "instrument_wet_airy_woodwind_over_voice_source": bool(wet_airy_woodwind_over_voice_source),
            "instrument_non_plucked_tonal_decoy": bool(non_plucked_tonal_decoy),
            "instrument_plucked_string_source_signal": bool(plucked_string_source_signal),
            "instrument_plucked_vs_woodwind_conflict": bool(
                plucked_string_source_signal and reed_woodwind_source_signal
            ),
            "instrument_short_bright_percussion_hit": bool(short_bright_percussion_hit),
            "instrument_plucked_branch_steal_guard_open": bool(pluck_can_steal),
            "instrument_electric_keys_panel_source": bool(electric_keys_panel_source),
            "instrument_clean_electric_keys_loop_signal": bool(clean_electric_keys_loop_source),
            "instrument_synth_arp_true_event_motion": round(float(true_arp_event_motion), 6),
            "instrument_synth_arp_pad_or_bass_decoy": bool(arp_pad_or_bass_decoy),
            "instrument_loop_periodicity": round(float(loop_periodicity), 6),
            "instrument_loop_role_score": round(float(loop_role_score), 6),
            "instrument_phrase_role_score": round(float(phrase_role_score), 6),
            **compound,
        }
        for branch_name, scores in panel_groups.items():
            selected_name, selected_score, selected_margin = panel_summary[branch_name]
            out[f"instrument_{branch_name}_subpanel_selected"] = selected_name
            out[f"instrument_{branch_name}_subpanel_confidence"] = round(float(selected_score), 6)
            out[f"instrument_{branch_name}_subpanel_margin"] = round(float(selected_margin), 6)
            for panel_name, panel_score in scores.items():
                out[f"instrument_panel_{branch_name}_{panel_name}"] = round(float(panel_score), 6)
        for name, value in branch_scores.items():
            out[f"instrument_branch_{name}"] = round(float(value), 6)
        return out
