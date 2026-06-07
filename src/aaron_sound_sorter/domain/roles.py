# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Measured audio role evidence.

These roles are not destination categories. They are broad functional readings
from audio physics that help voters separate things like a sustained bass loop
from a kick loop, or a rhythmic drum loop from long ambience.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MeasuredRoles:
    """Broad functional role strengths derived only from measured features."""

    bass_loop: float = 0.0
    voiced_one_shot: float = 0.0
    percussive_one_shot: float = 0.0
    pitched_music_phrase: float = 0.0
    bright_drum_loop: float = 0.0
    percussive_drum_loop: float = 0.0
    pitched_music_loop: float = 0.0
    low_rhythmic_drum_loop: float = 0.0
    vocal_music_phrase: float = 0.0
    evidence: dict[str, float] | None = None

    def to_evidence(self) -> dict[str, object]:
        data = asdict(self)
        data["primary_roles"] = [
            name
            for name in [
                "bass_loop",
                "voiced_one_shot",
                "percussive_one_shot",
                "pitched_music_phrase",
                "bright_drum_loop",
                "percussive_drum_loop",
                "pitched_music_loop",
                "low_rhythmic_drum_loop",
                "vocal_music_phrase",
            ]
            if float(data.get(name, 0.0) or 0.0) >= 0.70
        ]
        return data


def measured_roles_from_features(
    *,
    is_loop_like: bool,
    is_single_event_like: bool,
    is_short_hit_like: bool,
    is_long: bool,
    feature_values: Mapping[str, float],
) -> MeasuredRoles:
    """Return functional role strengths from measured audio facts."""
    sub = value(feature_values, "sub_bass_ratio_lt_150hz")
    bass = value(feature_values, "bass_ratio_150_500hz")
    mid = value(feature_values, "mid_ratio_500_2000hz")
    presence = value(feature_values, "presence_ratio_2000_8000hz")
    air = value(feature_values, "air_ratio_gt_8000hz")
    pitch_confidence = value(feature_values, "pitch_confidence")
    f0_voiced = value(feature_values, "f0_voiced_ratio")
    formant = value(feature_values, "formant_like_peak_spacing")
    flatness = value(feature_values, "spectral_flatness_mean")
    log_crest = value(feature_values, "log_crest")
    attack_rise = value(feature_values, "attack_rise_time_norm")
    temporal_centroid = value(feature_values, "temporal_centroid_ratio")
    tail_energy = value(feature_values, "tail_energy_ratio")
    log_transients = value(feature_values, "log_transient_count")
    loop_pitched = value(feature_values, "loop_pitched_event_ratio")
    loop_percussive = value(feature_values, "loop_percussive_event_ratio")
    loop_drumlike = value(feature_values, "loop_drumlike_frame_ratio")
    loop_balance = value(feature_values, "loop_tonal_to_percussive_balance")
    loop_sustained = value(feature_values, "loop_sustained_tonal_frame_ratio")
    loop_non_event_tonal = value(feature_values, "loop_non_event_tonal_ratio")
    low_peak_hz = value(feature_values, "low_peak_frequency_hz")

    low_total = min(1.0, sub + bass)
    high_total = min(1.0, presence + air)

    low_rhythmic_drum_loop = 0.0
    if is_loop_like and is_long:
        # Low-heavy drum and drum+bass loops can look like bass loops if the
        # low end dominates.  Treat repeated percussive/drumlike evidence as a
        # measured loop role before rewarding bass-loop labels.  This remains a
        # physics role, not a filename/category rule.
        low_rhythmic_raw = average_strength(
            ramp(low_total, 0.38, 0.78),
            ramp(loop_percussive, 0.35, 0.72),
            ramp(loop_drumlike, 0.32, 0.70),
            inverse_ramp(loop_sustained, 0.10, 0.45),
            inverse_ramp(loop_non_event_tonal, 0.08, 0.42),
        )
        low_rhythmic_drum_loop = low_rhythmic_raw * ramp(low_total, 0.25, 0.55)

        # A kick-loop can be strongly pitched and sustained between attacks, so
        # the older loop_percussive/loop_drumlike ratios under-read it as a
        # bass loop.  Add a separate pulse-based low drum-loop reading: many
        # fast low-frequency events, very low high-band content, and weak
        # sustained voicing.  This is deliberately broad, source-name blind,
        # and only affects repeated material, not one-shot kicks.
        low_pulse_loop = average_strength(
            ramp(low_total, 0.74, 0.94),
            ramp(log_transients, 2.75, 3.70),
            inverse_ramp(attack_rise, 0.012, 0.055),
            inverse_ramp(f0_voiced, 0.18, 0.58),
            inverse_ramp(high_total, 0.015, 0.10),
            ramp(pitch_confidence, 0.45, 0.88),
        )

        # Do not let the low-pulse drum-loop fallback swallow clean bass lines.
        # A repeated bass phrase can be very low, sharply articulated, and highly
        # pitched while still having no drumlike/percussive frames.  The drum-loop
        # reading should require event/body evidence, or it should back off and
        # let the bass/pitched-loop roles compete normally.
        tonal_bass_line_veto = average_strength(
            ramp(loop_pitched, 0.72, 0.96),
            ramp(loop_sustained, 0.45, 0.72),
            ramp(loop_non_event_tonal, 0.42, 0.72),
            inverse_ramp(loop_percussive, 0.02, 0.18),
            inverse_ramp(loop_drumlike, 0.02, 0.18),
        )
        # Only a *voiced* sustained tonal line should veto the low-pulse
        # drum-loop reading. Repeated tonal kick loops often have high pitch
        # confidence from autocorrelation, but low voiced-frame ratio and very
        # fast low-frequency attacks. Without this distinction, a kick-loop can
        # be misread as a clean bass line before the arbiter ever sees it.
        voiced_tonal_line_gate = average_strength(
            ramp(f0_voiced, 0.34, 0.72),
            inverse_ramp(attack_rise, 0.055, 0.16),
            inverse_ramp(log_transients, 2.40, 3.35),
        )
        tonal_bass_line_veto *= voiced_tonal_line_gate
        low_pulse_loop *= 1.0 - 0.82 * ramp(tonal_bass_line_veto, 0.50, 0.88)

        low_rhythmic_drum_loop = max(
            low_rhythmic_drum_loop,
            low_pulse_loop * ramp(low_total, 0.70, 0.90),
        )

    bass_loop = 0.0
    if is_loop_like and is_long:
        # Bass identity needs actual low-frequency dominance.  A sax or vocal
        # loop can be strongly pitched and sustained, so those traits refine
        # the role only after the low-end gate is present.
        # Do not let any low-mid dominated pitched loop become a Bass Loop.
        # Rhodes, organ, synth leads, and low keys can put most energy in
        # 150-500 Hz while still not being foundational bass.  A measured
        # bass-loop role should require real sub/foundation energy before it
        # can drive broad Bass routing.
        low_identity_gate = ramp(low_total, 0.30, 0.62)
        sub_foundation_gate = ramp(sub, 0.35, 0.70)
        raw_bass_loop = average_strength(
            ramp(pitch_confidence, 0.55, 0.90),
            ramp(loop_pitched, 0.65, 1.0),
            ramp(loop_sustained, 0.20, 0.62),
            ramp(loop_non_event_tonal, 0.20, 0.62),
            inverse_ramp(loop_percussive, 0.10, 0.35),
            inverse_ramp(loop_drumlike, 0.05, 0.25),
            inverse_ramp(high_total, 0.02, 0.20),
        )
        # Low-rhythmic drum-loop evidence should suppress Bass only when the
        # loop also has measured drum/percussive structure.  Pure sub-foundation
        # bass lines can be very low, highly periodic, and low in high-band
        # energy, which makes the low-rhythmic role fire even though the source
        # family is still Bass.  This keeps the role model from treating every
        # repeated sub line as a kick loop.
        drum_event_gate = max(
            ramp(loop_percussive, 0.08, 0.38),
            ramp(loop_drumlike, 0.06, 0.28),
            inverse_ramp(loop_sustained, 0.45, 0.88),
            inverse_ramp(loop_non_event_tonal, 0.42, 0.86),
        )
        rhythmic_suppression = ramp(low_rhythmic_drum_loop, 0.45, 0.85) * drum_event_gate
        bass_loop = low_identity_gate * sub_foundation_gate * raw_bass_loop * (1.0 - 0.80 * rhythmic_suppression)

    voiced_one_shot = 0.0
    if is_single_event_like or is_short_hit_like:
        formant_identity = ramp(formant, 1.05, 2.20)
        formant_light_voice_identity = average_strength(
            ramp(f0_voiced, 0.78, 0.95),
            ramp(mid + presence, 0.62, 0.95),
            inverse_ramp(low_total, 0.10, 0.36),
            inverse_ramp(loop_drumlike, 0.02, 0.22),
        )
        # Very short resonant percussion often produces a stable autocorrelation
        # pitch and even formant-like spectral spacing.  Do not treat that as
        # voice unless the event has enough internal vocal motion or avoids the
        # metallic/high-noise profile.  This keeps bongos, claps, hats, and
        # cymbals from becoming Voice/Human FX while preserving real vocal stabs.
        ultra_front_loaded_hit = average_strength(
            inverse_ramp(attack_rise, 0.012, 0.045),
            inverse_ramp(temporal_centroid, 0.08, 0.18),
            inverse_ramp(log_transients, 1.05, 1.55),
            ramp(log_crest, 1.75, 2.45),
        )
        metallic_noise_hit = average_strength(
            ramp(high_total, 0.55, 0.90),
            ramp(flatness, 0.35, 0.62),
            ramp(log_crest, 1.70, 2.45),
        )
        percussive_hit_veto = max(ultra_front_loaded_hit, metallic_noise_hit)
        clean_tonal_stab_veto = average_strength(
            ramp(pitch_confidence, 0.72, 0.95),
            ramp(loop_pitched, 0.88, 1.0),
            ramp(loop_sustained, 0.88, 1.0),
            ramp(loop_non_event_tonal, 0.88, 1.0),
            inverse_ramp(high_total, 0.03, 0.16),
            inverse_ramp(flatness, 0.14, 0.26),
            inverse_ramp(log_transients, 1.00, 1.65),
        )
        protected_voice_identity = max(formant_identity, formant_light_voice_identity * 0.85)
        protected_voice_identity *= 1.0 - 0.80 * ramp(percussive_hit_veto, 0.45, 0.85)
        protected_voice_identity *= 1.0 - 0.78 * ramp(clean_tonal_stab_veto, 0.58, 0.90)
        raw_voiced_one_shot = average_strength(
            ramp(f0_voiced, 0.70, 0.95),
            ramp(pitch_confidence, 0.35, 0.68),
            protected_voice_identity,
            ramp(mid + presence, 0.42, 0.78),
            inverse_ramp(low_total, 0.12, 0.40),
        )
        raw_voiced_one_shot *= 1.0 - 0.72 * ramp(percussive_hit_veto, 0.55, 0.95)
        raw_voiced_one_shot *= 1.0 - 0.85 * ramp(clean_tonal_stab_veto, 0.58, 0.90)
        # Resonant percussion can create a stable autocorrelation pitch.  Do
        # not let that alone become a voice/instrument one-shot when the
        # envelope is still a sharp, front-loaded hit.  Strong formant spacing
        # is real voice evidence, so it protects short vocal shouts from being
        # collapsed into percussion only because they are front-loaded.
        percussive_envelope = average_strength(
            ramp(log_crest, 1.45, 2.35),
            inverse_ramp(attack_rise, 0.05, 0.24),
            inverse_ramp(temporal_centroid, 0.16, 0.42),
        )
        voice_identity = max(formant_identity, formant_light_voice_identity)
        protected_percussive_envelope = percussive_envelope * (1.0 - 0.70 * voice_identity)
        voiced_one_shot = raw_voiced_one_shot * (1.0 - 0.65 * ramp(protected_percussive_envelope, 0.50, 0.90))

    # Long tonal phrases can have a fast first attack and a low transient count,
    # which made them look like percussion one-shots.  Keep this as a measured
    # functional role: sustained, strongly pitched, low-noise audio belongs in
    # musical instrument logic even when the loop detector does not see enough
    # repeated onsets to call it a loop.
    pitched_music_phrase = 0.0
    vocal_music_phrase = 0.0
    phrase_long_enough = bool(is_long or is_loop_like)
    if phrase_long_enough and (pitch_confidence >= 0.62 and f0_voiced >= 0.70 and loop_sustained >= 0.60):
        pitched_music_phrase = average_strength(
            ramp(pitch_confidence, 0.55, 0.88),
            ramp(f0_voiced, 0.62, 0.92),
            ramp(loop_pitched, 0.55, 0.92),
            ramp(loop_sustained, 0.45, 0.85),
            ramp(loop_non_event_tonal, 0.45, 0.85),
            inverse_ramp(loop_percussive, 0.05, 0.35),
            inverse_ramp(loop_drumlike, 0.03, 0.25),
            inverse_ramp(high_total, 0.08, 0.35),
        )
    # Short synth/key phrases can contain several clean pitched events without
    # enough voiced-frame continuity to satisfy the long-phrase gate above.
    # Treat that as musical phrase evidence when the body is tonal, repeated,
    # and non-drumlike. This keeps short repeated stabs out of FX/Human or
    # percussion conflict paths without making one-event pitched hits loops.
    short_repeated_tonal_phrase = 0.0
    if not is_long:
        low_sub_hit_exception = bool(
            low_total >= 0.88
            and (0.0 < low_peak_hz <= 95.0)
            and f0_voiced <= 0.25
            and high_total <= 0.08
            and attack_rise <= 0.04
            and tail_energy <= 0.22
        )
        short_repeated_tonal_phrase = average_strength(
            ramp(log_transients, 1.75, 2.55),
            ramp(pitch_confidence, 0.56, 0.84),
            ramp(loop_pitched, 0.82, 0.98),
            ramp(loop_sustained, 0.72, 0.96),
            ramp(loop_non_event_tonal, 0.72, 0.96),
            inverse_ramp(loop_percussive, 0.02, 0.14),
            inverse_ramp(loop_drumlike, 0.02, 0.14),
            inverse_ramp(high_total, 0.05, 0.24),
            inverse_ramp(flatness, 0.16, 0.38),
        )
        if low_sub_hit_exception:
            short_repeated_tonal_phrase = 0.0
        if short_repeated_tonal_phrase >= 0.58:
            pitched_music_phrase = max(pitched_music_phrase, short_repeated_tonal_phrase * 0.92)
    clean_pitched_hit = 0.0
    if is_single_event_like or is_short_hit_like:
        # A short chord, synth stab, electric-piano hit, or key stab can be
        # front-loaded like a drum while still being clean pitched material.
        # Single-F0 confidence is often weak on chords, so use the frame-shape
        # evidence that already separates pitched/tonal frames from
        # percussive/drumlike frames. This is a measured role guard, not a
        # source-name or category rule.
        clean_tonal_body = average_strength(
            ramp(loop_pitched, 0.70, 0.96),
            ramp(loop_sustained, 0.70, 0.96),
            ramp(loop_non_event_tonal, 0.70, 0.96),
            inverse_ramp(max(loop_percussive, loop_drumlike), 0.02, 0.16),
            inverse_ramp(flatness, 0.04, 0.20),
            inverse_ramp(high_total, 0.02, 0.18),
            ramp(attack_rise, 0.045, 0.16),
        )
        sub_kick_exception = (
            sub >= 0.18
            and (0.0 < low_peak_hz <= 150.0 or low_total >= 0.90)
            and high_total <= 0.16
            and f0_voiced <= 0.32
        )
        clean_pitched_hit = clean_tonal_body * (0.0 if sub_kick_exception else 1.0)
        if clean_pitched_hit >= 0.55:
            pitched_music_phrase = max(pitched_music_phrase, clean_pitched_hit * 0.90)

    if phrase_long_enough and pitch_confidence >= 0.45 and f0_voiced >= 0.55 and formant >= 1.05:
        # Sung/rap vocal phrases share the general pitched-phrase shape with
        # sax, reeds, piano, and synth leads.  Formant-like spacing alone is not
        # a voice claim because stable harmonic instruments can produce the same
        # spacing.  Require an independent voice texture gate before this role
        # can become strong; otherwise keep the sound in the pitched-music path
        # and let Human/Voice candidates compete in the final arbiter.
        voice_texture_gate = max(
            ramp(flatness, 0.16, 0.42),
            ramp(presence, 0.10, 0.32) * inverse_ramp(low_total, 0.18, 0.52),
        )
        raw_vocal_music_phrase = average_strength(
            ramp(formant, 1.05, 2.20),
            ramp(f0_voiced, 0.55, 0.88),
            ramp(pitch_confidence, 0.42, 0.78),
            ramp(mid + presence, 0.45, 0.78),
            inverse_ramp(loop_percussive, 0.05, 0.38),
            inverse_ramp(loop_drumlike, 0.05, 0.32),
        )
        vocal_music_phrase = raw_vocal_music_phrase * voice_texture_gate

    percussive_one_shot = 0.0
    low_pitched_hit = 0.0
    if is_single_event_like or is_short_hit_like:
        # This is a measured role, not a destination rule.  It captures
        # short front-loaded transient sounds that should make a drum one-shot
        # candidate more plausible and make unrelated FX/instrument labels less
        # plausible.  Suppress it when the same measured audio is a sustained
        # pitched phrase, which covers sax/reed/bass/guitar notes with fast
        # attacks and prevents both voters from calling them percussion.
        raw_percussive_one_shot = average_strength(
            1.0 if is_short_hit_like else 0.55,
            ramp(log_crest, 1.15, 2.35),
            inverse_ramp(attack_rise, 0.05, 0.28),
            inverse_ramp(temporal_centroid, 0.18, 0.46),
            inverse_ramp(tail_energy, 0.08, 0.48),
            inverse_ramp(log_transients, 1.10, 2.20),
            inverse_ramp(loop_drumlike, 0.02, 0.22),
        )
        metallic_percussive_hit = average_strength(
            1.0 if is_short_hit_like else 0.55,
            ramp(high_total, 0.48, 0.88),
            ramp(flatness, 0.30, 0.62),
            ramp(log_crest, 1.50, 2.40),
            inverse_ramp(attack_rise, 0.04, 0.20),
            inverse_ramp(log_transients, 1.35, 2.25),
        )
        raw_percussive_one_shot = max(raw_percussive_one_shot, metallic_percussive_hit)
        tonal_phrase_suppression = ramp(pitched_music_phrase, 0.45, 0.85)
        voice_suppression = ramp(voiced_one_shot, 0.55, 0.85)
        clean_pitched_hit_suppression = ramp(clean_pitched_hit, 0.45, 0.82)
        percussive_one_shot = (
            raw_percussive_one_shot
            * (1.0 - 0.85 * tonal_phrase_suppression)
            * (1.0 - 0.70 * voice_suppression)
            * (1.0 - 0.90 * clean_pitched_hit_suppression)
        )
        low_pitched_hit = average_strength(
            1.0 if is_single_event_like else 0.55,
            ramp(low_total, 0.55, 0.92),
            ramp(pitch_confidence, 0.62, 0.92),
            inverse_ramp(attack_rise, 0.02, 0.20),
            inverse_ramp(log_transients, 1.10, 2.20),
            inverse_ramp(high_total, 0.05, 0.30),
            inverse_ramp(f0_voiced, 0.08, 0.42),
        )
        low_pitched_hit *= 1.0 - 0.95 * clean_pitched_hit_suppression
        percussive_one_shot = max(percussive_one_shot, low_pitched_hit * (1.0 - 0.90 * voice_suppression))

    bright_drum_loop = 0.0
    if is_loop_like:
        bright_drum_loop = average_strength(
            ramp(high_total, 0.45, 0.82),
            ramp(loop_percussive, 0.58, 0.95),
            ramp(loop_drumlike, 0.50, 0.82),
            inverse_ramp(low_total, 0.08, 0.30),
            inverse_ramp(loop_sustained, 0.10, 0.40),
            inverse_ramp(loop_balance, -0.70, -0.20),
        )

    percussive_drum_loop = 0.0
    if is_loop_like:
        percussive_drum_loop = average_strength(
            ramp(loop_drumlike, 0.45, 0.75),
            ramp(loop_percussive, 0.35, 0.65),
            inverse_ramp(loop_balance, -0.65, -0.20),
            inverse_ramp(loop_sustained, 0.18, 0.45),
            inverse_ramp(loop_non_event_tonal, 0.15, 0.45),
        )

    pitched_music_loop = 0.0
    if is_loop_like and is_long:
        pitched_music_loop = average_strength(
            ramp(pitch_confidence, 0.50, 0.85),
            ramp(loop_pitched, 0.55, 0.92),
            ramp(loop_sustained, 0.45, 0.85),
            ramp(loop_non_event_tonal, 0.45, 0.85),
            inverse_ramp(loop_drumlike, 0.08, 0.32),
        )

    evidence = {
        "low_total": round(low_total, 6),
        "high_total": round(high_total, 6),
        "f0_voiced_ratio": round(f0_voiced, 6),
        "pitch_confidence": round(pitch_confidence, 6),
        "formant_like_peak_spacing": round(formant, 6),
        "formant_light_voice_identity": round(
            formant_light_voice_identity if (is_single_event_like or is_short_hit_like) else 0.0, 6
        ),
        "log_crest": round(log_crest, 6),
        "attack_rise_time_norm": round(attack_rise, 6),
        "temporal_centroid_ratio": round(temporal_centroid, 6),
        "tail_energy_ratio": round(tail_energy, 6),
        "log_transient_count": round(log_transients, 6),
        "loop_pitched_event_ratio": round(loop_pitched, 6),
        "loop_percussive_event_ratio": round(loop_percussive, 6),
        "loop_drumlike_frame_ratio": round(loop_drumlike, 6),
        "loop_tonal_to_percussive_balance": round(loop_balance, 6),
        "loop_sustained_tonal_frame_ratio": round(loop_sustained, 6),
        "loop_non_event_tonal_ratio": round(loop_non_event_tonal, 6),
        "pitched_music_phrase_raw": round(pitched_music_phrase, 6),
        "short_repeated_tonal_phrase_raw": round(short_repeated_tonal_phrase, 6),
        "clean_pitched_hit_raw": round(clean_pitched_hit, 6),
        "low_peak_frequency_hz": round(low_peak_hz, 6),
        "low_rhythmic_drum_loop_raw": round(low_rhythmic_drum_loop, 6),
        "vocal_music_phrase_raw": round(vocal_music_phrase, 6),
        "low_pitched_hit_raw": round(low_pitched_hit, 6),
    }
    return MeasuredRoles(
        bass_loop=round(clamp01(bass_loop), 6),
        voiced_one_shot=round(clamp01(voiced_one_shot), 6),
        percussive_one_shot=round(clamp01(percussive_one_shot), 6),
        pitched_music_phrase=round(clamp01(pitched_music_phrase), 6),
        bright_drum_loop=round(clamp01(bright_drum_loop), 6),
        percussive_drum_loop=round(clamp01(percussive_drum_loop), 6),
        pitched_music_loop=round(clamp01(pitched_music_loop), 6),
        low_rhythmic_drum_loop=round(clamp01(low_rhythmic_drum_loop), 6),
        vocal_music_phrase=round(clamp01(vocal_music_phrase), 6),
        evidence=evidence,
    )


def value(values: Mapping[str, float], name: str) -> float:
    try:
        return float(values.get(name, 0.0) or 0.0)
    except Exception:
        return 0.0


def clamp01(number: float) -> float:
    return max(0.0, min(1.0, float(number)))


def ramp(number: float, start: float, full: float) -> float:
    if full <= start:
        return 0.0
    return clamp01((float(number) - float(start)) / (float(full) - float(start)))


def inverse_ramp(number: float, good_at_or_below: float, bad_at_or_above: float) -> float:
    if bad_at_or_above <= good_at_or_below:
        return 0.0
    return clamp01((float(bad_at_or_above) - float(number)) / (float(bad_at_or_above) - float(good_at_or_below)))


def average_strength(*parts: float) -> float:
    usable = [clamp01(part) for part in parts]
    if not usable:
        return 0.0
    return sum(usable) / len(usable)
