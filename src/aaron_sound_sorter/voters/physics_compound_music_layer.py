# SOURCE-NAME BLINDNESS INVARIANT:
# This production sorting module must never use producer filenames, source
# folder names, ZIP member names, path tokens, or sample-pack labels as
# classification evidence. Names are allowed only for I/O/display/diagnostics
# after the audio decision. Run RUN_NO_SOURCE_NAME_SORTING_AUDIT.command before
# shipping any sorter-logic change.
"""Measured compound musical-loop physics layer."""

from __future__ import annotations

from typing import Any

from aaron_sound_sorter.voters.physics_layer_utils import *


class PhysicsCompoundMusicLayer:
    """Measured layer for produced, mixed, or compound musical loops.

    This layer does not block sax, voice, keys, or synth directly.  It answers a
    higher-level question first: does the audio behave like one isolated source,
    or like a layered musical loop where a single branch leaf is too specific?
    """

    def decide(
        self,
        *,
        raw_scores: dict[str, float],
        instrument_anchor: float,
        pitch_strength: float,
        tonal_clean: float,
        low_total: float,
        mid: float,
        high: float,
        event_low: float,
        event_mid: float,
        event_high: float,
        event_count: float,
        onset_span: float,
        flatness: float,
        entropy: float,
        body_noise: float,
        tail_noise: float,
        tail: float,
        loop_pitched: float,
        sustained_tonal: float,
        non_event_tonal: float,
        percussive_loop: float,
        drumlike_loop: float,
        is_loop: bool,
        is_long: bool,
        clean_bass_phrase: bool,
        wet_woodwind_signal: bool,
    ) -> dict[str, Any]:
        plausible_count = sum(1 for value in raw_scores.values() if value >= 0.46)
        sorted_scores = sorted((float(value) for value in raw_scores.values()), reverse=True)
        strongest = sorted_scores[0] if sorted_scores else 0.0
        second = sorted_scores[1] if len(sorted_scores) >= 2 else 0.0
        third = sorted_scores[2] if len(sorted_scores) >= 3 else 0.0
        strongest_gap = max(0.0, strongest - second)

        low_band = ramp(low_total, 0.14, 0.42)
        mid_band = ramp(mid, 0.24, 0.62)
        high_band = ramp(high, 0.055, 0.24)
        sorted_bands = sorted((low_band, mid_band, high_band))
        event_band_spread = clamp01(
            0.40 * ramp(event_low, 0.12, 0.42)
            + 0.35 * ramp(event_mid, 0.22, 0.58)
            + 0.25 * ramp(event_high, 0.05, 0.24)
        )
        band_spread = clamp01(0.48 * sorted_bands[0] + 0.32 * sorted_bands[1] + 0.20 * event_band_spread)

        branch_disagreement = clamp01(
            0.30 * ramp(plausible_count, 2.0, 5.0)
            + 0.24 * ramp(second, 0.38, 0.72)
            + 0.16 * ramp(third, 0.34, 0.66)
            + 0.18 * inverse_ramp(strongest, 0.52, 0.90)
            + 0.12 * inverse_ramp(strongest_gap, 0.035, 0.30)
        )
        layered_activity = clamp01(
            0.25 * ramp(event_count, 8.0, 44.0)
            + 0.20 * ramp(onset_span, 0.36, 0.88)
            + 0.18 * ramp(max(loop_pitched, sustained_tonal, non_event_tonal), 0.48, 0.96)
            + 0.17 * ramp(max(percussive_loop, drumlike_loop), 0.08, 0.46)
            + 0.12 * (1.0 if is_loop or is_long else 0.0)
            + 0.08 * event_band_spread
        )
        tonal_fx_wash = clamp01(
            0.28 * ramp(pitch_strength, 0.55, 0.96)
            + 0.22 * ramp(max(body_noise, tail_noise, flatness), 0.16, 0.48)
            + 0.18 * ramp(tail, 0.24, 0.78)
            + 0.17 * ramp(entropy, 0.34, 0.74)
            + 0.15 * ramp(high, 0.08, 0.30)
        )
        single_source_isolation = clamp01(
            0.30 * ramp(strongest, 0.70, 0.96)
            + 0.20 * inverse_ramp(plausible_count, 1.0, 3.0)
            + 0.18 * tonal_clean
            + 0.14 * inverse_ramp(event_count, 8.0, 36.0)
            + 0.10 * inverse_ramp(band_spread, 0.22, 0.62)
            + 0.08 * inverse_ramp(strongest_gap, 0.02, 0.24)
        )
        if clean_bass_phrase:
            single_source_isolation = max(single_source_isolation, 0.92)
        if wet_woodwind_signal and band_spread < 0.52 and plausible_count <= 3:
            single_source_isolation = max(single_source_isolation, 0.78)

        compound_strength = clamp01(
            0.22 * instrument_anchor
            + 0.24 * band_spread
            + 0.14 * branch_disagreement
            + 0.24 * layered_activity
            + 0.18 * tonal_fx_wash
            + 0.06 * ramp(max(body_noise, tail_noise), 0.18, 0.55)
            - 0.24 * single_source_isolation
        )
        prefer_broad_loop = bool(
            instrument_anchor >= 0.64
            and compound_strength >= 0.60
            and band_spread >= 0.50
            and layered_activity >= 0.62
            and (is_loop or is_long or event_count >= 10.0)
            and not (single_source_isolation >= 0.72 and strongest >= 0.76)
        )
        branch_score = clamp01(
            0.40 * instrument_anchor
            + 0.38 * compound_strength
            + 0.14 * ramp(plausible_count, 3.0, 5.0)
            + 0.08 * layered_activity
        )
        if prefer_broad_loop:
            branch_score = max(branch_score, min(0.94, 0.78 + 0.22 * compound_strength))
        if single_source_isolation >= 0.78 and not prefer_broad_loop:
            branch_score = min(branch_score, 0.54)

        return {
            "compound_music_strength": round(float(compound_strength), 6),
            "compound_music_band_spread": round(float(band_spread), 6),
            "compound_music_branch_disagreement": round(float(branch_disagreement), 6),
            "compound_music_layered_activity": round(float(layered_activity), 6),
            "compound_music_tonal_fx_wash": round(float(tonal_fx_wash), 6),
            "compound_music_single_source_isolation": round(float(single_source_isolation), 6),
            "compound_music_plausible_branch_count": int(plausible_count),
            "compound_music_strongest_branch_score": round(float(strongest), 6),
            "compound_music_second_branch_score": round(float(second), 6),
            "compound_music_prefer_broad_loop": prefer_broad_loop,
            "compound_music_mixed_branch_score": round(float(branch_score), 6),
        }
