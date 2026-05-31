"""Category-complete physics panel catalog.

This module owns broad category-panel coverage for the current production brain
labels.  It is source-name blind: panel scores are built only from measured audio
facts and lower-level physics subpanel scores, never from producer file names,
source folder names, ZIP members, or sample-pack labels.

The scores here are intentionally light-weight witnesses.  They do not make final
routing decisions.  They give PhysicsVoter and manifests a complete map of the
Drums, Instruments, and FX leaves that currently exist in the brain so future
calibration has a named place to land instead of creating ad-hoc rescue code.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class CategoryPanelSpec:
    """One source-name-blind category panel spec."""

    top_family: str
    category_path: str
    score_key: str
    base_key: str
    role: str = "any"
    modifiers: tuple[str, ...] = ()


@dataclass(frozen=True)
class CategoryCalibrationCurve:
    """Optional percentile calibration curve for one category panel.

    The category panel builder can run without these curves.  When future
    approved samples provide per-category percentile anchors, this curve maps
    raw witness scores into a more calibrated probability-like value without
    changing the source-name-blind panel formulas.
    """

    raw_p10: float
    raw_p50: float
    raw_p90: float


def load_category_panel_calibration(path: str | Path | None = None) -> dict[str, CategoryCalibrationCurve]:
    """Load optional category-panel percentile curves from JSON.

    Expected JSON shape::

        {
          "drums_kick_drums_sub_kick_one_shots_score": {
            "raw_p10": 0.22,
            "raw_p50": 0.56,
            "raw_p90": 0.86
          }
        }

    Missing files or malformed rows return an empty profile.  That keeps normal
    sorting deterministic until Aaron has approved enough clean calibration
    examples.
    """
    if path is None:
        env_path = os.environ.get("AARON_SOUND_SORTER_CATEGORY_PANEL_CALIBRATION", "").strip()
        if not env_path:
            return {}
        path = env_path
    calibration_path = Path(path).expanduser()
    if not calibration_path.exists():
        return {}
    try:
        raw = json.loads(calibration_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    curves: dict[str, CategoryCalibrationCurve] = {}
    for score_key, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        p10 = finite_float(payload.get("raw_p10"), float("nan"))
        p50 = finite_float(payload.get("raw_p50"), float("nan"))
        p90 = finite_float(payload.get("raw_p90"), float("nan"))
        if not (math.isfinite(p10) and math.isfinite(p50) and math.isfinite(p90)):
            continue
        if not (0.0 <= p10 <= p50 <= p90 <= 1.0):
            continue
        curves[str(score_key)] = CategoryCalibrationCurve(p10, p50, p90)
    return curves


def calibrate_category_panel_score(
    score_key: str,
    raw_score: float,
    calibration: dict[str, CategoryCalibrationCurve] | None,
) -> float:
    """Apply an optional percentile calibration curve to a panel score."""
    value = clamp01(raw_score)
    if not calibration:
        return value
    curve = calibration.get(score_key)
    if curve is None:
        return value
    if curve.raw_p90 <= curve.raw_p10:
        return value
    if value <= curve.raw_p10:
        return 0.10 * ramp(value, 0.0, max(curve.raw_p10, 1e-9))
    if value <= curve.raw_p50:
        local = (value - curve.raw_p10) / max(curve.raw_p50 - curve.raw_p10, 1e-9)
        return clamp01(0.10 + 0.40 * local)
    if value <= curve.raw_p90:
        local = (value - curve.raw_p50) / max(curve.raw_p90 - curve.raw_p50, 1e-9)
        return clamp01(0.50 + 0.40 * local)
    local = (value - curve.raw_p90) / max(1.0 - curve.raw_p90, 1e-9)
    return clamp01(0.90 + 0.10 * local)


DRUM_CATEGORY_SPECS: tuple[CategoryPanelSpec, ...] = (
    CategoryPanelSpec(
        "Drums",
        "Drums/Claps Snaps Slaps/Generic Clap/One Shots",
        "drums_claps_snaps_slaps_generic_clap_one_shots_score",
        "drum_clap_source_score",
        "one_shot",
        ("generic",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Claps Snaps Slaps/Hand Clap/One Shots",
        "drums_claps_snaps_slaps_hand_clap_one_shots_score",
        "drum_clap_source_score",
        "one_shot",
        ("organic_hand",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Cymbals/Crash Cymbal/One Shots",
        "drums_cymbals_crash_cymbal_one_shots_score",
        "drum_cymbal_source_score",
        "one_shot",
        ("crash_tail",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Cymbals/Generic Cymbal/One Shots",
        "drums_cymbals_generic_cymbal_one_shots_score",
        "drum_cymbal_source_score",
        "one_shot",
        ("generic",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Cymbals/Ride Cymbal/One Shots",
        "drums_cymbals_ride_cymbal_one_shots_score",
        "drum_cymbal_source_score",
        "one_shot",
        ("ride_ping",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Cymbals/Splash Cymbal/One Shots",
        "drums_cymbals_splash_cymbal_one_shots_score",
        "drum_cymbal_source_score",
        "one_shot",
        ("splash_short",),
    ),
    CategoryPanelSpec(
        "Drums", "Drums/Drum Loops/Loops", "drums_drum_loops_loops_score", "drum_loop_source_score", "loop"
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Hi Hats/Closed Hat/One Shots",
        "drums_hi_hats_closed_hat_one_shots_score",
        "drum_closed_hat_source_score",
        "one_shot",
        ("short_bright",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Hi Hats/Generic Hat/One Shots",
        "drums_hi_hats_generic_hat_one_shots_score",
        "drum_closed_hat_source_score",
        "one_shot",
        ("generic",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Hi Hats/Open Hat/One Shots",
        "drums_hi_hats_open_hat_one_shots_score",
        "drum_cymbal_source_score",
        "one_shot",
        ("open_hat_tail",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Kick Drums/Generic Kick/One Shots",
        "drums_kick_drums_generic_kick_one_shots_score",
        "drum_kick_source_score",
        "one_shot",
        ("generic",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Kick Drums/Short Kick/One Shots",
        "drums_kick_drums_short_kick_one_shots_score",
        "drum_kick_source_score",
        "one_shot",
        ("short_low",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Kick Drums/Sub Kick/One Shots",
        "drums_kick_drums_sub_kick_one_shots_score",
        "drum_kick_source_score",
        "one_shot",
        ("sub_low",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Percussion/Bells and Metallic Percussion/One Shots",
        "drums_percussion_bells_and_metallic_percussion_one_shots_score",
        "drum_metallic_percussion_source_score",
        "one_shot",
        ("metallic",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Percussion/Generic Percussion/One Shots",
        "drums_percussion_generic_percussion_one_shots_score",
        "drum_hit_score",
        "one_shot",
        ("generic",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Percussion/Guiros Scrapes and Rasps/One Shots",
        "drums_percussion_guiros_scrapes_and_rasps_one_shots_score",
        "drum_guiro_scrape_source_score",
        "one_shot",
        ("scrape",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Percussion/Shakers and Tambourines/One Shots",
        "drums_percussion_shakers_and_tambourines_one_shots_score",
        "drum_shaker_tambourine_source_score",
        "one_shot",
        ("shaker_roll",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Rims and Sticks/Generic Rim or Stick/One Shots",
        "drums_rims_and_sticks_generic_rim_or_stick_one_shots_score",
        "drum_rim_stick_source_score",
        "one_shot",
        ("generic",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Rims and Sticks/Rimshot/One Shots",
        "drums_rims_and_sticks_rimshot_one_shots_score",
        "drum_rim_stick_source_score",
        "one_shot",
        ("rimshot",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Rims and Sticks/Sidestick/One Shots",
        "drums_rims_and_sticks_sidestick_one_shots_score",
        "drum_rim_stick_source_score",
        "one_shot",
        ("sidestick",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Snares/Acoustic Snare/One Shots",
        "drums_snares_acoustic_snare_one_shots_score",
        "drum_snare_source_score",
        "one_shot",
        ("acoustic_snare",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Snares/Generic Snare/One Shots",
        "drums_snares_generic_snare_one_shots_score",
        "drum_snare_source_score",
        "one_shot",
        ("generic",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Toms/Floor Tom/One Shots",
        "drums_toms_floor_tom_one_shots_score",
        "drum_tom_conga_source_score",
        "one_shot",
        ("floor_tom",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Toms/Generic Tom/One Shots",
        "drums_toms_generic_tom_one_shots_score",
        "drum_tom_conga_source_score",
        "one_shot",
        ("generic",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/Toms/Low Tom/One Shots",
        "drums_toms_low_tom_one_shots_score",
        "drum_tom_conga_source_score",
        "one_shot",
        ("low_tom",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/World Percussion/Indian Percussion/Tabla/One Shots",
        "drums_world_percussion_indian_percussion_tabla_one_shots_score",
        "drum_tom_conga_source_score",
        "one_shot",
        ("tabla",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/World Percussion/Latin Percussion/Bongo/One Shots",
        "drums_world_percussion_latin_percussion_bongo_one_shots_score",
        "drum_tom_conga_source_score",
        "one_shot",
        ("bongo",),
    ),
    CategoryPanelSpec(
        "Drums",
        "Drums/World Percussion/Latin Percussion/Conga/One Shots",
        "drums_world_percussion_latin_percussion_conga_one_shots_score",
        "drum_tom_conga_source_score",
        "one_shot",
        ("conga",),
    ),
)

INSTRUMENT_CATEGORY_SPECS: tuple[CategoryPanelSpec, ...] = (
    CategoryPanelSpec(
        "Instruments",
        "Instruments/Bass/808 Bass/One Shots",
        "instruments_bass_808_bass_one_shots_score",
        "bass_808_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "Instruments",
        "Instruments/Bass/Electric Bass/One Shots",
        "instruments_bass_electric_bass_one_shots_score",
        "bass_electric_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "Instruments",
        "Instruments/Bass/Generic Bass/One Shots",
        "instruments_bass_generic_bass_one_shots_score",
        "low_end_source_score",
        "one_shot",
        ("generic_bass",),
    ),
    CategoryPanelSpec(
        "Instruments",
        "Instruments/Bass/Sub Bass/One Shots",
        "instruments_bass_sub_bass_one_shots_score",
        "bass_sub_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "Instruments",
        "Instruments/Bass/Synth Bass/One Shots",
        "instruments_bass_synth_bass_one_shots_score",
        "bass_synth_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "Instruments",
        "Instruments/Bass/Upright Bass/One Shots",
        "instruments_bass_upright_bass_one_shots_score",
        "bass_upright_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "Instruments",
        "Instruments/Brass/Trumpet/One Shots",
        "instruments_brass_trumpet_one_shots_score",
        "brass_trumpet_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "Instruments",
        "Instruments/Guitar/Acoustic Guitar/One Shots",
        "instruments_guitar_acoustic_guitar_one_shots_score",
        "guitar_acoustic_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "Instruments",
        "Instruments/Guitar/Electric Guitar/One Shots",
        "instruments_guitar_electric_guitar_one_shots_score",
        "guitar_electric_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "Instruments",
        "Instruments/Guitar/Nylon Guitar/One Shots",
        "instruments_guitar_nylon_guitar_one_shots_score",
        "guitar_nylon_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "Instruments",
        "Instruments/Instrument Loops/Loops",
        "instruments_instrument_loops_loops_score",
        "role_loop_score",
        "loop",
        ("instrument_loop",),
    ),
    CategoryPanelSpec(
        "Instruments",
        "Instruments/Keys/Rhodes/One Shots",
        "instruments_keys_rhodes_one_shots_score",
        "struck_keys_score",
        "one_shot",
        ("electric_piano",),
    ),
    CategoryPanelSpec(
        "Instruments",
        "Instruments/Strings Bowed/Cello/One Shots",
        "instruments_strings_bowed_cello_one_shots_score",
        "string_cello_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "Instruments",
        "Instruments/Strings Bowed/Violin/One Shots",
        "instruments_strings_bowed_violin_one_shots_score",
        "string_violin_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "Instruments",
        "Instruments/Synths/Synth Chord/One Shots",
        "instruments_synths_synth_chord_one_shots_score",
        "synth_chord_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "Instruments",
        "Instruments/Synths/Synth Lead/One Shots",
        "instruments_synths_synth_lead_one_shots_score",
        "synth_lead_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "Instruments",
        "Instruments/Synths/Synth Pad/One Shots",
        "instruments_synths_synth_pad_one_shots_score",
        "synth_pad_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "Instruments",
        "Instruments/Voice/Choir/One Shots",
        "instruments_voice_choir_one_shots_score",
        "voice_choir_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "Instruments",
        "Instruments/Woodwinds/Flute/One Shots",
        "instruments_woodwinds_flute_one_shots_score",
        "woodwind_flute_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "Instruments",
        "Instruments/Woodwinds/Saxophone/One Shots",
        "instruments_woodwinds_saxophone_one_shots_score",
        "woodwind_sax_score",
        "one_shot",
    ),
)

FX_CATEGORY_SPECS: tuple[CategoryPanelSpec, ...] = (
    CategoryPanelSpec(
        "FX",
        "FX/Animals and Creatures/Bird/Long FX",
        "fx_animals_and_creatures_bird_long_fx_score",
        "animal_bird_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Animals and Creatures/Bird/One Shots",
        "fx_animals_and_creatures_bird_one_shots_score",
        "animal_bird_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Animals and Creatures/Cat/Long FX",
        "fx_animals_and_creatures_cat_long_fx_score",
        "animal_cat_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Animals and Creatures/Cat/One Shots",
        "fx_animals_and_creatures_cat_one_shots_score",
        "animal_cat_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Animals and Creatures/Cricket/Long FX",
        "fx_animals_and_creatures_cricket_long_fx_score",
        "animal_cricket_insect_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Animals and Creatures/Cricket/One Shots",
        "fx_animals_and_creatures_cricket_one_shots_score",
        "animal_cricket_insect_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Animals and Creatures/Dog/Long FX",
        "fx_animals_and_creatures_dog_long_fx_score",
        "animal_dog_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Animals and Creatures/Dog/One Shots",
        "fx_animals_and_creatures_dog_one_shots_score",
        "animal_dog_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX", "FX/Designed Noise FX/Alarm/Long FX", "fx_designed_noise_fx_alarm_long_fx_score", "fx_alarm_score", "long"
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Designed Noise FX/Alarm/One Shots",
        "fx_designed_noise_fx_alarm_one_shots_score",
        "fx_alarm_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Designed Noise FX/Beep/One Shots",
        "fx_designed_noise_fx_beep_one_shots_score",
        "fx_beep_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Designed Noise FX/Blip/One Shots",
        "fx_designed_noise_fx_blip_one_shots_score",
        "fx_blip_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX", "FX/Designed Noise FX/Siren/Long FX", "fx_designed_noise_fx_siren_long_fx_score", "fx_siren_score", "long"
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Designed Noise FX/Siren/One Shots",
        "fx_designed_noise_fx_siren_one_shots_score",
        "fx_siren_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/Long FX",
        "fx_digital_mechanical_industrial_transport_glitches_and_stutters_generic_glitch_long_fx_score",
        "fx_glitch_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/One Shots",
        "fx_digital_mechanical_industrial_transport_glitches_and_stutters_generic_glitch_one_shots_score",
        "fx_glitch_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Stutter/Long FX",
        "fx_digital_mechanical_industrial_transport_glitches_and_stutters_generic_stutter_long_fx_score",
        "fx_stutter_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Stutter/One Shots",
        "fx_digital_mechanical_industrial_transport_glitches_and_stutters_generic_stutter_one_shots_score",
        "fx_stutter_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Everyday Foley/Doors/Generic Door/Long FX",
        "fx_everyday_foley_doors_generic_door_long_fx_score",
        "fx_door_foley_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Everyday Foley/Doors/Generic Door/One Shots",
        "fx_everyday_foley_doors_generic_door_one_shots_score",
        "fx_door_foley_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Everyday Foley/Keys Coins and Small Objects/Coins/Long FX",
        "fx_everyday_foley_keys_coins_and_small_objects_coins_long_fx_score",
        "fx_coin_object_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Everyday Foley/Keys Coins and Small Objects/Coins/One Shots",
        "fx_everyday_foley_keys_coins_and_small_objects_coins_one_shots_score",
        "fx_coin_object_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Everyday Foley/Keys Coins and Small Objects/Keys/Long FX",
        "fx_everyday_foley_keys_coins_and_small_objects_keys_long_fx_score",
        "fx_key_object_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Everyday Foley/Keys Coins and Small Objects/Keys/One Shots",
        "fx_everyday_foley_keys_coins_and_small_objects_keys_one_shots_score",
        "fx_key_object_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Everyday Foley/Machines/Engine/Long FX",
        "fx_everyday_foley_machines_engine_long_fx_score",
        "fx_engine_machine_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Everyday Foley/Machines/Engine/One Shots",
        "fx_everyday_foley_machines_engine_one_shots_score",
        "fx_engine_machine_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Everyday Foley/Machines/Generic Machine/Long FX",
        "fx_everyday_foley_machines_generic_machine_long_fx_score",
        "fx_machine_mechanical_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Everyday Foley/Machines/Generic Machine/One Shots",
        "fx_everyday_foley_machines_generic_machine_one_shots_score",
        "fx_machine_mechanical_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Everyday Foley/Machines/Motor/Long FX",
        "fx_everyday_foley_machines_motor_long_fx_score",
        "fx_motor_machine_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Everyday Foley/Machines/Motor/One Shots",
        "fx_everyday_foley_machines_motor_one_shots_score",
        "fx_motor_machine_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Human and Voice FX/Applause/Long FX",
        "fx_human_and_voice_fx_applause_long_fx_score",
        "human_applause_crowd_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Human and Voice FX/Applause/One Shots",
        "fx_human_and_voice_fx_applause_one_shots_score",
        "human_applause_crowd_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Human and Voice FX/Breath/Long FX",
        "fx_human_and_voice_fx_breath_long_fx_score",
        "human_breath_mouth_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Human and Voice FX/Crowd/Long FX",
        "fx_human_and_voice_fx_crowd_long_fx_score",
        "human_applause_crowd_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Human and Voice FX/Crowd/One Shots",
        "fx_human_and_voice_fx_crowd_one_shots_score",
        "human_applause_crowd_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Human and Voice FX/Mouth Sounds/Long FX",
        "fx_human_and_voice_fx_mouth_sounds_long_fx_score",
        "human_breath_mouth_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Human and Voice FX/Mouth Sounds/One Shots",
        "fx_human_and_voice_fx_mouth_sounds_one_shots_score",
        "human_breath_mouth_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Human and Voice FX/Scream/Long FX",
        "fx_human_and_voice_fx_scream_long_fx_score",
        "human_scream_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Human and Voice FX/Spoken Voice/Long FX",
        "fx_human_and_voice_fx_spoken_voice_long_fx_score",
        "human_spoken_voice_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Human and Voice FX/Spoken Voice/One Shots",
        "fx_human_and_voice_fx_spoken_voice_one_shots_score",
        "human_spoken_voice_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX", "FX/Impacts and Hits/Boom/Long FX", "fx_impacts_and_hits_boom_long_fx_score", "fx_boom_score", "long"
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Impacts and Hits/Boom/One Shots",
        "fx_impacts_and_hits_boom_one_shots_score",
        "fx_boom_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Impacts and Hits/Generic Impact/Long FX",
        "fx_impacts_and_hits_generic_impact_long_fx_score",
        "fx_impact_score",
        "long",
        ("generic",),
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Impacts and Hits/Generic Impact/One Shots",
        "fx_impacts_and_hits_generic_impact_one_shots_score",
        "fx_impact_score",
        "one_shot",
        ("generic",),
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Impacts and Hits/Short Impact/Long FX",
        "fx_impacts_and_hits_short_impact_long_fx_score",
        "fx_impact_score",
        "long",
        ("short_impact",),
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Impacts and Hits/Short Impact/One Shots",
        "fx_impacts_and_hits_short_impact_one_shots_score",
        "fx_impact_score",
        "one_shot",
        ("short_impact",),
    ),
    CategoryPanelSpec(
        "FX", "FX/Impacts and Hits/Slam/Long FX", "fx_impacts_and_hits_slam_long_fx_score", "fx_slam_score", "long"
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Impacts and Hits/Slam/One Shots",
        "fx_impacts_and_hits_slam_one_shots_score",
        "fx_slam_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Impacts and Hits/Sub Hit/One Shots",
        "fx_impacts_and_hits_sub_hit_one_shots_score",
        "fx_sub_hit_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX",
        "fx_structural_and_transitional_fx_drops_and_downlifters_generic_drop_or_downlifter_long_fx_score",
        "fx_drop_downlifter_score",
        "long",
        ("generic",),
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/One Shots",
        "fx_structural_and_transitional_fx_drops_and_downlifters_generic_drop_or_downlifter_one_shots_score",
        "fx_drop_downlifter_score",
        "one_shot",
        ("generic",),
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Structural and Transitional FX/Reverses and Tails/Generic Reverse/Long FX",
        "fx_structural_and_transitional_fx_reverses_and_tails_generic_reverse_long_fx_score",
        "fx_reverse_score",
        "long",
        ("generic",),
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Structural and Transitional FX/Reverses and Tails/Generic Reverse/One Shots",
        "fx_structural_and_transitional_fx_reverses_and_tails_generic_reverse_one_shots_score",
        "fx_reverse_score",
        "one_shot",
        ("generic",),
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Structural and Transitional FX/Reverses and Tails/Reverse Cymbal/Long FX",
        "fx_structural_and_transitional_fx_reverses_and_tails_reverse_cymbal_long_fx_score",
        "fx_reverse_cymbal_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Structural and Transitional FX/Risers and Builds/Generic Build/Long FX",
        "fx_structural_and_transitional_fx_risers_and_builds_generic_build_long_fx_score",
        "fx_build_score",
        "long",
        ("generic",),
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX",
        "fx_structural_and_transitional_fx_risers_and_builds_generic_riser_long_fx_score",
        "fx_riser_score",
        "long",
        ("generic",),
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Structural and Transitional FX/Risers and Builds/Generic Riser/One Shots",
        "fx_structural_and_transitional_fx_risers_and_builds_generic_riser_one_shots_score",
        "fx_riser_score",
        "one_shot",
        ("generic",),
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Structural and Transitional FX/Risers and Builds/Granular Riser/Long FX",
        "fx_structural_and_transitional_fx_risers_and_builds_granular_riser_long_fx_score",
        "fx_granular_riser_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Structural and Transitional FX/Risers and Builds/Long Riser/Long FX",
        "fx_structural_and_transitional_fx_risers_and_builds_long_riser_long_fx_score",
        "fx_long_riser_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Structural and Transitional FX/Risers and Builds/Short Riser/Long FX",
        "fx_structural_and_transitional_fx_risers_and_builds_short_riser_long_fx_score",
        "fx_short_riser_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Structural and Transitional FX/Risers and Builds/Short Riser/One Shots",
        "fx_structural_and_transitional_fx_risers_and_builds_short_riser_one_shots_score",
        "fx_short_riser_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX",
        "fx_structural_and_transitional_fx_risers_and_builds_synth_riser_long_fx_score",
        "fx_synth_riser_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Structural and Transitional FX/Sweeps and Whooshes/Generic Whoosh or Sweep/Long FX",
        "fx_structural_and_transitional_fx_sweeps_and_whooshes_generic_whoosh_or_sweep_long_fx_score",
        "fx_whoosh_sweep_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Structural and Transitional FX/Sweeps and Whooshes/Generic Whoosh or Sweep/One Shots",
        "fx_structural_and_transitional_fx_sweeps_and_whooshes_generic_whoosh_or_sweep_one_shots_score",
        "fx_whoosh_sweep_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Textures/Natural Ambience/Coffee Shop Ambience/Long FX",
        "fx_textures_natural_ambience_coffee_shop_ambience_long_fx_score",
        "texture_room_crowd_ambience_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Textures/Natural Ambience/Fire/Long FX",
        "fx_textures_natural_ambience_fire_long_fx_score",
        "texture_fire_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Textures/Natural Ambience/Fire/One Shots",
        "fx_textures_natural_ambience_fire_one_shots_score",
        "texture_fire_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Textures/Natural Ambience/Ocean/Long FX",
        "fx_textures_natural_ambience_ocean_long_fx_score",
        "texture_ocean_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Textures/Natural Ambience/Ocean/One Shots",
        "fx_textures_natural_ambience_ocean_one_shots_score",
        "texture_ocean_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Textures/Natural Ambience/Rain/Long FX",
        "fx_textures_natural_ambience_rain_long_fx_score",
        "texture_rain_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Textures/Natural Ambience/Rain/One Shots",
        "fx_textures_natural_ambience_rain_one_shots_score",
        "texture_rain_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Textures/Natural Ambience/Thunder/Long FX",
        "fx_textures_natural_ambience_thunder_long_fx_score",
        "texture_thunder_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Textures/Natural Ambience/Thunder/One Shots",
        "fx_textures_natural_ambience_thunder_one_shots_score",
        "texture_thunder_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Textures/Natural Ambience/Water/Long FX",
        "fx_textures_natural_ambience_water_long_fx_score",
        "texture_water_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Textures/Natural Ambience/Water/One Shots",
        "fx_textures_natural_ambience_water_one_shots_score",
        "texture_water_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Textures/Natural Ambience/Waves/Long FX",
        "fx_textures_natural_ambience_waves_long_fx_score",
        "texture_waves_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Textures/Natural Ambience/Waves/One Shots",
        "fx_textures_natural_ambience_waves_one_shots_score",
        "texture_waves_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Textures/Natural Ambience/Wind/Long FX",
        "fx_textures_natural_ambience_wind_long_fx_score",
        "texture_wind_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Textures/Natural Ambience/Wind/One Shots",
        "fx_textures_natural_ambience_wind_one_shots_score",
        "texture_wind_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Textures/Noise and Static/Hiss/Long FX",
        "fx_textures_noise_and_static_hiss_long_fx_score",
        "texture_hiss_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Textures/Noise and Static/Hiss/One Shots",
        "fx_textures_noise_and_static_hiss_one_shots_score",
        "texture_hiss_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Textures/Noise and Static/Static/Long FX",
        "fx_textures_noise_and_static_static_long_fx_score",
        "texture_static_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Textures/Noise and Static/Vinyl Noise/Long FX",
        "fx_textures_noise_and_static_vinyl_noise_long_fx_score",
        "texture_vinyl_noise_score",
        "long",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Textures/Noise and Static/Vinyl Noise/One Shots",
        "fx_textures_noise_and_static_vinyl_noise_one_shots_score",
        "texture_vinyl_noise_score",
        "one_shot",
    ),
    CategoryPanelSpec(
        "FX",
        "FX/Textures/Noise and Static/White Noise/Long FX",
        "fx_textures_noise_and_static_white_noise_long_fx_score",
        "texture_white_noise_score",
        "long",
    ),
)

ALL_CATEGORY_SPECS: tuple[CategoryPanelSpec, ...] = DRUM_CATEGORY_SPECS + INSTRUMENT_CATEGORY_SPECS + FX_CATEGORY_SPECS
EXPECTED_CATEGORY_PANEL_COUNT = len(ALL_CATEGORY_SPECS)


def finite_float(value: Any, default: float = 0.0) -> float:
    """Return a finite float or default."""
    try:
        number = float(value)
        if math.isfinite(number):
            return number
    except Exception:
        pass
    return float(default)


def clamp01(value: float) -> float:
    """Clamp a score to 0..1."""
    return max(0.0, min(1.0, finite_float(value, 0.0)))


def ramp(value: float, start: float, full: float) -> float:
    """Linear 0..1 ramp."""
    if full <= start:
        return 0.0
    return clamp01((finite_float(value, 0.0) - float(start)) / (float(full) - float(start)))


def inverse_ramp(value: float, good_at_or_below: float, bad_at_or_above: float) -> float:
    """Inverse linear 1..0 ramp."""
    if bad_at_or_above <= good_at_or_below:
        return 0.0
    return clamp01(
        (float(bad_at_or_above) - finite_float(value, 0.0)) / (float(bad_at_or_above) - float(good_at_or_below))
    )


def get_score(scores: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Read a bounded source score."""
    return clamp01(finite_float(scores.get(key, default), default))


def get_value(values: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Read a raw measured feature."""
    return finite_float(values.get(key, default), default)


def role_modifier(role: str, values: dict[str, Any], source_scores: dict[str, Any]) -> float:
    """Return role fit for one-shot/long/loop categories."""
    role_one_shot = get_score(source_scores, "role_one_shot_score")
    role_loop = get_score(source_scores, "role_loop_score")
    role_phrase = get_score(source_scores, "role_phrase_score")
    event_count = max(0.0, math.expm1(max(0.0, get_value(values, "log_transient_count", 0.0))))
    onset_span = get_value(values, "onset_span_ratio", 0.0)
    tail = get_value(values, "tail_energy_ratio", 0.0)
    sustained = get_value(values, "loop_sustained_tonal_frame_ratio", 0.0)
    if role == "one_shot":
        return clamp01(
            0.56 * role_one_shot
            + 0.22 * inverse_ramp(event_count, 1.0, 8.0)
            + 0.12 * inverse_ramp(onset_span, 0.06, 0.50)
            + 0.10 * inverse_ramp(tail, 0.04, 0.66)
        )
    if role == "long":
        return clamp01(
            0.32 * role_phrase
            + 0.25 * role_loop
            + 0.18 * ramp(onset_span, 0.22, 0.86)
            + 0.15 * ramp(tail, 0.12, 0.78)
            + 0.10 * ramp(sustained, 0.30, 0.92)
        )
    if role == "loop":
        return clamp01(
            0.55 * role_loop
            + 0.20 * role_phrase
            + 0.15 * ramp(onset_span, 0.32, 0.88)
            + 0.10 * ramp(event_count, 4.0, 20.0)
        )
    return 0.70


def modifier_score(modifier: str, values: dict[str, Any], source_scores: dict[str, Any]) -> float:
    """Return additional feature fit for specific leaf flavors."""
    flatness = get_value(values, "spectral_flatness_mean", 0.0)
    get_value(values, "spectral_entropy_mean", 0.0)
    attack = get_value(values, "attack_rise_time_norm", 1.0)
    tail = get_value(values, "tail_energy_ratio", 0.0)
    high = get_value(values, "presence_ratio_2000_8000hz", 0.0) + get_value(values, "air_ratio_gt_8000hz", 0.0)
    mid = get_value(values, "mid_ratio_500_2000hz", 0.0)
    sub = get_value(values, "sub_bass_ratio_lt_150hz", 0.0)
    bass = get_value(values, "bass_ratio_150_500hz", 0.0)
    low_total = sub + bass
    inharm = get_value(values, "inharmonicity", 0.0)
    event_count = max(0.0, math.expm1(max(0.0, get_value(values, "log_transient_count", 0.0))))
    onset_span = get_value(values, "onset_span_ratio", 0.0)
    pitch = max(get_value(values, "pitch_confidence", 0.0), get_value(values, "loop_mean_event_pitch_confidence", 0.0))
    harmonic = get_value(values, "harmonic_energy_ratio", 0.0)
    get_value(values, "zcr_mean", 0.0)
    get_value(values, "centroid_slope_norm", 0.0)
    get_value(values, "stereo_width", 0.0)
    low_peak = get_value(values, "low_peak_frequency_hz", 0.0)

    lookup: dict[str, Callable[[], float]] = {
        "generic": lambda: 0.58,
        "organic_hand": lambda: clamp01(
            0.35 * get_score(source_scores, "drum_clap_source_score")
            + 0.25 * ramp(mid, 0.22, 0.72)
            + 0.20 * ramp(flatness, 0.18, 0.56)
            + 0.20 * inverse_ramp(tail, 0.03, 0.34)
        ),
        "crash_tail": lambda: clamp01(
            0.34 * ramp(high, 0.18, 0.72)
            + 0.30 * ramp(tail, 0.18, 0.86)
            + 0.20 * ramp(inharm, 0.10, 0.48)
            + 0.16 * inverse_ramp(event_count, 1.0, 5.0)
        ),
        "ride_ping": lambda: clamp01(
            0.32 * ramp(pitch, 0.26, 0.76)
            + 0.28 * ramp(inharm, 0.10, 0.54)
            + 0.22 * ramp(high, 0.16, 0.58)
            + 0.18 * inverse_ramp(tail, 0.12, 0.82)
        ),
        "splash_short": lambda: clamp01(
            0.36 * ramp(high, 0.24, 0.78)
            + 0.32 * inverse_ramp(tail, 0.06, 0.42)
            + 0.18 * ramp(inharm, 0.08, 0.42)
            + 0.14 * inverse_ramp(event_count, 1.0, 4.0)
        ),
        "short_bright": lambda: clamp01(
            0.40 * inverse_ramp(attack, 0.004, 0.08)
            + 0.34 * ramp(high, 0.20, 0.82)
            + 0.26 * inverse_ramp(tail, 0.02, 0.24)
        ),
        "open_hat_tail": lambda: clamp01(
            0.42 * ramp(high, 0.22, 0.78)
            + 0.32 * ramp(tail, 0.10, 0.66)
            + 0.16 * ramp(flatness, 0.14, 0.55)
            + 0.10 * inverse_ramp(low_total, 0.03, 0.30)
        ),
        "short_low": lambda: clamp01(
            0.42 * ramp(low_total, 0.45, 0.94)
            + 0.30 * inverse_ramp(tail, 0.04, 0.34)
            + 0.28 * inverse_ramp(high, 0.01, 0.18)
        ),
        "sub_low": lambda: clamp01(
            0.56 * ramp(sub, 0.38, 0.90)
            + 0.22 * inverse_ramp(low_peak or 300.0, 45.0, 150.0)
            + 0.22 * inverse_ramp(high, 0.01, 0.16)
        ),
        "metallic": lambda: clamp01(
            0.44 * ramp(inharm, 0.14, 0.62) + 0.30 * ramp(high, 0.12, 0.58) + 0.26 * ramp(pitch, 0.20, 0.76)
        ),
        "scrape": lambda: clamp01(
            0.38 * ramp(onset_span, 0.18, 0.82)
            + 0.30 * ramp(flatness, 0.20, 0.62)
            + 0.20 * ramp(event_count, 2.0, 12.0)
            + 0.12 * inverse_ramp(pitch, 0.05, 0.48)
        ),
        "shaker_roll": lambda: clamp01(
            0.42 * ramp(event_count, 3.0, 18.0)
            + 0.30 * ramp(high, 0.16, 0.70)
            + 0.18 * ramp(flatness, 0.22, 0.66)
            + 0.10 * inverse_ramp(low_total, 0.04, 0.30)
        ),
        "rimshot": lambda: clamp01(
            0.36 * inverse_ramp(attack, 0.004, 0.06)
            + 0.26 * ramp(mid + high, 0.28, 0.84)
            + 0.22 * inverse_ramp(tail, 0.02, 0.24)
            + 0.16 * ramp(pitch, 0.12, 0.56)
        ),
        "sidestick": lambda: clamp01(
            0.34 * inverse_ramp(attack, 0.004, 0.06)
            + 0.30 * ramp(mid, 0.28, 0.78)
            + 0.22 * inverse_ramp(high, 0.08, 0.42)
            + 0.14 * inverse_ramp(tail, 0.02, 0.24)
        ),
        "acoustic_snare": lambda: clamp01(
            0.36 * get_score(source_scores, "drum_snare_source_score")
            + 0.24 * ramp(mid + high, 0.36, 0.88)
            + 0.22 * ramp(flatness, 0.14, 0.50)
            + 0.18 * inverse_ramp(tail, 0.04, 0.44)
        ),
        "floor_tom": lambda: clamp01(
            0.42 * ramp(low_total, 0.40, 0.86)
            + 0.26 * ramp(pitch, 0.28, 0.76)
            + 0.20 * inverse_ramp(high, 0.04, 0.32)
            + 0.12 * ramp(tail, 0.05, 0.40)
        ),
        "low_tom": lambda: clamp01(
            0.38 * ramp(low_total, 0.36, 0.82)
            + 0.26 * ramp(pitch, 0.24, 0.72)
            + 0.22 * inverse_ramp(high, 0.04, 0.34)
            + 0.14 * inverse_ramp(tail, 0.04, 0.40)
        ),
        "tabla": lambda: clamp01(
            0.30 * ramp(pitch, 0.30, 0.82)
            + 0.26 * ramp(mid, 0.20, 0.62)
            + 0.24 * inverse_ramp(attack, 0.004, 0.09)
            + 0.20 * ramp(event_count, 1.0, 8.0)
        ),
        "bongo": lambda: clamp01(
            0.32 * ramp(mid, 0.24, 0.66)
            + 0.28 * ramp(pitch, 0.25, 0.72)
            + 0.22 * inverse_ramp(high, 0.04, 0.34)
            + 0.18 * inverse_ramp(tail, 0.03, 0.34)
        ),
        "conga": lambda: clamp01(
            0.34 * ramp(low_total, 0.24, 0.70)
            + 0.24 * ramp(mid, 0.18, 0.58)
            + 0.22 * ramp(pitch, 0.25, 0.70)
            + 0.20 * ramp(tail, 0.04, 0.42)
        ),
        "generic_bass": lambda: clamp01(
            0.42 * ramp(low_total, 0.28, 0.82)
            + 0.24 * ramp(pitch, 0.28, 0.78)
            + 0.20 * inverse_ramp(high, 0.01, 0.22)
            + 0.14 * inverse_ramp(flatness, 0.03, 0.36)
        ),
        "instrument_loop": lambda: clamp01(
            0.34 * get_score(source_scores, "role_loop_score")
            + 0.24 * ramp(pitch, 0.30, 0.86)
            + 0.20 * ramp(event_count, 4.0, 24.0)
            + 0.12 * inverse_ramp(get_score(source_scores, "drum_loop_source_score"), 0.10, 0.70)
            + 0.10 * inverse_ramp(get_score(source_scores, "fx_motion_score"), 0.10, 0.70)
        ),
        "electric_piano": lambda: clamp01(
            0.34 * get_score(source_scores, "struck_keys_score")
            + 0.24 * ramp(mid, 0.34, 0.88)
            + 0.20 * inverse_ramp(flatness, 0.006, 0.16)
            + 0.12 * ramp(tail, 0.10, 0.56)
            + 0.10 * ramp(harmonic, 0.18, 0.72)
        ),
        "short_impact": lambda: clamp01(
            0.42 * get_score(source_scores, "fx_impact_score")
            + 0.28 * inverse_ramp(tail, 0.06, 0.48)
            + 0.18 * inverse_ramp(attack, 0.004, 0.10)
            + 0.12 * inverse_ramp(event_count, 1.0, 5.0)
        ),
    }
    return clamp01(lookup.get(modifier, lambda: 0.60)())


def build_derived_source_scores(values: dict[str, Any], source_scores: dict[str, Any]) -> dict[str, float]:
    """Build derived intermediate scores needed by the complete catalog."""
    high = get_value(values, "presence_ratio_2000_8000hz", 0.0) + get_value(values, "air_ratio_gt_8000hz", 0.0)
    mid = get_value(values, "mid_ratio_500_2000hz", 0.0)
    low_total = get_value(values, "sub_bass_ratio_lt_150hz", 0.0) + get_value(values, "bass_ratio_150_500hz", 0.0)
    flatness = get_value(values, "spectral_flatness_mean", 0.0)
    entropy = get_value(values, "spectral_entropy_mean", 0.0)
    tail = get_value(values, "tail_energy_ratio", 0.0)
    slope = get_value(values, "centroid_slope_norm", 0.0)
    event_count = max(0.0, math.expm1(max(0.0, get_value(values, "log_transient_count", 0.0))))
    pitch = max(get_value(values, "pitch_confidence", 0.0), get_value(values, "loop_mean_event_pitch_confidence", 0.0))
    get_value(values, "harmonic_energy_ratio", 0.0)
    zcr = get_value(values, "zcr_mean", 0.0)
    onset_span = get_value(values, "onset_span_ratio", 0.0)
    get_value(values, "inharmonicity", 0.0)
    stereo = get_value(values, "stereo_width", 0.0)

    blip_beep = get_score(source_scores, "fx_blip_beep_score")
    riser = get_score(source_scores, "fx_riser_build_score")
    reverse = get_score(source_scores, "fx_reverse_score")
    texture_water = get_score(source_scores, "texture_water_ocean_score")
    texture_noise = get_score(source_scores, "texture_noise_static_score")
    machine = get_score(source_scores, "fx_machine_mechanical_score")
    glitch_stutter = get_score(source_scores, "fx_glitch_stutter_score")

    return {
        "fx_beep_score": clamp01(
            0.62 * blip_beep
            + 0.18 * ramp(pitch, 0.45, 0.92)
            + 0.12 * inverse_ramp(tail, 0.02, 0.22)
            + 0.08 * inverse_ramp(flatness, 0.02, 0.28)
        ),
        "fx_blip_score": clamp01(
            0.58 * blip_beep
            + 0.18 * ramp(high + mid, 0.10, 0.48)
            + 0.14 * inverse_ramp(event_count, 1.0, 3.5)
            + 0.10 * inverse_ramp(tail, 0.02, 0.26)
        ),
        "fx_glitch_score": clamp01(
            0.66 * glitch_stutter + 0.18 * ramp(entropy, 0.44, 0.86) + 0.16 * ramp(event_count, 3.0, 20.0)
        ),
        "fx_stutter_score": clamp01(
            0.58 * glitch_stutter + 0.24 * ramp(event_count, 4.0, 28.0) + 0.18 * ramp(onset_span, 0.28, 0.86)
        ),
        "fx_riser_score": clamp01(0.60 * riser + 0.25 * ramp(slope, 0.05, 0.24) + 0.15 * ramp(tail, 0.18, 0.78)),
        "fx_build_score": clamp01(
            0.48 * riser
            + 0.26 * ramp(onset_span, 0.32, 0.90)
            + 0.16 * ramp(event_count, 4.0, 28.0)
            + 0.10 * ramp(slope, 0.02, 0.20)
        ),
        "fx_granular_riser_score": clamp01(
            0.46 * riser
            + 0.24 * ramp(flatness, 0.18, 0.62)
            + 0.18 * ramp(entropy, 0.42, 0.82)
            + 0.12 * ramp(onset_span, 0.28, 0.88)
        ),
        "fx_long_riser_score": clamp01(
            0.62 * riser + 0.26 * role_modifier("long", values, source_scores) + 0.12 * ramp(tail, 0.22, 0.86)
        ),
        "fx_short_riser_score": clamp01(
            0.60 * riser
            + 0.26 * role_modifier("one_shot", values, source_scores)
            + 0.14 * inverse_ramp(tail, 0.04, 0.44)
        ),
        "fx_synth_riser_score": clamp01(
            0.44 * riser
            + 0.24 * get_score(source_scores, "synth_tonal_source_score")
            + 0.18 * ramp(pitch, 0.34, 0.86)
            + 0.14 * inverse_ramp(flatness, 0.02, 0.28)
        ),
        "fx_reverse_cymbal_score": clamp01(
            0.54 * reverse
            + 0.24 * get_score(source_scores, "drum_cymbal_source_score")
            + 0.12 * ramp(high, 0.16, 0.60)
            + 0.10 * ramp(tail, 0.18, 0.80)
        ),
        "texture_ocean_score": clamp01(
            0.50 * texture_water
            + 0.20 * ramp(low_total + mid, 0.22, 0.74)
            + 0.18 * ramp(onset_span, 0.30, 0.90)
            + 0.12 * ramp(stereo, 0.16, 0.72)
        ),
        "texture_water_score": clamp01(
            0.55 * texture_water
            + 0.20 * ramp(flatness, 0.18, 0.60)
            + 0.15 * ramp(high, 0.05, 0.34)
            + 0.10 * ramp(onset_span, 0.18, 0.78)
        ),
        "texture_waves_score": clamp01(
            0.52 * texture_water
            + 0.20 * ramp(onset_span, 0.36, 0.92)
            + 0.18 * ramp(low_total + mid, 0.20, 0.72)
            + 0.10 * ramp(tail, 0.18, 0.80)
        ),
        "texture_hiss_score": clamp01(
            0.54 * texture_noise
            + 0.24 * ramp(high, 0.14, 0.62)
            + 0.14 * ramp(zcr, 0.08, 0.34)
            + 0.08 * inverse_ramp(low_total, 0.02, 0.26)
        ),
        "texture_static_score": clamp01(
            0.56 * texture_noise
            + 0.24 * ramp(flatness, 0.24, 0.72)
            + 0.12 * ramp(entropy, 0.48, 0.90)
            + 0.08 * ramp(event_count, 1.0, 12.0)
        ),
        "texture_white_noise_score": clamp01(
            0.62 * texture_noise
            + 0.18 * ramp(flatness, 0.34, 0.82)
            + 0.12 * ramp(entropy, 0.62, 0.96)
            + 0.08 * inverse_ramp(pitch, 0.02, 0.26)
        ),
        "texture_vinyl_noise_score": clamp01(
            0.48 * texture_noise
            + 0.22 * ramp(event_count, 2.0, 18.0)
            + 0.16 * ramp(mid, 0.18, 0.58)
            + 0.14 * ramp(flatness, 0.18, 0.58)
        ),
        "fx_motor_machine_score": clamp01(
            0.56 * machine
            + 0.18 * ramp(pitch, 0.18, 0.62)
            + 0.16 * ramp(onset_span, 0.30, 0.88)
            + 0.10 * ramp(low_total + mid, 0.24, 0.76)
        ),
        "fx_engine_machine_score": clamp01(
            0.54 * get_score(source_scores, "fx_engine_machine_score")
            + 0.18 * ramp(low_total, 0.20, 0.70)
            + 0.16 * ramp(onset_span, 0.30, 0.90)
            + 0.12 * ramp(flatness, 0.12, 0.46)
        ),
    }


def build_category_panel_scores(
    feature_values: dict[str, Any],
    source_scores: dict[str, Any],
    calibration_profile: dict[str, CategoryCalibrationCurve] | None = None,
) -> dict[str, Any]:
    """Return complete current-brain category panel scores.

    Every active Drums, Instruments, and FX label from the current brain has one
    named score in the returned flat map.  The optional calibration profile lets
    future approved sample percentiles reshape raw witness scores without
    changing category formulas or using filenames.
    """
    values = feature_values or {}
    scores: dict[str, Any] = dict(source_scores or {})
    calibration = calibration_profile if calibration_profile is not None else load_category_panel_calibration()
    scores.update(build_derived_source_scores(values, scores))

    flat: dict[str, float] = {}
    nested: dict[str, dict[str, float]] = {"Drums": {}, "Instruments": {}, "FX": {}}
    for spec in ALL_CATEGORY_SPECS:
        base = get_score(scores, spec.base_key)
        role_fit = role_modifier(spec.role, values, scores)
        modifier_values = [modifier_score(modifier, values, scores) for modifier in spec.modifiers]
        modifier_fit = sum(modifier_values) / len(modifier_values) if modifier_values else 0.62
        raw_category_score = clamp01(0.64 * base + 0.22 * role_fit + 0.14 * modifier_fit)
        category_score = calibrate_category_panel_score(spec.score_key, raw_category_score, calibration)
        flat[spec.score_key] = category_score
        nested.setdefault(spec.top_family, {})[spec.score_key] = category_score

    missing: list[str] = []
    return {
        "drums": {key: round(value, 6) for key, value in nested["Drums"].items()},
        "instruments": {key: round(value, 6) for key, value in nested["Instruments"].items()},
        "fx": {key: round(value, 6) for key, value in nested["FX"].items()},
        "flat": {key: round(value, 6) for key, value in flat.items()},
        "coverage": {
            "expected_category_panel_count": EXPECTED_CATEGORY_PANEL_COUNT,
            "implemented_category_panel_count": len(flat),
            "missing_category_panels": missing,
            "coverage_status": "complete_for_current_stage4_brain_labels",
            "calibration_status": "loaded" if calibration else "uncalibrated_default_curves",
            "calibration_curve_count": len(calibration),
            "calibration_note": "All current Drums/Instruments/FX brain labels have source-name-blind category panels. Optional percentile calibration is implemented and stays disabled unless a calibration JSON is supplied.",
        },
    }
