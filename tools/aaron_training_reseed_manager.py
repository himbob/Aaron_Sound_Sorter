#!/usr/bin/env python3
"""Review-first training taxonomy sync, reseed, and symlink normalizer.

This tool is for repairing Aaron Sound Sorter training data without silently
training bad examples.

Core workflow:
  1. audit-training
  2. normalize-symlinks  (dry run first)
  3. sync-taxonomy       (dry run first, creates missing category slots)
  4. stage-candidates    (creates listening review symlinks)
  5. apply-reviewed-candidates (only after Aaron deletes bad candidates)

Important design rules:
  - File/folder names are allowed only for candidate discovery and review staging.
  - The final sorter brain still learns from reviewed folder placement.
  - FX does not use loop slots. FX uses _ONE_SHOTS and _LONG_FX.
  - Electric piano, Rhodes, Wurli, Organ, Clav, etc. must not contaminate clean Piano.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import time
import wave
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

AUDIO_EXTENSIONS = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".m4a", ".ogg"}
WAV_EXTENSIONS = {".wav"}
MUSICAL_STRUCTURES = ("_LOOPS", "_ONE_SHOTS")
FX_STRUCTURES = ("_ONE_SHOTS", "_LONG_FX")
STRUCTURE_NAMES = {"_LOOPS", "_ONE_SHOTS", "_LONG_FX"}
DEFAULT_REPORT_DIR_NAME = "_reports/training_reseed"
RECOVERED_REAL_FILES_DIR = "_TRAINING_REAL_FILES_RECOVERED"
DEFAULT_TRAINING_ROOT = "/Volumes/T9/testbed/Aaron_Sound_Sorter/training/locked_curated_v1"
DEFAULT_SAMPLES_ROOT = "/Volumes/T9/music_production/samples"
DEFAULT_PROJECT_DIR = "/Volumes/T9/testbed/Aaron_Sound_Sorter"

LOOP_HINT_RE = re.compile(
    r"\b(loop|loops|looped|bpm|bar|bars|groove|grooves|riff|riffs|phrase|phrases|"
    r"melody|melodic|chord|chords|progression|progressions|line|pattern|beat|beats|break|breakbeat)\b",
    re.I,
)
ONE_SHOT_HINT_RE = re.compile(
    r"\b(one[ _-]?shot|oneshot|shot|hit|hits|stab|stabs|single|kick|snare|clap|snap|hat|tom|"
    r"rim|stick|pluck|pluck[ _-]?one|note|slam|boom|thud|impact|beep|blip)\b",
    re.I,
)
EXPLICIT_ONE_SHOT_STRUCTURE_RE = re.compile(
    r"\b(one[ _-]?shot|oneshot|single|single\s+hit|shot|hit|hits|stab|stabs|note|pluck[ _-]?one)\b",
    re.I,
)
BPM_RE = re.compile(r"\b\d{2,3}\s*bpm\b", re.I)


@dataclass(frozen=True)
class AudioStats:
    """Small audio-health summary used only for candidate review ranking."""

    path: Path
    readable: bool
    reason: str
    duration_sec: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    rms: float = 0.0
    peak: float = 0.0
    zcr: float = 0.0
    spectral_centroid_hz: float = 0.0
    transient_ratio: float = 0.0


@dataclass(frozen=True)
class TrainingSlot:
    """One training target folder such as Instruments/Keys/Piano/_ONE_SHOTS."""

    label_rel: Path
    structure: str
    folder: Path


@dataclass(frozen=True)
class Candidate:
    """One source-library candidate staged for listening."""

    source: Path
    label_rel: Path
    structure: str
    score: float
    rank_reason: str
    stats: AudioStats


@dataclass(frozen=True)
class TaxonomyEntry:
    """A training category we know how to create and mine candidates for."""

    label_rel: Path
    family: str
    include_patterns: tuple[str, ...]
    exclude_patterns: tuple[str, ...]
    structures: tuple[str, ...]
    source: str
    aliases: tuple[str, ...] = ()


def rx(*patterns: str) -> tuple[str, ...]:
    return tuple(patterns)


def rel(path: str) -> Path:
    return Path(*[p.strip() for p in path.split("/") if p.strip()])


def fx(path: str, include: tuple[str, ...], exclude: tuple[str, ...] = (), source: str = "embedded") -> TaxonomyEntry:
    return TaxonomyEntry(rel(path), "FX", include, exclude, FX_STRUCTURES, source)


def inst(path: str, include: tuple[str, ...], exclude: tuple[str, ...] = (), source: str = "embedded") -> TaxonomyEntry:
    return TaxonomyEntry(rel(path), "Instruments", include, exclude, MUSICAL_STRUCTURES, source)


def drum(path: str, include: tuple[str, ...], exclude: tuple[str, ...] = (), source: str = "embedded") -> TaxonomyEntry:
    return TaxonomyEntry(rel(path), "Drums", include, exclude, MUSICAL_STRUCTURES, source)


CLEAN_PIANO_EXCLUDE = rx(
    r"\b(electric\s+piano|electric\s+.*piano|e[\s_-]?piano|epiano|rhodes|wurli|wurlitzer|organ|hammond|b3|drawbar|"
    r"clav|clavinet|harpsichord|synth\s+piano|processed\s+piano|reverse\s+piano|fx\s+piano|vibes?|vibraphone)\b"
)
ELECTRIC_PIANO_INCLUDE = rx(
    r"\b(electric\s+piano|electric\s+.*piano|e[\s_-]?piano|epiano|rhodes|wurli|wurlitzer|mk\s?[i1]|suitcase\s+piano|stage\s+piano)\b",
    r"\belec(?:tric)?\s+.*\bpiano\b",
)

# Embedded broad taxonomy. This is intentionally broader than the current training
# tree. sync-taxonomy validates against Aaron's sample library before creating slots.
EMBEDDED_TAXONOMY: tuple[TaxonomyEntry, ...] = (
    # Drums.
    drum(
        "Drums/Kick Drums/Generic Kick",
        rx(r"\b(kick|kik|bd|bass\s*drum)\b"),
        rx(
            r"\b(loop|bass\s+guitar|synth\s+bass|short\s+kick|tight\s+kick|dry\s+kick|sub\s+kick|808\s+kick|low\s+kick|deep\s+kick|acoustic\s+kick|live\s+kick|real\s+kick)\b"
        ),
    ),
    drum("Drums/Kick Drums/Short Kick", rx(r"\b(short\s+kick|tight\s+kick|dry\s+kick)\b")),
    drum("Drums/Kick Drums/Sub Kick", rx(r"\b(sub\s+kick|808\s+kick|low\s+kick|deep\s+kick)\b")),
    drum(
        "Drums/Snares/Generic Snare",
        rx(r"\b(snare|snr|sd)\b"),
        rx(
            r"\b(loop|roll|fill|acoustic\s+snare|live\s+snare|real\s+snare|electronic\s+snare|synth\s+snare|808\s+snare|909\s+snare|clap\s+snare|snare\s+clap)\b"
        ),
    ),
    drum("Drums/Snares/Acoustic Snare", rx(r"\b(acoustic\s+snare|live\s+snare|real\s+snare)\b")),
    drum(
        "Drums/Snares/Electronic Snare",
        rx(r"\b(electronic\s+snare|synth\s+snare|drum\s+machine\s+snare|tr\s?808\s+snare|909\s+snare)\b"),
    ),
    drum("Drums/Snares/Clap Snare", rx(r"\b(clap\s+snare|snare\s+clap|clapstack|clap\s+stack)\b")),
    drum("Drums/Claps Snaps Slaps/Hand Clap", rx(r"\b(clap|hand\s+clap)\b"), rx(r"\b(snare|loop)\b")),
    drum("Drums/Claps Snaps Slaps/Snap", rx(r"\b(snap|finger\s+snap|fingersnap)\b")),
    drum("Drums/Claps Snaps Slaps/Body Slap", rx(r"\b(body\s+slap|slap|smack)\b"), rx(r"\b(bass\s+slap|guitar)\b")),
    drum("Drums/Hi Hats/Closed Hat", rx(r"\b(closed\s+hats?|closedhat|chh|tight\s+hats?)\b")),
    drum("Drums/Hi Hats/Open Hat", rx(r"\b(open\s+hats?|openhat|ohh)\b")),
    drum("Drums/Hi Hats/Pedal Hat", rx(r"\b(pedal\s+hats?|foot\s+hats?)\b")),
    drum(
        "Drums/Hi Hats/Generic Hat",
        rx(r"\b(hi\s*hat|hihat|hats?|hh)\b"),
        rx(
            r"\b(cowboy\s+hat|hatchet|open\s+hats?|openhat|ohh|closed\s+hats?|closedhat|chh|pedal\s+hats?|foot\s+hats?)\b"
        ),
    ),
    drum("Drums/Cymbals/Crash Cymbal", rx(r"\b(crash\s+cymbal|cymbal\s+crash|crash)\b"), rx(r"\b(glass|car|door)\b")),
    drum("Drums/Cymbals/Ride Cymbal", rx(r"\b(ride\s+cymbal|ride)\b"), rx(r"\b(car|bike|horse)\b")),
    drum("Drums/Cymbals/Splash Cymbal", rx(r"\b(splash\s+cymbal|splash)\b"), rx(r"\b(water)\b")),
    drum(
        "Drums/Cymbals/Generic Cymbal",
        rx(r"\b(cymbal|cymbals)\b"),
        rx(r"\b(crash\s+cymbal|cymbal\s+crash|ride\s+cymbal|splash\s+cymbal)\b"),
    ),
    drum(
        "Drums/Toms/Generic Tom",
        rx(r"\b(tom|toms|tomtom)\b"),
        rx(r"\b(high\s+tom|rack\s+tom|floor\s+tom|low\s+tom|mid\s+tom|middle\s+tom)\b"),
    ),
    drum("Drums/Toms/Floor Tom", rx(r"\b(floor\s+tom|low\s+tom)\b")),
    drum("Drums/Toms/High Tom", rx(r"\b(high\s+tom|rack\s+tom)\b")),
    drum("Drums/Toms/Mid Tom", rx(r"\b(mid\s+tom|middle\s+tom|tom\s+mid|tom\s+middle)\b")),
    drum("Drums/Rims and Sticks/Rimshot", rx(r"\b(rimshot|rim\s+shot|rim)\b")),
    drum("Drums/Rims and Sticks/Sidestick", rx(r"\b(side\s+stick|sidestick|cross\s+stick)\b")),
    drum(
        "Drums/Rims and Sticks/Drum Sticks",
        rx(r"\b(drum\s+stick|stick\s+hit|sticks?)\b"),
        rx(r"\b(walking\s+stick|wood\s+stick\s+break)\b"),
    ),
    drum("Drums/Rims and Sticks/Claves and Wood Blocks", rx(r"\b(clave|claves|wood\s*block|woodblock)\b")),
    drum("Drums/Percussion/Shakers and Tambourines", rx(r"\b(shaker|shakers|tamb|tambourine|maraca|cabasa)\b")),
    drum(
        "Drums/Percussion/Guiros Scrapes and Rasps",
        rx(r"\b(guiro|scrape|scraper|rasp|ratchet)\b"),
        rx(r"\b(guitar\s+scrape)\b"),
    ),
    drum(
        "Drums/Percussion/Bells and Metallic Percussion",
        rx(r"\b(cowbell|triangle|agogo|metallic\s+perc|bell\s+perc|perc\s+bell)\b"),
    ),
    drum("Drums/Percussion/Generic Percussion", rx(r"\b(perc|percussion|percussive)\b"), rx(r"\b(loop)\b")),
    drum("Drums/World Percussion/Latin Percussion/Bongo", rx(r"\b(bongo|bongos)\b")),
    drum("Drums/World Percussion/Latin Percussion/Conga", rx(r"\b(conga|congas)\b")),
    drum("Drums/World Percussion/Latin Percussion/Timbale", rx(r"\b(timbale|timbales)\b")),
    drum("Drums/World Percussion/Indian Percussion/Tabla", rx(r"\b(tabla)\b")),
    drum("Drums/World Percussion/African Percussion/Djembe", rx(r"\b(djembe|jembe)\b")),
    drum("Drums/World Percussion/Hand Percussion/Cajon", rx(r"\b(cajon)\b")),
    drum(
        "Drums/Drum Loops/Full Drum Loops", rx(r"\b(drum\s+loop|drumloop|drums\s+loop|beat|breakbeat|break\s+beat)\b")
    ),
    drum("Drums/Drum Fills and Rolls/Drum Fill", rx(r"\b(drum\s+fill|fill)\b"), rx(r"\b(synth\s+fill)\b")),
    drum("Drums/Drum Fills and Rolls/Drum Roll", rx(r"\b(drum\s+roll|snare\s+roll|tom\s+roll|roll)\b")),
    drum(
        "Drums/Percussion Loops/Percussion Groove",
        rx(r"\b(perc(?:ussion)?\s+loop|percussion\s+groove|perc\s+groove)\b"),
    ),
    # Instruments, bass.
    inst(
        "Instruments/Bass/808 Bass",
        rx(r"\b(808|808s|eight\s*oh\s*eight)\b"),
        rx(r"\b(808\s+kick|808\s+snare|808\s+hat)\b"),
    ),
    inst("Instruments/Bass/Sub Bass", rx(r"\b(sub\s+bass|subbass|sub)\b"), rx(r"\b(sub\s+kick)\b")),
    inst("Instruments/Bass/Synth Bass", rx(r"\b(synth\s+bass|bass\s+synth|reese|reece)\b")),
    inst("Instruments/Bass/Electric Bass", rx(r"\b(electric\s+bass|bass\s+guitar|bassgtr|bass\s+gtr|ebass)\b")),
    inst("Instruments/Bass/Upright Bass", rx(r"\b(upright\s+bass|double\s+bass|contrabass)\b")),
    inst(
        "Instruments/Bass/Generic Bass",
        rx(r"\b(bass|bassline|bass\s+line)\b"),
        rx(
            r"\b(kick|bass\s+drum|bassoon|808|sub\s+bass|subbass|synth\s+bass|bass\s+synth|reese|reece|electric\s+bass|bass\s+guitar|upright\s+bass|double\s+bass|contrabass)\b"
        ),
    ),
    # Guitar.
    inst("Instruments/Guitar/Acoustic Guitar", rx(r"\b(acoustic\s+guitar|ac\s*gtr|acgtr|steel\s+string)\b")),
    inst("Instruments/Guitar/Nylon Guitar", rx(r"\b(nylon\s+guitar|classical\s+guitar|spanish\s+guitar)\b")),
    inst(
        "Instruments/Guitar/Electric Guitar",
        rx(r"\b(electric\s+guitar|el\s*gtr|elgtr|egtr|distorted\s+guitar|wah\s+guitar)\b"),
    ),
    inst("Instruments/Guitar/Guitar Plucks", rx(r"\b(guitar\s+pluck|gtr\s+pluck|pluck\s+guitar|guitar\s+pick)\b")),
    inst("Instruments/Guitar/Guitar Chords", rx(r"\b(guitar\s+chord|gtr\s+chord|guitarschord|gtrchord)\b")),
    inst("Instruments/Guitar/Guitar Loops", rx(r"\b(guitar|gtr|guitars?)\b"), rx(r"\b(piano|rhodes|kick|snare|hat)\b")),
    # Keys.
    inst(
        "Instruments/Keys/Piano",
        rx(r"\b(piano|grand\s+piano|upright\s+piano|acoustic\s+piano|pno)\b"),
        CLEAN_PIANO_EXCLUDE,
    ),
    inst(
        "Instruments/Keys/Electric Piano",
        ELECTRIC_PIANO_INCLUDE,
        rx(r"\b(organ|hammond|clav|harpsichord|guitar|kick|snare|hat)\b"),
    ),
    inst("Instruments/Keys/Rhodes", rx(r"\b(rhodes|suitcase\s+piano|stage\s+piano)\b")),
    inst("Instruments/Keys/Wurlitzer", rx(r"\b(wurli|wurlitzer)\b")),
    inst(
        "Instruments/Keys/Organ",
        rx(r"\b(organ|hammond|b3|drawbar|farfisa|vox\s+continental)\b"),
        rx(r"\b(organic|organism)\b"),
    ),
    inst("Instruments/Keys/Clavinet", rx(r"\b(clav|clavinet)\b")),
    inst("Instruments/Keys/Harpsichord", rx(r"\b(harpsichord|cembalo)\b")),
    inst(
        "Instruments/Keys/Processed Keys",
        rx(
            r"\b(reverse\s+piano|processed\s+keys?|processed\s+piano|synth\s+piano|fx\s+piano|cmpzr|vibes?|vibraphone)\b"
        ),
    ),
    inst(
        "Instruments/Keys/Keys Loops",
        rx(r"\b(keys?|keyboard|keyboards)\b"),
        rx(r"\b(car\s+keys?|house\s+keys?|carkeys|coin)\b"),
    ),
    # Mallets, bells, plucked/special instruments.
    inst("Instruments/Mallets and Bells/Marimba", rx(r"\b(marimba)\b")),
    inst("Instruments/Mallets and Bells/Xylophone", rx(r"\b(xylophone|xylo)\b")),
    inst("Instruments/Mallets and Bells/Vibraphone", rx(r"\b(vibraphone|vibes|vibe)\b")),
    inst("Instruments/Mallets and Bells/Glockenspiel", rx(r"\b(glockenspiel|glock)\b")),
    inst("Instruments/Mallets and Bells/Kalimba", rx(r"\b(kalimba|thumb\s+piano)\b")),
    inst(
        "Instruments/Mallets and Bells/Bells and Mallets",
        rx(r"\b(bell|bells|mallet|mallets|chime|chimes)\b"),
        rx(r"\b(door\s+bell|bike\s+bell|alarm)\b"),
    ),
    inst("Instruments/Plucked Strings/Harp", rx(r"\b(harp)\b"), rx(r"\b(harpsichord)\b")),
    inst("Instruments/Plucked Strings/Sitar", rx(r"\b(sitar)\b")),
    inst("Instruments/Plucked Strings/Koto", rx(r"\b(koto)\b")),
    inst("Instruments/Plucked Strings/Banjo", rx(r"\b(banjo)\b")),
    inst("Instruments/Plucked Strings/Mandolin", rx(r"\b(mandolin)\b")),
    inst("Instruments/World and Special Instruments/Oud", rx(r"\b(oud)\b")),
    inst("Instruments/World and Special Instruments/Accordion", rx(r"\b(accordion)\b")),
    inst("Instruments/World and Special Instruments/Harmonica", rx(r"\b(harmonica|mouth\s+organ)\b")),
    inst("Instruments/World and Special Instruments/Bagpipe", rx(r"\b(bagpipe|bagpipes)\b")),
    # Strings.
    inst("Instruments/Strings Bowed/Violin", rx(r"\b(violin|fiddle)\b")),
    inst("Instruments/Strings Bowed/Viola", rx(r"\b(viola)\b")),
    inst("Instruments/Strings Bowed/Cello", rx(r"\b(cello)\b")),
    inst(
        "Instruments/Strings Bowed/String Section", rx(r"\b(string\s+section|strings\s+section|orchestral\s+strings)\b")
    ),
    inst(
        "Instruments/Strings/Strings Sustains",
        rx(r"\b(strings?|violin|viola|cello).*(sustain|sustained|long|legato|bowed)\b"),
    ),
    inst(
        "Instruments/Strings/String Plucks",
        rx(r"\b(strings?|violin|viola|cello).*(pluck|pizz|pizzicato)\b|\bpizzicato\b"),
    ),
    inst("Instruments/Strings/String Drones", rx(r"\b(strings?|violin|viola|cello).*(drone|drones)\b")),
    inst("Instruments/Strings/String Loops", rx(r"\b(strings?|violin|viola|cello).*(loop|phrase|riff|bpm|chord)\b")),
    # Brass and woodwinds.
    inst("Instruments/Brass/Trumpet", rx(r"\b(trumpet)\b")),
    inst("Instruments/Brass/Trombone", rx(r"\b(trombone)\b")),
    inst("Instruments/Brass/Tuba", rx(r"\b(tuba)\b")),
    inst(
        "Instruments/Brass/Horns",
        rx(r"\b(horn|horns|french\s+horn|brass)\b"),
        rx(r"\b(car\s+horn|airhorn|air\s+horn)\b"),
    ),
    inst("Instruments/Woodwinds/Saxophone", rx(r"\b(sax|saxophone)\b")),
    inst("Instruments/Woodwinds/Flute", rx(r"\b(flute|flutes)\b")),
    inst("Instruments/Woodwinds/Clarinet", rx(r"\b(clarinet)\b")),
    inst("Instruments/Woodwinds/Oboe", rx(r"\b(oboe)\b")),
    inst("Instruments/Woodwinds/Bassoon", rx(r"\b(bassoon)\b")),
    inst("Instruments/Woodwinds/Pan Pipe Recorder Whistle", rx(r"\b(pan\s*pipe|panpipe|recorder|whistle)\b")),
    inst("Instruments/Brass and Woodwinds/Stabs", rx(r"\b(brass|horns?|sax|flute|woodwind).*(stab|stabs|hit|hits)\b")),
    inst(
        "Instruments/Brass and Woodwinds/Sustains",
        rx(r"\b(brass|horns?|sax|flute|woodwind).*(sustain|sustained|long)\b"),
    ),
    inst(
        "Instruments/Brass and Woodwinds/Loops",
        rx(r"\b(brass|horns?|sax|flute|woodwind|trumpet|trombone).*(loop|phrase|riff|bpm|chord)\b"),
    ),
    # Synths.
    inst(
        "Instruments/Synths/Synth Lead",
        rx(r"\b(synth\s+lead|lead\s+synth|lead)\b"),
        rx(r"\b(guitar\s+lead|vocal\s+lead)\b"),
    ),
    inst("Instruments/Synths/Synth Pad", rx(r"\b(synth\s+pad|pad|pads)\b"), rx(r"\b(mouse\s+pad)\b")),
    inst("Instruments/Synths/Synth Pluck", rx(r"\b(synth\s+pluck|pluck\s+synth|pluck)\b"), rx(r"\b(guitar|string)\b")),
    inst("Instruments/Synths/Synth Arp", rx(r"\b(arp|arps|arpeggio|arpeggiated)\b")),
    inst("Instruments/Synths/Synth Drone", rx(r"\b(synth\s+drone|drone\s+synth)\b")),
    inst("Instruments/Synths/Synth Stab", rx(r"\b(synth\s+stab|stab\s+synth|chord\s+stab)\b")),
    inst("Instruments/Synths/Synth Chord", rx(r"\b(synth\s+chord|chord\s+synth|poly\s+synth)\b")),
    inst(
        "Instruments/Synths/Synth Loops", rx(r"\b(synth|synths|analog|modular).*(loop|riff|phrase|bpm|arp|sequence)\b")
    ),
    inst(
        "Instruments/Synths/Synth One Shots",
        rx(r"\b(synth|synths|analog|modular).*(one[ _-]?shot|hit|stab|pluck|note)\b"),
    ),
    # Voice.
    inst("Instruments/Voice/Vocal Chops", rx(r"\b(vocal\s+chop|vox\s+chop|chopped\s+vocal|chops)\b")),
    inst(
        "Instruments/Voice/Vocal One Shots",
        rx(r"\b(vocal|vocals|vox|voice).*(one[ _-]?shot|hit|stab|shot|adlib|ad\s+lib)\b"),
    ),
    inst("Instruments/Voice/Vocal Loops", rx(r"\b(vocal|vocals|vox|voice).*(loop|phrase|hook|bpm|melody)\b")),
    inst(
        "Instruments/Voice/Spoken or Processed Voice",
        rx(r"\b(spoken|speech|talk|talking|processed\s+voice|processed\s+vocal|voice\s+fx|vocal\s+fx)\b"),
    ),
    inst("Instruments/Voice/Choir", rx(r"\b(choir|choral|voices)\b")),
    inst("Instruments/Voice/Breath and Mouth Sounds", rx(r"\b(breath|breathy|mouth|lip|lips|whisper|whispers)\b")),
    inst(
        "Instruments/Mixed Musical Loops/Multi Instrument",
        rx(r"\b(full\s+mix|fullmix|songstarter|song\s+starter|multi\s+instrument|melodic\s+loop|music\s+loop)\b"),
    ),
    # FX and textures under FX.
    fx(
        "FX/Structural and Transitional FX/Risers and Builds/Generic Riser",
        rx(r"\b(riser|risers|rise|rising|uplifter|up\s+lifter|build\s*up|buildup|swell)\b"),
    ),
    fx(
        "FX/Structural and Transitional FX/Risers and Builds/Short Riser",
        rx(r"\b(short\s+riser|small\s+riser|quick\s+riser)\b"),
    ),
    fx(
        "FX/Structural and Transitional FX/Drops and Downlifters/Generic Downlifter",
        rx(r"\b(downlifter|down\s+lifter|down\s*sweep|drop|falling|suckback|suck\s+back)\b"),
    ),
    fx(
        "FX/Structural and Transitional FX/Sweeps and Whooshes/Generic Whoosh or Sweep",
        rx(r"\b(whoosh|swoosh|swish|sweep|sweeps|flyby|fly\s+by|passby|pass\s+by)\b"),
    ),
    fx(
        "FX/Structural and Transitional FX/Reverses and Tails/Generic Reverse",
        rx(
            r"\b(reverse|reversed|reverse\s+fx|reverse\s+hit|reverse\s+cymbal|reverse\s+crash|reverse\s+reverb|rev\s+hit|rev\s+tail)\b"
        ),
    ),
    fx(
        "FX/Impacts and Hits/Generic Impact",
        rx(r"\b(impact|impacts|hit\s+fx|fx\s+hit|cinematic\s+hit)\b"),
        rx(r"\b(kick|snare|tom|guitar|piano)\b"),
    ),
    fx("FX/Impacts and Hits/Short Impact", rx(r"\b(short\s+impact|short\s+hit|punch\s+hit)\b")),
    fx("FX/Impacts and Hits/Boom", rx(r"\b(boom|booms|boomer|explosion|thunder\s+boom)\b")),
    fx("FX/Impacts and Hits/Sub Hit", rx(r"\b(sub\s+hit|bass\s+hit|low\s+hit|lowend\s+hit|sub\s+boom)\b")),
    fx("FX/Impacts and Hits/Slam", rx(r"\b(slam|slams|thud|thump)\b")),
    fx(
        "FX/Crashes and Breaks/Glass and Metal",
        rx(r"\b(glass|shatter|shattered|smash|break|breaking|metal\s+crash|crash\s+metal|debris)\b"),
        rx(r"\b(cymbal|ride|crash\s+cymbal)\b"),
    ),
    fx(
        "FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch",
        rx(r"\b(glitch|glitches|digital\s+glitch|malfunction|buffer)\b"),
    ),
    fx(
        "FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Stutter",
        rx(r"\b(stutter|stutters|stuttering|chop|chopped|slice|sliced)\b"),
    ),
    fx("FX/Digital Mechanical Industrial Transport/Machines/Engine", rx(r"\b(engine|engines|vehicle\s+engine)\b")),
    fx("FX/Digital Mechanical Industrial Transport/Machines/Motor", rx(r"\b(motor|motors|machine\s+motor)\b")),
    fx("FX/Designed Noise FX/Alarm", rx(r"\b(alarm|alarms)\b")),
    fx("FX/Designed Noise FX/Siren", rx(r"\b(siren|sirens|police\s+siren)\b")),
    fx("FX/Designed Noise FX/Beep", rx(r"\b(beep|beeps|bleep|bleeps)\b")),
    fx("FX/Designed Noise FX/Blip", rx(r"\b(blip|blips|bloop|bloops)\b")),
    fx(
        "FX/Designed Noise FX/Radio and Electrical",
        rx(r"\b(radio|electrical|electric\s+noise|static\s+burst|zaps?|buzz|hum)\b"),
    ),
    fx("FX/Designed Noise FX/Formant FX", rx(r"\b(formant|vowel\s+fx|talkbox|talk\s+box)\b")),
    fx(
        "FX/Designed Noise FX/Hybrid Designed FX",
        rx(r"\b(designed\s+fx|complex\s+fx|hybrid\s+fx|sci\s*fi|scifi|energy\s+fx|crystal\s+fx)\b"),
    ),
    fx("FX/Human and Voice FX/Applause", rx(r"\b(applause|clapping\s+crowd)\b")),
    fx("FX/Human and Voice FX/Crowd", rx(r"\b(crowd|crowds|audience)\b")),
    fx("FX/Human and Voice FX/Mouth Sounds", rx(r"\b(mouth\s+sound|mouth\s+fx|lip\s+smack|smack\s+mouth)\b")),
    fx("FX/Human and Voice FX/Spoken Voice", rx(r"\b(spoken\s+fx|speech\s+fx|voice\s+fx|vocal\s+fx)\b")),
    fx("FX/Animals and Creatures/Bird", rx(r"\b(bird|birds|chirp|tweet|tweets)\b")),
    fx("FX/Animals and Creatures/Cat", rx(r"\b(cat|meow|kitten)\b")),
    fx("FX/Animals and Creatures/Dog", rx(r"\b(dog|bark|puppy)\b")),
    fx("FX/Animals and Creatures/Cricket", rx(r"\b(cricket|crickets)\b")),
    fx("FX/Everyday Foley/Doors/Generic Door", rx(r"\b(door|doors|door\s+slam|door\s+close|door\s+open)\b")),
    fx(
        "FX/Everyday Foley/Keys Coins and Small Objects/Carkeys",
        rx(r"\b(car\s+keys?|carkeys|keys\s+jingle|key\s+jingle)\b"),
    ),
    fx("FX/Everyday Foley/Keys Coins and Small Objects/Coins", rx(r"\b(coin|coins|money\s+jingle)\b")),
    fx("FX/Everyday Foley/Cloth and Movement/Cloth Rustle", rx(r"\b(cloth|rustle|clothes|fabric)\b")),
    fx("FX/Everyday Foley/Footsteps/Footstep", rx(r"\b(footstep|footsteps|steps|walking)\b")),
    fx("FX/Everyday Foley/Water Foley/Splash", rx(r"\b(splash|splashes)\b")),
    fx("FX/Textures/Natural Ambience/Ocean", rx(r"\b(ocean|sea\s+ambience)\b")),
    fx("FX/Textures/Natural Ambience/Waves", rx(r"\b(waves|wave\s+ambience)\b")),
    fx("FX/Textures/Natural Ambience/Water", rx(r"\b(water|stream|river|creek)\b"), rx(r"\b(water\s+drop|splash)\b")),
    fx("FX/Textures/Natural Ambience/Rain", rx(r"\b(rain|rainfall)\b")),
    fx("FX/Textures/Natural Ambience/Wind", rx(r"\b(wind|windy)\b"), rx(r"\b(woodwind)\b")),
    fx("FX/Textures/Natural Ambience/Thunder", rx(r"\b(thunder|storm)\b")),
    fx("FX/Textures/Natural Ambience/Fire", rx(r"\b(fire|flame|flames|campfire)\b")),
    fx("FX/Textures/Noise and Static/Hiss", rx(r"\b(hiss|hissing)\b")),
    fx("FX/Textures/Noise and Static/Vinyl Noise", rx(r"\b(vinyl\s+noise|vinyl\s+crackle|record\s+crackle)\b")),
    fx("FX/Textures/Noise and Static/White Noise", rx(r"\b(white\s+noise|pink\s+noise|brown\s+noise|noise\s+bed)\b")),
    fx("FX/Textures/Drones and Atmospheres/Atmosphere", rx(r"\b(atmosphere|atmo|ambience|ambient|soundscape)\b")),
    fx("FX/Textures/Drones and Atmospheres/Drone", rx(r"\b(drone|drones|dronic)\b")),
    fx(
        "FX/Textures/Industrial Drones/Industrial Drone",
        rx(r"\b(industrial\s+drone|machine\s+drone|factory\s+ambience)\b"),
    ),
    fx("FX/Textures/Granular and Sandy Textures/Granular", rx(r"\b(granular|sandy|sand|grain|grains)\b")),
    fx(
        "FX/Textures/High Pitched Noise and Tones/High Tone",
        rx(r"\b(high\s+tone|high\s+pitched|whine|whistle\s+tone|test\s+tone)\b"),
    ),
    fx(
        "FX/Textures/Room Tone and Field Recordings/Room Tone",
        rx(r"\b(room\s+tone|field\s+recording|field\s+recordings|location\s+sound)\b"),
    ),
)


def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTENSIONS


def is_wav_file(path: Path) -> bool:
    return path.suffix.lower() in WAV_EXTENSIONS


def safe_rel(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except Exception:
        cleaned = str(path).strip("/").replace("/", "__")
        return Path(cleaned)


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._+()\[\] -]+", "_", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "audio_file"


def unique_path(path: Path) -> Path:
    if not path.exists() and not path.is_symlink():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for index in range(2, 100000):
        candidate = parent / f"{stem}__{index}{suffix}"
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise RuntimeError(f"Could not create a unique path for {path}")


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def iter_audio_files(root: Path, follow_symlinks: bool = False) -> Iterator[Path]:
    for current, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", "_reports"} and not d.startswith("._")]
        for filename in filenames:
            if filename.startswith("._") or filename == ".DS_Store" or filename.endswith(".asd"):
                continue
            path = Path(current) / filename
            if is_audio_file(path):
                yield path


def read_wav_stats(path: Path, max_seconds: float = 20.0) -> AudioStats:
    if np is None:
        return AudioStats(path=path, readable=False, reason="numpy_not_available")
    if not is_wav_file(path):
        return AudioStats(path=path, readable=False, reason="not_wav")
    try:
        with wave.open(str(path), "rb") as handle:
            channels = int(handle.getnchannels())
            sample_width = int(handle.getsampwidth())
            sample_rate = int(handle.getframerate())
            total_frames = int(handle.getnframes())
            read_frames = min(total_frames, int(sample_rate * max_seconds)) if sample_rate > 0 else total_frames
            raw = handle.readframes(read_frames)
    except Exception as exc:
        return AudioStats(path=path, readable=False, reason=f"wav_read_error:{exc}")

    if sample_rate <= 0 or total_frames <= 0 or not raw:
        return AudioStats(path=path, readable=False, reason="empty_or_bad_header")

    try:
        if sample_width == 1:
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            data = (data - 128.0) / 128.0
        elif sample_width == 2:
            data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 3:
            bytes_array = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            signed = (
                bytes_array[:, 0].astype(np.int32)
                | (bytes_array[:, 1].astype(np.int32) << 8)
                | (bytes_array[:, 2].astype(np.int32) << 16)
            )
            signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
            data = signed.astype(np.float32) / 8388608.0
        elif sample_width == 4:
            data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            return AudioStats(path=path, readable=False, reason=f"unsupported_sample_width:{sample_width}")
    except Exception as exc:
        return AudioStats(path=path, readable=False, reason=f"decode_error:{exc}")

    if channels > 1:
        frame_count = len(data) // channels
        if frame_count <= 0:
            return AudioStats(path=path, readable=False, reason="no_decoded_frames")
        data = data[: frame_count * channels].reshape(frame_count, channels).mean(axis=1)

    duration = float(total_frames) / float(sample_rate)
    if len(data) == 0:
        return AudioStats(path=path, readable=False, reason="no_samples")

    peak = float(np.max(np.abs(data)))
    rms = float(math.sqrt(float(np.mean(np.square(data)))))
    signs = np.signbit(data)
    zcr = float(np.mean(signs[1:] != signs[:-1])) if len(signs) > 1 else 0.0

    max_fft = min(len(data), 262144)
    if max_fft > 2048:
        windowed = data[:max_fft] * np.hanning(max_fft)
        magnitude = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(max_fft, d=1.0 / sample_rate)
        denom = float(np.sum(magnitude))
        centroid = float(np.sum(freqs * magnitude) / denom) if denom > 0.0 else 0.0
    else:
        centroid = 0.0

    block = max(256, int(sample_rate * 0.01))
    block_count = len(data) // block
    trimmed = data[: block_count * block]
    if block_count > 0 and len(trimmed) == block_count * block:
        blocks = trimmed.reshape(block_count, block)
        energies = np.sqrt(np.mean(np.square(blocks), axis=1))
        median_energy = float(np.median(energies)) if len(energies) else 0.0
        max_energy = float(np.max(energies)) if len(energies) else 0.0
        transient_ratio = max_energy / (median_energy + 1e-9)
    else:
        transient_ratio = 0.0

    return AudioStats(
        path=path,
        readable=True,
        reason="ok",
        duration_sec=duration,
        sample_rate=sample_rate,
        channels=channels,
        rms=rms,
        peak=peak,
        zcr=zcr,
        spectral_centroid_hz=centroid,
        transient_ratio=float(transient_ratio),
    )


def discover_training_slots(training_root: Path) -> list[TrainingSlot]:
    slots: list[TrainingSlot] = []
    for folder in sorted(training_root.rglob("*")):
        if not folder.is_dir():
            continue
        if folder.name not in STRUCTURE_NAMES:
            continue
        try:
            label_rel = folder.parent.relative_to(training_root)
        except ValueError:
            continue
        if not str(label_rel) or str(label_rel) == ".":
            continue
        slots.append(TrainingSlot(label_rel=label_rel, structure=folder.name, folder=folder))
    return slots


def text_tokens(path: Path | str) -> str:
    text = str(path)
    text = text.replace("_", " ").replace("-", " ").replace("/", " ")
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"(?i)(bpm)([a-z])", r"\1 \2", text)
    text = text.lower()
    # Common misspellings seen in user/sample names.
    text = text.replace("eletric", "electric").replace("electic", "electric")
    text = text.replace("wurlitzer", "wurlitzer").replace("wurly", "wurli")
    text = re.sub(r"[^a-z0-9+ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def regex_any(patterns: Sequence[str], text: str) -> tuple[bool, str]:
    for pattern in patterns:
        if re.search(pattern, text, flags=re.I):
            return True, pattern
    return False, ""


DRUM_LOOP_BRANCHES = {"drum loops", "percussion loops"}
DRUM_ONE_SHOT_BRANCHES = {
    "kick drums",
    "snares",
    "claps snaps slaps",
    "hi hats",
    "cymbals",
    "toms",
    "rims and sticks",
    "percussion",
    "world percussion",
    "drum fills and rolls",
}
INSTRUMENT_ONE_SHOT_TERMS = {
    "one shot",
    "one shots",
    "stab",
    "stabs",
    "hit",
    "hits",
    "pluck",
    "plucks",
    "chop",
    "chops",
    "note",
    "notes",
    "sustain",
    "sustains",
    "drone",
    "drones",
}


def label_parts_text(label_rel: Path) -> tuple[str, ...]:
    return tuple(text_tokens(part) for part in label_rel.parts)


def label_family(label_rel: Path, family: str = "") -> str:
    if family:
        return family
    return label_rel.parts[0] if label_rel.parts else ""


def label_terminal(label_rel: Path) -> str:
    return text_tokens(label_rel.parts[-1]) if label_rel.parts else ""


def label_has_text(label_rel: Path, *needles: str) -> bool:
    parts = label_parts_text(label_rel)
    return any(part in needles for part in parts)


def explicit_loop_label(label_rel: Path) -> bool:
    label_terminal(label_rel)
    parts = label_parts_text(label_rel)
    return any(re.search(r"\bloops?\b", part) for part in parts) or label_has_text(label_rel, *DRUM_LOOP_BRANCHES)


def explicit_instrument_one_shot_label(label_rel: Path) -> bool:
    terminal = label_terminal(label_rel)
    return any(re.search(r"\b" + re.escape(term) + r"\b", terminal) for term in INSTRUMENT_ONE_SHOT_TERMS)


def structures_for_label(label_rel: Path, family: str = "") -> tuple[str, ...]:
    """Return the category-owned structure slots that are safe to create.

    Aaron's training tree is intentionally uniform: every musical/drum category
    gets loop and one-shot lanes, while FX gets one-shot and long-FX lanes.
    Duration and filename hints decide which samples fit those lanes.
    """

    resolved_family = label_family(label_rel, family)
    if resolved_family == "FX" or (label_rel.parts and label_rel.parts[0] == "FX"):
        return FX_STRUCTURES
    return MUSICAL_STRUCTURES


def is_structure_allowed_for_label(label_rel: Path, structure: str, family: str = "") -> bool:
    return structure in structures_for_label(label_rel, family)


def phrase_guard_allows(entry: TaxonomyEntry, text: str) -> tuple[bool, str]:
    label = str(entry.label_rel)
    terminal = label_terminal(entry.label_rel)

    if label == "Drums/Drum Fills and Rolls/Drum Roll":
        if re.search(r"\b(drum|snare|tom|perc|percussion)\b", text):
            return True, "phrase_guard:drum_roll_context"
        return False, "phrase_guard:roll_without_drum_context"

    if label == "Drums/Drum Fills and Rolls/Drum Fill":
        if re.search(r"\b(drum|snare|tom|perc|percussion)\b", text):
            return True, "phrase_guard:drum_fill_context"
        return False, "phrase_guard:fill_without_drum_context"

    if label == "Drums/Cymbals/Splash Cymbal":
        if re.search(r"\bsplash\s+cymbals?\b|\bcymbals?\s+splash\b", text):
            return True, "phrase_guard:splash_cymbal_phrase"
        return False, "phrase_guard:splash_without_cymbal_context"

    if label == "Drums/Cymbals/Crash Cymbal":
        if re.search(r"\bcrash\s+cymbals?\b|\bcymbals?\s+crash\b", text):
            return True, "phrase_guard:crash_cymbal_phrase"
        if re.search(r"\bcrash\b", text) and re.search(r"\bcymbals?\b", text):
            return True, "phrase_guard:crash_and_cymbal_context"
        return False, "phrase_guard:crash_without_cymbal_context"

    if label == "Drums/Cymbals/Generic Cymbal" or terminal.endswith("cymbal"):
        if re.search(r"\bcymbals?\b", text):
            return True, "phrase_guard:cymbal_context"
        return False, "phrase_guard:cymbal_label_without_cymbal_text"

    if label == "FX/Crashes and Breaks/Glass and Metal":
        if re.search(r"\b(glass|shatter|shattered|smash|metal|debris)\b", text):
            return True, "phrase_guard:glass_or_metal_context"
        if re.search(r"\b(break|breaking)\b", text) and re.search(r"\b(glass|metal)\b", text):
            return True, "phrase_guard:break_with_material_context"
        return False, "phrase_guard:break_without_glass_or_metal_context"

    if label == "Instruments/Keys/Keys Loops":
        if re.search(r"\b(keys|keyboard|keyboards)\b", text):
            return True, "phrase_guard:keys_loop_context"
        return False, "phrase_guard:key_marker_without_keys_context"

    if label.startswith("Instruments/Strings/") or label.startswith("Instruments/Strings Bowed/"):
        bowed_context = r"\b(violin|viola|cello|orchestral\s+strings?|string\s+section|strings\s+section|bowed\s+strings?|string\s+ensemble)\b"
        if re.search(r"\bguitars?\b", text) and not re.search(bowed_context, text):
            return False, "phrase_guard:guitar_string_not_bowed_strings"

    if label == "Instruments/Mallets and Bells/Vibraphone":
        if re.search(r"\b(vibraphone|vibes)\b", text):
            return True, "phrase_guard:vibraphone_context"
        return False, "phrase_guard:vibe_without_vibraphone_context"

    return True, "phrase_guard:ok"


def entry_matches_text(entry: TaxonomyEntry, text: str) -> tuple[bool, float, str]:
    excluded, exclude_pattern = regex_any(entry.exclude_patterns, text)
    if excluded:
        return False, -20.0, f"excluded_by:{exclude_pattern}"
    included, include_pattern = regex_any(entry.include_patterns, text)
    if included:
        allowed, guard_reason = phrase_guard_allows(entry, text)
        if not allowed:
            return False, -10.0, guard_reason
        return True, 8.0, f"included_by:{include_pattern}|{guard_reason}"
    return False, 0.0, "no_taxonomy_match"


def load_ledger_taxonomy(project_dir: Path, ledger_json: str | None = None) -> tuple[TaxonomyEntry, ...]:
    candidates: list[Path] = []
    if ledger_json:
        candidates.append(Path(ledger_json).expanduser())
    candidates.extend(
        [
            project_dir / "Aaron_Master_Attribute_Classification_Ledger_v1_0.json",
            project_dir / "docs" / "Aaron_Master_Attribute_Classification_Ledger_v1_0.json",
        ]
    )
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ()
    cats = payload.get("classification_categories", {})
    entries: list[TaxonomyEntry] = []
    for key, item in cats.items():
        if not isinstance(item, dict):
            continue
        folder = str(item.get("primary_folder_path") or "").strip()
        if not folder:
            continue
        folder = folder.replace(" / ", "/").replace("/ ", "/").replace(" /", "/")
        parts = [p.strip() for p in folder.split("/") if p.strip()]
        if not parts:
            continue
        # Current project convention keeps textures inside FX. If the ledger says
        # Textures as a top-level family, map it into FX/Textures for training repair.
        family = parts[0]
        if family == "Textures":
            parts = ["FX", "Textures"] + parts[1:]
            family = "FX"
        if family not in {"Drums", "Instruments", "FX"}:
            if str(item.get("label_family", "")) == "FX":
                parts = ["FX"] + parts
                family = "FX"
            else:
                parts = ["Instruments"] + parts
                family = "Instruments"
        label_rel = Path(*parts)
        hints: list[str] = []
        for value in [item.get("display_name"), item.get("source_name_template"), key.replace("_", " ")]:
            if value:
                hints.append(str(value))
        for value in item.get("classifier_label_hints", []) or []:
            if value:
                hints.append(str(value))
        escaped = [r"\b" + re.escape(text_tokens(h)).replace(r"\ ", r"\s+") + r"\b" for h in hints if text_tokens(h)]
        if not escaped:
            continue
        structures = structures_for_label(label_rel, family)
        entries.append(TaxonomyEntry(label_rel, family, tuple(escaped), (), structures, f"ledger:{path.name}"))
    return tuple(entries)


def all_taxonomy_entries(project_dir: Path, ledger_json: str | None = None) -> tuple[TaxonomyEntry, ...]:
    entries: dict[str, TaxonomyEntry] = {}
    for entry in EMBEDDED_TAXONOMY + load_ledger_taxonomy(project_dir, ledger_json):
        key = str(entry.label_rel)
        if key not in entries:
            entries[key] = entry
        else:
            old = entries[key]
            entries[key] = TaxonomyEntry(
                label_rel=old.label_rel,
                family=old.family,
                include_patterns=tuple(dict.fromkeys(old.include_patterns + entry.include_patterns)),
                exclude_patterns=tuple(dict.fromkeys(old.exclude_patterns + entry.exclude_patterns)),
                structures=structures_for_label(old.label_rel, old.family),
                source=old.source + "+" + entry.source,
                aliases=tuple(dict.fromkeys(old.aliases + entry.aliases)),
            )
    return tuple(
        TaxonomyEntry(
            label_rel=entry.label_rel,
            family=entry.family,
            include_patterns=entry.include_patterns,
            exclude_patterns=entry.exclude_patterns,
            structures=structures_for_label(entry.label_rel, entry.family),
            source=entry.source,
            aliases=entry.aliases,
        )
        for entry in entries.values()
    )


def entry_for_label(label_rel: Path, entries: Sequence[TaxonomyEntry]) -> TaxonomyEntry | None:
    wanted = str(label_rel)
    for entry in entries:
        if str(entry.label_rel) == wanted:
            return entry
    return None


def fallback_aliases(label_rel: Path) -> tuple[str, ...]:
    aliases: list[str] = []
    for part in label_rel.parts:
        cleaned = text_tokens(part)
        if cleaned and cleaned not in {"drums", "instruments", "fx", "one shots", "loops", "generic"}:
            aliases.append(cleaned)
    terminal = text_tokens(label_rel.parts[-1]) if label_rel.parts else ""
    if terminal.startswith("generic "):
        aliases.append(terminal.replace("generic ", ""))
    return tuple(dict.fromkeys(aliases))


def fallback_category_score(text: str, label_rel: Path) -> tuple[float, str]:
    score = 0.0
    matched: list[str] = []
    for alias in fallback_aliases(label_rel):
        if not alias or len(alias) <= 1:
            continue
        pattern = r"(^|\s)" + re.escape(alias) + r"(\s|$)"
        if re.search(pattern, text, flags=re.I):
            score += 2.5
            matched.append(alias)
    return score, ";".join(matched) if matched else "fallback_no_category_hint"


def structure_guess(path_text: str, stats: AudioStats, family: str) -> set[str]:
    guesses: set[str] = set()
    has_loop_text = bool(LOOP_HINT_RE.search(path_text) or BPM_RE.search(path_text))
    has_one_text = bool(EXPLICIT_ONE_SHOT_STRUCTURE_RE.search(path_text))
    has_hit_identity_text = bool(ONE_SHOT_HINT_RE.search(path_text))
    duration = stats.duration_sec if stats.readable else 0.0

    if family == "FX":
        long_fx_text = bool(
            re.search(
                r"\b(riser|rise|sweep|whoosh|downlifter|tail|atmo|ambience|ambient|drone|texture|noise|wind|rain|ocean|waves)\b",
                path_text,
            )
        )
        short_fx_text = bool(re.search(r"\b(hit|impact|boom|slam|thud|beep|blip|stab|one[ _-]?shot|shot)\b", path_text))
        if has_loop_text or long_fx_text or (stats.readable and duration >= 2.0 and not short_fx_text):
            guesses.add("_LONG_FX")
        if short_fx_text or has_one_text or (stats.readable and 0.05 < duration <= 2.8 and not long_fx_text):
            guesses.add("_ONE_SHOTS")
        if not guesses:
            guesses.add("_LONG_FX")
        return guesses

    if has_loop_text or (stats.readable and duration >= 2.5 and not has_one_text):
        guesses.add("_LOOPS")
    if (
        has_one_text
        or (stats.readable and 0.05 < duration <= 2.5 and not has_loop_text)
        or (not stats.readable and has_hit_identity_text and not has_loop_text)
    ):
        guesses.add("_ONE_SHOTS")
    if not guesses:
        guesses.update(MUSICAL_STRUCTURES)
    return guesses


def structure_score(path_text: str, structure: str, stats: AudioStats, family: str) -> tuple[float, str]:
    guesses = structure_guess(path_text, stats, family)
    if structure in guesses:
        if structure == "_LONG_FX":
            return 3.0, "structure_guess:long_fx"
        if structure == "_LOOPS":
            return 3.0, "structure_guess:loop"
        return 3.0, "structure_guess:one_shot"
    if stats.readable:
        if structure == "_LOOPS" and stats.duration_sec < 1.0:
            return -6.0, "too_short_for_loop"
        if structure == "_LONG_FX" and stats.duration_sec < 0.8 and not LOOP_HINT_RE.search(path_text):
            return -4.0, "too_short_for_long_fx"
        if (
            structure == "_ONE_SHOTS"
            and stats.duration_sec > 2.5
            and not EXPLICIT_ONE_SHOT_STRUCTURE_RE.search(path_text)
        ):
            return -5.0, "too_long_for_one_shot"
    return -1.0, "structure_guess_mismatch"


def audio_quality_score(stats: AudioStats) -> tuple[float, str]:
    if not stats.readable:
        return -0.5, stats.reason
    score = 0.0
    reasons: list[str] = []
    if stats.duration_sec <= 0.05:
        score -= 10.0
        reasons.append("tiny_file")
    elif stats.duration_sec >= 0.08:
        score += 1.0
        reasons.append("duration_ok")
    if stats.rms < 0.0003:
        score -= 8.0
        reasons.append("near_silent")
    else:
        score += 1.0
        reasons.append("rms_ok")
    if stats.peak >= 0.999:
        score -= 0.4
        reasons.append("possibly_clipped")
    if stats.spectral_centroid_hz > 0.0:
        score += 0.4
        reasons.append("spectrum_ok")
    return score, ";".join(reasons)


def is_eval_named_file(path: Path) -> bool:
    return bool(re.search(r"\beval\b", text_tokens(path.name)))


def parents_with_non_eval_audio(samples: Sequence[Path]) -> set[Path]:
    return {sample.parent for sample in samples if not is_eval_named_file(sample)}


def should_skip_eval_candidate(sample: Path, non_eval_parents: set[Path]) -> bool:
    return is_eval_named_file(sample) and sample.parent in non_eval_parents


def review_link_rank(path: Path) -> int:
    match = re.match(r"^(\d+)__", path.name)
    if match:
        return int(match.group(1))
    return 1_000_000


def stable_shuffle_value(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def spread_review_links(label_rel: Path, structure: str, links: Sequence[Path], limit: int) -> set[Path]:
    if limit <= 0 or len(links) <= limit:
        return set(links)

    seed = f"{label_rel}/{structure}"
    groups: dict[str, list[Path]] = {}
    for link in links:
        try:
            target = Path(os.readlink(link))
            group_key = str(target.parent)
        except OSError:
            group_key = str(link.parent)
        groups.setdefault(group_key, []).append(link)

    for group_links in groups.values():
        group_links.sort(key=lambda path: (review_link_rank(path), stable_shuffle_value(seed, str(path))))

    group_order = sorted(groups, key=lambda group: stable_shuffle_value(seed, group))
    selected: list[Path] = []
    while len(selected) < limit:
        made_progress = False
        for group in group_order:
            group_links = groups[group]
            if not group_links:
                continue
            selected.append(group_links.pop(0))
            made_progress = True
            if len(selected) >= limit:
                break
        if not made_progress:
            break
    return set(selected)


def build_sample_hash_index(samples_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in iter_audio_files(samples_root, follow_symlinks=False):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            digest = file_sha256(path)
        except Exception:
            continue
        index.setdefault(digest, path)
    return index


def write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def latest_run_dir(project_dir: Path, name: str) -> Path:
    stamp = time.strftime("run_%Y%m%d_%H%M%S")
    root = project_dir / DEFAULT_REPORT_DIR_NAME / name / stamp
    root.mkdir(parents=True, exist_ok=True)
    return root


def normalize_symlinks(args: argparse.Namespace) -> int:
    training_root = Path(args.training_root).expanduser().resolve()
    samples_root = Path(args.samples_root).expanduser().resolve()
    project_dir = Path(args.project_dir).expanduser().resolve()
    apply = bool(args.apply)
    run_dir = latest_run_dir(project_dir, "normalize_symlinks")
    manifest_rows: list[dict[str, object]] = []

    print(f"Scanning training root: {training_root}")
    print(f"Samples root: {samples_root}")
    print(f"Mode: {'APPLY' if apply else 'DRY RUN'}")
    print("Building sample hash index. This can take a while on a large library.")
    sample_hashes = build_sample_hash_index(samples_root)

    real_training_files = [
        p for p in iter_audio_files(training_root, follow_symlinks=False) if p.is_file() and not p.is_symlink()
    ]
    recovered_root = samples_root / RECOVERED_REAL_FILES_DIR

    for source in sorted(real_training_files):
        try:
            digest = file_sha256(source)
        except Exception as exc:
            manifest_rows.append(
                {
                    "action": "error_hashing_training_file",
                    "training_path": str(source),
                    "sample_target": "",
                    "sha256": "",
                    "reason": str(exc),
                    "applied": False,
                }
            )
            continue

        existing = sample_hashes.get(digest)
        if existing is not None:
            target = existing
            action = "replace_real_training_file_with_symlink_to_existing_sample"
            reason = "same_content_already_exists_in_samples_root"
        else:
            target = unique_path(
                recovered_root / safe_rel(source, training_root).parent / sanitize_filename(source.name)
            )
            action = "move_real_training_file_to_recovered_samples_and_symlink_back"
            reason = "no_identical_sample_found_in_samples_root"

        if apply:
            target.parent.mkdir(parents=True, exist_ok=True)
            if existing is None:
                shutil.move(str(source), str(target))
                sample_hashes[digest] = target
            else:
                source.unlink()
            os.symlink(str(target), str(source))

        manifest_rows.append(
            {
                "action": action,
                "training_path": str(source),
                "sample_target": str(target),
                "sha256": digest,
                "reason": reason,
                "applied": apply,
            }
        )

    write_csv(
        run_dir / "normalize_symlinks_manifest.csv",
        manifest_rows,
        ["action", "training_path", "sample_target", "sha256", "reason", "applied"],
    )
    (run_dir / "README_NORMALIZE_SYMLINKS.txt").write_text(
        "Aaron training symlink normalization\n"
        f"Mode: {'APPLY' if apply else 'DRY RUN'}\n"
        f"Training root: {training_root}\n"
        f"Samples root: {samples_root}\n"
        f"Real training audio files found: {len(real_training_files)}\n"
        f"Manifest: {run_dir / 'normalize_symlinks_manifest.csv'}\n",
        encoding="utf-8",
    )
    print(f"Real training audio files found: {len(real_training_files)}")
    print(f"Report folder: {run_dir}")
    return 0


def count_existing_audio(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in iter_audio_files(folder, follow_symlinks=False))


def sample_records(samples_root: Path) -> list[tuple[Path, str]]:
    records: list[tuple[Path, str]] = []
    for sample in sorted(iter_audio_files(samples_root, follow_symlinks=False)):
        if not is_candidate_source_path(sample, samples_root):
            continue
        try:
            rel_path = sample.relative_to(samples_root)
        except ValueError:
            rel_path = sample
        records.append((sample, text_tokens(rel_path)))
    return records


def is_candidate_source_path(path: Path, samples_root: Path) -> bool:
    try:
        rel_path = path.relative_to(samples_root)
    except ValueError:
        rel_path = path
    raw_parts = {part.strip().lower() for part in rel_path.parts}
    text_parts = {text_tokens(part) for part in rel_path.parts}
    if raw_parts & {"_loops", "_one_shots", "_long_fx"}:
        return False
    return not ("sorted samples" in text_parts or "aaron sorted sounds" in text_parts)


BROAD_SOURCE_EVIDENCE_TOKENS = {
    "all",
    "audio",
    "cd",
    "content",
    "contents",
    "data",
    "dataset",
    "dev",
    "eval",
    "file",
    "files",
    "library",
    "one",
    "pack",
    "percussive",
    "percussion",
    "sample",
    "samples",
    "shot",
    "shots",
    "sound",
    "sounds",
    "test",
    "train",
}


def meaningful_source_tokens(text: str) -> list[str]:
    return [token for token in text.split() if re.search(r"[a-z]", token) and token not in BROAD_SOURCE_EVIDENCE_TOKENS]


def is_opaque_audio_filename(path: Path) -> bool:
    return len(meaningful_source_tokens(text_tokens(path.stem))) == 0


def is_broad_source_part(text: str) -> bool:
    tokens = text.split()
    if not tokens:
        return True
    meaningful = meaningful_source_tokens(text)
    return len(meaningful) == 0


def source_path_has_specific_evidence(entry: TaxonomyEntry, sample: Path, samples_root: Path) -> bool:
    filename_text = text_tokens(sample.stem)
    filename_match, _ = regex_any(entry.include_patterns, filename_text)
    if filename_match:
        return True
    if not is_opaque_audio_filename(sample):
        return True

    try:
        rel_path = sample.relative_to(samples_root)
    except ValueError:
        rel_path = sample

    for part in rel_path.parent.parts:
        part_text = text_tokens(part)
        if is_broad_source_part(part_text):
            continue
        part_match, _ = regex_any(entry.include_patterns, part_text)
        if part_match:
            return True
    return False


def sync_taxonomy(args: argparse.Namespace) -> int:
    training_root = Path(args.training_root).expanduser().resolve()
    samples_root = Path(args.samples_root).expanduser().resolve()
    project_dir = Path(args.project_dir).expanduser().resolve()
    apply = bool(args.apply)
    bool(args.create_all)
    entries = all_taxonomy_entries(project_dir, args.ledger_json)
    run_dir = latest_run_dir(project_dir, "sync_taxonomy")

    print(f"Taxonomy entries loaded: {len(entries)}")
    print(f"Scanning samples root for validation: {samples_root}")
    records = sample_records(samples_root)
    print(f"Sample audio files scanned: {len(records)}")

    rows: list[dict[str, object]] = []
    created_count = 0
    validated_count = 0
    for entry in sorted(entries, key=lambda e: str(e.label_rel)):
        allowed_structures = structures_for_label(entry.label_rel, entry.family)
        matched_sources: list[Path] = []
        structure_sources = {s: [] for s in allowed_structures}
        structure_counts = {s: 0 for s in allowed_structures}
        for sample, text in records:
            matched, _, _ = entry_matches_text(entry, text)
            if not matched:
                continue
            if not source_path_has_specific_evidence(entry, sample, samples_root):
                continue
            matched_sources.append(sample)
            stats = AudioStats(path=sample, readable=False, reason="not_scanned_for_sync")
            for structure in structure_guess(text, stats, entry.family):
                if structure in structure_counts:
                    structure_counts[structure] += 1
                    if len(structure_sources[structure]) < 5:
                        structure_sources[structure].append(sample)
        validated = any(count > 0 for count in structure_counts.values())
        if validated:
            validated_count += 1
        for structure in allowed_structures:
            structure_match_count = structure_counts.get(structure, 0)
            structure_matches = structure_sources.get(structure, [])
            should_create = True
            folder = training_root / entry.label_rel / structure
            exists = folder.exists()
            existing_count = count_existing_audio(folder) if exists else 0
            action = "none"
            if should_create and not exists:
                action = "create_folder"
                if apply:
                    folder.mkdir(parents=True, exist_ok=True)
                    created_count += 1
            elif should_create and exists:
                action = "already_exists"
            elif not should_create:
                action = "skip_no_matching_sample"
            rows.append(
                {
                    "label_rel": str(entry.label_rel),
                    "family": entry.family,
                    "structure": structure,
                    "folder": str(folder),
                    "exists_before": exists,
                    "existing_audio_count": existing_count,
                    "matching_sample_count": structure_match_count,
                    "matching_structure_guess_count": structure_match_count,
                    "first_matching_sample": str(structure_matches[0]) if structure_matches else "",
                    "source": entry.source,
                    "action": action,
                    "applied": apply,
                }
            )

    fieldnames = [
        "label_rel",
        "family",
        "structure",
        "folder",
        "exists_before",
        "existing_audio_count",
        "matching_sample_count",
        "matching_structure_guess_count",
        "first_matching_sample",
        "source",
        "action",
        "applied",
    ]
    write_csv(run_dir / "taxonomy_sync_manifest.csv", rows, fieldnames)
    (run_dir / "README_SYNC_TAXONOMY.txt").write_text(
        "Aaron training taxonomy sync\n"
        f"Mode: {'APPLY' if apply else 'DRY RUN'}\n"
        f"Training root: {training_root}\n"
        f"Samples root: {samples_root}\n"
        f"Taxonomy entries loaded: {len(entries)}\n"
        f"Sample audio files scanned: {len(records)}\n"
        f"Validated categories with at least one matching sample: {validated_count}\n"
        f"Folders created: {created_count}\n"
        "\nFX categories use _ONE_SHOTS and _LONG_FX, not _LOOPS.\n"
        "Clean Piano excludes Electric Piano/Rhodes/Wurli/Organ/Clav/Synth Piano evidence.\n",
        encoding="utf-8",
    )
    print(f"Validated categories with at least one matching sample: {validated_count}")
    print(f"Folders created: {created_count}")
    print(f"Report folder: {run_dir}")
    return 0


def stage_candidates(args: argparse.Namespace) -> int:
    training_root = Path(args.training_root).expanduser().resolve()
    samples_root = Path(args.samples_root).expanduser().resolve()
    project_dir = Path(args.project_dir).expanduser().resolve()
    per_structure = int(args.per_structure)
    max_candidates_per_slot = int(args.max_candidates_per_slot)
    min_score = float(args.min_score)
    entries = all_taxonomy_entries(project_dir, args.ledger_json)
    run_dir = latest_run_dir(project_dir, "stage_candidates")
    review_root = run_dir / "REVIEW_CANDIDATES_LISTEN_FIRST"
    slots = discover_training_slots(training_root)

    print(f"Training slots found: {len(slots)}")
    print(f"Review root: {review_root}")

    training_targets = {str(p.resolve()) for p in iter_audio_files(training_root, follow_symlinks=True) if p.exists()}
    all_samples = [
        sample
        for sample in iter_audio_files(samples_root, follow_symlinks=False)
        if is_candidate_source_path(sample, samples_root)
    ]
    non_eval_parents = parents_with_non_eval_audio(all_samples)
    samples = [p for p in all_samples if str(p.resolve()) not in training_targets]
    records = []
    for sample in sorted(samples):
        try:
            rel_path = sample.relative_to(samples_root)
        except ValueError:
            rel_path = sample
        records.append((sample, text_tokens(rel_path)))

    stats_cache: dict[Path, AudioStats] = {}
    candidate_rows: list[dict[str, object]] = []
    staged_rows: list[dict[str, object]] = []

    for slot in slots:
        entry = entry_for_label(slot.label_rel, entries)
        if entry is None:
            continue
        family = entry.family if entry else (slot.label_rel.parts[0] if slot.label_rel.parts else "")
        if not is_structure_allowed_for_label(slot.label_rel, slot.structure, family):
            continue
        candidates: list[Candidate] = []
        for sample, text in records:
            if should_skip_eval_candidate(sample, non_eval_parents):
                continue
            if entry:
                matched, cat_score, cat_reason = entry_matches_text(entry, text)
                if not matched:
                    continue
                if not source_path_has_specific_evidence(entry, sample, samples_root):
                    continue
            else:
                cat_score, cat_reason = fallback_category_score(text, slot.label_rel)
                if cat_score <= 0.0:
                    continue
            if sample not in stats_cache:
                stats_cache[sample] = read_wav_stats(sample)
            stats = stats_cache[sample]
            struct_score, struct_reason = structure_score(text, slot.structure, stats, family)
            quality, quality_reason = audio_quality_score(stats)
            score = cat_score + struct_score + quality
            if score < min_score:
                continue
            reason = f"category={cat_reason}|structure={struct_reason}|quality={quality_reason}"
            candidates.append(Candidate(sample, slot.label_rel, slot.structure, score, reason, stats))

        candidates.sort(key=lambda c: (-c.score, c.source.name.lower()))
        selected = candidates[:max_candidates_per_slot]
        dest_dir = review_root / slot.label_rel / slot.structure
        if selected:
            dest_dir.mkdir(parents=True, exist_ok=True)
        for index, cand in enumerate(selected, start=1):
            link_name = sanitize_filename(f"{index:02d}__score_{cand.score:.2f}__{cand.source.name}")
            link_path = unique_path(dest_dir / link_name)
            os.symlink(str(cand.source), str(link_path))
            row = {
                "label_rel": str(cand.label_rel),
                "structure": cand.structure,
                "rank": index,
                "score": f"{cand.score:.3f}",
                "review_symlink": str(link_path),
                "source_path": str(cand.source),
                "duration_sec": f"{cand.stats.duration_sec:.3f}",
                "rms": f"{cand.stats.rms:.6f}",
                "peak": f"{cand.stats.peak:.6f}",
                "zcr": f"{cand.stats.zcr:.6f}",
                "spectral_centroid_hz": f"{cand.stats.spectral_centroid_hz:.2f}",
                "transient_ratio": f"{cand.stats.transient_ratio:.3f}",
                "reason": cand.rank_reason,
            }
            candidate_rows.append(row)
            if index <= per_structure:
                staged_rows.append(row)

    fieldnames = [
        "label_rel",
        "structure",
        "rank",
        "score",
        "review_symlink",
        "source_path",
        "duration_sec",
        "rms",
        "peak",
        "zcr",
        "spectral_centroid_hz",
        "transient_ratio",
        "reason",
    ]
    write_csv(run_dir / "candidate_manifest_all_ranked.csv", candidate_rows, fieldnames)
    write_csv(run_dir / "candidate_manifest_top_default_apply_set.csv", staged_rows, fieldnames)
    (run_dir / "README_RESEED_REVIEW.txt").write_text(
        "Aaron training reseed candidate review\n\n"
        "Listen before applying. Delete bad candidate symlinks from REVIEW_CANDIDATES_LISTEN_FIRST.\n"
        "When the remaining files are the samples you want, run apply-reviewed-candidates.\n\n"
        f"Training root: {training_root}\n"
        f"Samples root: {samples_root}\n"
        f"Training slots found: {len(slots)}\n"
        f"Candidates staged: {len(candidate_rows)}\n"
        f"Review root: {review_root}\n"
        "\nCandidate ranking uses source path/name hints plus lightweight WAV physics only for review staging.\n"
        "FX long sounds are staged into _LONG_FX, not _LOOPS.\n",
        encoding="utf-8",
    )
    print(f"Candidates staged: {len(candidate_rows)}")
    print(f"Default top apply set rows: {len(staged_rows)}")
    print(f"Report folder: {run_dir}")
    print(f"Review folder: {review_root}")
    return 0


def apply_reviewed_candidates(args: argparse.Namespace) -> int:
    review_root = Path(args.review_root).expanduser().resolve()
    training_root = Path(args.training_root).expanduser().resolve()
    project_dir = Path(args.project_dir).expanduser().resolve()
    apply = bool(args.apply)
    max_per_structure = int(getattr(args, "max_per_structure", 10))
    run_dir = latest_run_dir(project_dir, "apply_reviewed_candidates")
    rows: list[dict[str, object]] = []
    slot_links: dict[tuple[Path, str], list[Path]] = {}

    if not review_root.exists():
        raise SystemExit(f"Review root not found: {review_root}")

    for link in sorted(review_root.rglob("*")):
        if not link.is_symlink() or not is_audio_file(link):
            continue
        try:
            rel_path = link.parent.relative_to(review_root)
        except ValueError:
            continue
        if len(rel_path.parts) < 2 or rel_path.parts[-1] not in STRUCTURE_NAMES:
            continue
        label_rel = Path(*rel_path.parts[:-1])
        structure = rel_path.parts[-1]
        slot_links.setdefault((label_rel, structure), []).append(link)

    for (label_rel, structure), links in sorted(slot_links.items(), key=lambda item: (str(item[0][0]), item[0][1])):
        accepted_count = 0
        dest_dir = training_root / label_rel / structure
        existing_count = count_existing_audio(dest_dir)
        capacity = max_per_structure if max_per_structure <= 0 else max(0, max_per_structure - existing_count)
        selected_links = spread_review_links(label_rel, structure, links, capacity)
        for link in sorted(links, key=lambda path: (review_link_rank(path), path.name.lower())):
            target = Path(os.readlink(link))
            family = label_rel.parts[0] if label_rel.parts else ""
            if not is_structure_allowed_for_label(label_rel, structure, family):
                rows.append(
                    {
                        "action": "skip_invalid_structure_policy",
                        "label_rel": str(label_rel),
                        "structure": structure,
                        "review_link": str(link),
                        "source_target": str(target),
                        "training_link": "",
                        "applied": False,
                    }
                )
                continue
            if max_per_structure > 0 and existing_count >= max_per_structure:
                rows.append(
                    {
                        "action": "skip_existing_slot_at_or_above_limit",
                        "label_rel": str(label_rel),
                        "structure": structure,
                        "review_link": str(link),
                        "source_target": str(target),
                        "training_link": "",
                        "applied": False,
                    }
                )
                continue
            if link not in selected_links:
                rows.append(
                    {
                        "action": "skip_not_selected_for_spread_limit",
                        "label_rel": str(label_rel),
                        "structure": structure,
                        "review_link": str(link),
                        "source_target": str(target),
                        "training_link": "",
                        "applied": False,
                    }
                )
                continue
            accepted_count += 1
            dest_name = sanitize_filename(target.name)
            dest = unique_path(dest_dir / dest_name)
            if apply:
                dest_dir.mkdir(parents=True, exist_ok=True)
                os.symlink(str(target), str(dest))
            rows.append(
                {
                    "action": "add_reviewed_training_symlink",
                    "label_rel": str(label_rel),
                    "structure": structure,
                    "review_link": str(link),
                    "source_target": str(target),
                    "training_link": str(dest),
                    "applied": apply,
                }
            )

    write_csv(
        run_dir / "apply_reviewed_candidates_manifest.csv",
        rows,
        ["action", "label_rel", "structure", "review_link", "source_target", "training_link", "applied"],
    )
    (run_dir / "README_APPLY_REVIEWED_CANDIDATES.txt").write_text(
        "Aaron training reseed apply-reviewed-candidates\n"
        f"Mode: {'APPLY' if apply else 'DRY RUN'}\n"
        f"Review root: {review_root}\n"
        f"Training root: {training_root}\n"
        f"Reviewed symlinks found: {len(rows)}\n"
        f"Target total per label/structure: {max_per_structure if max_per_structure > 0 else 'unlimited'}\n",
        encoding="utf-8",
    )
    print(f"Reviewed symlinks found: {len(rows)}")
    print(f"Target total per label/structure: {max_per_structure if max_per_structure > 0 else 'unlimited'}")
    print(f"Mode: {'APPLY' if apply else 'DRY RUN'}")
    print(f"Report folder: {run_dir}")
    return 0


def audit_training(args: argparse.Namespace) -> int:
    training_root = Path(args.training_root).expanduser().resolve()
    project_dir = Path(args.project_dir).expanduser().resolve()
    run_dir = latest_run_dir(project_dir, "audit_training")
    rows: list[dict[str, object]] = []
    total_audio = 0
    real_audio = 0
    symlink_audio = 0
    broken_symlink = 0

    for path in sorted(iter_audio_files(training_root, follow_symlinks=False)):
        total_audio += 1
        is_link = path.is_symlink()
        exists = path.exists()
        if is_link:
            symlink_audio += 1
            if not exists:
                broken_symlink += 1
        else:
            real_audio += 1
        rows.append(
            {
                "path": str(path),
                "kind": "symlink" if is_link else "real_file",
                "exists": exists,
                "target": os.readlink(path) if is_link else "",
            }
        )

    slots = discover_training_slots(training_root)
    slot_rows: list[dict[str, object]] = []
    for slot in slots:
        count = count_existing_audio(slot.folder)
        slot_rows.append(
            {"label_rel": str(slot.label_rel), "structure": slot.structure, "count": count, "folder": str(slot.folder)}
        )

    write_csv(run_dir / "training_file_audit.csv", rows, ["path", "kind", "exists", "target"])
    write_csv(run_dir / "training_slot_counts.csv", slot_rows, ["label_rel", "structure", "count", "folder"])
    (run_dir / "README_AUDIT_TRAINING.txt").write_text(
        "Aaron training audit\n"
        f"Training root: {training_root}\n"
        f"Slots: {len(slots)}\n"
        f"Audio files: {total_audio}\n"
        f"Real audio files: {real_audio}\n"
        f"Symlink audio files: {symlink_audio}\n"
        f"Broken symlinks: {broken_symlink}\n"
        "Structure folders counted: _LOOPS, _ONE_SHOTS, _LONG_FX\n",
        encoding="utf-8",
    )
    print(f"Slots: {len(slots)}")
    print(f"Audio files: {total_audio}")
    print(f"Real audio files: {real_audio}")
    print(f"Symlink audio files: {symlink_audio}")
    print(f"Broken symlinks: {broken_symlink}")
    print(f"Report folder: {run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aaron training taxonomy sync, reseed, and symlink manager")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
        p.add_argument(
            "--ledger-json", default="", help="Optional master ledger JSON. If omitted, auto-detects in project root."
        )

    audit = sub.add_parser("audit-training")
    audit.add_argument("--training-root", default=DEFAULT_TRAINING_ROOT)
    add_common(audit)
    audit.set_defaults(func=audit_training)

    norm = sub.add_parser("normalize-symlinks")
    norm.add_argument("--training-root", default=DEFAULT_TRAINING_ROOT)
    norm.add_argument("--samples-root", default=DEFAULT_SAMPLES_ROOT)
    norm.add_argument("--apply", action="store_true", help="Actually move/replace files. Omit for dry run.")
    add_common(norm)
    norm.set_defaults(func=normalize_symlinks)

    sync = sub.add_parser("sync-taxonomy")
    sync.add_argument("--training-root", default=DEFAULT_TRAINING_ROOT)
    sync.add_argument("--samples-root", default=DEFAULT_SAMPLES_ROOT)
    sync.add_argument("--apply", action="store_true", help="Actually create missing folders. Omit for dry run.")
    sync.add_argument(
        "--create-all",
        action="store_true",
        help="Compatibility flag. Taxonomy sync now always creates every expected structure folder.",
    )
    add_common(sync)
    sync.set_defaults(func=sync_taxonomy)

    stage = sub.add_parser("stage-candidates")
    stage.add_argument("--training-root", default=DEFAULT_TRAINING_ROOT)
    stage.add_argument("--samples-root", default=DEFAULT_SAMPLES_ROOT)
    stage.add_argument(
        "--per-structure", type=int, default=10, help="Target kept count per label/structure after review."
    )
    stage.add_argument(
        "--max-candidates-per-slot",
        type=int,
        default=25,
        help="How many ranked candidates to stage for listening per label/structure.",
    )
    stage.add_argument("--min-score", type=float, default=8.0)
    add_common(stage)
    stage.set_defaults(func=stage_candidates)

    apply_p = sub.add_parser("apply-reviewed-candidates")
    apply_p.add_argument("--review-root", required=True)
    apply_p.add_argument("--training-root", default=DEFAULT_TRAINING_ROOT)
    apply_p.add_argument("--apply", action="store_true", help="Actually add symlinks. Omit for dry run.")
    apply_p.add_argument(
        "--max-per-structure",
        type=int,
        default=10,
        help="Fill each label/structure to this total training count. Use 0 for unlimited.",
    )
    add_common(apply_p)
    apply_p.set_defaults(func=apply_reviewed_candidates)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
