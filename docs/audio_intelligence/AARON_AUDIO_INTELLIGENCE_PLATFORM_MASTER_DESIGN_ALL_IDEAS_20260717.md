# Aaron Audio Intelligence Platform — Master Design Plan

**Date:** 2026-07-17  
**Status:** Consolidated master plan  
**Supersedes / incorporates:**

- `AARON_SOUND_SORTER_TRAINABLE_VOTER_BRAINS_OPEN_WORLD_DESIGN_20260717.md`
- `AARON_SOUND_SORTER_TRAINABLE_BRAINS_COMPONENTS_MUSIC_PROPERTIES_V2_20260717.md`
- `AARON_SOUND_SORTER_TRAINABLE_BRAINS_COMPONENTS_MUSIC_PROPERTIES_STEM_INTELLIGENCE_V3_20260717.md`
- `AARON_AUDIO_INTELLIGENCE_PLATFORM_V4_PRODUCT_FAMILY_PLAN_20260717.md`

---

## 1. Executive summary

The project should no longer be designed as only a sample sorter.

The stronger long-term design is:

```text
Aaron Audio Intelligence Core
  -> Sound Sorter
  -> Loop Analyzer
  -> Stem Intelligence / Component Finder
  -> Sample Search / Flavor Browser
  -> Training / Correction Console
  -> Dataset / Atlas Builder
```

The current sorter remains important because it gives immediate value and creates correction data. But the deeper invention is a reusable audio-intelligence layer that can understand sound structure, components, music properties, open-world unknowns, and use-case roles.

The core principle:

```text
Do not only ask: "What folder does this go in?"
Ask first: "What is physically, musically, and functionally happening in this sound?"
```

The folder is only one output. The reusable evidence sidecar is the real asset.

---

## 2. Product philosophy

### 2.1 Keep final user folders simple

The Sound Sorter output should remain producer-friendly:

```text
Drums
Instruments
Textures
FX
_TO_REVIEW
```

Do not create dozens of top-level folders. The user wants a clean library, not an academic taxonomy dumped into Finder.

### 2.2 Make internal understanding much richer

Internally, every file should produce a rich sidecar:

```text
identity-like evidence
role evidence
shape evidence
component evidence
music properties
open-world novelty scores
usable-as ratings
warnings
training links
```

Example:

```text
A sound can be sax-like, but not useful as a normal sax instrument.
It might be better treated as ambient texture or FX because of huge reverb, smear, distortion, or motion.
```

That distinction is central.

### 2.3 Separate “sounds like” from “used as”

A sound may contain evidence of a source without belonging in that source’s folder.

Examples:

```text
sax-like + huge reverb tail + smeared attack -> Texture or FX, not clean sax
piano-like + reversed wash + no playable onset -> Texture / FX, not piano loop
guitar-like + feedback drone -> Texture / FX, not guitar phrase
human-like + formant sweep -> FX/Formant FX, not Voice phrase
bird-like + synthetic laser chirp -> FX or synthetic chirp, not animal recording
```

The system must preserve both facts:

```text
source_like = sax
usable_role = ambient_texture
final_folder = Textures / Hybrid Textures or FX / Hybrid Designed FX
```

### 2.4 Review is a product feature

Review is not failure. Review is honesty.

The system should send unclear, conflicting, unknown, or open-world sounds to review with useful traits rather than faking confidence.

A bad confident folder is worse than a good review report.

---

## 3. Shared platform architecture

### 3.1 High-level flow

```text
Audio input
  ↓
Audio loader / cleanup / integrity checks
  ↓
Universal feature extraction
  ↓
Trait brains
  ↓
Shape brains
  ↓
Owner brains
  ↓
Component analyzer
  ↓
Music property analyzer
  ↓
Open-world novelty / unknown detector
  ↓
Sidecar evidence writer
  ↓
Product adapters
      - Sound Sorter
      - Loop Analyzer
      - Stem Intelligence
      - Search / Browser
      - Training Console
```

### 3.2 Product adapters

The same intelligence core should serve multiple tools.

#### Sound Sorter

Answers:

```text
Where should this file go?
```

Outputs:

```text
final folder
manifest row
review reason
secondary traits
```

#### Loop Analyzer

Answers:

```text
What is inside this loop?
What BPM?
What key?
What chords?
What instruments?
What drum parts?
What ambience or FX layers?
```

#### Stem Intelligence / Component Finder

Answers:

```text
Can this loop be broken into useful parts?
Should we output MIDI, reconstruction, broad stems, or description-only?
```

#### Search / Flavor Browser

Answers:

```text
Find dark wide evolving FX.
Find A minor piano loops around 90 BPM.
Find forest loops with birds and water but no traffic.
Find punchy dry kicks.
```

#### Training / Correction Console

Answers:

```text
What did Aaron correct?
What should be a positive example?
What should be a negative counterexample?
Which owner brain is weak?
Can this correction be learned safely?
```

---

## 4. Non-negotiable engineering rules

### 4.1 Measured audio first

Production routing must not use filenames, source folder names, ZIP member names, or pack labels as evidence.

Names can be used only for:

```text
audit
external evaluation
sample selection
human review helpers
training/correction UI
```

They must not drive production classification.

### 4.2 Structure and role before identity

Before deciding a folder, ask:

```text
Is this a one-shot, loop, phrase, texture, transition, impact, ambience, field recording, mixed loop, or compound sound?
```

Then ask identity:

```text
kick, snare, sax, piano, bird, machine, water, formant FX, etc.
```

This prevents bad shortcuts like:

```text
pitch detected -> instrument
many onsets -> drum loop
bright noisy transient -> cymbal
low repeating body -> bass loop
```

### 4.3 Trainable brains need negative examples

Every trainable owner brain must learn:

```text
positive examples
negative counterexamples
```

Example:

```text
DesignedMotionFXOwnerBrain positives:
  risers
  down sweeps
  whooshes
  laser sweeps
  scrape drops
  pulsed motion FX

DesignedMotionFXOwnerBrain negatives:
  real drum loops
  synth arps
  full music loops
  sax phrases
  guitar loops
  clean hat loops
```

Without negative examples, FX becomes a vacuum and steals drums/instruments.

### 4.4 Hard safety gates stay non-trainable

Not everything should become a black-box brain.

Hard safety gates should remain deterministic:

```text
silent/broken/too short -> review
low-end phase problem -> not clean kick or bass
strong real drum loop anchor -> protect Drums
strong clean voiced phrase -> protect Voice
strong clean sax/reed evidence -> protect Brass/Woodwind
strong full-music-loop evidence -> protect Instruments/Mixed Musical Loops
```

Trainable models can improve probabilities. Hard gates prevent catastrophic category theft.

### 4.5 Sidecar-first design

Every analyzed file should produce reusable evidence, even if the sorter only needs one folder.

The sidecar is what allows future products to exist.

---

## 5. Universal feature extraction layer

The feature extractor should produce global and segment-level descriptors.

### 5.1 Global file features

```text
duration
sample rate
integrity flags
RMS / peak / crest factor
spectral centroid
spectral bandwidth
spectral rolloff
spectral flatness
spectral entropy
spectral contrast
MFCC / delta MFCC summaries
zero-crossing rate
onset count
onset density
pulse clarity
tempo candidates
spectral flux
centroid slope
band-energy slope
temporal centroid
decay / tail length
stereo width
mid/side ratio
low-side ratio
pitch confidence
harmonicity
inharmonicity
chroma / HPCP / tonal centroid
```

### 5.2 Segment-level features

Global averages flatten important events. The system also needs time-segment evidence:

```text
intro / body / tail summaries
bar-level or beat-level summaries for loops
onset-region summaries for percussive events
moving spectral slope windows
local pitch/chroma frames
local component activations
local novelty/outlier events
```

### 5.3 Time-frequency regions

For component detection, the system should inspect regions, not just whole files:

```text
low band: sub/kick/bass/boom
low-mid band: body/toms/wood/voice warmth
mid band: snares/reeds/guitar/piano/body
high-mid band: bite/sax/formants/metallic attacks
high band: hats/shakers/air/noise/insects/birds
wide stereo bands: ambience/reverb/field recording
```

---

## 6. Trainable brain layers

### 6.1 Trait brains

Trait brains identify low-level or mid-level traits.

Examples:

```text
punchy
plucked
bowed
voiced
breathy
metallic
wooden
glassy
liquid
windy
granular
machine-like
animal-like
synthetic
organic
rising
falling
sweep-motion
impact-tail
loop-grid
clean-bass-foundation
designed-low-boom
glitch-stutter
```

Trait brains do not decide folders. They produce evidence.

### 6.2 Shape brains

Shape brains describe the sound’s time structure.

Examples:

```text
drum_groove_shape
drum_fill_shape
one_shot_hit_shape
clean_musical_phrase_shape
voiced_phrase_shape
bass_phrase_shape
texture_bed_shape
transition_riser_shape
transition_drop_shape
whoosh_sweep_shape
impact_with_tail_shape
glitch_stutter_shape
designed_tonal_motion_shape
field_ambience_shape
compound_loop_shape
```

Shape may own structural role, but shape should not directly force a final folder without owner-gate permission.

### 6.3 Owner brains

Owner brains decide which role/family is allowed to own the sound.

Recommended owner brains:

```text
DrumLoopOwnerBrain
DrumOneShotOwnerBrain
BassFoundationOwnerBrain
CleanInstrumentPhraseOwnerBrain
VoicedPhraseOwnerBrain
MixedMusicLoopOwnerBrain
TextureBedOwnerBrain
AmbientTailOwnerBrain
DesignedMotionFXOwnerBrain
DesignedImpactFXOwnerBrain
DesignedTonalFXOwnerBrain
GlitchMechanicalFXOwnerBrain
BiologicalFoleyOwnerBrain
MachineMechanicalOwnerBrain
WaterWindWeatherOwnerBrain
HouseholdObjectFoleyOwnerBrain
ElectricalRadioDataOwnerBrain
UnknownOwnerBrain
```

Each owner outputs:

```text
score
confidence
positive evidence
negative evidence
allowed final top levels
blocked final top levels
training match IDs
reason text
```

### 6.4 Folder brain

The folder brain converts owner decisions into producer-friendly folders.

Example:

```text
owner = DesignedMotionFXOwnerBrain
shape = down_sweep
final = FX / Drops and Downlifters
```

Example:

```text
owner = CleanInstrumentPhraseOwnerBrain
source_like = sax
usable_as = normal_instrument
final = Instruments / Brass and Woodwinds / Sax
```

Example:

```text
owner = AmbientTailOwnerBrain
source_like = sax
usable_as = ambient_texture
final = Textures / Hybrid Textures
secondary_traits = sax_like, huge_reverb_tail, smeared_attack
```

---

## 7. Sidecar evidence schema

Every file should get a JSON sidecar.

### 7.1 Example sidecar

```json
{
  "schema_version": "aai_sidecar_v1",
  "file_id": "sha256-or-stable-id",
  "original_filename": "example_loop.wav",
  "analysis_version": "2026-07-17",
  "final_sort": {
    "folder": "Instruments/Keys/Piano Loops",
    "confidence": 0.86,
    "review_reason": null
  },
  "roles": {
    "primary_owner": "CleanInstrumentPhraseOwnerBrain",
    "owner_scores": {
      "CleanInstrumentPhraseOwnerBrain": 0.86,
      "TextureBedOwnerBrain": 0.18,
      "DesignedMotionFXOwnerBrain": 0.04,
      "UnknownOwnerBrain": 0.07
    },
    "blocked_by": []
  },
  "source_like": {
    "piano": 0.92,
    "keys": 0.88,
    "bass": 0.18,
    "drums": 0.04,
    "reverb_tail": 0.41
  },
  "usable_as": {
    "clean_instrument_loop": 0.84,
    "ambient_texture": 0.16,
    "fx_transition": 0.02
  },
  "music": {
    "bpm": 90,
    "bpm_confidence": 0.78,
    "tempo_candidates": [90, 180, 45],
    "bars_estimated": 4,
    "key": "A minor",
    "key_confidence": 0.69,
    "scale": "minor",
    "chords": [
      {"bar": 1, "chord": "Am", "confidence": 0.74},
      {"bar": 2, "chord": "G", "confidence": 0.68},
      {"bar": 3, "chord": "F", "confidence": 0.66},
      {"bar": 4, "chord": "E", "confidence": 0.61}
    ],
    "chord_confidence_policy": "usable_with_review"
  },
  "components": {
    "drums": {
      "present": 0.04,
      "kick": 0.01,
      "snare": 0.01,
      "hats": 0.02
    },
    "instruments": {
      "piano": 0.92,
      "bass": 0.18,
      "guitar": 0.05,
      "synth_pad": 0.09
    },
    "ambience": {
      "room": 0.31,
      "reverb_tail": 0.41
    }
  },
  "traits": [
    "tonal",
    "warm",
    "loop_like",
    "medium_reverb",
    "stable_pitch"
  ],
  "warnings": [],
  "training": {
    "safe_to_auto_train": false,
    "suggested_positive_owner": "CleanInstrumentPhraseOwnerBrain",
    "suggested_negative_owners": ["DesignedMotionFXOwnerBrain", "DrumLoopOwnerBrain"]
  }
}
```

### 7.2 Manifest additions

The CSV manifest should stay readable but expose key evidence:

```text
final_folder
review_reason
primary_owner
owner_confidence
blocked_by_owner
shape_primary
shape_secondary
source_like_primary
usable_as_primary
component_summary
bpm
bpm_confidence
key
key_confidence
chord_summary
chord_confidence
novelty_score
unknown_cluster_id
warnings
```

---

## 8. Component analysis inside loops

Loops should not only get one folder label. They should get an inventory.

### 8.1 Drum loop components

For drum loops, identify likely components:

```text
kick
snare
clap
closed hat
open hat
ride
crash
rim/stick
shaker
tambourine
tom
conga/bongo
metal percussion
wood percussion
noise percussion
FX hit / transition layer
```

Important output:

```text
contains_kick: yes/no/confidence
contains_snare_or_clap: yes/no/confidence
contains_hats: yes/no/confidence
contains_shaker: yes/no/confidence
drum_grid_confidence
swing_or_shuffle_estimate
fill_vs_loop estimate
```

### 8.2 Mixed musical loop components

For mixed loops, identify broad instrument layers:

```text
drums
bass
piano/keys
guitar
synth lead
synth pad
strings
brass/woodwind
voice/vocal chop
FX transition
texture/ambience
```

Do not pretend to perfectly isolate every instrument. The first product should describe what is probably present.

### 8.3 Forest / ambience / field-recording components

For forest/ambience loops, identify:

```text
birds
insects
frogs
mammal calls
wind
rain
water stream/ocean
leaves/brush
human voice contamination
traffic contamination
machine hum
room tone
microphone handling
```

Example output:

```text
Forest ambience:
  birds: strong
  insects: medium
  wind: medium
  distant traffic: possible
  water: none
  loop-grid: none
  final role: Textures / Organic Textures
```

### 8.4 FX loop components

For FX loops, identify:

```text
riser layer
downlifter layer
whoosh/sweep layer
glitch/stutter events
impact events
tonality/pitch center
noise wash
mechanical movement
formant/vocal-like layer
```

---

## 9. Music property analyzer

The sorter should become a music-property intelligence tool too.

### 9.1 BPM / rhythm properties

Estimate:

```text
BPM
tempo confidence
tempo candidates
half-time candidate
double-time candidate
beat grid confidence
bar count estimate
loop length in beats/bars
swing/shuffle estimate
onset grid density
```

Important: Always expose tempo candidates, not just one BPM.

Example:

```text
bpm_primary = 90
bpm_candidates = 90, 180, 45
confidence = 0.78
```

### 9.2 Pitch / key properties

Estimate:

```text
root note
single-note pitch
pitch confidence
key
scale
key confidence
chroma profile
tuning deviation
```

Use cases:

```text
bass one-shot -> root note
piano loop -> key and chord timeline
synth stab -> root/chord estimate
bell/metal FX -> pitch center with low confidence
animal call -> pitch contour, not musical key unless stable
```

### 9.3 Chord timeline

For piano, keys, guitar, synth, and clean musical loops:

```text
bar-level chord estimate
beat-level chord changes when confident
confidence per chord
slash-chord or ambiguity marker when needed
```

Example:

```text
4-bar piano loop:
  key: A minor
  chords: Am | G | F | E
  chord confidence: medium-high
```

### 9.4 Chord safety rule

Do not force chord labels onto:

```text
drum loops
noise FX
field recordings
atonal textures
whooshes
animal ambiences
machine hums
glitch clips
```

If a sound is tonal but not musical, report:

```text
pitch_center: possible
key: not applicable
chords: not applicable
```

### 9.5 Usability metadata

Add producer-useful metadata:

```text
clean_loop_ready
needs_warping
has_reverb_tail
has_clicks_or_noise
has_room_bleed
mono_compatible
wide_stereo
phase_risk
low_end_clean
loop_start_quality
loop_end_quality
```

---

## 10. Stem Intelligence / Component Finder product

Modern stem splitters are useful, but too broad for many producer tasks. They often give buckets like:

```text
vocals
bass
drums
other
```

That is not enough when a musician wants to know:

```text
What drums are in this loop?
Can I get the kick pattern?
Can I rebuild the hats?
Is there shaker under the hats?
What instrument is the chord loop?
Can I get MIDI instead of ugly separated audio?
```

### 10.1 Do not promise perfect audio separation

The system should choose the best output type:

```text
A. Describe-only
B. MIDI transcription / reconstruction
C. Audio stem separation
D. Hybrid: broad stem + MIDI/reconstruction
E. No separation recommended
```

### 10.2 Stem action decision logic

```text
If component is clearly detectable but audio separation would be artifact-heavy:
  output MIDI/reconstruction, not separated audio.

If component is broad and separable:
  output broad audio stem.

If component is present but overlapped/noisy:
  describe only and show confidence.

If component is unknown/open-world:
  cluster and describe traits before attempting separation.
```

### 10.3 Drum loop path

For drum loops:

```text
1. Detect drum grid.
2. Identify kick/snare/hat/shaker/percussion events.
3. Estimate MIDI pattern.
4. Decide whether original audio stems are useful.
5. Optionally reconstruct clean drum parts using replacement samples.
```

Example report:

```text
This loop contains:
  kick: strong
  snare/clap: strong
  closed hats: strong
  shaker: medium
  crash: weak at bar 4

Recommended output:
  MIDI drum map + reconstructed clean stems

Not recommended:
  hat-only audio separation, because hats and shaker overlap too much
```

### 10.4 Mixed loop path

For mixed musical loops:

```text
1. Detect broad components.
2. Estimate tempo/key/chords.
3. Identify dominant source layers.
4. Decide whether MIDI/chord transcription is more useful than audio stems.
```

Example:

```text
Piano+bass+drum loop:
  BPM: 92
  key: C minor
  chords: Cm | Ab | Bb | Gm
  components: piano, bass, drums
  recommendation: chord/MIDI extraction + broad drum/bass stems only
```

### 10.5 Ambience/open-world path

For field recordings:

```text
1. Detect event classes.
2. Detect background bed.
3. Mark contamination.
4. Avoid fake musical separation.
```

Example:

```text
Forest loop:
  birds: strong
  insects: medium
  wind: low
  traffic: possible
  recommended: describe and tag; no audio separation recommended
```

---

## 11. Open-world unknown handling

There are almost infinite sounds. The system must not require every sound to fit a known category.

### 11.1 Unknown is not useless

Unknown sounds should still get:

```text
traits
shape
component guesses
novelty score
similar known examples
possible use roles
review reason
cluster ID
```

Example:

```text
Unknown alien scrape:
  traits: metallic, granular, rising, wide, noisy
  role candidates: FX motion, texture, foley scrape
  known similarity: metal scrape FX, glitch texture
  final: _TO_REVIEW / Unknown Compound Sound
```

### 11.2 UnknownOwnerBrain

The UnknownOwnerBrain should trigger when:

```text
no owner brain is confident
multiple owners conflict strongly
nearest-neighbor distance is high
open-world novelty score is high
known training prototypes do not match
```

It should not just dump the sound in review silently. It should describe why.

### 11.3 Unknown clustering

Repeated unknowns should form clusters:

```text
unknown_cluster_001: metallic rising scrapes
unknown_cluster_002: synthetic bird-like chirps
unknown_cluster_003: watery machine pulses
unknown_cluster_004: low organic monster roars
```

These clusters become training candidates.

---

## 12. Expanded edge-case library

### 12.1 Instrument transformed into texture/FX

#### Sax with huge reverb tail

Evidence:

```text
source_like: sax / reed / brass-woodwind
condition: huge reverb, smeared attack, wide tail, low direct-signal clarity
usable_as: ambient texture or FX
```

Possible outputs:

```text
Textures / Hybrid Textures
FX / Hybrid Designed FX
_TO_REVIEW / Possible Instrument or Texture
```

Do not blindly route to clean sax.

#### Guitar feedback drone

```text
source_like: electric guitar
shape: sustained drone / feedback texture
usable_as: texture or FX
not: guitar loop
```

#### Piano reverse reverb

```text
source_like: piano
shape: reverse swell / tail wash
usable_as: transition FX or texture
not: clean piano one-shot
```

#### Vocal chop vs vocal FX vs texture

```text
clean phrase -> Instruments / Voice
short rhythmic vocal chop -> Instruments / Voice / Vocal Chops
formant sweep / robot vowel -> FX / Formant FX
breathy smear / crowd wash -> Texture or FX depending role
```

#### Bass that becomes FX

```text
clean stable low pitch -> bass / 808 / sub
low roar with motion, unstable pitch, wide tail -> FX / Designed Low Boom or Texture
```

### 12.2 Drums and percussion edge cases

#### Real drum loop with processing

Protect Drums if:

```text
kick/snare/hat alternation exists
pulse grid is strong
drum-band roles are stable
transient pattern resembles groove
```

Do not let broad FX motion steal it.

#### Drum loop with transition layer

```text
primary: drum loop
secondary: riser / sweep layer
folder: Drums / Drum Loops
sidecar: contains_transition_layer = true
```

#### Crash cymbal vs metal scrape FX

Crash cymbal:

```text
single bright hit
cymbal-like decay
drum-kit role
```

Metal scrape/drop FX:

```text
moving spectrum
scrape body
non-cymbal temporal shape
transition role
```

#### Tiny zap vs kick/click

```text
kick: sub punch, drum envelope, centered, foundational low role
zap: electrical transient, synthetic high/mid motion, no drum body
click: short high transient, may be percussion or UI depending owner evidence
```

### 12.3 Non-instrument open-world edge cases

#### Household objects

```text
doors
keys
cups
paper
plastic bags
cloth rustle
footsteps
tools
kitchen sounds
```

Need material/action traits:

```text
metallic
wooden
glassy
plastic
cloth
liquid
scrape
rattle
thud
snap
crack
```

#### Machines and vehicles

```text
engine idle
engine rev
brake squeal
motor hum
fan
industrial drone
gear movement
hydraulic hiss
train/plane/car passby
```

Do not call machine rhythm a drum loop unless actual drum anchors exist.

#### Nature / weather / water

```text
rain
wind
thunder
stream
ocean waves
fire
leaves
mud
ice cracking
```

Use texture/ambience/foley roles before forcing FX.

#### Animals and biological sounds

```text
single bird
bird chorus
insects
frogs
dogs
cats
crowd/human group
monster-like processed voice
```

Animal-like does not always mean real animal. Synthetic chirps and formant FX need separate routing.

#### Electrical / radio / data noise

```text
static
hum
buzz
morse-like beeps
digital glitch
radio tuning
EMI whine
modem/data chirps
```

Can route to FX, Textures, or review depending duration/role.

#### UI / game / synthetic one-shots

```text
beeps
blips
coin sounds
menu clicks
confirmation tones
laser shots
power-ups
```

Need to separate:

```text
tonally useful synth one-shot
UI/game FX
drum/percussion click
```

#### Full music loops

A full loop can be highly processed and weird but still be music.

Protect full music loops when:

```text
multi-instrument structure
musical harmonic continuity
bar/grid stability
chord or bass movement
strong drum + instrument co-presence
```

#### Unknown alien sounds

Do not guess. Describe:

```text
metallic/noisy/tonal/granular
rising/falling/static/pulsed
wide/mono
impact/texture/loop/phrase
similar known clusters
```

---

## 13. Training and correction system

### 13.1 Final folder is not enough

If Aaron corrects a file to `FX`, the system must not blindly teach:

```text
this is FX and not everything else
```

It should capture layered correction info:

```text
final folder
source-like identity
usable role
shape
owner brain positive label
owner brain negative labels
component labels
music property corrections
```

### 13.2 Correction example

For a sax with huge reverb tail:

```json
{
  "final_folder": "Textures/Hybrid Textures",
  "source_like": "sax",
  "usable_role": "ambient_texture",
  "positive_owner": "AmbientTailOwnerBrain",
  "negative_owners": ["CleanInstrumentPhraseOwnerBrain"],
  "traits": ["sax_like", "huge_reverb_tail", "wide", "smeared_attack"]
}
```

### 13.3 Positive and negative training

Every correction should create:

```text
positive examples for the correct owner
negative examples for the owner that made the mistake
counterexamples for protection gates
```

Example:

```text
If real drum loop was stolen by FX:
  positive -> DrumLoopOwnerBrain
  negative -> DesignedMotionFXOwnerBrain
```

### 13.4 Active learning queue

The system should ask for labels on the most useful examples:

```text
high confidence wrong candidates
high novelty unknown clusters
owner conflicts
near-threshold protectors
frequent review families
```

Do not ask Aaron to label everything.

### 13.5 Calibration and rollback

Training must be versioned and reversible:

```text
brain version
feature schema version
training run ID
positive/negative counts
before/after regression results
rollback point
canary failures
```

No training should be applied if canary tests fail.

---

## 14. Phased engineering roadmap

### Phase 0 — Stabilize current sorter

Goal:

```text
Stop breaking drums/instruments while improving FX.
```

Tasks:

```text
freeze current regression failures
maintain no-source-name invariant
keep owner-arbitration safety gates
make obvious FX anchors pass without stealing drum/music loops
```

Exit criteria:

```text
known RealFX anchors route correctly
known drum loops stay Drums
known instrument loops stay Instruments
wet sax does not become review/voice/drums
full music loop does not become FX
```

### Phase 1 — Sidecar schema and evidence export

Goal:

```text
Make every classification produce reusable evidence.
```

Tasks:

```text
write JSON sidecar per file
add manifest columns for owner/shape/component/music fields
no routing changes yet
```

Exit criteria:

```text
sorter output includes sidecars
old folder output still works
regression tests unchanged
```

### Phase 2 — Read-only music properties MVP

Goal:

```text
Add BPM, tempo candidates, root note, key estimates, and confidence.
```

Tasks:

```text
BPM estimator
tempo candidates
half/double-time candidates
root note/pitch estimate
key estimate for tonal loops
confidence policy
```

Exit criteria:

```text
piano/bass/synth loops get useful music metadata
noise/drums/FX do not get fake chord labels
```

### Phase 3 — Chord timeline MVP

Goal:

```text
Estimate chords for clean piano/keys/guitar/synth loops.
```

Tasks:

```text
bar segmentation
chroma/HPCP summaries
chord candidate scoring
confidence per chord
not-applicable policy
```

Exit criteria:

```text
simple piano loops get plausible chord timelines
ambience/FX/drums do not get fake chords
```

### Phase 4 — Component analyzer MVP

Goal:

```text
Detect components inside loops without changing sorting.
```

Tasks:

```text
drum component inventory
mixed loop component inventory
forest/ambience component inventory
FX loop component inventory
```

Exit criteria:

```text
drum loops report kick/snare/hat/shaker guesses
mixed loops report instrument layers
forest loops report birds/wind/water/traffic guesses
```

### Phase 5 — Owner brain interfaces

Goal:

```text
Create trainable owner-brain interface without replacing rules yet.
```

Tasks:

```text
OwnerBrain protocol
score/confidence/support/blocker outputs
positive/negative prototype storage
owner diagnostics in sidecar
```

Exit criteria:

```text
owners can run read-only beside existing rules
no routing behavior changes unless explicitly enabled
```

### Phase 6 — DesignedMotionFXOwnerBrain pilot

Goal:

```text
Train one specialist brain for FX motion.
```

Tasks:

```text
positive examples from risers/sweeps/whooshes/laser motion
negative examples from drum loops/synth arps/full music loops/sax/guitar loops
routing only through owner-arbitration gates
```

Exit criteria:

```text
obvious sweeps/rises improve
real drum/music/instrument protectors still pass
```

### Phase 7 — Protector owner brains

Goal:

```text
Prevent FX from stealing real drums and instruments.
```

Owner brains:

```text
DrumLoopOwnerBrain
CleanInstrumentPhraseOwnerBrain
MixedMusicLoopOwnerBrain
VoicedPhraseOwnerBrain
BassFoundationOwnerBrain
```

Exit criteria:

```text
real drum loops protected
full music loops protected
wet sax handled based on usable role
voice phrases protected
bass foundations protected
```

### Phase 8 — AmbientTail / Degraded Instrument brain

Goal:

```text
Handle sounds that are source-like but no longer clean source instruments.
```

Examples:

```text
sax with huge reverb
piano reverse wash
guitar feedback drone
vocal smear
bass roar
```

Exit criteria:

```text
source-like identity preserved
usable role can become Texture/FX
clean instruments remain clean instruments
```

### Phase 9 — Unknown / novelty brain

Goal:

```text
Open-world handling for sounds that do not fit.
```

Tasks:

```text
novelty score
unknown clusters
similar known examples
review reason generation
active learning queue
```

Exit criteria:

```text
unknowns are described usefully
repeated unknown families become visible training candidates
```

### Phase 10 — Loop Analyzer product adapter

Goal:

```text
Separate tool that analyzes loops without necessarily sorting them.
```

Outputs:

```text
BPM/key/chords
components
usable role
warnings
loop quality
```

Exit criteria:

```text
single-file and folder-based loop analysis works
exports readable report and JSON sidecars
```

### Phase 11 — Stem Intelligence prototype

Goal:

```text
Recommend useful musician outputs from component intelligence.
```

Tasks:

```text
drum MIDI transcription MVP
component usefulness scoring
separation-not-recommended warnings
broad stem integration only when useful
```

Exit criteria:

```text
reports kick/snare/hat patterns for some drum loops
recommends MIDI/reconstruction vs audio separation honestly
```

### Phase 12 — Search / Flavor Browser

Goal:

```text
Search by traits, components, music properties, and roles.
```

Examples:

```text
A minor piano loops around 90 BPM
forest with birds and water but no traffic
dark wide evolving FX
dry punchy kicks
sax-like textures with huge reverb
```

Exit criteria:

```text
sidecar index searchable locally
results show confidence and warnings
```

---

## 15. Testing strategy

### 15.1 Balanced panels

Every improvement needs positive anchors and counteranchors.

Example FX panel:

```text
positive FX:
  down sweep
  riser
  whoosh
  laser
  scrape drop
  glitch loop

negative protectors:
  real drum loop
  real synth arp
  real sax phrase
  real guitar loop
  full music loop
  real bass loop
```

### 15.2 Component tests

```text
drum loop with kick/snare/hats
drum loop with hats+shaker ambiguity
mixed loop with piano+bass+drums
forest loop with birds+wind+traffic
FX loop with riser+impact
```

### 15.3 Music property tests

```text
clean piano loop with known BPM/key/chords
bass one-shot with known root
synth stab with known chord/no chord ambiguity
drum loop where key/chords must be not-applicable
noise FX where key/chords must be not-applicable
```

### 15.4 Stem intelligence tests

```text
drum loop -> MIDI/reconstruction recommended
hat+shaker overlap -> no hat-only separation recommended
mixed music loop -> broad description + chord/MIDI recommendation
field recording -> describe-only recommended
```

### 15.5 Regression standard

A patch is not acceptable if it fixes one family but breaks another.

Required check categories:

```text
FX anchors
Drum protectors
Instrument protectors
Music-loop protectors
Voice protectors
Bass protectors
Texture/ambience protectors
Unknown/review behavior
No-source-name invariant
```

---

## 16. Recommended repository structure

```text
src/aaron_audio_intelligence/
  core/
    audio_loader.py
    feature_extractor.py
    sidecar_schema.py
    evidence_types.py

  traits/
    trait_brain_base.py
    timbre_traits.py
    material_traits.py
    motion_traits.py

  shapes/
    shape_brain_base.py
    loop_shapes.py
    transition_shapes.py
    texture_shapes.py

  owners/
    owner_brain_base.py
    drum_loop_owner.py
    clean_instrument_owner.py
    mixed_music_owner.py
    designed_motion_fx_owner.py
    ambient_tail_owner.py
    unknown_owner.py

  components/
    drum_components.py
    mixed_loop_components.py
    ambience_components.py
    fx_components.py

  music/
    tempo_analyzer.py
    pitch_analyzer.py
    key_analyzer.py
    chord_timeline.py

  stem_intelligence/
    stem_planner.py
    drum_transcription.py
    reconstruction_policy.py
    separation_policy.py

  products/
    sorter_adapter.py
    loop_analyzer_adapter.py
    stem_intelligence_adapter.py
    search_adapter.py
    training_console_adapter.py

  training/
    correction_schema.py
    active_learning_queue.py
    trainer.py
    rollback.py

  tests/
    panels/
      fx_motion_panel.py
      drum_protector_panel.py
      instrument_protector_panel.py
      component_panel.py
      music_property_panel.py
      open_world_panel.py
```

---

## 17. Immediate next action

The next practical implementation chunk should **not** be Stem Intelligence yet.

Do this first:

```text
Phase 1: Sidecar schema and evidence export
```

Why:

```text
It is low risk.
It does not change routing.
It gives every future tool reusable data.
It helps debug current sorter failures.
It creates the foundation for Loop Analyzer, Training Console, Search, and Stem Intelligence.
```

Concrete first task:

```text
Add read-only sidecar generation to the current sorter.
Do not change folder decisions.
Expose current shapes, voters, owner-like scores, top matches, features, warnings, and music placeholders.
```

Then add music properties and components as read-only analysis before letting them influence routing.

---

## 18. What not to do

Do not:

```text
build one giant app that does everything at once
let shape directly force final folders without owner gates
train only from final folders
use filenames/source paths as production evidence
promise perfect stem separation
fake chord labels on noise/drums/FX
make unknown sounds disappear into useless review
mutate broad ledgers without evidence
hide secondary traits from the manifest
```

---

## 19. Final recommendation

The long-term product should be:

```text
Aaron Audio Intelligence Platform
```

The Sound Sorter is the first product, but the shared core should be built to support:

```text
sorting
loop analysis
music-property extraction
component inventory
stem intelligence
search and browsing
training and correction
open-world sound discovery
```

The most important engineering move is:

```text
Make every classification produce reusable evidence.
```

Once the sidecar evidence exists, the project can grow into a family of tools without rewriting the same audio intelligence over and over.

