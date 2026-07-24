# Archived: Aaron Sound Sorter — Shape Owner FX Repair Note

Date: 2026-06-11

## Problem

The RealFX probe showed that several obvious designed FX files were not failing because ShapeVoter could not see them. They were failing because later drum/instrument authority layers treated repeated transients, bright noisy energy, or metallic high-band content as stronger than the high-confidence shape.

The clearest example was `scraping_string_drop_long.aif`:

- ShapeVoter: `transition_riser`, confidence `0.986111`
- Previous final folder: `Drums/Cymbals/Crash Cymbal/One Shots`
- Corrected final folder: `FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX`

This proves the architecture issue was authority ownership, not simply missing shape labels.

## Design decision

ShapeVoter should become a structural authority when the shape is unusually clear, but it still should not become a final folder classifier by itself.

The rule is:

> Shape may own the parent only when high-confidence structural FX morphology is supported by measured FX panels and the competing drum/instrument reading is explainable as a decoy.

This avoids the bad rule:

> weird = FX

## External references used for the design

There is no usable drop-in “FX shape library” that maps directly to this sorter’s folder tree. The closest public foundations are:

- Librosa feature extraction: spectral and rhythm descriptors such as centroid, flatness, RMS, tempogram, and delta features.
- Essentia/Freesound extractor: batch spectral, time-domain, rhythm, and tonal descriptors for sound collections.
- AudioCommons timbral models: timbral attributes such as brightness, roughness, sharpness, boominess, and reverberation for sound-effect search.
- AudioSet ontology: a broad sound-event ontology. Useful for vocabulary, but it is an event taxonomy, not a sample-pack shape hierarchy.

So the safest implementation is a small internal shape ontology built on the sorter’s existing measured features.

## New shape-owner behavior

### 1. Transition FX ownership

High-confidence `transition_riser` / `transition_drop` shapes may now beat crash/cymbal/drum decoys even when the body is tonal, as long as it has noisy/motion evidence and is not a clean low-noise instrument phrase.

This fixes short/odd risers that contain tonal scrapes, strings, bells, or metallic elements.

### 2. Designed FX loop/stutter ownership

Loop-looking shapes such as `top_loop`, `beat_loop`, and `repeated_phrase_loop` may now be treated as designed FX when all of the following are true:

- high shape confidence,
- low true pulse or drum identity support,
- strong designed-FX shape scores such as `designed_tonal_fx`, `designed_motion_fx_loop`, `glitch_stutter`, or `siren_alarm_tone`,
- supporting FX panels such as glitch/stutter, siren/alarm, radio/electrical, or formant FX.

This fixes laser/stutter cases where repeated transient structure looked like a drum loop.

### 3. Pulsed sweep motion over drum-loop authority

Some FX loops are rhythmic and pulsed. For these, the earlier “low rhythmic drum loop” rescue and high-hat/break-loop invariants were too strong. They now stand down when the shape stack shows a broad sweep/stutter motion body with strong spectral slope and FX panel support.

This fixes pulsed down-sweeps/uplifters without globally demoting normal drum loops.

## Guardrails

Shape ownership stands down for:

- clean stable tonal instrument loops,
- clear drum material with strong pulse/drum-source anchors,
- unsupported weirdness with no FX candidate or FX panel support,
- short percussion one-shots with strong drum anchors.

Production routing remains source-name blind. Filenames and ZIP paths are still used only by probe/audit tooling, not by sorter decisions.

## RealFX probe anchors fixed

| File | Previous problem | New result |
|---|---|---|
| `scraping_string_drop_long.aif` | Crash cymbal / drum one-shot | `FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX` |
| `MajorLaser.wav` | Drum loop steal | `FX/Digital Mechanical Industrial Transport/Glitches and Stutters/Generic Glitch/Long FX` |
| `DeepTechKitFour4-Fx-Uplifter_120bpm.wav` | Drum loop steal | `FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX` |
| `5.FX down sweep 128bpm.wav` | Drum loop steal | `FX/Structural and Transitional FX/Drops and Downlifters` |

## Code areas changed

- `shape_voter.py` already had the designed FX shapes from the previous pass.
- `measured_transition_fx.py` now trusts noisy/motion tonal transitions instead of blocking them as clean instruments.
- `eligibility.py` now has designed FX loop/motion eligibility roles.
- `final_drum_loop.py` now blocks drum-loop claims for designed FX loop decoys.
- `measured_early_drum_fx.py` now prevents low-rhythmic drum rescue from stealing motion FX.
- `measured_drum_structures.py` now blocks hi-hat loop claims for strong sweep/stutter motion bodies.
- `measured_music_structures.py` now blocks rhythmic-break drum claims for strong sweep/stutter motion bodies.
- `family_claim_arbiter.py` now lets high-confidence transition FX claims and designed-motion FX replacements win over drum-loop broadening.

## Remaining risk

This is not a full FX solution. It fixes the worst architecture seam: shape evidence being ignored after it already saw designed FX. The next big missing layer is a broader “designed material vs acoustic material” model for object hits, metallic scrapes, junk rattles, and creature/voice-like FX.
