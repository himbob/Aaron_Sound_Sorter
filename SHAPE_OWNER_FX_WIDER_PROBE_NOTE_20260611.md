# Shape-owner FX wider probe repair note — 2026-06-11

## Goal

Continue the RealFX repair without using source names as evidence and without globally boosting FX over real drums/instruments.

This pass sampled 40 rows from `realFX.zip` as 20 two-file probes spread across the archive. The probe harness used source paths only as an external audit label. Production routing remains source-name blind.

## Wider probe result

Rows completed: 40

Source-FX rows completed: 29

Source-FX status after this pass:

- PASS_FX: 20
- FAIL_INSTRUMENT: 4
- QUESTIONABLE_DRUM: 3
- PASS_DRUM_ALLOWED: 2

The remaining misses are not all the same class. The safest next fixes are separate lower-level producers for:

1. low boom / bass-loop ambiguity,
2. tiny zap / kick ambiguity,
3. clean tonal FX-loop / instrument-loop ambiguity,
4. vocal or ambience FX that has stable tonal/voice evidence.

Do not solve those with a global FX boost.

## What changed

### 1. Measured FX decoy shape body

Added a conservative bridge from ShapeVoter morphology to the measured transition-FX claim producer.

It can now rescue FX when:

- the primary shape is a decoy such as texture/noise/top-loop/beat-loop/solo/pitched phrase,
- secondary shape pressure indicates designed FX motion, tonal FX, whoosh, reverse, riser/drop, siren/alarm, or impact tail,
- PhysicsFX subpanels also support FX motion / transition / glitch / whoosh / radio / formant / boom / siren / alarm,
- clean stable instrument evidence is not strong,
- real drum material is not strong enough to own the parent.

This lets ShapeVoter own the parent for strong FX morphology without making ShapeVoter a filename-like category shortcut.

### 2. Designed low-FX support widened carefully

Added support for:

- short directional stutter bodies,
- longer glitch/formant loops with measured FX support.

This fixed short falling stutters and circuit-bent/groove FX that had enough motion evidence but were previously routed as drums or instrument loops.

### 3. Arbiter mirror logic

The final arbiter now mirrors the same measured FX-decoy body check. This prevents a valid measured transition-FX claim from being undone by older arbitration gates.

### 4. One-hit percussion guard

A noisy single-event texture can look FX-like, but it is also exactly how some real percussion fixtures look. Added a guard:

- one or two onsets,
- no onset span,
- tiny tail,
- no strong measured motion/transition owner,

means the broad texture-to-FX rule stands down.

This protects short percussion hit tests while still allowing multi-event scrapes/stutters to route to FX.

## Fixed probe families

Examples fixed or protected by this pass:

- `falling_chopped_stutter.wav` no longer becomes snare.
- `short_metal_plate_crash.aif` no longer becomes drum loop.
- `shaky_tremor_glitch_loop.wav` no longer falls to shape conflict review.
- `SL_Circuit_Bent_Groove_03_085_BPM.wav` no longer becomes generic instrument loop.

## Deliberately not forced yet

Some rows stayed conservative because forcing them would damage normal material:

- dry low body hits can look like kicks,
- loose junk/object hits can reasonably land in percussion,
- spaceship/boom low tonal material can look like bass loops,
- clean ambience/tonal loops can look like sax/synth/instrument loops,
- clean pan-pipe-like or tonal riser bodies need a better tonal-FX-vs-real-instrument model.

These need new measured producers, not a bigger FX override.

## Tests added

Added regression coverage for:

- noisy single-event texture hits standing down from FX,
- noisy multi-event designed texture bodies owning broad FX.

## Files changed

- `src/aaron_sound_sorter/engine/claim_producers/measured_transition_fx.py`
- `src/aaron_sound_sorter/engine/family_claim_arbiter.py`
- `tests/test_designed_fx_shape_claims.py`
