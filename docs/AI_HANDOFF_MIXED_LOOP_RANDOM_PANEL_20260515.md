# AI Handoff: Mixed Loop Over-Narrowing and Random Panel Update

Date: 2026-05-15

## Problem confirmed

Aaron identified `WS_KIT_3_West_Coast_Melody_Loop_Gnarly_Dm_100BPM.wav` as a mixed instrumental melody loop with bells/chimes/layers, not a woodwind-only loop.

The old result was:

```text
Instruments/Brass and Woodwinds/Loops
```

The bad mechanism was not the brain. The raw brain/consensus was already safe enough:

```text
Instruments/Instrument Loops/Loops
```

Then parent eligibility over-narrowed it because `reed_like_sustained_loop` fired.

## Architecture-safe fix

Do not use parent eligibility to promote generic/mixed instrument loops into Brass/Woodwinds.

Changed broad fallback for:

```text
pitched_reed_or_instrument_loop
pitched_reed_or_instrument_phrase
```

to:

```text
Instruments/Instrument Loops/Loops
```

This still blocks Drums, FX, Human/Voice FX, animals, machines, ambience, etc. It just does not invent a specific Brass/Woodwind branch.

`DecisionCoreV2._broad_reed_bucket_for_generic_instrument()` now refuses to broaden if the eligibility fallback is not itself a Brass/Woodwind-style path.

## Tests

New:

```text
tests/test_parent_eligibility_synthetic_mixed_loop_over_narrowing.py
```

Updated:

```text
tests/test_parent_eligibility_fact_roles_v27.py
```

The old test expected reed-like evidence to force Brass/Woodwinds. That expectation was the bug. It now expects broad Instrument Loops unless voters already chose Brass/Woodwinds.

## Real validation

`WS_KIT_3_West_Coast_Melody_Loop_Gnarly_Dm_100BPM.wav` now sorts to:

```text
Instruments/Instrument Loops/Loops
```

## Random panel updates

The random panel sampled AKWF and Sorted samples. Aaron clarified AKWF are tiny wavetable files, so Broken Or Tiny is correct, and we should avoid them in broad sweeps.

Defaults now skip:

```text
AKWF/wavetable folders and files
Sorted samples
FX_Aaron
one_shot_percussive_sounds
Loop 4
Combined training/regression paths
```

Override flags are available through environment variables in the `.command` file.

## Validation result

```text
59 passed
```

## Next investigation

Rerun random panel after this patch. Focus on:

- real non-vocal material still going to Human and Voice FX
- real non-alarm material going to Alarm
- keys/Rhodes/synth/guitar/string material going to Bass Loops
- mixed or vocal loops still going to Brass and Woodwinds
- pitched instruments going to Drums

Do not treat AKWF/Broken Or Tiny as a problem.
