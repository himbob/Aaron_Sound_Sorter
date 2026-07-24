# Archived: Designed FX shapes patch

## Purpose

This patch gives the lower shape/claim layers a way to recognize designed FX structures that imitate normal music or drum evidence:

- low designed FX that looks like bass/kick material
- tonal/formant/robotic FX that looks like guitar/sax/synth
- motion/stutter/sweep FX that looks like a drum loop

The patch keeps the production blind-sorting rule intact. It does not use filenames, source folders, ZIP member names, or pack labels as production evidence.

## Changed files

- `src/aaron_sound_sorter/voters/shape_voter.py`
  - Adds structural shape labels:
    - `designed_low_fx`
    - `designed_motion_fx_loop`
    - `designed_tonal_fx`
  - Lets ShapeVoter read measured physics subpanel evidence from `facts.evidence["physics_subpanels"]["flat"]`.
  - Uses real drum and clean instrument guards before promoting designed-FX shapes.

- `src/aaron_sound_sorter/engine/claim_producers/measured_transition_fx.py`
  - Allows designed-FX shapes to emit broad `FX/Hybrid Designed FX` claims.
  - Requires compatible FX subpanel support and candidate support.
  - Blocks obvious real drum material and clean instrument loops.

- `src/aaron_sound_sorter/engine/family_claim_arbiter.py`
  - Updates the arbiter legality check so the new designed-FX claims are allowed to compete.
  - Mirrors the same real-drum guard used by the claim producer.

- `src/aaron_sound_sorter/engine/claim_producers/measured_buckets.py`
  - Lets measured transition/designed-FX claims ride alongside true-bucket claims instead of being skipped when a false true-bucket instrument/drum rescue fires first.

- `tests/test_designed_fx_shape_claims.py`
  - Adds regressions for designed tonal FX rescue, real drum blocking, and clean instrument blocking.

## Spot-probe impact

Key realFX rows improved:

- Row 2 `APLSFX_COMPLEX_Reverse Heartbeat.wav`: `Instruments/Bass/Bass Loops` -> `FX/Hybrid Designed FX`
- Row 6 `falling_chopped_stutter.wav`: `Drums/Snares/...` -> `FX/Hybrid Designed FX`
- Row 15 `darkEnergy.wav`: `FX/Impacts and Hits/Short Impact/Long FX` -> `FX/Hybrid Designed FX` top-family still FX
- Row 21 `robot_bell_melody_fragment.wav`: `Instruments/Guitar/Electric Guitar/One Shots` -> `FX/Hybrid Designed FX`
- Row 22 `sliced_bell_glitch_run.wav`: `Instruments/Woodwinds/Saxophone/Loops` -> `FX/Hybrid Designed FX`
- Row 73 `Sound Fx Spanish Crowd.wav`: `Instruments/Synths/Synth Lead/One Shots` -> `FX/Hybrid Designed FX`
- Row 81 `WS_CFX2_C_Synth_Sweep_4bars.wav`: `Instruments/Instrument Loops/Loops` -> `FX/Hybrid Designed FX`

Still not solved safely in this pass:

- very pure low roars that look like clean bass (`APL_SFX_Whoosh_Roar_01.wav`)
- strongly percussive/hi-hat-like sweeps where the drum panels are very high (`MajorLaser.wav`, `5.FX down sweep 128bpm.wav`)
- some long FX/music hybrid loops that still look like stable instrument loops

Those need a separate measured motion-vs-drum-loop claim or a stronger low-roar-vs-bass body model; forcing them here would likely hurt normal drums and instruments.
