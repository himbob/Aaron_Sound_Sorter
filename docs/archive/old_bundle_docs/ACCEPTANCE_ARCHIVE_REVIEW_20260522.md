# Locked smoke acceptance archive review, 2026-05-22

Reviewed upload archive: `Archive.zip`

The locked smoke acceptance harness ran correctly. It processed all 17 protected cases and produced the expected reports:

- `actual_results.csv`
- `failures.csv`
- `summary.txt`
- per-case manifests and brain audit files under `case_outputs/`

## Result

```text
Cases: 17
Protected cases: 17
Protected failures: 3
PASS: 14
FAIL_WRONG_FOLDER: 3
```

## Current protected failures

```text
kick_clean_cs_ne_monroe
  Expected: Drums/Kick-style destination
  Actual:   Drums/Percussion/Generic Percussion/One Shots

snare_clean_jackbaby
  Expected: Drums/Snare-style destination
  Actual:   Drums/Claps Snaps Slaps/Generic Clap/One Shots

synth_bells_no_safety
  Expected: Instruments/Synths, Instruments/Mallets and Bells, broad Instruments, or Review
  Actual:   Instruments/Voice/Vocal Loops/Loops
```

## Interpretation

The QA gate is doing what it is supposed to do. It blocks a future AI from claiming the sorter is fixed while clear sentinel failures remain.

The next classifier/debugging work should focus on these failures first, but only after the AI runs the locked smoke panel itself and proves no protected cases regress.
