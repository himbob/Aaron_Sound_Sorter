# AI Handoff: Golden FX Matrix Recovery and Testing Rule

Date: 2026-05-16

## Why this patch exists

A previous sequence of resolver patches optimized the wrong metric: reducing `_TO_REVIEW` in the curated FX_Aaron2 smoke run. That made the summary look better while damaging true broad buckets, especially:

- Bass loops flattened to `Instruments/Instrument Loops/Loops`.
- Drum loops flattened to `Instruments/Instrument Loops/Loops`.
- Sax/pitched reed loops routed to `Drums` or `FX/Human and Voice FX`.
- True vocal shouts stolen by `Instruments/Synths/Synth Chord`.

The new rule is: **summary counts are not proof.** The curated FX smoke pack must pass a broad-bucket golden audit.

## Permanent process rule

No future sorter-logic bundle may be called fixed unless all of these are true:

1. Add a failing synthetic or unit regression test for the exact failure class.
2. Prove the test fails before the fix, or document why the failure already exists in an uploaded manifest.
3. Patch the smallest safe seam.
4. Run all pytest tests, not only the new tests.
5. Run the FX_Aaron2 one-by-one golden audit on Aaron's Mac for production acceptance.
6. If the full FX audit was not run, say the bundle is a candidate patch, not fully validated.

## Source-name rule

Production sorting, voting, eligibility, roles, consensus, and final placement must never use producer file names, source folder names, ZIP member names, path tokens, or sample-pack labels as classification evidence.

The golden audit and review-pack tools may use source paths only after sorting, as a test oracle.

## Code seams changed

- `src/aaron_sound_sorter/engine/consensus.py`
  - Restores Bass Loop broad bucket when measured `bass_loop` plus BrainVoter Bass evidence are strong.

- `src/aaron_sound_sorter/engine/decision_core_v2.py`
  - Prevents useful raw Bass/Drum/Voice buckets from being demoted to generic Instrument Loops.
  - Prevents sax/reed-like pitched musical loops with vocal-like shape from becoming Human/Voice FX.
  - Prevents drum-loop true-bucket rescue from stealing sax/vocal pitched music into Drums unless measured drum/percussion evidence actually supports it.
  - Preserves true Human/Voice winners when measured voice evidence is present.

## New/updated tests and tools

- `tests/test_decision_core_fx_smoke_matrix_hard_failures.py`
- `tools/run_fx_aaron2_one_by_one_golden_audit.py`
- `commands/quality/RUN_FX_AARON2_ONE_BY_ONE_GOLDEN_AUDIT.command`
- `commands/quality/RUN_FULL_PYTEST_AND_GOLDEN_CHECKS.command`

The all-pytest run in this patch tree passed. Some audio-fixture tests skip when their external WAV fixtures are not present in a code-only bundle.

## Local validation completed here

- Full pytest: passed with skips for missing external audio fixtures.
- Real individual spot checks passed:
  - `03_bass_Emn_178bpm.wav` -> `Instruments/Bass/Bass Loops`
  - `04.bass_92bpm_Em.wav` -> `Instruments/Bass/Bass Loops`
  - `SCY097_03_Sax_Loop_KeyEm_90bpm_01.wav` -> `Instruments/Instrument Loops/Loops`
  - `Drum_Loop_10_94BPM.wav` -> `Drums/Drum Loops/Loops`
  - `DOJO_FBP_Female_Vocal_Shout.wav` -> `FX/Human and Voice FX`
- FX_Aaron2 one-by-one golden audit was run only with `LIMIT=5` in this sandbox due runtime limits; it passed those first five files.

## Required acceptance step on Aaron's Mac

Run:

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
./commands/quality/RUN_FULL_PYTEST_AND_GOLDEN_CHECKS.command
RUN_FX_ONE_BY_ONE=1 ./commands/quality/RUN_FULL_PYTEST_AND_GOLDEN_CHECKS.command
```

Or just the FX audit:

```bash
./commands/quality/RUN_FX_AARON2_ONE_BY_ONE_GOLDEN_AUDIT.command
```

The acceptance target is zero golden failures.
