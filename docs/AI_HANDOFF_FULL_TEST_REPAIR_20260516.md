# AI Handoff: Full Test Repair and DecisionCore Safety Lattice

Date: 2026-05-16

## Why this patch exists

The previous bundle shipped after a narrow pytest subset. Aaron then ran the full regression command and exposed many failures. That was a process failure and a code failure.

The root problem was that `DecisionCoreV2` had accumulated competing rescue/review rules. Several true-bucket rules were guarded too tightly and failed the exact regressions they were meant to protect:

- Kick/drum loops stayed in `Instruments/Bass` or generic `Instrument Loops`.
- Drum breaks stayed in `FX/Human and Voice FX`.
- Non-voice FX stayed in `FX/Human and Voice FX/Crowd`.
- Growing metallic bell FX stayed in generic `Instrument Loops`.
- Pitched percussion stayed in generic `Instrument Loops` or could be stolen by FX.
- Generic conflict rules sometimes reviewed or rescued too broadly.

## Permanent process rule

No future sorter logic bundle may be called fixed unless the full regression suite is run.

Required gates:

1. No-source-name audit passes.
2. All pytest files pass. If all-at-once pytest is too slow or opaque, run `commands/quality/RUN_ALL_PYTESTS_CHUNKED.command` and require every test file to pass or skip/xfail according to pytest.
3. The FX_Aaron2 one-by-one golden audit is run on Aaron's Mac before production acceptance.
4. If the full FX audit was not run, the bundle is only a candidate patch.
5. Do not claim a bundle is validated from a narrow subset.

## Runtime sorting source-name rule

Production sorting, voting, eligibility, roles, consensus, and final placement must never use producer filenames, source folder names, ZIP member names, path tokens, sample-pack labels, or any source-name text as evidence. Reports and golden audits may use source paths only after sorting as test oracles.

## Code changed

- `src/aaron_sound_sorter/engine/decision_core_v2.py`
  - Fixed shape parsing for simple string `shape_vote` evidence.
  - Added an early candidate adjudicator that runs before broad smoke fallbacks.
  - Restored true-bucket rescue for kick/drum/percussion loops.
  - Restored non-voice FX rescue out of Human/Voice false positives.
  - Restored tonal metallic/bell FX rescue, but only when a bell/chime/hybrid/metallic anchor exists, so ordinary piano/guitar/string loops are not stolen by generic riser/build candidates.
  - Protected vocal-loop conflicts from being forced into Drum Loops.
  - Protected weak reed/woodwind over-narrowing from stealing raw FX/vocal material.
  - Kept measured pitched percussion loops in Drums when candidate evidence is distant but role is structurally decisive.

## Tooling changed

- `commands/quality/RUN_ALL_PYTESTS_CHUNKED.command`
  - Runs every `tests/test_*.py` file one at a time and writes logs under `reports/pytest_chunked/run_*`.
  - Fails if any test file fails.

- `commands/quality/RUN_FULL_PYTEST_AND_GOLDEN_CHECKS.command`
  - Runs full pytest.
  - Optional `RUN_CHUNKED_PYTEST=1` runs chunked pytest after full pytest.
  - Optional `RUN_FX_ONE_BY_ONE=1` runs the FX_Aaron2 golden audit.

## Validation completed in sandbox

- The exact failure group pasted by Aaron now passes.
- The first 25 pytest files passed as a grouped run.
- The remaining pytest files were run individually because the full all-at-once run exceeded the sandbox timeout. Every test file run individually passed or skipped/xfail according to pytest.
- No-source-name audit passed and scanned 21 production sorting files.
- Real CLI spot checks passed:
  - `03_bass_Emn_178bpm.wav` -> `Instruments/Bass/Bass Loops`
  - `04.bass_92bpm_Em.wav` -> `Instruments/Bass/Bass Loops`
  - `SCY097_03_Sax_Loop_KeyEm_90bpm_01.wav` -> `Instruments/Instrument Loops/Loops`

## Acceptance required on Aaron's Mac

Run:

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command
./commands/quality/RUN_ALL_PYTESTS_CHUNKED.command
RUN_FX_ONE_BY_ONE=1 ./commands/quality/RUN_FULL_PYTEST_AND_GOLDEN_CHECKS.command
```

The production acceptance target is zero pytest failures and zero FX golden audit failures.
