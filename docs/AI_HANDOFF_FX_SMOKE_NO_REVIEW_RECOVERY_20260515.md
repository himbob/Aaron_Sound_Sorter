# AI Handoff: FX Smoke No-Review Recovery

## Context

Aaron reported that the old curated FX smoke test regressed after the conflict
resolver patches. The uploaded summary showed review rows in a pack that is
supposed to contain curated valid examples.

## Diagnosis

The resolver became too aggressive and treated several broad measured conflicts
as mandatory review even when the raw winner was already inside an allowed broad
family or could be safely broadened to a real bucket.

Examples from the manifest:

- Sax/reed-like loops with `pitched_percussion_conflict_loop` went to
  `_TO_REVIEW/Measured Role Conflict` even though raw was broad Instruments.
- Drum-top loops with `drum_loop` went to review due close Human/Voice candidate.
- Raw FX/drop leaves with measured `drum_loop` went to review instead of broad
  `Drums/Drum Loops/Loops`.
- Vocal one-shot/phrase rows with raw machine/motor FX went to review instead of
  broad `FX/Human and Voice FX`.

## Rule going forward

Do not fix this using filenames. Do not add FX-pack-specific code. The resolver
must operate only on voter candidates, measured facts, eligibility, raw folder,
and shape/role evidence.

## New tests

`tests/test_decision_core_fx_smoke_no_review_v2.py` locks five resolver behaviors:

1. Allowed broad Instrument Loop conflict stays Instruments, not Review.
2. Stable Drum Loops stay Drums, not Review.
3. Measured drum loop with incompatible FX/drop raw broadens to Drums.
4. Measured vocal phrase with incompatible machine/motor FX broadens to Human/Voice FX.
5. Generic pitched loop with close FX candidate does not review solely because FX is close.

## Validation

Run:

```bash
./commands/quality/RUN_FX_SMOKE_NO_REVIEW_REGRESSION_TESTS.command
```

Expected: `42 passed`.

Then rerun the real FX smoke sort on Aaron's Mac and audit the manifest:

```bash
MANIFEST="/path/to/Aaron_Sorted_Sounds_manifest.csv" \
./commands/quality/RUN_MANIFEST_NO_REVIEW_AUDIT.command
```

Target for `FX_Aaron2.zip`: zero review rows.
