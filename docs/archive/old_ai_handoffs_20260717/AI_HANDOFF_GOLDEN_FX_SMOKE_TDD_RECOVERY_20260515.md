# AI Handoff: Golden FX Smoke TDD Recovery

Date: 2026-05-15

## What went wrong

The resolver patches had green unit tests but still broke the full FX smoke run.
Two failure patterns were confirmed:

1. Decisive bass-loop evidence was flattened to generic `Instruments/Instrument Loops/Loops` or stolen by FX/Risers.
2. Pitched musical loops were over-rescued into FX/Risers or FX/Alarm without actual transition-shape evidence.

The earlier test suite was too narrow. It tested the code that changed, not the
full behavior that mattered.

## What this patch does

### Production code

Changed:

```text
src/aaron_sound_sorter/engine/decision_core_v2.py
```

Main fixes:

- Candidate true-bucket rescue now runs before broad smoke-stability fallback.
- `bass_loop` rescue requires actual Bass/808/sub/synth-bass candidate support.
- False bass roles without Bass candidate support no longer force Bass Loops.
- Pitched music loops are not stolen by close FX/Riser candidates unless shape is a strong transition.
- `fx_tonal_alert_or_siren` cannot steal normal synth/piano/guitar/string loops into FX/Alarm unless transition evidence is present.
- Shape parsing now correctly reads `facts.evidence["shape_vote"]` dictionaries.

### Tests

Added:

```text
tests/test_decision_core_golden_failure_regressions.py
tests/test_fx_smoke_expected_bucket_audit.py
```

Updated stale expectation in:

```text
tests/test_decision_core_v2.py
```

The old expectation promoted weak reed-like generic instrument loops into
`Brass and Woodwinds`. That is no longer acceptable because it over-narrowed
mixed/vocal/melodic loops. Broad `Instruments/Instrument Loops/Loops` is the
safer fallback unless the voters independently choose Brass/Woodwinds.

### Tools

Added:

```text
tools/audit_fx_smoke_expected_buckets.py
commands/quality/RUN_FX_GOLDEN_SMOKE_AUDIT.command
commands/quality/RUN_GOLDEN_TDD_REGRESSION_TESTS.command
```

## Validation performed in this environment

Ran:

```bash
PYTHONPATH=src:. pytest -q \
  tests/test_decision_core_golden_failure_regressions.py \
  tests/test_fx_smoke_expected_bucket_audit.py \
  tests/test_decision_core_v2.py
```

Result:

```text
19 passed
```

## Required validation on Aaron's Mac

After install:

```bash
cd /path/to/Aaron_Sound_Sorter
./commands/quality/RUN_GOLDEN_TDD_REGRESSION_TESTS.command
```

Then rerun the exact FX smoke sort and audit the newest manifest:

```bash
MANIFEST="$(find ./_real_sort_tests -name 'Aaron_Sorted_Sounds_manifest.csv' -type f -print0 \
  | xargs -0 ls -t 2>/dev/null \
  | head -1)" \
./commands/quality/RUN_FX_GOLDEN_SMOKE_AUDIT.command
```

Do not accept the patch if this golden audit fails.

## Important note

The golden audit uses source folder names only as a post-sort test oracle. This
is not runtime sorting logic and does not violate the no-source-name rule.
