# AI Safe Locked Smoke Acceptance Testing

This project has a locked smoke acceptance panel under:

```text
tests/acceptance/locked_smoke_v1/
```

The normal all-case wrapper can be too quiet for some AI sandboxes. A quiet audio-analysis subprocess may be killed by the sandbox before the wrapper writes `PASS` or `FAIL`. Treat a wrapper timeout as a harness/runtime problem first, not automatically as a classifier failure.

## Recommended command for AI helpers

Use the one-case-at-a-time runner:

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command
```

This runner reads all case IDs from:

```text
tests/acceptance/locked_smoke_v1/expected_results.json
```

and runs:

```text
commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --case-id <case_id>
```

for each case, with heartbeat output during quiet analysis periods.

## Run one case

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command --case-id clap_clean_clap2
```

## List case IDs

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command --list-cases
```

## Stop on first failure

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command --stop-on-fail
```

## Use a specific Python

Use the project virtual environment when possible:

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
PROJECT_ROOT="$PWD" \
PYTHON_BIN="$PWD/.venv_phase4/bin/python" \
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command
```

For Python 3.13 comparison runs:

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
PROJECT_ROOT="$PWD" \
PYTHON_BIN="$PWD/.venv_py313_clean/bin/python" \
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command
```

## Full wrapper command

The full wrapper can still be tried first on a local Mac:

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
PROJECT_ROOT="$PWD" \
PYTHON_BIN="$PWD/.venv_phase4/bin/python" \
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command
```

If it hangs or gets killed, switch to the one-by-one runner above.

## Interpretation rules

A classifier failure is when a case produces a real failing verdict such as wrong folder, missing output, missing sample, or sorter nonzero exit.

A harness/runtime failure is when the sorter appears to finish or is quiet but the acceptance wrapper is killed before writing a verdict. In that situation, rerun the case by itself with heartbeat output.

Do not claim acceptance passed unless every protected case reports pass, either through the full wrapper or through the one-by-one runner.

## Report location

The one-by-one runner writes logs to:

```text
_reports/locked_smoke_one_by_one/run_YYYYMMDD_HHMMSS/
```

Each case has its own log file plus:

```text
case_results.tsv
```

Use `case_results.tsv` as the summary when reporting results.
