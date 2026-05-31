# Acceptance Testing Note for Future AI Helpers

When validating code changes, do not rely only on the all-at-once locked smoke acceptance wrapper. In constrained AI sandboxes, the wrapper may be killed during quiet audio analysis even when the sorter itself is still working or has returned successfully.

Preferred validation command:

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command
```

To run a single case:

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command --case-id clap_clean_clap2
```

To force the same Python used by the local project virtual environment:

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
PROJECT_ROOT="$PWD" \
PYTHON_BIN="$PWD/.venv_phase4/bin/python" \
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command
```

Rules:

- A wrapper timeout is not automatically a classifier failure.
- A case is only accepted when it writes a real pass verdict.
- If the full wrapper times out, rerun one case at a time with heartbeat output.
- Do not claim acceptance passed unless every protected case passed.
