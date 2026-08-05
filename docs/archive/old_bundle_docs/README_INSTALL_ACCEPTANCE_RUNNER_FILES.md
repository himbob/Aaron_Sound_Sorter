# Install Acceptance Runner Documentation Files

Copy these files into the project tree:

```text
commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command
docs/AI_SAFE_ACCEPTANCE_TESTING.md
AI_READ_THIS_FIRST_ACCEPTANCE_APPEND.md
```

Then make the command executable:

```bash
cd /path/to/Aaron_Sound_Sorter
chmod +x commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command
```

Optional: paste the contents of `AI_READ_THIS_FIRST_ACCEPTANCE_APPEND.md` into `AI_READ_THIS_FIRST.md`, or leave it as a separate top-level AI note.

Test the runner:

```bash
cd /path/to/Aaron_Sound_Sorter
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command --list-cases
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command --case-id clap_clean_clap2
```
