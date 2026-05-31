# AI Handoff Template

## What changed

## What did not change

## Tests added or updated

## Tests run

## Tests not run

## Known failures

## Next safe step

## Files changed

## Architecture notes

---

## Absolute blind-sorting rule

Production sorting logic must never use producer filenames, source folder names,
ZIP member names, path tokens, sample-pack labels, or any other source-name text
as classification evidence. Names are allowed only for I/O, manifest display,
post-decision diagnostics, test fixture selection, and human review. Voters,
roles, eligibility, consensus, conflict resolution, and final placement must use
audio measurements, learned brain candidates, physics candidates, shape/role
facts, and explicit manual corrections only.

Every sorter-logic change must preserve this invariant and must pass:

```bash
./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command
```

