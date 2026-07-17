# AI Handoff: Blind Sorting Invariant and True-Bucket Resolver Patch

Date: 2026-05-15

## Why this bundle exists

Aaron caught the most dangerous possible failure class: any sorter logic that uses filenames, source folders, ZIP member names, or source-path words can make the system look better than it really is. The project requirement is absolute: source names are for diagnostics after the decision, never for sorting evidence.

## Hard invariant

Production sorting code must never use source-name text as classification evidence.

Banned in sorting/voting/role/eligibility/consensus/resolver/final placement:

- producer filename words
- source folder names
- ZIP member names
- path tokens
- old category names in source paths
- sample-pack names
- source-hint helper functions
- `trust_input_names` style switches

Allowed:

- file I/O
- ZIP staging
- manifest display
- report-only suspicious diagnostics after placement
- test fixture selection
- human review packs
- explicit manual correction/training folder labels chosen by Aaron

## New audit

Added:

```text
tools/audit_no_source_name_sorting.py
tests/test_no_source_name_sorting_invariant.py
commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command
```

This audit scans the active production sorting/voting/decision code for banned source-name evidence markers.

It passed on this bundle:

```text
PASS: production sorting/voting/decision code contains no banned source-name evidence markers.
```

## Existing architecture patch preserved

This bundle also preserves the true-bucket conflict resolver work:

- kick/drum/percussion loops can be rescued to Drums/Drum Loops when candidate evidence supports it
- keys/Rhodes one-shots can be rescued to instrument/keys instead of FX
- non-voice FX can be rescued out of Human/Voice FX
- growing bell/tonal FX can remain FX instead of generic Instrument Loops
- ambiguous conflicts still go to `_TO_REVIEW/Measured Role Conflict`

## Required workflow for future fixes

1. Add a synthetic regression test first.
2. Prove it fails before the fix.
3. Patch the smallest evidence-based seam.
4. Prove the same test passes.
5. Run the no-source-name sorting audit.
6. Run the focused regression panel.
7. Use file/folder names only after placement for diagnostics.

## Validation run here

```bash
./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command
./commands/quality/RUN_TRUE_BUCKET_CONFLICT_REGRESSION_TESTS.command
```

Results:

```text
No-source-name audit: PASS
Focused regression panel: passed
```

## Warning to future AIs

Do not add filename, source-folder, ZIP-member, sample-pack, or source-path text to sorter logic. If a filename appears to explain a bug, use it only to pick a test fixture or flag a report row. Then encode the actual audio-behavior pattern as synthetic facts and fix the evidence seam.
