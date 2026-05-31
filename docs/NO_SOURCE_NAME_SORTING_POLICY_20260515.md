# No Source-Name Sorting Policy

Date: 2026-05-15

## Non-negotiable rule

Production sorting logic must never use any source-name text as classification evidence.

Banned as sorting/voting/decision evidence:

- producer filename words
- source folder names
- ZIP member names
- path tokens
- sample-pack names
- old category labels in source paths
- regexes over source paths
- any variable or helper whose purpose is source-name hints

Allowed uses of names:

- opening files
- staging ZIP members safely
- preserving original paths in manifests
- creating output filenames
- selecting test fixtures
- post-decision diagnostics and suspicious-row reports
- human review packs
- manual corrections explicitly made by Aaron
- training folder paths as labels when Aaron intentionally curates training folders

## Reason

The project goal is audio-physics and learned-reference sorting. Source names are often wrong, marketing-oriented, copied from old bad folders, or misleading. A sorter that uses names can look good while failing the actual audio problem.

## Required architecture boundary

The sorting path must be blind:

```text
audio file
  -> measured physics
  -> brain candidates
  -> physics candidates
  -> shape/role facts
  -> eligibility / consensus / resolver
  -> final folder
```

Names may enter only after final placement:

```text
final folder + original path
  -> manifest display
  -> suspicious diagnostics
  -> human review pack
```

## Permanent audit

This project now includes a hard audit:

```bash
./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command
```

It scans the production sorter/voter/decision code for source-name evidence markers such as:

```text
source_hint
name_hint
path_hint
filename_hint
folder_hint
trust_input_names
producer_name
producer_file
source_context_for_input_path
source_hint_tokens_for_path
source_backed_review_rescue
```

If this audit fails, do not run tuning tests and do not ship a patch.

## Required bug-fix workflow

For any sorter-logic bug:

1. Add a synthetic regression test first.
2. Prove it fails before the fix.
3. Patch the smallest evidence-based seam.
4. Prove the same test passes.
5. Run the no-source-name audit.
6. Run the focused regression panel.
7. Use filenames only in diagnostics after the decision.

## Instruction to future AIs

Do not reintroduce filename, folder-name, source-path, ZIP-member, or source-pack text into sorting logic. If you are tempted to use a filename clue because it seems obvious, stop. Add a report-only diagnostic instead.
