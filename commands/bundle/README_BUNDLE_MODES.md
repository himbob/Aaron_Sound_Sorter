# Aaron Sound Sorter bundle modes

## Default: AI handoff with brains

Run:

```bash
./commands/bundle/bundle.sh
```

This is the normal bundle for giving another AI or engineer the working project state.
It includes:

- `Aaron_Sound_Sorter.py`
- `src/`
- `tests/`
- `tools/`
- `commands/`
- `docs/`
- `AI_READ_THIS_FIRST.md`
- requirements/config files
- locked smoke acceptance JSON and small fixtures
- active root brain JSON files, such as:
  - `stage4_folder_brain.json`
  - `stage4_folder_brain_core_baby.json`
  - `stage4_folder_brain_spread_baby.json`
  - `stage4_folder_brain_outlier_baby.json`
  - `stage4_folder_brain_baby.json`
  - `v31_brain.json`

Before staging the bundle, this script runs:

```bash
make clean-for-bundle
```

This removes generated reports, old run folders, cache files, root upload-back ZIPs, and other junk that should not confuse the next AI.

To skip cleanup:

```bash
SKIP_CLEAN=1 ./commands/bundle/bundle.sh
```

## Code-only mode

Run either:

```bash
CODE_ONLY=1 ./commands/bundle/bundle.sh
```

or:

```bash
./commands/bundle/bundle_code_only.sh
```

This excludes root brain JSONs.

## Acceptance sample audio

By default, small locked acceptance samples under:

```text
tests/acceptance/locked_smoke_v1/samples/
```

are included because they are QA fixtures, not the full sample library.

To exclude them:

```bash
INCLUDE_ACCEPTANCE_AUDIO=0 ./commands/bundle/bundle.sh
```

## Output location

Bundles are written to:

```text
_reports/bundles/
```

Override:

```bash
BUNDLE_OUTPUT_DIR=/path/to/output ./commands/bundle/bundle.sh
```

## Do not put bundle ZIPs in the project root

Bundle ZIP files belong in `_reports/bundles/`. Root-level ZIP files are treated as generated junk and removed by `make clean`.
