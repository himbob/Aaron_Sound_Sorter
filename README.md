# Aaron Sound Sorter Phase 4

Aaron Sound Sorter is a practical sample-library sorter for music production. Give it a ZIP or folder of samples, and it creates a producer-friendly sorted library with reports for review.

## Phase 4 objective

Phase 4 is the production sorter path. The user-facing goal is simple:

```bash
python3 Aaron_Sound_Sorter.py sort "/path/to/samples.zip" "/path/to/output_folder"
```

The output should contain:

```text
Aaron_Sorted_Sounds/
Aaron_Sorted_Sounds_manifest.csv
Aaron_Sorted_Sounds_summary.txt
Aaron_Sorted_Sounds.zip
```

Top-level production folders stay simple:

```text
Drums
Instruments
Textures
FX
_TO_REVIEW
```

The sorter should place obvious sounds, send uncertain sounds to review, and preserve enough manifest evidence to debug mistakes.

## Current project layout

```text
Aaron_Sound_Sorter.py        Thin runner. Calls the package in src/.
src/aaron_sound_sorter/      Real Stage 4 application code.
stage4_folder_brain.json     Current 100-feature Stage 4 folder brain.
training/                    Locked curated training tree. Mostly symlinks to T9 sample files.
tests/                       Active pytest tests.
tests/fixtures/              Small test fixtures only.
commands/                    Useful shell commands, organized by purpose.
docs/legacy_docs/            Old status/handoff notes kept out of the root.
docs/archive/                Older working notes moved out of the active path.
_archive/                    Local debug/smoke artifacts and macOS sidecars.
```

## Useful commands

Run the active pytest suite:

```bash
./commands/quality/RUN_PYTEST.command
```

Run the built-in Stage 4 self-test:

```bash
./commands/smoke/RUN_SELF_TEST.command
```

Run a small real percussion ZIP sort on your T9 drive:

```bash
./commands/smoke/RUN_REAL_PERCUSSION_SORT.command
```

Rebuild the brain from the locked curated training symlink tree:

```bash
./commands/training/RUN_TRAIN_LOCKED_CURATED.command
```

## Training folder rule

Do not replace the `training/locked_curated_v1` symlinks with copied audio. Those symlinks point back to the real files on `/Volumes/T9/music_production/samples`. This keeps the project small and avoids duplicating sample libraries.

## What was removed from the messy working folder

The cleaned bundle intentionally excludes:

- `.git/`
- `.venv_phase4/`
- `.pytest_cache/`
- `__pycache__/`
- `__MACOSX/` and `.DS_Store`
- old backup ZIPs
- old root command scripts
- `Aaron_Sound_Sorter_Test_Runner.py`
- old duplicate sorter files such as `_old.py`, `_last.py`, and `.bak` files
- broken/obsolete compatibility tests
- one-off debug run folders in the project root
- root-level Phase 4 patch status notes

Testing is now done through pytest and direct CLI smoke commands only.

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

