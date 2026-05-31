# Master Project Goals — Aaron Sound Sorter

## Product goal

Build a practical sound librarian for music production.

The final user should be able to give the sorter a ZIP file or a folder of samples and get back a clean, producer-friendly library with useful review folders and clear diagnostics.

The final product is not supposed to pretend to be perfect AI. It should sort obvious files, explain uncertain evidence, and send conflicts to review instead of faking certainty.

## Top-level final sorter folders

The final production sorter should stay simple:

```text
Drums
Instruments
Textures
FX
_TO_REVIEW
```

## Current phase

This bundle is **Phase 4: Production sorter**.

Phase 4 is the final user workflow: give the sorter a ZIP file or folder of samples, load the saved folder-supervised brain, place obvious files, and route uncertain or conflicted files to review with enough manifest evidence to debug mistakes.

## Architecture rule for future AI helpers

Take a step back before changing code.

Do not make a tiny patch that wins one file but damages the system goal. Every change should preserve the larger architecture:

1. Measured audio evidence first.
2. Structure/role before specific identity.
3. Character/flavor separate from source identity.
4. JSON ledger preserved as a broad evidence database.
5. Python interpretation layer handles phase-specific reporting policy.
6. Developer-only tests are allowed, but user-facing workflow must stay simple.

## JSON ledger policy

The master JSON ledger is a major project asset. Treat it as the evidence rulebook.

Do not mutate it just because a tone-card report overcalls a label. First decide whether the problem is:

- measurement extraction
- feature normalization
- role interpretation
- report wording
- actual ledger category logic

For the v7.7 to v7.9 problem, the fix belongs in Python because long-loop over-specific labels are a report interpretation problem, not a reason to weaken the ledger.

## Future phases

### Phase 1: Sound Atlas Research

Current bundle. Describe the library honestly.

### Phase 2: Curation and training candidate selection

Use Phase 1 reports to identify clean one-shots, clean loops, outliers, and reliable training examples.

### Phase 3: Brain training and validation

Build or improve the reference brain using curated examples and regression tests.

### Phase 4: Production sorter

Simple user workflow:

```bash
python3 Aaron_Sound_Sorter.py "/path/to/samples.zip" "/path/to/output_folder"
```

### Phase 5: Sound Atlas search and flavor browser

Use the same evidence to search by flavor:

- dark warm loops
- bright dry claps
- sub-heavy mono-safe hits
- wide evolving FX
- breathy vocal fragments
- gritty guitar loops
