# Aaron Sound Sorter v0.4.61 Fast Folder Brain

## What this bundle fixes

The v0.4.60 folder-brain training run was not hanging on the last audio file. It was burning CPU in slow developer-only post-training diagnostics after the training scan finished.

v0.4.61 changes the normal `train-brain` workflow so it builds the reusable folder brain without running those expensive diagnostics by default.

## Run this first

```bash
./RUN_PHASE4_v0461_FAST_TRAIN_SORT_REPORTS.command
```

That command builds a persistent `stage4_folder_brain.json` at the project root, sorts `FX_Aaron2.zip` with that brain, and creates a logs/reports upload ZIP.

## Optional slow mode

Only use this when you specifically want the heavy developer reports:

```bash
python3 Aaron_Sound_Sorter.py train-brain "/Volumes/T9/music_production/samples/Sorted samples" --full-diagnostics
```

## Current architecture

Folder-supervised brain remains the target:

- training folder path is the label
- no filename routing
- no Sax-specific or category-specific boost
- support balancing uses only folder training counts
- tiny folders still need stronger evidence before auto-place
