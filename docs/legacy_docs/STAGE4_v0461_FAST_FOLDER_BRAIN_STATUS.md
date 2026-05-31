# Stage 4 v0.4.61 Fast Folder Brain Build

## Purpose

This bundle fixes the v0.4.60 test workflow problem where full folder-brain training could appear to hang after the last training files were analyzed.

## Root cause

The process was not stuck on a WAV file. It was burning CPU in developer-only post-training diagnostics, especially O(training rows × labels) label-isolation and training-recall scans. Those reports are useful for deep architecture debugging, but they are too expensive for the normal `train-brain` product workflow.

## Code changes

- `train-brain` now skips expensive developer-only training diagnostics by default.
- `train-brain --full-diagnostics` can still run those slow reports when needed.
- Brain building now prints progress checkpoints after curation, during brain build, during lightweight report writing, and during final brain JSON write.
- The reusable brain is saved to the requested `--save-brain` path.
- The latest brain pointer is written to `PHASE4_LATEST_FOLDER_BRAIN_PATH.txt`.
- The support-balanced folder-brain scoring from v0.4.60 remains active.

## Main test command

Run:

```bash
./RUN_PHASE4_v0461_FAST_TRAIN_SORT_REPORTS.command
```

This command:

1. Finds the training tree.
2. Builds a persistent `stage4_folder_brain.json` at the project root.
3. Sorts `FX_Aaron2.zip` using that persistent brain.
4. Zips only logs and reports for upload.
5. Opens the run folder on macOS.

## Important behavior

This is still folder-supervised. Folder paths from the training tree are labels. The support balancing uses only folder training counts. It does not inspect filenames or hard-code category names.
