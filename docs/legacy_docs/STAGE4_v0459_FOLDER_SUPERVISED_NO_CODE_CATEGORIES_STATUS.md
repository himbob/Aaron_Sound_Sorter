# Stage 4 v0.4.59 folder-supervised brain status

## Purpose

Aaron changed the architecture target: the sorter should not use a fixed list of named terminal categories anymore.

The folder path in training data is now the truth label. The brain learns whatever folders Aaron gives it, then sorts future files into those learned folders.

## Main behavior

### Train a brain from a trusted folder tree

```bash
python3 Aaron_Sound_Sorter.py train-brain "/path/to/training_folder" \
  --mode trusted-tree \
  --save-brain stage4_folder_brain.json
```

### Train a brain from an approved sorted output

```bash
python3 Aaron_Sound_Sorter.py train-brain "/path/to/approved/Aaron_Sorted_Sounds" \
  --mode approved-sort \
  --save-brain stage4_folder_brain.json
```

`approved-sort` skips `_TO_REVIEW` through the folder scanner.

### Sort with a saved folder brain

```bash
python3 Aaron_Sound_Sorter.py sort "/path/to/samples.zip" "/path/to/output_folder" \
  --brain stage4_folder_brain.json
```

The shortcut also works:

```bash
python3 Aaron_Sound_Sorter.py "/path/to/samples.zip" "/path/to/output_folder" \
  --brain stage4_folder_brain.json
```

## Architecture changes

- Added `train-brain` command.
- Added saved brain loading for `sort`.
- `sort` no longer rebuilds a training brain every time by default.
- Added final Phase 4 output files at the output root:
  - `Aaron_Sorted_Sounds/`
  - `Aaron_Sorted_Sounds_manifest.csv`
  - `Aaron_Sorted_Sounds_summary.txt`
  - `Aaron_Sorted_Sounds.zip`
- Fixed flat `Instruments/Instrument Loops` so it is learned as a loop folder.
- Replaced the active fixed-label fallback so compatibility calls return the folder-derived label.
- Disabled named physical-role candidate folders. Physics can still send a file to review, but it does not fabricate a new category path.
- The scanner skips `_TO_REVIEW` folders so approved sorted output can be used as training data safely.

## Remaining caution

Some old diagnostic helper names and historical tests still contain words like drums, FX, or loop because those are folder names and report language. The active sorter target is no longer a fixed terminal category table.

## Validation run

```bash
python3 -S -m py_compile Aaron_Sound_Sorter.py src/aaron_stage4/phase3_pure_brain_lab.py
pytest -q
```

Result:

```text
20 passed
```

Tiny local train/sort smoke also passed:

- trained from synthetic folders:
  - `Drums/MyLowHits`
  - `FX/MyBrightBeeps`
- saved `/mnt/data/folder_brain_smoke/brain.json`
- sorted `test_low.wav` into:
  - `Aaron_Sorted_Sounds/Drums/MyLowHits/One Shots/`
