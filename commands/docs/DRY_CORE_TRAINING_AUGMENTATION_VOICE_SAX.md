# Experimental dry-core training augmentation for Voice and Saxophone

This patch adds an opt-in training augmentation path for one experiment only:
trusted Voice/Vocal and Saxophone training folders can train with both the normal
full-audio fingerprint and a harmonic-core dry-view fingerprint.

## What it does

When enabled, each readable training row whose trusted folder label is Voice/Vocal
or Saxophone gets one additional in-memory training row:

- same folder-truth label
- same source file
- same training tree
- full audio file remains unchanged
- added fingerprint is computed from the existing harmonic-core analysis view
- eval rows are not augmented
- FX Human/Voice FX labels are not included

This is not filename routing. The augmentation decision uses the trusted training
folder path and label only.

## Why this exists

Wet sax and wet voice examples can smear source identity into generic synth,
voice, keys, or broad instrument buckets. This lets a rebuilt brain see both:

1. the real wet sample
2. the reduced-tail harmonic/core identity view

without destructively editing any training WAVs.

## How to enable it for the brain-family command

```bash
cd /path/to/Aaron_Sound_Sorter

PATH="/path/to/Aaron_Sound_Sorter/.venv_phase4/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
TRAINING_ROOT="/path/to/sample-library/Sorted samples" \
CORE_ANCHORS=4 \
SPREAD_ANCHORS=4 \
OUTLIER_ANCHORS=4 \
FULL_MAX_CENTROIDS=10 \
BABY_MAX_CENTROIDS=4 \
MAX_FILES_PER_LABEL_TO_SCAN=0 \
FINGERPRINT_TIMEOUT_SEC=45 \
DRY_CORE_AUGMENT_VOICE_SAX=1 \
DRY_CORE_AUGMENT_TIMEOUT_SEC=45 \
/bin/bash commands/legacy_root_commands/RUN_TRAIN_BRAIN_FAMILY.command
```

## How to enable it directly

```bash
python3 Aaron_Sound_Sorter.py train-brain-family \
  "/path/to/sample-library/Sorted samples" \
  --augment-dry-core-voice-sax
```

## How to verify it ran

Open the latest training reports and inspect `train_feature_manifest.csv`.
There should be rows where:

```text
training_view = dry_core_voice
training_view = dry_core_sax
structure_remap_reason contains dry_core_training_augmentation
```

## Do not assume this is permanent

This is an experiment. If sax and voice improve but cymbals, hats, bells, or FX
get worse, disable it and rebuild without `DRY_CORE_AUGMENT_VOICE_SAX=1`.
