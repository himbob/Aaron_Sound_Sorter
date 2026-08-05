#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/path/to/Aaron_Sound_Sorter"
TRAINING_ROOT="${TRAINING_ROOT:-/path/to/sample-library/Sorted samples}"

cd "$PROJECT_ROOT"

echo "Training full/core/spread/outlier brain family"
echo "Training root: $TRAINING_ROOT"

EXTRA_TRAIN_ARGS=()
if [ "${DRY_CORE_AUGMENT_VOICE_SAX:-0}" = "1" ]; then
  echo "Dry-core augmentation: ENABLED for trusted Voice/Vocal and Saxophone training folders only"
  EXTRA_TRAIN_ARGS+=(--augment-dry-core-voice-sax)
  EXTRA_TRAIN_ARGS+=(--dry-core-augment-timeout-sec "${DRY_CORE_AUGMENT_TIMEOUT_SEC:-45}")
else
  echo "Dry-core augmentation: off"
fi
echo

PYTHONPATH=src python3 Aaron_Sound_Sorter.py train-brain-family "$TRAINING_ROOT" \
  --project-dir "$PROJECT_ROOT" \
  --save-full "$PROJECT_ROOT/stage4_folder_brain.json" \
  --save-core-baby "$PROJECT_ROOT/stage4_folder_brain_core_baby.json" \
  --save-spread-baby "$PROJECT_ROOT/stage4_folder_brain_spread_baby.json" \
  --save-outlier-baby "$PROJECT_ROOT/stage4_folder_brain_outlier_baby.json" \
  --core-anchors "${CORE_ANCHORS:-3}" \
  --spread-anchors "${SPREAD_ANCHORS:-3}" \
  --outlier-anchors "${OUTLIER_ANCHORS:-3}" \
  --full-max-centroids "${FULL_MAX_CENTROIDS:-6}" \
  --baby-max-centroids "${BABY_MAX_CENTROIDS:-3}" \
  --max-files-per-label-to-scan "${MAX_FILES_PER_LABEL_TO_SCAN:-0}" \
  --fingerprint-timeout-sec "${FINGERPRINT_TIMEOUT_SEC:-45}" \
  --training-preview-per-label 3 \
  --min-active-train-per-label 1 \
  "${EXTRA_TRAIN_ARGS[@]}"

echo
echo "DONE. Brain files:"
ls -lh stage4_folder_brain.json stage4_folder_brain_core_baby.json stage4_folder_brain_spread_baby.json stage4_folder_brain_outlier_baby.json

open "$PROJECT_ROOT/stage4_brain_family_training" 2>/dev/null || true
