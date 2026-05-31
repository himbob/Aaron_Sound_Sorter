#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

# Keep generated work out of the project root.
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$ROOT/_reports/brain_lane_big_variety/run_${RUN_ID}"
INPUT_ROOT="$RUN_ROOT/input_good_expected_groups"
OUTPUT_ROOT="$RUN_ROOT/sort_output"
UPLOAD_DIR="$RUN_ROOT/upload_back"
MAX_PER_GROUP="${MAX_PER_GROUP:-80}"

mkdir -p "$RUN_ROOT" "$UPLOAD_DIR"

# Clean macOS metadata before validation and packaging.
find . -name '._*' -type f -delete
find . -name '.DS_Store' -type f -delete

SOURCE_ARGS=()
for SRC in \
  "$ROOT/training/locked_curated_v1" \
  "/Volumes/T9/testbed/Aaron_Sound_Sorter/training/locked_curated_v1" \
  "/Volumes/T9/music_production/samples/Sorted samples"
do
  if [ -d "$SRC" ]; then
    SOURCE_ARGS+=(--source-root "$SRC")
  fi
done

ZIP_ARGS=()
for ZIP in \
  "$ROOT/FX_Aaron2.zip" \
  "/Volumes/T9/music_production/samples/FX_Aaron2.zip" \
  "$ROOT/one_shot_percussive_sounds.zip" \
  "/Volumes/T9/music_production/samples/one_shot_percussive_sounds.zip" \
  "$ROOT/Combined_Training1_2_3_4.zip" \
  "/Volumes/T9/music_production/samples/Combined_Training1_2_3_4.zip"
do
  if [ -f "$ZIP" ]; then
    ZIP_ARGS+=(--zip "$ZIP")
  fi
done

if [ ${#SOURCE_ARGS[@]} -eq 0 ] && [ ${#ZIP_ARGS[@]} -eq 0 ]; then
  echo "No source folders or project ZIPs found."
  echo "Checked training/locked_curated_v1, Sorted samples, FX_Aaron2.zip, one_shot_percussive_sounds.zip, and Combined_Training1_2_3_4.zip."
  exit 2
fi

python3 tools/build_brain_lane_variety_test_input.py \
  --output "$INPUT_ROOT" \
  --manifest "$RUN_ROOT/source_selection_manifest.csv" \
  --max-per-group "$MAX_PER_GROUP" \
  "${SOURCE_ARGS[@]}" \
  "${ZIP_ARGS[@]}" \
  | tee "$RUN_ROOT/build_input.log"

python3 Aaron_Sound_Sorter.py sort \
  "$INPUT_ROOT" \
  "$OUTPUT_ROOT" \
  --brain "$ROOT/stage4_folder_brain.json" \
  --core-baby-brain "$ROOT/stage4_folder_brain_core_baby.json" \
  --spread-baby-brain "$ROOT/stage4_folder_brain_spread_baby.json" \
  --outlier-baby-brain "$ROOT/stage4_folder_brain_outlier_baby.json" \
  --harmonic-core-baby-brain "$ROOT/stage4_folder_brain_harmonic_core_baby.json" \
  --harmonic-spread-baby-brain "$ROOT/stage4_folder_brain_harmonic_spread_baby.json" \
  --harmonic-outlier-baby-brain "$ROOT/stage4_folder_brain_harmonic_outlier_baby.json" \
  --candidate-count 100 \
  --no-zip \
  | tee "$RUN_ROOT/sort.log"

python3 tools/analyze_existing_brain_lane_run.py "$OUTPUT_ROOT" --rebuild-summary | tee "$RUN_ROOT/analyze.log"

ZIP_OUT="$UPLOAD_DIR/UPLOAD_BACK_${RUN_ID}_brain_lane_big_variety_reports.zip"
zip -j "$ZIP_OUT" \
  "$RUN_ROOT/source_selection_manifest.csv" \
  "$RUN_ROOT/build_input.log" \
  "$RUN_ROOT/sort.log" \
  "$RUN_ROOT/analyze.log" \
  "$OUTPUT_ROOT"/Aaron_Brain_Lane_*.csv \
  "$OUTPUT_ROOT"/Aaron_Brain_Lane_Study.md \
  "$OUTPUT_ROOT"/Aaron_Brain_Ensemble_Audit.csv \
  "$OUTPUT_ROOT"/Aaron_Sorted_Sounds_manifest.csv \
  "$OUTPUT_ROOT"/Aaron_Sorted_Sounds_summary.txt

echo ""
echo "DONE"
echo "Run folder: $RUN_ROOT"
echo "Sort output: $OUTPUT_ROOT"
echo "Upload ZIP: $ZIP_OUT"
open "$UPLOAD_DIR"
