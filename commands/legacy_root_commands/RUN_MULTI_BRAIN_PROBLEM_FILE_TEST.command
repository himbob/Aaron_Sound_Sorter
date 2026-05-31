#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/Volumes/T9/testbed/Aaron_Sound_Sorter"
INPUT_PATH="${1:-}"
if [ -z "$INPUT_PATH" ]; then
  echo "Usage: ./RUN_MULTI_BRAIN_PROBLEM_FILE_TEST.command /path/to/problem.wav"
  echo
  echo "Example:"
  echo "  ./RUN_MULTI_BRAIN_PROBLEM_FILE_TEST.command /tmp/AA_JBL_74bpm_Am_Sax_Loop_18.wav"
  exit 1
fi

cd "$PROJECT_ROOT"
RUN_ROOT="/tmp/aaron_multi_baby_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"

PYTHONPATH=src python3 Aaron_Sound_Sorter.py sort "$INPUT_PATH" "$RUN_ROOT" \
  --brain "$PROJECT_ROOT/stage4_folder_brain.json" \
  --core-baby-brain "$PROJECT_ROOT/stage4_folder_brain_core_baby.json" \
  --spread-baby-brain "$PROJECT_ROOT/stage4_folder_brain_spread_baby.json" \
  --outlier-baby-brain "$PROJECT_ROOT/stage4_folder_brain_outlier_baby.json" \
  --no-zip

echo
echo "Manifest brain columns:"
head -1 "$RUN_ROOT/Aaron_Sorted_Sounds_manifest.csv" | tr ',' '\n' | grep -E 'brain|baby|physics|shape' || true

echo
echo "Manifest row:"
cat "$RUN_ROOT/Aaron_Sorted_Sounds_manifest.csv"

echo
echo "Output: $RUN_ROOT"
open "$RUN_ROOT" 2>/dev/null || true
