#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="_reports/manual_bad_file_panel/run_${STAMP}"
INPUT_DIR="$RUN_DIR/input_bad_files"
OUTPUT_DIR="$RUN_DIR/sort_output"
UPLOAD_DIR="$RUN_DIR/upload_back"
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR" "$UPLOAD_DIR"

find . -name '._*' -type f -delete
find . -name '.DS_Store' -type f -delete

for root in \
  "/path/to/Aaron_Sound_Sorter" \
  "/path/to/sample-library" \
  "$HOME/Downloads"; do
  [ -d "$root" ] || continue
  find "$root" -type f ! -name '._*' \( \
    -name 'clipped_sub_hit.wav' -o \
    -name 'AA_TSS_D_RUMBLER_808_BASS.wav' -o \
    -name 'Clap 17.wav' -o \
    -name 'boom_eval_108640.wav' \
  \) -exec cp -n "{}" "$INPUT_DIR/" \;
done

find "$INPUT_DIR" -name '._*' -type f -delete
find "$INPUT_DIR" -name '.DS_Store' -type f -delete

echo "Input files:"
find "$INPUT_DIR" -maxdepth 1 -type f -print | sort

if [ -x ".venv_phase4/bin/python" ]; then
  PYTHON=".venv_phase4/bin/python"
elif [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python3"
fi

"$PYTHON" Aaron_Sound_Sorter.py sort "$INPUT_DIR" "$OUTPUT_DIR" --no-zip 2>&1 | tee "$RUN_DIR/sort_console.log"

REPORT_ZIP="$UPLOAD_DIR/UPLOAD_BACK_${STAMP}_manual_bad_file_panel.zip"
zip -j "$REPORT_ZIP" \
  "$OUTPUT_DIR"/Aaron_Sorted_Sounds_manifest.csv \
  "$OUTPUT_DIR"/Aaron_Sorted_Sounds_summary.txt \
  "$OUTPUT_DIR"/Aaron_Brain_Ensemble_Audit.csv \
  "$OUTPUT_DIR"/Aaron_Brain_Lane_Validation_Matrix.csv \
  "$OUTPUT_DIR"/Aaron_Brain_Lane_Competence_Summary.csv \
  "$RUN_DIR"/sort_console.log 2>/dev/null || true

open "$RUN_DIR"
echo "Upload this: $REPORT_ZIP"
