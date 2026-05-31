#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$PROJECT_ROOT/.venv_phase4/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv_phase4/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$PROJECT_ROOT/_reports/v31108_uploaded_samples_one_by_one_$STAMP"
RESULTS_CSV="$RUN_ROOT/results.csv"
FAILURES_CSV="$RUN_ROOT/failures.csv"
mkdir -p "$RUN_ROOT"

find "$PROJECT_ROOT" -name '._*' -type f -delete
find "$PROJECT_ROOT" -name '.DS_Store' -type f -delete

SAMPLES=(
  "tests/regression_audio/01_WCS_No_Safety_BPM92_D#min__Bells.wav"
  "tests/regression_audio/belize87bpm_8bars_UNKWN (Bbm).wav"
  "tests/regression_audio/AV5_5_94bpm_Hit 2.wav"
)

printf 'index,source,neutral_file,final_label,status,consensus_status,case_dir\n' > "$RESULTS_CSV"
printf 'index,source,neutral_file,final_label,status,consensus_status,case_dir\n' > "$FAILURES_CSV"
INDEX=0
for REL in "${SAMPLES[@]}"; do
  INDEX=$((INDEX + 1))
  SRC="$PROJECT_ROOT/$REL"
  if [ ! -f "$SRC" ]; then
    echo "Missing sample: $REL" >&2
    exit 2
  fi
  CASE_DIR="$RUN_ROOT/case_$(printf '%04d' "$INDEX")"
  INPUT_DIR="$CASE_DIR/input"
  OUT_DIR="$CASE_DIR/output"
  LOG_FILE="$CASE_DIR/sorter.log"
  mkdir -p "$INPUT_DIR" "$OUT_DIR"
  EXT=".${SRC##*.}"
  NEUTRAL="sample_$(printf '%04d' "$INDEX")$EXT"
  cp "$SRC" "$INPUT_DIR/$NEUTRAL"
  echo "[$INDEX/3] $REL -> $NEUTRAL"
  "$PYTHON_BIN" Aaron_Sound_Sorter.py "$INPUT_DIR" "$OUT_DIR" --no-zip --workers 1 > "$LOG_FILE" 2>&1
  MANIFEST="$OUT_DIR/Aaron_Sorted_Sounds_manifest.csv"
  ROW="$($PYTHON_BIN - "$MANIFEST" <<'PY'
import csv
import sys
from pathlib import Path
p = Path(sys.argv[1])
with p.open(newline='', encoding='utf-8', errors='replace') as f:
    row = next(csv.DictReader(f), {})
label = row.get('final_label') or row.get('folder_path') or row.get('new_relative_path') or ''
status = row.get('consensus_status') or row.get('confidence_status') or ''
print(label.replace('\n', ' ') + '\t' + status.replace('\n', ' '))
PY
)"
  FINAL_LABEL="${ROW%%$'\t'*}"
  CONSENSUS_STATUS="${ROW#*$'\t'}"
  LOW_LABEL="$(printf '%s' "$FINAL_LABEL" | tr '[:upper:]' '[:lower:]')"
  STATUS="PASS"
  if [[ "$LOW_LABEL" == *voice* || "$LOW_LABEL" == *vocal* ]]; then
    STATUS="FAIL_VOICE_FALSE_POSITIVE"
  fi
  if [[ "$REL" == *"Hit 2"* && "$LOW_LABEL" == _to_review/measured\ role\ conflict* ]]; then
    STATUS="FAIL_MEASURED_ROLE_CONFLICT"
  fi
  printf '%s,%q,%q,%q,%q,%q,%q\n' "$INDEX" "$REL" "$NEUTRAL" "$FINAL_LABEL" "$STATUS" "$CONSENSUS_STATUS" "$CASE_DIR" >> "$RESULTS_CSV"
  if [ "$STATUS" != "PASS" ]; then
    printf '%s,%q,%q,%q,%q,%q,%q\n' "$INDEX" "$REL" "$NEUTRAL" "$FINAL_LABEL" "$STATUS" "$CONSENSUS_STATUS" "$CASE_DIR" >> "$FAILURES_CSV"
  fi
  echo "$STATUS -> $FINAL_LABEL"
done

FAIL_COUNT="$(( $(wc -l < "$FAILURES_CSV" | tr -d ' ') - 1 ))"
echo "Done. Failures: $FAIL_COUNT"
echo "Results: $RESULTS_CSV"
echo "Failures: $FAILURES_CSV"
open "$RUN_ROOT" 2>/dev/null || true
if [ "$FAIL_COUNT" -ne 0 ]; then
  exit 1
fi
