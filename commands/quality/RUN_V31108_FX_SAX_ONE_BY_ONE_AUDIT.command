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

FX_ZIP="${FX_ZIP:-/Volumes/T9/music_production/samples/FX_Aaron2.zip}"
if [ ! -f "$FX_ZIP" ]; then
  echo "Missing FX zip: $FX_ZIP" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$PROJECT_ROOT/_reports/v31108_fx_sax_one_by_one_$STAMP"
EXTRACT_DIR="$RUN_ROOT/fx_zip_extracted"
RESULTS_CSV="$RUN_ROOT/results.csv"
FAILURES_CSV="$RUN_ROOT/failures.csv"
LIST_FILE="$RUN_ROOT/sax_members.txt"
mkdir -p "$EXTRACT_DIR"

find "$PROJECT_ROOT" -name '._*' -type f -delete
find "$PROJECT_ROOT" -name '.DS_Store' -type f -delete

/usr/bin/unzip -q "$FX_ZIP" -d "$EXTRACT_DIR"
find "$EXTRACT_DIR" -name '._*' -type f -delete
find "$EXTRACT_DIR" -name '.DS_Store' -type f -delete

find "$EXTRACT_DIR" -type f \( -iname '*.wav' -o -iname '*.aif' -o -iname '*.aiff' -o -iname '*.flac' -o -iname '*.ogg' -o -iname '*.au' \) \
  | grep -Ei 'sax|saxophone' \
  | sort > "$LIST_FILE"

TOTAL="$(wc -l < "$LIST_FILE" | tr -d ' ')"
echo "Found sax-selected files: $TOTAL"
echo "Run root: $RUN_ROOT"

printf 'index,original_path,neutral_file,final_label,status,consensus_status,case_dir\n' > "$RESULTS_CSV"
printf 'index,original_path,neutral_file,final_label,status,consensus_status,case_dir\n' > "$FAILURES_CSV"

INDEX=0
while IFS= read -r SRC; do
  INDEX=$((INDEX + 1))
  CASE_DIR="$RUN_ROOT/case_$(printf '%04d' "$INDEX")"
  INPUT_DIR="$CASE_DIR/input"
  OUT_DIR="$CASE_DIR/output"
  LOG_FILE="$CASE_DIR/sorter.log"
  mkdir -p "$INPUT_DIR" "$OUT_DIR"
  EXT=".${SRC##*.}"
  NEUTRAL="sample_$(printf '%04d' "$INDEX")$EXT"
  cp "$SRC" "$INPUT_DIR/$NEUTRAL"

  echo "============================================================"
  echo "[$INDEX/$TOTAL] $SRC"
  echo "Neutral: $NEUTRAL"

  "$PYTHON_BIN" Aaron_Sound_Sorter.py "$INPUT_DIR" "$OUT_DIR" --no-zip --workers 1 > "$LOG_FILE" 2>&1

  MANIFEST="$OUT_DIR/Aaron_Sorted_Sounds_manifest.csv"
  STATUS="FAIL_NO_MANIFEST"
  FINAL_LABEL=""
  CONSENSUS_STATUS=""
  if [ -f "$MANIFEST" ]; then
    ROW="$($PYTHON_BIN - "$MANIFEST" <<'PY'
import csv
import sys
from pathlib import Path
p = Path(sys.argv[1])
with p.open(newline='', encoding='utf-8', errors='replace') as f:
    rows = list(csv.DictReader(f))
row = rows[0] if rows else {}
label = row.get('final_label') or row.get('folder_path') or row.get('new_relative_path') or ''
status = row.get('consensus_status') or row.get('confidence_status') or ''
print(label.replace('\n', ' ') + '\t' + status.replace('\n', ' '))
PY
)"
    FINAL_LABEL="${ROW%%$'\t'*}"
    CONSENSUS_STATUS="${ROW#*$'\t'}"
    LOW_LABEL="$(printf '%s' "$FINAL_LABEL" | tr '[:upper:]' '[:lower:]')"
    if [[ "$LOW_LABEL" == *instruments* && ( "$LOW_LABEL" == *sax* || "$LOW_LABEL" == *woodwind* ) ]]; then
      STATUS="PASS"
    else
      STATUS="FAIL_NOT_SAX_OR_WOODWIND"
    fi
  fi

  printf '%s,%q,%q,%q,%q,%q,%q\n' "$INDEX" "$SRC" "$NEUTRAL" "$FINAL_LABEL" "$STATUS" "$CONSENSUS_STATUS" "$CASE_DIR" >> "$RESULTS_CSV"
  if [ "$STATUS" != "PASS" ]; then
    printf '%s,%q,%q,%q,%q,%q,%q\n' "$INDEX" "$SRC" "$NEUTRAL" "$FINAL_LABEL" "$STATUS" "$CONSENSUS_STATUS" "$CASE_DIR" >> "$FAILURES_CSV"
  fi
  echo "$STATUS -> $FINAL_LABEL"
done < "$LIST_FILE"

FAIL_COUNT="$(( $(wc -l < "$FAILURES_CSV" | tr -d ' ') - 1 ))"
echo "============================================================"
echo "Done. Total: $TOTAL  Failures: $FAIL_COUNT"
echo "Results: $RESULTS_CSV"
echo "Failures: $FAILURES_CSV"
open "$RUN_ROOT" 2>/dev/null || true
if [ "$FAIL_COUNT" -ne 0 ]; then
  exit 1
fi
