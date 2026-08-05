#!/bin/bash
set -u
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR" || exit 1
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="$PROJECT_DIR/reports/v3193_code_simplicity_full_checks_$STAMP"
mkdir -p "$REPORT_DIR"

echo "Cleaning macOS metadata files..."
find . -name '._*' -type f -delete
find . -name '.DS_Store' -type f -delete

echo "Running compileall..."
python3 -S -m compileall -q Aaron_Sound_Sorter.py src tests > "$REPORT_DIR/compileall.log" 2>&1 || {
  echo "FAIL: compileall"
  cat "$REPORT_DIR/compileall.log"
  exit 1
}
echo "PASS: compileall"

echo "Running no-source-name audit..."
python3 tools/audit_no_source_name_sorting.py --project-root . > "$REPORT_DIR/no_source_name_audit.log" 2>&1 || {
  echo "FAIL: no-source-name audit"
  cat "$REPORT_DIR/no_source_name_audit.log"
  exit 1
}
cat "$REPORT_DIR/no_source_name_audit.log"

echo "Running self-test..."
python3 Aaron_Sound_Sorter.py self-test > "$REPORT_DIR/self_test.log" 2>&1 || {
  echo "FAIL: self-test"
  cat "$REPORT_DIR/self_test.log"
  exit 1
}
cat "$REPORT_DIR/self_test.log"

echo "Running pytest files one at a time..."
PYTEST_SUMMARY="$REPORT_DIR/pytest_file_summary.txt"
: > "$PYTEST_SUMMARY"
FAIL_COUNT=0
TEST_COUNT=0
while IFS= read -r TEST_FILE; do
  TEST_COUNT=$((TEST_COUNT + 1))
  TEST_NAME="$(basename "$TEST_FILE" .py)"
  LOG_FILE="$REPORT_DIR/${TEST_NAME}.log"
  echo "=== $TEST_FILE" | tee -a "$PYTEST_SUMMARY"
  if command -v timeout >/dev/null 2>&1; then
    timeout 180s python3 -m pytest -q "$TEST_FILE" > "$LOG_FILE" 2>&1
    RC=$?
  else
    python3 -m pytest -q "$TEST_FILE" > "$LOG_FILE" 2>&1
    RC=$?
  fi
  if [ "$RC" -eq 0 ]; then
    tail -n 2 "$LOG_FILE" | tee -a "$PYTEST_SUMMARY"
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo "FAILED_OR_TIMEOUT rc=$RC" | tee -a "$PYTEST_SUMMARY"
    tail -n 120 "$LOG_FILE" | tee -a "$PYTEST_SUMMARY"
  fi
done < <(find tests -maxdepth 1 -name 'test*.py' -type f | sort)

echo "Pytest files checked: $TEST_COUNT" | tee -a "$PYTEST_SUMMARY"
echo "Pytest failures/timeouts: $FAIL_COUNT" | tee -a "$PYTEST_SUMMARY"
if [ "$FAIL_COUNT" -ne 0 ]; then
  echo "FAIL: one or more pytest files failed or timed out. See $REPORT_DIR"
  exit 1
fi

echo "Running representative FX one-file checks if FX_Aaron2.zip is available..."
FX_ZIP="/path/to/sample-library/FX_Aaron2.zip"
REAL_DIR="$REPORT_DIR/real_fx_samples"
REAL_OUT="$REPORT_DIR/real_fx_outputs"
mkdir -p "$REAL_DIR" "$REAL_OUT"
if [ -f "$FX_ZIP" ]; then
  unzip -q -j "$FX_ZIP" 'FX_Aaron2/Loop/Brass_Woodwind/AA_JBL_74bpm_Am_Sax_Loop_13.wav' -d "$REAL_DIR" || true
  unzip -q -j "$FX_ZIP" 'FX_Aaron2/Premium Bounce/Loop/Drums/Drumloop_hats_Dark_Rap_140BPM.wav' -d "$REAL_DIR" || true
  unzip -q -j "$FX_ZIP" 'FX_Aaron2/Premium Bounce/Loop/Vocals/Halfway_Vocal_keyC#m_82BPM.wav' -d "$REAL_DIR" || true
  unzip -q -j "$FX_ZIP" 'FX_Aaron2/Premium Bounce/One_Shot/Drums/CS_NE_Kick_OneShot_Monroe.wav' -d "$REAL_DIR" || true
  for WAV_FILE in "$REAL_DIR"/*.wav; do
    [ -f "$WAV_FILE" ] || continue
    BASE="$(basename "$WAV_FILE" .wav)"
    OUT_DIR="$REAL_OUT/$BASE"
    echo "Sorting $BASE" | tee -a "$REPORT_DIR/real_fx_summary.txt"
    if command -v timeout >/dev/null 2>&1; then
      timeout 180s python3 Aaron_Sound_Sorter.py sort "$WAV_FILE" "$OUT_DIR" --brain stage4_folder_brain.json --no-zip > "$OUT_DIR.log" 2>&1
      RC=$?
    else
      python3 Aaron_Sound_Sorter.py sort "$WAV_FILE" "$OUT_DIR" --brain stage4_folder_brain.json --no-zip > "$OUT_DIR.log" 2>&1
      RC=$?
    fi
    if [ "$RC" -ne 0 ]; then
      echo "FAILED $BASE rc=$RC" | tee -a "$REPORT_DIR/real_fx_summary.txt"
      tail -n 80 "$OUT_DIR.log" | tee -a "$REPORT_DIR/real_fx_summary.txt"
    fi
  done
  python3 - <<PY > "$REPORT_DIR/real_fx_manifest_summary.txt"
import csv
import glob
import pathlib
import sys
csv.field_size_limit(sys.maxsize)
for manifest in sorted(glob.glob(r"$REAL_OUT/*/Aaron_Sorted_Sounds_manifest.csv")):
    with open(manifest, newline="", encoding="utf-8", errors="replace") as handle:
        row = next(csv.DictReader(handle))
    print(f"{pathlib.Path(manifest).parent.name} => {row.get('folder_path')} | {row.get('consensus_status')}")
PY
  cat "$REPORT_DIR/real_fx_manifest_summary.txt"
else
  echo "SKIP: $FX_ZIP not found" | tee -a "$REPORT_DIR/real_fx_summary.txt"
fi

find "$REPORT_DIR" -name '._*' -type f -delete
find "$REPORT_DIR" -name '.DS_Store' -type f -delete
(cd "$(dirname "$REPORT_DIR")" && zip -qr "$(basename "$REPORT_DIR").zip" "$(basename "$REPORT_DIR")")

echo "PASS: v31.93 code simplicity full checks"
echo "Report folder: $REPORT_DIR"
echo "Report ZIP: $REPORT_DIR.zip"
open "$REPORT_DIR" 2>/dev/null || true
