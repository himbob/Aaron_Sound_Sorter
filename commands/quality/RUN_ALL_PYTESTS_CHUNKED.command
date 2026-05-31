#!/bin/bash
set -euo pipefail

# Run every pytest file one at a time. macOS Bash 3.2 safe.
# This command is the required full-regression gate before any future bundle.

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
REPORT_DIR="$PROJECT_ROOT/reports/pytest_chunked/run_$STAMP"
mkdir -p "$REPORT_DIR"
SUMMARY="$REPORT_DIR/pytest_chunked_summary.txt"
FAILED_LIST="$REPORT_DIR/pytest_failed_files.txt"
PASSED_LIST="$REPORT_DIR/pytest_passed_files.txt"
SKIPPED_LIST="$REPORT_DIR/pytest_no_tests_or_skipped_files.txt"
: > "$SUMMARY"; : > "$FAILED_LIST"; : > "$PASSED_LIST"; : > "$SKIPPED_LIST"

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:${PYTHONPATH:-}"

echo "Aaron Sound Sorter chunked pytest run" | tee -a "$SUMMARY"
echo "Project: $PROJECT_ROOT" | tee -a "$SUMMARY"
echo "Python:  $PYTHON_BIN" | tee -a "$SUMMARY"
echo "Report:  $REPORT_DIR" | tee -a "$SUMMARY"
echo | tee -a "$SUMMARY"

TEST_LIST="$REPORT_DIR/test_file_list.txt"
find "$PROJECT_ROOT/tests" -type f -name 'test_*.py' ! -name '._*' ! -path '*/__pycache__/*' | sort > "$TEST_LIST"
TOTAL=0; PASSED=0; FAILED=0; NO_TESTS_OR_SKIPPED=0

while IFS= read -r TEST_FILE; do
  [ -n "$TEST_FILE" ] || continue
  TOTAL=$((TOTAL + 1))
  REL="${TEST_FILE#$PROJECT_ROOT/}"
  SAFE_NAME="$(printf '%s' "$REL" | sed 's#[^A-Za-z0-9._-]#_#g')"
  LOG_FILE="$REPORT_DIR/${SAFE_NAME}.log"
  echo "[$TOTAL] pytest $REL" | tee -a "$SUMMARY"
  set +e
  "$PYTHON_BIN" -m pytest -q "$TEST_FILE" > "$LOG_FILE" 2>&1
  STATUS=$?
  set -e
  if [ "$STATUS" -eq 0 ]; then
    PASSED=$((PASSED + 1)); echo "  PASS" | tee -a "$SUMMARY"; echo "$REL" >> "$PASSED_LIST"
  elif grep -qiE 'no tests ran|collected 0 items' "$LOG_FILE"; then
    NO_TESTS_OR_SKIPPED=$((NO_TESTS_OR_SKIPPED + 1)); echo "  NO TESTS / SKIPPED" | tee -a "$SUMMARY"; echo "$REL" >> "$SKIPPED_LIST"
  else
    FAILED=$((FAILED + 1)); echo "  FAIL: $LOG_FILE" | tee -a "$SUMMARY"; echo "$REL" >> "$FAILED_LIST"; tail -100 "$LOG_FILE" | sed 's/^/    /' | tee -a "$SUMMARY"
  fi
  echo | tee -a "$SUMMARY"
done < "$TEST_LIST"

echo "Final chunked pytest result" | tee -a "$SUMMARY"
echo "Total files: $TOTAL" | tee -a "$SUMMARY"
echo "Passed:      $PASSED" | tee -a "$SUMMARY"
echo "No tests:    $NO_TESTS_OR_SKIPPED" | tee -a "$SUMMARY"
echo "Failed:      $FAILED" | tee -a "$SUMMARY"
echo "Report dir:  $REPORT_DIR" | tee -a "$SUMMARY"

if [ "$FAILED" -ne 0 ]; then
  echo | tee -a "$SUMMARY"
  echo "FAILED TEST FILES:" | tee -a "$SUMMARY"
  cat "$FAILED_LIST" | tee -a "$SUMMARY"
  open "$REPORT_DIR" >/dev/null 2>&1 || true
  exit 1
fi
open "$REPORT_DIR" >/dev/null 2>&1 || true
