#!/usr/bin/env bash
# RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command
#
# AI-safe locked-smoke acceptance runner.
# Runs each protected acceptance case one at a time with heartbeat output so
# quiet audio analysis does not look hung in constrained AI sandboxes.
#
# Intended location inside project:
#   commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command
#
# Common use:
#   ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command
#
# One case:
#   ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command --case-id clap_clean_clap2
#
# List cases:
#   ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command --list-cases

set -u

main() {
  PROJECT_ROOT="${PROJECT_ROOT:-}"
  PYTHON_BIN="${PYTHON_BIN:-}"
  HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-10}"
  STOP_ON_FAIL=0
  LIST_CASES=0
  ONLY_CASE_ID=""

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --case-id)
        shift
        ONLY_CASE_ID="${1:-}"
        ;;
      --list-cases)
        LIST_CASES=1
        ;;
      --stop-on-fail)
        STOP_ON_FAIL=1
        ;;
      --heartbeat-seconds)
        shift
        HEARTBEAT_SECONDS="${1:-10}"
        ;;
      --help|-h)
        print_help
        return 0
        ;;
      *)
        echo "ERROR: Unknown argument: $1"
        print_help
        return 2
        ;;
    esac
    shift || true
  done

  if [ -z "$PROJECT_ROOT" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
  fi

  if [ -z "$PYTHON_BIN" ]; then
    if [ -x "$PROJECT_ROOT/.venv_phase4/bin/python" ]; then
      PYTHON_BIN="$PROJECT_ROOT/.venv_phase4/bin/python"
    else
      PYTHON_BIN="$(command -v python3)"
    fi
  fi

  ACCEPTANCE_CMD="$PROJECT_ROOT/commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command"
  EXPECTED_JSON="$PROJECT_ROOT/tests/acceptance/locked_smoke_v1/expected_results.json"

  if [ ! -d "$PROJECT_ROOT" ]; then
    echo "ERROR: PROJECT_ROOT not found: $PROJECT_ROOT"
    return 1
  fi

  if [ ! -x "$ACCEPTANCE_CMD" ]; then
    echo "ERROR: AI-safe acceptance command not executable: $ACCEPTANCE_CMD"
    echo "Try: chmod +x \"$ACCEPTANCE_CMD\""
    return 1
  fi

  if [ ! -f "$EXPECTED_JSON" ]; then
    echo "ERROR: Expected results JSON not found: $EXPECTED_JSON"
    return 1
  fi

  if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN"
    return 1
  fi

  STAMP="$(date +%Y%m%d_%H%M%S)"
  REPORT_DIR="$PROJECT_ROOT/_reports/locked_smoke_one_by_one/run_$STAMP"
  mkdir -p "$REPORT_DIR"

  CASE_LIST="$REPORT_DIR/case_ids.txt"
  RESULTS_TSV="$REPORT_DIR/case_results.tsv"

  "$PYTHON_BIN" - <<'PY' "$EXPECTED_JSON" "$CASE_LIST"
import json
import sys
from pathlib import Path

expected_json = Path(sys.argv[1])
out_path = Path(sys.argv[2])
data = json.loads(expected_json.read_text(encoding="utf-8"))

if isinstance(data, dict) and isinstance(data.get("cases"), list):
    cases = data["cases"]
elif isinstance(data, list):
    cases = data
elif isinstance(data, dict):
    cases = list(data.values())
else:
    raise SystemExit("Unknown expected_results.json format")

ids = []
for case in cases:
    if not isinstance(case, dict):
        continue
    case_id = case.get("case_id") or case.get("id") or case.get("name")
    if case_id:
        ids.append(str(case_id))

if not ids:
    raise SystemExit("No case ids found in expected_results.json")

out_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
PY

  if [ "$LIST_CASES" = "1" ]; then
    cat "$CASE_LIST"
    return 0
  fi

  if [ -n "$ONLY_CASE_ID" ]; then
    printf '%s\n' "$ONLY_CASE_ID" > "$CASE_LIST"
  fi

  {
    echo -e "case_id\tstatus\tseconds\tlog_file"
  } > "$RESULTS_TSV"

  echo "Project root: $PROJECT_ROOT"
  echo "Python:       $PYTHON_BIN"
  echo "Runner:       $ACCEPTANCE_CMD"
  echo "Expected:     $EXPECTED_JSON"
  echo "Report dir:   $REPORT_DIR"
  echo "Heartbeat:    ${HEARTBEAT_SECONDS}s"
  echo ""

  TOTAL=0
  PASSED=0
  FAILED=0

  while IFS= read -r CASE_ID; do
    [ -z "$CASE_ID" ] && continue
    TOTAL=$((TOTAL + 1))

    SAFE_CASE_NAME="$(printf '%s' "$CASE_ID" | tr -c 'A-Za-z0-9._-' '_')"
    LOG_FILE="$REPORT_DIR/${SAFE_CASE_NAME}.log"

    echo "============================================================"
    echo "RUNNING CASE $TOTAL: $CASE_ID"
    echo "============================================================"

    START_TS="$(date +%s)"

    (
      cd "$PROJECT_ROOT" || return 1
      PROJECT_ROOT="$PROJECT_ROOT" PYTHON_BIN="$PYTHON_BIN" \
        "$ACCEPTANCE_CMD" --case-id "$CASE_ID"
    ) > "$LOG_FILE" 2>&1 &

    PID=$!

    while kill -0 "$PID" 2>/dev/null; do
      echo "HEARTBEAT: $CASE_ID still running at $(date)"
      sleep "$HEARTBEAT_SECONDS"
    done

    wait "$PID"
    STATUS=$?
    END_TS="$(date +%s)"
    SECONDS=$((END_TS - START_TS))

    if [ "$STATUS" = "0" ]; then
      PASSED=$((PASSED + 1))
      echo "PASS: $CASE_ID (${SECONDS}s)"
      printf '%s\tPASS\t%s\t%s\n' "$CASE_ID" "$SECONDS" "$LOG_FILE" >> "$RESULTS_TSV"
    else
      FAILED=$((FAILED + 1))
      echo "FAIL: $CASE_ID (${SECONDS}s, status $STATUS)"
      echo "Log: $LOG_FILE"
      tail -80 "$LOG_FILE" || true
      printf '%s\tFAIL_%s\t%s\t%s\n' "$CASE_ID" "$STATUS" "$SECONDS" "$LOG_FILE" >> "$RESULTS_TSV"
      if [ "$STOP_ON_FAIL" = "1" ]; then
        break
      fi
    fi

    echo "HEARTBEAT: completed $CASE_ID at $(date)"
    echo ""
  done < "$CASE_LIST"

  echo "============================================================"
  echo "ONE-BY-ONE ACCEPTANCE SUMMARY"
  echo "============================================================"
  echo "Total:  $TOTAL"
  echo "Passed: $PASSED"
  echo "Failed: $FAILED"
  echo "Report: $REPORT_DIR"
  echo "Table:  $RESULTS_TSV"

  if [ "$FAILED" = "0" ]; then
    return 0
  fi
  return 1
}

print_help() {
  cat <<'HELP'
AI-safe one-by-one locked smoke acceptance runner.

Usage:
  ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command
  ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command --case-id CASE_ID
  ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command --list-cases

Options:
  --case-id CASE_ID          Run only one case.
  --list-cases               Print case ids from expected_results.json.
  --stop-on-fail             Stop after first failing case.
  --heartbeat-seconds N      Print heartbeat every N seconds. Default: 10.

Environment:
  PROJECT_ROOT               Optional project root override.
  PYTHON_BIN                 Optional Python executable override.
  HEARTBEAT_SECONDS          Optional heartbeat interval override.
HELP
}

main "$@"
