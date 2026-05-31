#!/bin/bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/Volumes/T9/testbed/Aaron_Sound_Sorter}"
cd "$PROJECT_ROOT" || exit 2

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$PROJECT_ROOT/.venv_phase4/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv_phase4/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

# AI-safe wrapper: same acceptance harness, but with progress visible and
# optional chunk controls. It does not change sorter behavior.
# Examples:
#   ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command
#   ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --start-index 1 --max-cases 4
#   ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --case-id kick_clean_cs_ne_monroe
"$PYTHON_BIN" "$PROJECT_ROOT/tools/locked_smoke_acceptance.py" run --project-root "$PROJECT_ROOT" "$@"
STATUS=$?

RUN_DIR_FILE="$PROJECT_ROOT/_reports/locked_smoke_acceptance/latest_run_path.txt"
if [ -f "$RUN_DIR_FILE" ]; then
  RUN_DIR="$(cat "$RUN_DIR_FILE")"
  if [ -d "$RUN_DIR" ]; then
    echo "RUN_DIR=$RUN_DIR"
    echo "SUMMARY=$RUN_DIR/summary.txt"
    open "$RUN_DIR" >/dev/null 2>&1 || true
  fi
fi

exit "$STATUS"
