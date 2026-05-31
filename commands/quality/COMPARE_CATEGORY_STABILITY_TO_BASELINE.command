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

"$PYTHON_BIN" "$PROJECT_ROOT/tools/category_stability_gate.py" compare --project-root "$PROJECT_ROOT" "$@"
STATUS=$?

RUN_DIR_FILE="$PROJECT_ROOT/_reports/category_stability_gate/latest_run_path.txt"
if [ -f "$RUN_DIR_FILE" ]; then
  RUN_DIR="$(cat "$RUN_DIR_FILE")"
  if [ -d "$RUN_DIR" ]; then
    open "$RUN_DIR" >/dev/null 2>&1 || true
  fi
fi

exit "$STATUS"
