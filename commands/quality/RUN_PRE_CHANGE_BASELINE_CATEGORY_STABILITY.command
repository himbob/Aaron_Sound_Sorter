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

"$PYTHON_BIN" "$PROJECT_ROOT/tools/category_stability_gate.py" baseline --project-root "$PROJECT_ROOT" "$@"
STATUS=$?

BASELINE_FILE="$PROJECT_ROOT/_reports/category_stability_gate/latest_baseline_path.txt"
if [ -f "$BASELINE_FILE" ]; then
  BASELINE_DIR="$(cat "$BASELINE_FILE")"
  if [ -d "$BASELINE_DIR" ]; then
    open "$BASELINE_DIR" >/dev/null 2>&1 || true
  fi
fi

exit "$STATUS"
