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
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"
STOP_AFTER="${STOP_AFTER:-0}"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:${PYTHONPATH:-}"
"$PYTHON_BIN" "$PROJECT_ROOT/tools/run_pytest_node_by_node.py" --project-root "$PROJECT_ROOT" --timeout-sec "$TIMEOUT_SEC" --stop-after "$STOP_AFTER" "${@:-tests}"
