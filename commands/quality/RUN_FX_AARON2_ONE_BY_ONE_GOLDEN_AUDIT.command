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
FX_ZIP="${FX_ZIP:-}"
LIMIT="${LIMIT:-0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-0}"
ARGS=("$PROJECT_ROOT/tools/run_fx_aaron2_one_by_one_golden_audit.py" --project-root "$PROJECT_ROOT" --limit "$LIMIT" --timeout-sec "$TIMEOUT_SEC")
if [ -n "$FX_ZIP" ]; then
  ARGS+=(--fx-zip "$FX_ZIP")
fi
"$PYTHON_BIN" "${ARGS[@]}"
