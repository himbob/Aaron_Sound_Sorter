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

echo
if [ -x "$PROJECT_ROOT/commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command" ]; then
  echo "Running no-source-name audit first..."
  "$PROJECT_ROOT/commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command"
fi

echo
echo "Running FX smoke stability hotfix regression..."
"$PYTHON_BIN" -m pytest -q tests/test_decision_core_fx_smoke_no_review_guard.py
