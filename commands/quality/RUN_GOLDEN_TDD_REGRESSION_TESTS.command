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

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:${PYTHONPATH:-}"

echo
echo "Running golden TDD regression panel"
echo "Project: $PROJECT_ROOT"
echo "Python:  $PYTHON_BIN"
echo

"$PYTHON_BIN" -m py_compile \
  src/aaron_sound_sorter/engine/decision_core_v2.py \
  tools/audit_fx_smoke_expected_buckets.py

"$PYTHON_BIN" -m pytest -q \
  tests/test_decision_core_golden_failure_regressions.py \
  tests/test_fx_smoke_expected_bucket_audit.py \
  tests/test_decision_core_v2.py

echo
echo "PASS: golden TDD regression panel passed."
