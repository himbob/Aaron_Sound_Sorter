#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-}"
if [ -z "$PROJECT_ROOT" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$PROJECT_ROOT/.venv_phase4/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv_phase4/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

echo "Project: $PROJECT_ROOT"
echo "Python:  $PYTHON_BIN"

echo "Cleaning macOS metadata..."
find . -name '._*' -type f -delete
find . -name '.DS_Store' -type f -delete

echo "Compiling changed decision code..."
"$PYTHON_BIN" -m py_compile src/aaron_sound_sorter/engine/family_claim_arbiter.py tests/conftest.py

echo "Running decision-core golden failure regressions..."
PYTHONPATH=src "$PYTHON_BIN" -m pytest -q tests/test_decision_core_golden_failure_regressions.py

echo "Running related FX smoke decision-core regressions..."
PYTHONPATH=src "$PYTHON_BIN" -m pytest -q tests/test_decision_core_fx_smoke_matrix_hard_failures.py

echo "V31.139 golden bass-loop decision-core checks completed."
