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
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
echo "Running FX smoke no-review resolver regression tests..."
"$PYTHON_BIN" -m py_compile src/aaron_sound_sorter/engine/decision_core_v2.py
"$PYTHON_BIN" -m pytest -q \
  tests/test_decision_core_v2.py \
  tests/test_decision_core_fx_smoke_no_review_guard.py \
  tests/test_decision_core_fx_smoke_no_review_v2.py \
  tests/test_decision_core_ground_truth_samples_tdd.py \
  tests/test_decision_core_ground_truth_true_bucket_tdd.py
