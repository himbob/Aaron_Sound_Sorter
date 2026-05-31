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

echo "Compiling changed files..."
"$PYTHON_BIN" -m py_compile \
  src/aaron_sound_sorter/engine/decision_core_v2.py \
  src/aaron_sound_sorter/engine/eligibility.py \
  tools/run_random_sample_folder_log_panel.py

echo "Running second conflict resolver regression panel..."
"$PYTHON_BIN" -m pytest -q \
  tests/test_decision_core_second_conflict_resolver_tdd.py \
  tests/test_decision_core_conflict_resolver_tdd.py \
  tests/test_decision_core_v2.py \
  tests/test_parent_eligibility_synthetic_mixed_loop_over_narrowing.py \
  tests/test_parent_eligibility_synthetic_bass_alert_overreach_surrogates.py \
  tests/test_parent_eligibility_synthetic_voice_false_positive_surrogates.py \
  tests/test_parent_eligibility_fact_roles_v27.py \
  tests/test_phase4_v060_physical_family_guard_regressions.py \
  tests/test_phase4_voice_guard.py \
  tests/test_phase4_shape_voter.py \
  tests/test_phase4_committee_physics_gap_locks.py \
  tests/test_two_voter_redesign.py
