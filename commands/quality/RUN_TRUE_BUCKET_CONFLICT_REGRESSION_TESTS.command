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

echo "Running no-source-name audit first..."
"$PYTHON_BIN" tools/audit_no_source_name_sorting.py --project-root "$PROJECT_ROOT"

echo
"$PYTHON_BIN" -m py_compile \
  Aaron_Sound_Sorter.py \
  src/aaron_sound_sorter/engine/decision_core_v2.py \
  src/aaron_sound_sorter/engine/eligibility.py \
  src/aaron_sound_sorter/domain/roles.py \
  tools/audit_no_source_name_sorting.py

echo
"$PYTHON_BIN" -m pytest -q \
  tests/test_no_source_name_sorting_invariant.py \
  tests/test_decision_core_ground_truth_true_bucket_tdd.py \
  tests/test_decision_core_second_conflict_resolver_tdd.py \
  tests/test_decision_core_conflict_resolver_tdd.py \
  tests/test_parent_eligibility_synthetic_bass_alert_overreach_surrogates.py \
  tests/test_parent_eligibility_synthetic_mixed_loop_over_narrowing.py \
  tests/test_parent_eligibility_synthetic_voice_false_positive_surrogates.py \
  tests/test_decision_core_v2.py \
  tests/test_phase4_voice_guard.py \
  tests/test_two_voter_redesign.py
