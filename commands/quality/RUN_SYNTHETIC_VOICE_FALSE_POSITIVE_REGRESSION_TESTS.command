#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PY=".venv_phase4/bin/python"
if [ ! -x "$PY" ]; then
  PY="python3"
fi

echo "Running synthetic voice false-positive regression panel..."
"$PY" -m pytest -q \
  tests/test_parent_eligibility_synthetic_voice_false_positive_surrogates.py \
  tests/test_parent_eligibility_fact_roles_v27.py \
  tests/test_decision_core_v2.py \
  tests/test_two_voter_redesign.py \
  tests/test_phase4_shape_voter.py \
  tests/test_phase4_v060_physical_family_guard_regressions.py \
  tests/test_phase4_committee_physics_gap_locks.py \
  tests/test_phase4_voice_guard.py \
  tests/test_phase4_v066_loop_family_policy_regressions.py

echo
echo "PASS: synthetic voice false-positive regression panel completed."
