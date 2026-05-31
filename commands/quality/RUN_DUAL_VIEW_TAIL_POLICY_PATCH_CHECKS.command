#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

echo "Running no-source-name production sorting audit..."
./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command

echo "Running dual-view tail/body policy regression checks..."
python3 -m pytest -q \
  tests/test_direct_body_dual_view_regressions.py \
  tests/test_direct_body_voice_and_low_drum_loop_policy.py \
  tests/test_decision_core_kick_demote_regression.py \
  tests/test_consensus_concrete_fx_gate_regressions.py \
  tests/test_decision_core_fx_aaron2_pattern_regressions.py \
  tests/test_decision_core_fx_smoke_matrix_hard_failures.py \
  tests/test_decision_core_fx_smoke_remaining_review_rows.py \
  tests/test_decision_core_golden_failure_regressions.py \
  tests/test_decision_core_v2.py \
  tests/test_two_voter_redesign.py \
  tests/test_no_source_name_sorting_invariant.py

echo "PASS: dual-view tail/body policy patch checks completed."
