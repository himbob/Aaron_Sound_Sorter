#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python3 -m compileall -q src tests Aaron_Sound_Sorter.py
./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command
PYTHONPATH=src pytest -q \
  tests/test_fx_bundle_regression_patterns.py \
  tests/test_direct_body_voice_and_low_drum_loop_policy.py \
  tests/test_direct_body_dual_view_regressions.py \
  tests/test_decision_core_kick_demote_regression.py \
  tests/test_consensus_concrete_fx_gate_regressions.py \
  tests/test_parent_eligibility_synthetic_mixed_loop_over_narrowing.py \
  tests/test_decision_core_v2.py
printf '\nPASS: family-claim arbitration patch checks completed.\n'
