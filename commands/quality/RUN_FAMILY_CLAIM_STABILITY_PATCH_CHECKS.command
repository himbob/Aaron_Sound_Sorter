#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:${PYTHONPATH:-}"

echo "Running compile check..."
"$PYTHON_BIN" -m compileall -q src tests tools

echo "Running no-source-name production sorting audit..."
./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command

echo "Running family-claim stability regression checks..."
"$PYTHON_BIN" -m pytest -q \
  tests/test_family_claim_arbitration_matrix.py \
  tests/test_decision_core_second_conflict_resolver_tdd.py \
  tests/test_parent_eligibility_fx_instrument_steal_guard.py \
  tests/test_parent_eligibility_synthetic_bass_alert_overreach_surrogates.py \
  tests/test_parent_eligibility_v25_transition_reverb.py \
  tests/test_parent_eligibility_v26_fx_smoke_reverb.py \
  tests/test_uploaded_regression_audio.py \
  tests/test_fx_bundle_regression_patterns.py \
  tests/test_direct_body_voice_and_low_drum_loop_policy.py \
  tests/test_decision_core_kick_demote_regression.py \
  tests/test_consensus_concrete_fx_gate_regressions.py

echo
 echo "PASS: family-claim stability patch checks completed."
