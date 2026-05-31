#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:${PYTHONPATH:-}"

# Remove macOS AppleDouble/resource-fork junk so Python never tries to compile binary ._* files.
find . -name '._*' -type f -delete 2>/dev/null || true
find . -name '__MACOSX' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo "Running compile check..."
"$PYTHON_BIN" -m compileall -q src tests tools

echo "Running no-source-name production sorting audit..."
./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command

echo "Running family-claim v31.75 synthetic/stability checks..."
"$PYTHON_BIN" -m pytest -q \
  tests/test_family_claim_stability_followup_matrix.py \
  tests/test_family_claim_arbitration_matrix.py \
  tests/test_decision_core_second_conflict_resolver_tdd.py \
  tests/test_parent_eligibility_synthetic_bass_alert_overreach_surrogates.py \
  tests/test_parent_eligibility_v25_transition_reverb.py \
  tests/test_fx_bundle_regression_patterns.py \
  tests/test_direct_body_voice_and_low_drum_loop_policy.py \
  tests/test_decision_core_kick_demote_regression.py \
  tests/test_consensus_concrete_fx_gate_regressions.py

echo "Running focused real-audio regression checks one group at a time..."
"$PYTHON_BIN" -m pytest -q tests/test_parent_eligibility_fx_instrument_steal_guard.py
"$PYTHON_BIN" -m pytest -q tests/test_parent_eligibility_v26_fx_smoke_reverb.py::test_wet_sax_is_brass_woodwind_not_human_voice_or_drums
"$PYTHON_BIN" -m pytest -q tests/test_uploaded_regression_audio.py::test_bass_loops_land_in_bass_instruments_not_kicks

echo
 echo "PASS: family-claim v31.75 stability checks completed."
