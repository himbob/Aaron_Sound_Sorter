#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project: $PROJECT_ROOT"
echo "Checking Python syntax..."
python3 -m compileall -q Aaron_Sound_Sorter.py src tests tools

echo "Running source-name blindness audit..."
./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command

echo "Running architectural rescue policy regression tests one file at a time..."
pytest -q tests/test_decision_core_fx_aaron2_pattern_regressions.py
pytest -q tests/test_consensus_concrete_fx_gate_regressions.py
pytest -q tests/test_decision_core_fx_smoke_matrix_hard_failures.py
pytest -q tests/test_decision_core_fx_smoke_remaining_review_rows.py
pytest -q tests/test_decision_core_golden_failure_regressions.py
pytest -q tests/test_decision_core_ground_truth_true_bucket_tdd.py
pytest -q tests/test_decision_core_v2.py
pytest -q tests/test_two_voter_redesign.py
PYTHONPATH=. pytest -q tests/test_no_source_name_sorting_invariant.py

echo "PASS: architectural rescue policy patch checks completed."
