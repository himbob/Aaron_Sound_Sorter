#!/bin/bash
set -euo pipefail
PROJECT_ROOT="/path/to/Aaron_Sound_Sorter"
REPORT_ROOT="$PROJECT_ROOT/_reports/claim_validity_refactor/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPORT_ROOT"
cd "$PROJECT_ROOT"
find . -name '._*' -type f -delete
find . -name '.DS_Store' -type f -delete
export PYTHONPATH="$PROJECT_ROOT/src"

run_test() {
  local name="$1"
  echo "RUN $name" | tee -a "$REPORT_ROOT/pytest_node_by_node.log"
  python3 -m pytest -q "$name" 2>&1 | tee -a "$REPORT_ROOT/pytest_node_by_node.log"
}

python3 -S -m py_compile \
  src/aaron_sound_sorter/engine/family_claim_arbiter.py \
  tests/test_claim_validity_refactor_v3196.py \
  tests/test_family_claim_arbitration_matrix.py \
  tests/test_two_voter_redesign.py \
  2>&1 | tee "$REPORT_ROOT/py_compile.log"

run_test tests/test_claim_validity_refactor_v3196.py
run_test tests/test_claim_arbiter_architecture.py
run_test tests/test_family_claim_arbitration_matrix.py
run_test tests/test_two_voter_redesign.py
run_test tests/test_broad_bucket_claim_producer.py
run_test tests/test_brain_competence_policy.py
run_test tests/test_v3187_architecture_doc_and_lane_competence.py
run_test tests/test_no_source_name_sorting_invariant.py
run_test tests/test_profile_candidate_claim_producer.py::test_profile_candidate_producer_returns_kick_claim_from_measured_kick_context
run_test tests/test_profile_candidate_claim_producer.py::test_profile_candidate_producer_does_not_promote_reed_without_specific_reed_role

python3 tools/audit_no_source_name_sorting.py 2>&1 | tee "$REPORT_ROOT/no_source_name_audit.log"

echo "Claim validity refactor checks finished. Reports: $REPORT_ROOT"
open "$REPORT_ROOT"
