#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
REPORT_ROOT="_reports/raw_voter_architecture_checks/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPORT_ROOT"

find . -name '._*' -type f -delete
find . -name '.DS_Store' -type f -delete

if [ -x ".venv_phase4/bin/python" ]; then
  PYTHON=".venv_phase4/bin/python"
elif [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python3"
fi

run_node() {
  local node="$1"
  echo "RUN $node" | tee -a "$REPORT_ROOT/pytest_nodes.log"
  "$PYTHON" -m pytest -q "$node" 2>&1 | tee -a "$REPORT_ROOT/pytest_nodes.log"
}

run_node tests/test_v3197_no_role_score_shaping_architecture.py::test_role_compatibility_never_changes_voter_score
run_node tests/test_v3197_no_role_score_shaping_architecture.py::test_winner_preselector_returns_raw_best_shared_candidate
run_node tests/test_v3197_no_role_score_shaping_architecture.py::test_profile_kick_shortcut_cannot_cross_family_from_specific_bass_raw
run_node tests/test_v3184_deep_voter_and_baby_recall.py::test_role_evidence_is_diagnostic_only_for_incompatible_fx_candidate
run_node tests/test_claim_validity_refactor_v3196.py
run_node tests/test_two_voter_redesign.py
run_node tests/test_broad_bucket_claim_producer.py
run_node tests/test_no_source_name_sorting_invariant.py

"$PYTHON" -m py_compile \
  src/aaron_sound_sorter/voters/scoring_tools.py \
  src/aaron_sound_sorter/engine/consensus_winner.py \
  src/aaron_sound_sorter/engine/family_claim_arbiter.py \
  tests/test_v3197_no_role_score_shaping_architecture.py \
  2>&1 | tee "$REPORT_ROOT/py_compile.log"

"$PYTHON" tools/audit_no_source_name_sorting.py 2>&1 | tee "$REPORT_ROOT/no_source_name_audit.log"

open "$REPORT_ROOT"
echo "Reports written to: $REPORT_ROOT"
