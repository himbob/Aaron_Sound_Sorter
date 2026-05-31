#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

REPORT_ROOT="reports/claim_arbiter_refactor_qa"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$REPORT_ROOT/run_$STAMP"
mkdir -p "$OUT"

{
  echo "Project: $(pwd)"
  echo "Started: $(date)"
  echo
  echo "== No-source-name audit =="
  ./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command
  echo
  echo "== Self-test =="
  python Aaron_Sound_Sorter.py self-test
  echo
  echo "== Compile src =="
  find src -name '*.py' -print0 | xargs -0 python -m py_compile
  echo
  echo "== Focused pytest =="
  python -m pytest \
    tests/test_claim_arbiter_architecture.py \
    tests/test_two_voter_redesign.py \
    tests/test_family_claim_arbitration_matrix.py \
    tests/test_no_source_name_sorting_invariant.py \
    -q
  echo
  echo "Finished: $(date)"
} 2>&1 | tee "$OUT/claim_arbiter_refactor_qa.log"

(cd "$REPORT_ROOT" && zip -qr "claim_arbiter_refactor_qa_$STAMP.zip" "run_$STAMP")
open "$OUT" 2>/dev/null || true
