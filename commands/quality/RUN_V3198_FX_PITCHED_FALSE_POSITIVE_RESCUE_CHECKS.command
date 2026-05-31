#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
find . -name '._*' -type f -delete
find . -name '.DS_Store' -type f -delete
if [ -x ".venv_phase4/bin/python" ]; then
  PYTHON=".venv_phase4/bin/python"
elif [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python3"
fi
export PYTHONPATH="src"
echo "Running v31.98 checks one file at a time..."
$PYTHON -m py_compile \
  src/aaron_sound_sorter/engine/family_claim_arbiter.py \
  src/aaron_sound_sorter/engine/claim_producers/baby_recall.py \
  tests/test_v3198_fx_pitched_false_positive_and_baby_recall.py
for t in \
  tests/test_v3197_no_role_score_shaping_architecture.py \
  tests/test_claim_validity_refactor_v3196.py \
  tests/test_v3184_deep_voter_and_baby_recall.py \
  tests/test_v3198_fx_pitched_false_positive_and_baby_recall.py \
  tests/test_claim_arbiter_architecture.py \
  tests/test_broad_bucket_claim_producer.py \
  tests/test_no_source_name_sorting_invariant.py \
  tests/test_brain_competence_policy.py \
  tests/test_v3187_architecture_doc_and_lane_competence.py
do
  echo "=== $t"
  $PYTHON -m pytest -q "$t"
done
$PYTHON tools/audit_no_source_name_sorting.py
echo "PASS: v31.98 FX pitched false-positive rescue checks completed."
