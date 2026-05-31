#!/bin/bash
set -euo pipefail
PROJECT_DIR="/Volumes/T9/testbed/Aaron_Sound_Sorter"
cd "$PROJECT_DIR"

echo "Cleaning macOS metadata files..."
find . -name '._*' -type f -delete
find . -name '.DS_Store' -type f -delete

export PYTHONPATH="$PROJECT_DIR/src"

echo "Compile check..."
python3 -S -m compileall -q Aaron_Sound_Sorter.py src tests

echo "No-source-name audit..."
python3 tools/audit_no_source_name_sorting.py --project-root .

echo "Broad bucket compatibility test..."
python3 -m pytest -q tests/test_broad_bucket_claim_producer.py

echo "Claim arbiter architecture test..."
python3 -m pytest -q tests/test_claim_arbiter_architecture.py

echo "Real audio architecture regression test..."
python3 -m pytest -q tests/test_v3189_real_audio_architecture_regressions.py

echo "Self-test..."
python3 Aaron_Sound_Sorter.py self-test

echo "v31.94C import compatibility checks passed."
