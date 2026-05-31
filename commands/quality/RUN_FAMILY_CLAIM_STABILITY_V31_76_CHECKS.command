#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/Volumes/T9/testbed/Aaron_Sound_Sorter"
if [ ! -d "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi
cd "$PROJECT_ROOT"

find . -name '._*' -type f -delete
find . -name '__MACOSX' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

PYTHON="${PROJECT_ROOT}/.venv_phase4/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

export PYTHONPATH="$PROJECT_ROOT/src"

echo "Running compile check..."
"$PYTHON" -m compileall -q src tests

echo "Running no-source-name production sorting audit..."
./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command

echo "Running v31.76 family-claim follow-up checks..."
"$PYTHON" -m pytest -q \
  tests/test_two_voter_redesign.py \
  tests/test_family_claim_arbitration_matrix.py \
  tests/test_family_claim_stability_followup_matrix.py \
  tests/test_family_claim_stability_v31_76_followup.py \
  tests/test_direct_body_voice_and_low_drum_loop_policy.py \
  tests/test_parent_eligibility_v24_drum_loop_steal_guard.py

echo "PASS: v31.76 family-claim follow-up checks completed."
