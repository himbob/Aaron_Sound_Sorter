#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/path/to/Aaron_Sound_Sorter"
if [ ! -d "$PROJECT_ROOT" ]; then
  PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi
cd "$PROJECT_ROOT"

find . -name '._*' -type f -delete
find . -name '__MACOSX' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

PYTHON="$PROJECT_ROOT/.venv_phase4/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi
export PYTHONPATH="$PROJECT_ROOT/src"

echo "Running compile check..."
"$PYTHON" -m compileall -q src tests

echo "Running no-source-name production sorting audit..."
./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command

echo "Running family-claim arbiter regression matrix..."
"$PYTHON" -m pytest -q \
  tests/test_fx_bundle_regression_patterns.py \
  tests/test_parent_eligibility_fx_instrument_steal_guard.py \
  tests/test_family_claim_arbitration_matrix.py \
  tests/test_family_claim_stability_followup_matrix.py \
  tests/test_family_claim_stability_v31_76_followup.py \
  tests/test_direct_body_voice_and_low_drum_loop_policy.py \
  tests/test_two_voter_redesign.py

echo "Running slow real-audio drum/bass/voice guard one node at a time..."
"$PYTHON" - <<'PY'
from __future__ import annotations
import re
import subprocess
import sys

path = "tests/test_parent_eligibility_v24_drum_loop_steal_guard.py"
collect = subprocess.run(
    [sys.executable, "-m", "pytest", "--collect-only", "-vv", path],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    check=True,
)
nodes = []
for line in collect.stdout.splitlines():
    match = re.search(r"<Function (.*?)>", line)
    if match:
        nodes.append(path + "::" + match.group(1))
if not nodes:
    raise SystemExit("No pytest nodes collected from " + path)
for index, node in enumerate(nodes, start=1):
    print(f"[{index}/{len(nodes)}] {node}", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", node],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
    print(result.stdout, flush=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
PY

echo "PASS: v31.77 family-claim arbiter checks completed."
