#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
python3 -m pytest -q \
  tests/test_phase4_percussive_one_shot_role.py \
  tests/test_scoring_tools_edge_coverage.py \
  tests/test_parent_role_audit.py \
  tests/test_dynamic_branch_identity_gate.py \
  --cov=src/aaron_sound_sorter/domain \
  --cov=src/aaron_sound_sorter/voters \
  --cov-report=term-missing
