#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pytest -q \
  tests/test_phase4_more_project_zip_patterns.py \
  tests/test_phase4_pitched_phrase_role.py \
  tests/test_phase4_percussive_one_shot_role.py \
  tests/test_scoring_tools_edge_coverage.py \
  tests/test_stage4_brain_category_simulation_matrix.py
