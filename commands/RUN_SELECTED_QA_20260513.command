#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
python3 -m py_compile Aaron_Sound_Sorter.py $(find src/aaron_sound_sorter -name '*.py' ! -name '._*')
python3 -m pytest -q \
  tests/test_phase4_percussive_one_shot_role.py \
  tests/test_scoring_tools_edge_coverage.py \
  tests/test_stage4_brain_category_simulation_matrix.py \
  tests/test_parent_role_audit.py \
  tests/test_dynamic_branch_identity_gate.py \
  tests/test_phase4_real_project_zip_smoke.py::test_real_percussive_family_guard_regression_never_auto_places_as_fx_or_instruments
