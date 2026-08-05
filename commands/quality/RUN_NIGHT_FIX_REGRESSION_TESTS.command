#!/bin/bash
set -euo pipefail
ROOT="${ROOT:-/path/to/Aaron_Sound_Sorter}"
cd "$ROOT"
PY=".venv_phase4/bin/python"
if [ ! -x "$PY" ]; then
  PY="python3"
fi

echo "Compiling changed files..."
"$PY" -m py_compile \
  src/aaron_sound_sorter/domain/roles.py \
  src/aaron_sound_sorter/engine/eligibility.py \
  src/aaron_sound_sorter/engine/decision_core_v2.py \
  tools/diff_sort_manifests.py \
  tools/run_single_file_log_panel.py \
  tools/run_random_sample_folder_log_panel.py

echo "Running safe synthetic and non-audio regression panel..."
"$PY" -m pytest -q \
  tests/test_parent_eligibility_synthetic_voice_false_positive_surrogates.py \
  tests/test_parent_eligibility_synthetic_bass_alert_overreach_surrogates.py \
  tests/test_parent_eligibility_synthetic_mixed_loop_over_narrowing.py \
  tests/test_parent_eligibility_fact_roles_v27.py \
  tests/test_decision_core_v2.py \
  tests/test_phase4_v060_physical_family_guard_regressions.py \
  tests/test_phase4_voice_guard.py \
  tests/test_phase4_shape_voter.py \
  tests/test_phase4_committee_physics_gap_locks.py \
  tests/test_two_voter_redesign.py

echo "PASS: night fix regression panel completed."
