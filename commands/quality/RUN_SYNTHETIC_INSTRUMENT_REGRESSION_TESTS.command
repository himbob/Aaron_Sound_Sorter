#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/Volumes/T9/testbed/Aaron_Sound_Sorter}"
cd "$PROJECT_ROOT"
python3 -m pytest -q \
  tests/test_parent_eligibility_synthetic_instrument_regressions.py \
  tests/test_parent_eligibility_fact_roles_v27.py \
  tests/test_decision_core_v2.py \
  tests/test_phase4_v060_physical_family_guard_regressions.py \
  tests/test_phase4_voice_guard.py \
  tests/test_phase4_shape_voter.py \
  tests/test_phase4_committee_physics_gap_locks.py \
  tests/test_two_voter_redesign.py
