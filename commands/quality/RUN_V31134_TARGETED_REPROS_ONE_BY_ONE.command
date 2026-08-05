#!/bin/bash
set -euo pipefail
cd /path/to/Aaron_Sound_Sorter
find . -name '._*' -type f -delete
find . -name '.DS_Store' -type f -delete

PYTEST=".venv_phase4/bin/python -m pytest"
if [ ! -x .venv_phase4/bin/python ]; then
  PYTEST="python3 -m pytest"
fi

run_one() {
  echo ""
  echo "================================================================"
  echo "RUNNING: $1"
  echo "================================================================"
  $PYTEST -q "$1"
}

run_one "tests/stability/test_category_neighbor_traps.py::test_anchor_case_ids_are_unique_and_protected_samples_exist"
run_one "tests/test_parent_eligibility_fx_instrument_steal_guard.py::test_short_percussion_hits_do_not_go_to_fx_or_instruments[33157.wav]"
run_one "tests/test_parent_eligibility_fx_instrument_steal_guard.py::test_short_percussion_hits_do_not_go_to_fx_or_instruments[50728.wav]"
run_one "tests/test_parent_eligibility_v28_fx_zip_matrix.py::test_fx_zip_voice_and_vocal_stabs_stay_voice_human[Money_vocals_female_rap_110bpm.wav]"
run_one "tests/test_phase4_shape_voter.py::test_shape_voter_does_not_promote_ambiguous_musical_phrase_to_drum_loop"
run_one "tests/test_uploaded_regression_audio.py::test_bass_loops_land_in_bass_instruments_not_kicks[04_Dmn_176bpm_bass.wav]"
run_one "tests/test_v31101_instrument_subpanels.py::test_mallet_bell_panels_report_metallic_pitched_hits"

echo ""
echo "Targeted v31.134 repro tests completed."
open "_reports" 2>/dev/null || true
