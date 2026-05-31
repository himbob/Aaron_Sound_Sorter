#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src"
python3 -m pytest -q \
  tests/test_decision_core_v2.py \
  tests/test_parent_eligibility_uploaded_audio.py \
  tests/test_parent_eligibility_percussion_oneshots.py \
  tests/test_parent_eligibility_fx_instrument_steal_guard.py
rm -rf _qa_parent_eligibility_v2
python3 Aaron_Sound_Sorter.py sort \
  tests/regression_audio_uploaded_current \
  _qa_parent_eligibility_v2 \
  --brain stage4_folder_brain.json \
  --no-zip
rm -rf _qa_parent_eligibility_v2_percussion
python3 Aaron_Sound_Sorter.py sort \
  tests/regression_audio_percussion_fx_guard \
  _qa_parent_eligibility_v2_percussion \
  --brain stage4_folder_brain.json \
  --no-zip
if command -v open >/dev/null 2>&1; then
  open "$PWD/_qa_parent_eligibility_v2"
  open "$PWD/_qa_parent_eligibility_v2_percussion"
fi
