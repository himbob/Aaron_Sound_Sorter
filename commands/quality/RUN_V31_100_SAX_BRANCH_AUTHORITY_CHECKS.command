#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$PROJECT_ROOT/.venv_phase4/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv_phase4/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:${PYTHONPATH:-}"
"$PYTHON_BIN" -m pytest -q \
  tests/test_physics_voter_reed_sax_identity.py \
  tests/test_physics_layers.py \
  tests/test_v3199_arbiter_authority_contract.py
./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command
for CASE_ID in \
  dark_low_mid_sax_grimy_hiphop_no_guitar \
  dark_low_mid_sax_hiphoptapes_31_no_guitar \
  electric_keys_ews_not_sax_or_guitar \
  police_fx_siren_not_piano_or_sax \
  electric_piano_not_sax_ws2_101_fm \
  wet_sax_hiphoptapes_no_voice \
  wet_sax_jazzhiphop_15 \
  wet_reverb_sax_intro_fminor_96 \
  wet_sax_scy097_no_voice
  do
    PROJECT_ROOT="$PROJECT_ROOT" PYTHON_BIN="$PYTHON_BIN" ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --case-id "$CASE_ID"
  done
