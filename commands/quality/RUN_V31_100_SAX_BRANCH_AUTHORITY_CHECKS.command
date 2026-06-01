#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$PROJECT_ROOT/.venv_phase4/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv_phase4/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

run_pytest_if_exists() {
  local node="$1"
  local file="${node%%::*}"
  if [ -f "$file" ]; then
    echo
    echo "RUN: $PYTHON_BIN -m pytest -q $node"
    "$PYTHON_BIN" -m pytest -q "$node"
  else
    echo
    echo "SKIP missing pytest file: $file"
  fi
}

run_command_if_exists() {
  local cmd="$1"
  if [ -x "$cmd" ]; then
    echo
    echo "RUN: $cmd"
    "$cmd"
  else
    echo
    echo "SKIP missing command: $cmd"
  fi
}

run_pytest_if_exists tests/test_physics_voter_reed_sax_identity.py
run_pytest_if_exists tests/test_physics_layers.py
run_pytest_if_exists tests/test_v3199_arbiter_authority_contract.py

run_command_if_exists ./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command

if [ ! -x ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command ]; then
  echo
  echo "SKIP missing command: ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command"
  exit 0
fi

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
    PROJECT_ROOT="$PROJECT_ROOT" PYTHON_BIN="$PYTHON_BIN" \
      ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --case-id "$CASE_ID"
  done

echo
echo "v31.100 sax branch authority checks complete."
