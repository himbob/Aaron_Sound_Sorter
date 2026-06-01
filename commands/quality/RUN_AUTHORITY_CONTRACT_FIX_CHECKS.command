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

find "$PROJECT_ROOT" -name '._*' -type f -delete
find "$PROJECT_ROOT" -name '.DS_Store' -type f -delete

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

run_pytest_if_exists tests/test_v3199_arbiter_authority_contract.py
run_pytest_if_exists tests/test_v3189_real_audio_architecture_regressions.py
run_pytest_if_exists tests/test_claim_arbiter_real_panel_surrogates.py
run_pytest_if_exists tests/test_family_claim_arbitration_matrix.py
run_pytest_if_exists tests/test_decision_core_v2.py
run_pytest_if_exists tests/test_parent_eligibility_uploaded_audio.py
run_pytest_if_exists tests/test_consensus_concrete_fx_gate_regressions.py

run_command_if_exists ./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command
