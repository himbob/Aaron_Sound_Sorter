#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$PROJECT_ROOT/.venv_phase4/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv_phase4/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

echo "Project: $PROJECT_ROOT"
echo "Python:  $PYTHON_BIN"

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/private/tmp/aaron_sound_sorter_pycache}"

echo "Compile check..."
"$PYTHON_BIN" -S -m py_compile Aaron_Sound_Sorter.py $(find src -name '*.py' ! -name '._*' | sort) $(find tests -name '*.py' ! -name '._*' | sort)

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

echo "Focused pytest panel, one file at a time..."
run_pytest_if_exists tests/test_v3199_arbiter_authority_contract.py
run_pytest_if_exists tests/test_physics_voter_reed_sax_identity.py
run_pytest_if_exists tests/test_v31101_instrument_subpanels.py
run_pytest_if_exists tests/test_physics_voter_piano_struck_identity.py
run_pytest_if_exists tests/test_no_source_name_sorting_invariant.py

echo "No-source-name sorting audit..."
if [ -x ./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command ]; then
  ./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command
else
  echo "SKIP missing command: ./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command"
fi

echo "Locked smoke acceptance, one-by-one..."
if [ -x ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command ]; then
  ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command
else
  echo "SKIP missing command: ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command"
fi

echo "v31.104 checks complete."
