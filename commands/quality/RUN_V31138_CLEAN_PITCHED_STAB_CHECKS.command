#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-}"
if [ -z "$PROJECT_ROOT" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
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

echo "Cleaning macOS metadata..."
find . -name '._*' -type f -delete
find . -name '.DS_Store' -type f -delete

echo "Compiling changed code..."
"$PYTHON_BIN" -m py_compile src/aaron_sound_sorter/domain/roles.py tests/test_v31138_clean_pitched_stab_role_guard.py

echo "Running clean pitched stab regression..."
PYTHONPATH=src "$PYTHON_BIN" -m pytest -q tests/test_v31138_clean_pitched_stab_role_guard.py

echo "Running percussion one-shot safety tests..."
PYTHONPATH=src "$PYTHON_BIN" -m pytest -q tests/test_parent_eligibility_percussion_oneshots.py

echo "Running pitched percussion guard tests..."
PYTHONPATH=src "$PYTHON_BIN" -m pytest -q tests/test_phase4_pitched_percussion_guard.py

echo "Running keys/synth guard tests..."
PYTHONPATH=src "$PYTHON_BIN" -m pytest -q tests/test_physics_voter_piano_struck_identity.py
PYTHONPATH=src "$PYTHON_BIN" -m pytest -q tests/test_v31116_synth_pad_keys_decoy_guard.py
echo "Running source-name blindness audit..."
PROJECT_ROOT="$PROJECT_ROOT" PYTHON_BIN="$PYTHON_BIN" ./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command

echo "V31.138 clean pitched stab checks completed."
