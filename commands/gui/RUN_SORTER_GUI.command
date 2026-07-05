#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/Volumes/T9/testbed/Aaron_Sound_Sorter}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv_phase4/bin/python}"
export TK_SILENCE_DEPRECATION=1
exec "$PYTHON_BIN" "$PROJECT_ROOT/tools/aaron_sorter_gui.py" "$@"
