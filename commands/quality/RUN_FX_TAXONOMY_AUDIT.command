#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/Volumes/T9/testbed/Aaron_Sound_Sorter}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv_phase4/bin/python}"
PYTHONPATH=src "$PYTHON_BIN" tools/fx_taxonomy_audit.py --project-root "$PROJECT_ROOT" "$@"
