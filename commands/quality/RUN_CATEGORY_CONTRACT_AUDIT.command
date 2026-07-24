#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/Volumes/T9/testbed/Aaron_Sound_Sorter}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv_phase4/bin/python}"

cd "$PROJECT_ROOT"
PYTHONPATH=src "$PYTHON_BIN" tools/category_contract_audit.py --project-root "$PROJECT_ROOT" "$@"
