#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null)}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv_phase4/bin/python}"

cd "$PROJECT_ROOT"
PYTHONPATH=src "$PYTHON_BIN" tools/category_contract_audit.py --project-root "$PROJECT_ROOT" "$@"
