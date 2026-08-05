#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null)}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv_phase4/bin/python}"
PYTHONPATH=src "$PYTHON_BIN" tools/trusted_memory_seed.py --project-root "$PROJECT_ROOT" "$@"
