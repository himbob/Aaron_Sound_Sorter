#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
cd "$PROJECT_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
        PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
    else
        PYTHON_BIN="$PROJECT_ROOT/.venv_phase4/bin/python"
    fi
fi
export TK_SILENCE_DEPRECATION=1
exec "$PYTHON_BIN" "$PROJECT_ROOT/tools/aaron_sorter_gui.py" "$@"
