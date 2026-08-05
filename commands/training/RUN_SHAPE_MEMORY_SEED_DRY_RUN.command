#!/bin/bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null)}"
cd "$PROJECT_ROOT" || exit 2

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$PROJECT_ROOT/.venv_phase4/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv_phase4/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" "$PROJECT_ROOT/tools/shape_memory_seed.py" --project-root "$PROJECT_ROOT" "$@"
