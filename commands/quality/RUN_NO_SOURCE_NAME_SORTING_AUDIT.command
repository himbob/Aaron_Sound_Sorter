#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$PROJECT_ROOT/.venv_phase4/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv_phase4/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

echo
echo "Running no-source-name production sorting audit..."
echo "Project: $PROJECT_ROOT"
echo

"$PYTHON_BIN" "$PROJECT_ROOT/tools/audit_no_source_name_sorting.py" --project-root "$PROJECT_ROOT" --verbose
