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
MANIFEST="${MANIFEST:-}"
ARGS=("$PROJECT_ROOT/tools/audit_manifest_no_review.py" --project-root "$PROJECT_ROOT")
if [ -n "$MANIFEST" ]; then
  ARGS+=(--manifest "$MANIFEST")
fi
"$PYTHON_BIN" "${ARGS[@]}"
