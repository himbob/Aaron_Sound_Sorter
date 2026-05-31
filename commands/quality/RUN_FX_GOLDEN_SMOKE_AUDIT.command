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
if [ -z "$MANIFEST" ]; then
  MANIFEST="$(find ./_real_sort_tests -name 'Aaron_Sorted_Sounds_manifest.csv' -type f -print0 2>/dev/null | xargs -0 ls -t 2>/dev/null | head -1 || true)"
fi
if [ -z "$MANIFEST" ] || [ ! -f "$MANIFEST" ]; then
  echo "ERROR: no manifest found. Set MANIFEST=/path/to/Aaron_Sorted_Sounds_manifest.csv"
  exit 1
fi

OUT="${OUT:-reports/fx_golden_smoke_audit/fx_golden_smoke_failures.csv}"
"$PYTHON_BIN" "$PROJECT_ROOT/tools/audit_fx_smoke_expected_buckets.py" --manifest "$MANIFEST" --out "$OUT"
