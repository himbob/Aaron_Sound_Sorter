#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

RUN_ID="$(date +%Y%m%d_%H%M%S)"
DEFAULT_ROOT="$PWD/training/locked_curated_v1"
SCAN_ROOT="${1:-$DEFAULT_ROOT}"
OUT="$PWD/reports/metallic_percussion_review/run_${RUN_ID}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Metallic percussion review"
echo "Project: $PWD"
echo "Scan root: $SCAN_ROOT"
echo "Output: $OUT"
echo

"$PYTHON_BIN" tools/aaron_metallic_percussion_reviewer.py \
  --root "$SCAN_ROOT" \
  --out "$OUT" \
  --mode symlink

echo
if command -v open >/dev/null 2>&1; then
  open "$OUT"
fi

echo "DONE"
echo "Review folder: $OUT"
