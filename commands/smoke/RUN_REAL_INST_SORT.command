#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/_real_sort_tests/inst_${RUN_ID}"
python3 Aaron_Sound_Sorter.py sort \
  "/path/to/sample-library/MixedInst1.zip" \
  "$OUT" \
  --brain stage4_folder_brain.json \
  --no-zip
open "$OUT" >/dev/null 2>&1 || true
