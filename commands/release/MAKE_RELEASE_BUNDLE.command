#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$(dirname "$ROOT")"
NAME="$(basename "$ROOT")"
OUT="${NAME}_release_$(date +%Y%m%d_%H%M%S).zip"
zip -yr "$OUT" "$NAME" \
  -x "*/.git/*" "*/.venv*/*" "*/__pycache__/*" "*/.pytest_cache/*" "*/.DS_Store" "*/._*" "*/.__*" "*/__MACOSX/*" "*/_archive/*" "*/docs/archive/*" "*/_real_sort_tests/*" "*/stage4_folder_brain_training/*"
echo "Created: $OUT"
open "$(pwd)"
