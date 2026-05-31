#!/bin/bash
set -euo pipefail
ROOT="${ROOT:-/Volumes/T9/testbed/Aaron_Sound_Sorter}"
cd "$ROOT"
PY=".venv_phase4/bin/python"
if [ ! -x "$PY" ]; then
  PY="python3"
fi
ZIP_PATH="${ZIP_PATH:-/Volumes/T9/music_production/samples/FX_Aaron2.zip}"
EXPECTED_TOP="${EXPECTED_TOP:-}"
MAX_TOTAL="${MAX_TOTAL:-36}"
PER_CATEGORY="${PER_CATEGORY:-3}"
TIMEOUT_SEC="${TIMEOUT_SEC:-75}"

ARGS=(
  tools/run_single_file_log_panel.py
  --project-root "$ROOT"
  --zip "$ZIP_PATH"
  --brain stage4_folder_brain.json
  --per-category "$PER_CATEGORY"
  --max-total "$MAX_TOTAL"
  --timeout-sec "$TIMEOUT_SEC"
)
if [ -n "$EXPECTED_TOP" ]; then
  ARGS+=(--expected-top "$EXPECTED_TOP")
fi

"$PY" "${ARGS[@]}"

LATEST=$(ls -dt reports/single_file_log_panel/run_* 2>/dev/null | head -1 || true)
if [ -n "$LATEST" ]; then
  open "$LATEST" >/dev/null 2>&1 || true
fi
