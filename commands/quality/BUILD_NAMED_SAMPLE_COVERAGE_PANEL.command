#!/bin/bash
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/Volumes/T9/testbed/Aaron_Sound_Sorter}"
cd "$PROJECT_ROOT" || exit 2

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$PROJECT_ROOT/.venv_phase4/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv_phase4/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

SAMPLES_ROOT="${SAMPLES_ROOT:-/Volumes/T9/music_production/samples}"
PANEL_SIZE="${PANEL_SIZE:-5000}"
PER_CATEGORY_LIMIT="${PER_CATEGORY_LIMIT:-120}"
LINK_MODE="${LINK_MODE:-symlink}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/_reports/named_sample_coverage/run_$STAMP}"

mkdir -p "$RUN_DIR"

"$PYTHON_BIN" "$PROJECT_ROOT/tools/named_sample_coverage_panel.py" build \
  --sample-root "$SAMPLES_ROOT" \
  --output-dir "$RUN_DIR" \
  --panel-size "$PANEL_SIZE" \
  --per-category-limit "$PER_CATEGORY_LIMIT" \
  --link-mode "$LINK_MODE"

STATUS=$?
echo "$RUN_DIR" > "$PROJECT_ROOT/_reports/named_sample_coverage/latest_run_path.txt"
echo "Named sample panel: $RUN_DIR"
exit "$STATUS"
