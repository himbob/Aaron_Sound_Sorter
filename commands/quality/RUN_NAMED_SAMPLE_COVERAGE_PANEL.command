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
SORT_WORKERS="${SORT_WORKERS:-6}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/_reports/named_sample_coverage/run_$STAMP}"

mkdir -p "$RUN_DIR"
echo "$RUN_DIR" > "$PROJECT_ROOT/_reports/named_sample_coverage/latest_run_path.txt"

echo "Building named sample coverage panel..."
"$PYTHON_BIN" "$PROJECT_ROOT/tools/named_sample_coverage_panel.py" build \
  --sample-root "$SAMPLES_ROOT" \
  --output-dir "$RUN_DIR" \
  --panel-size "$PANEL_SIZE" \
  --per-category-limit "$PER_CATEGORY_LIMIT" \
  --link-mode "$LINK_MODE"
BUILD_STATUS=$?
if [ "$BUILD_STATUS" -ne 0 ]; then
  echo "Panel build failed: $BUILD_STATUS"
  exit "$BUILD_STATUS"
fi

echo "Sorting panel with $SORT_WORKERS worker(s)..."
AARON_SORT_WORKERS="$SORT_WORKERS" "$PYTHON_BIN" "$PROJECT_ROOT/Aaron_Sound_Sorter.py" sort \
  "$RUN_DIR/input_panel" \
  "$RUN_DIR/sort_output" \
  --no-zip \
  --workers "$SORT_WORKERS" \
  > "$RUN_DIR/sort_console.log" 2>&1
SORT_STATUS=$?

echo "Auditing panel..."
"$PYTHON_BIN" "$PROJECT_ROOT/tools/named_sample_coverage_panel.py" audit \
  --expected "$RUN_DIR/named_sample_panel_expected.csv" \
  --manifest "$RUN_DIR/sort_output/Aaron_Sorted_Sounds_manifest.csv" \
  --output-dir "$RUN_DIR/audit"
AUDIT_STATUS=$?

echo "Run folder: $RUN_DIR"
echo "Sort status: $SORT_STATUS"
echo "Audit status: $AUDIT_STATUS"

if [ "$SORT_STATUS" -ne 0 ]; then
  exit "$SORT_STATUS"
fi
exit "$AUDIT_STATUS"
