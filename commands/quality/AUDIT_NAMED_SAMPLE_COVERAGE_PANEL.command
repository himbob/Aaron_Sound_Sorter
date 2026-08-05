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

RUN_DIR="${RUN_DIR:-}"
if [ -z "$RUN_DIR" ]; then
  RUN_DIR_FILE="$PROJECT_ROOT/_reports/named_sample_coverage/latest_run_path.txt"
  if [ ! -f "$RUN_DIR_FILE" ]; then
    echo "No RUN_DIR supplied and no latest named-sample run exists."
    exit 2
  fi
  RUN_DIR="$(cat "$RUN_DIR_FILE")"
fi

"$PYTHON_BIN" "$PROJECT_ROOT/tools/named_sample_coverage_panel.py" audit \
  --expected "$RUN_DIR/named_sample_panel_expected.csv" \
  --manifest "$RUN_DIR/sort_output/Aaron_Sorted_Sounds_manifest.csv" \
  --output-dir "$RUN_DIR/audit"
