#!/bin/bash
set -euo pipefail

# Random real-folder log panel for Aaron Sound Sorter.
#
# Defaults:
#   SAMPLES_ROOT=/path/to/sample-library
#   FOLDER_COUNT=20
#   FILES_PER_FOLDER=3
#   TIMEOUT_SEC=0   # no timeout on Aaron's Mac
#   SKIP_AKWF=1
#   SKIP_SORTED_SAMPLES=1
#   SKIP_KNOWN_TEST_MATERIAL=1
#
# Output:
#   reports/random_sample_folder_log_panel/run_*/random_folder_log_panel_upload_back.zip
#
# Upload back only that zip. It contains logs/CSVs only, not audio.

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

SAMPLES_ROOT="${SAMPLES_ROOT:-/path/to/sample-library}"
BRAIN="${BRAIN:-stage4_folder_brain.json}"
FOLDER_COUNT="${FOLDER_COUNT:-20}"
FILES_PER_FOLDER="${FILES_PER_FOLDER:-3}"
MAX_TOTAL="${MAX_TOTAL:-0}"
SEED="${SEED:-$(date +%s)}"
TIMEOUT_SEC="${TIMEOUT_SEC:-0}"
CONTAINS="${CONTAINS:-}"
FOLLOW_SYMLINKS="${FOLLOW_SYMLINKS:-1}"
COPY_AUDIO="${COPY_AUDIO:-0}"
SKIP_AKWF="${SKIP_AKWF:-1}"
SKIP_SORTED_SAMPLES="${SKIP_SORTED_SAMPLES:-1}"
SKIP_KNOWN_TEST_MATERIAL="${SKIP_KNOWN_TEST_MATERIAL:-1}"

ARGS=(
  "$PROJECT_ROOT/tools/run_random_sample_folder_log_panel.py"
  --project-root "$PROJECT_ROOT"
  --sample-root "$SAMPLES_ROOT"
  --brain "$BRAIN"
  --folder-count "$FOLDER_COUNT"
  --files-per-folder "$FILES_PER_FOLDER"
  --max-total "$MAX_TOTAL"
  --seed "$SEED"
  --timeout-sec "$TIMEOUT_SEC"
)

if [ -n "$CONTAINS" ]; then
  ARGS+=(--contains "$CONTAINS")
fi

if [ "$FOLLOW_SYMLINKS" = "1" ]; then
  ARGS+=(--follow-symlinks)
fi

if [ "$COPY_AUDIO" = "1" ]; then
  ARGS+=(--copy-audio)
fi

if [ "$SKIP_AKWF" != "1" ]; then
  ARGS+=(--include-akwf)
fi

if [ "$SKIP_SORTED_SAMPLES" != "1" ]; then
  ARGS+=(--include-sorted-samples)
fi

if [ "$SKIP_KNOWN_TEST_MATERIAL" != "1" ]; then
  ARGS+=(--include-known-test-material)
fi

echo
echo "Aaron Sound Sorter random folder log panel"
echo "Project:          $PROJECT_ROOT"
echo "Sample root:      $SAMPLES_ROOT"
echo "Folders:          $FOLDER_COUNT"
echo "Files per folder: $FILES_PER_FOLDER"
echo "Seed:             $SEED"
echo "Timeout seconds:  $TIMEOUT_SEC (0 means no timeout)"
echo "Skip AKWF:        $SKIP_AKWF"
echo "Skip Sorted:      $SKIP_SORTED_SAMPLES"
echo "Skip known tests: $SKIP_KNOWN_TEST_MATERIAL"
echo "Brain:            $BRAIN"
echo

"$PYTHON_BIN" "${ARGS[@]}"

LATEST_ZIP="$(ls -t "$PROJECT_ROOT"/reports/random_sample_folder_log_panel/run_*/random_folder_log_panel_upload_back.zip 2>/dev/null | head -1 || true)"
if [ -n "$LATEST_ZIP" ]; then
  echo
  echo "Upload this logs-only zip:"
  echo "$LATEST_ZIP"
  open "$(dirname "$LATEST_ZIP")" >/dev/null 2>&1 || true
fi
