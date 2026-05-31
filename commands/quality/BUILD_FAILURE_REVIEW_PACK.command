#!/bin/bash
set -euo pipefail

# Build a local failure review pack from the latest log-panel run.
#
# Default:
#   - Uses latest random/single-file log-panel run
#   - Includes suspicious rows only
#   - Creates symlinks to audio, not copies
#   - Zips the folder preserving symlinks
#
# For an uploadable mini-pack with actual audio:
#   COPY_AUDIO=1 MAX_AUDIO_FILES=20 ./commands/quality/BUILD_FAILURE_REVIEW_PACK.command

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

REPORT_DIR="${REPORT_DIR:-}"
REPORT_ZIP="${REPORT_ZIP:-}"
MAX_CASES="${MAX_CASES:-80}"
COPY_AUDIO="${COPY_AUDIO:-0}"
MAX_AUDIO_FILES="${MAX_AUDIO_FILES:-20}"
INCLUDE_ALL="${INCLUDE_ALL:-0}"
INCLUDE_REVIEWS="${INCLUDE_REVIEWS:-1}"
ONLY_CATEGORY="${ONLY_CATEGORY:-}"

ARGS=(
  "$PROJECT_ROOT/tools/build_failure_review_pack.py"
  --project-root "$PROJECT_ROOT"
  --max-cases "$MAX_CASES"
)

if [ -n "$REPORT_DIR" ]; then
  ARGS+=(--report-dir "$REPORT_DIR")
fi

if [ -n "$REPORT_ZIP" ]; then
  ARGS+=(--report-zip "$REPORT_ZIP")
fi

if [ "$COPY_AUDIO" = "1" ]; then
  ARGS+=(--copy-audio --max-audio-files "$MAX_AUDIO_FILES")
fi

if [ "$INCLUDE_ALL" = "1" ]; then
  ARGS+=(--include-all)
fi

if [ "$INCLUDE_REVIEWS" = "1" ]; then
  ARGS+=(--include-reviews)
fi

if [ -n "$ONLY_CATEGORY" ]; then
  ARGS+=(--only-category "$ONLY_CATEGORY")
fi

echo
echo "Building Aaron failure review pack"
echo "Project:        $PROJECT_ROOT"
echo "Max cases:      $MAX_CASES"
echo "Copy audio:     $COPY_AUDIO"
echo "Max audio:      $MAX_AUDIO_FILES"
echo "Include review: $INCLUDE_REVIEWS"
echo

"$PYTHON_BIN" "${ARGS[@]}"

LATEST="$(ls -td "$PROJECT_ROOT"/reports/failure_review_packs/run_* 2>/dev/null | head -1 || true)"
if [ -n "$LATEST" ]; then
  echo
  echo "Latest review pack:"
  echo "$LATEST"
  open "$LATEST" >/dev/null 2>&1 || true
fi
