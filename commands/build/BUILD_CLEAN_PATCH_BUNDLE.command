#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
AI_BASE="${AI_BASE:-HEAD}"
BUNDLE_NAME="${BUNDLE_NAME:-Aaron_Sound_Sorter_AI_patch}"
BUNDLE_OUTPUT_DIR="${BUNDLE_OUTPUT_DIR:-_reports/bundles}"

cd "$PROJECT_ROOT"

"$PYTHON_BIN" tools/build_clean_patch_bundle.py \
  --project-root "$PROJECT_ROOT" \
  --base "$AI_BASE" \
  --bundle-name "$BUNDLE_NAME" \
  --output-dir "$BUNDLE_OUTPUT_DIR" \
  "$@"

case "$(uname -s)" in
  Darwin)
    if [ -d "$BUNDLE_OUTPUT_DIR" ]; then
      if ! open "$BUNDLE_OUTPUT_DIR" >/dev/null 2>&1; then
        printf 'Bundle ready at %s (Finder could not open this folder).\n' "$BUNDLE_OUTPUT_DIR"
      fi
    fi
    ;;
esac
