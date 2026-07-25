#!/usr/bin/env bash
set -euo pipefail

# Build clean Aaron Sound Sorter distribution archives.
#
# Default output (split mode):
#   1. Aaron_Sound_Sorter_Code_Only_<timestamp>.zip
#      Small source/review archive without trained runtime assets.
#   2. Aaron_Sound_Sorter_End_User_Pack_<timestamp>_Part_XX_of_YY.zip
#      Balanced upload-safe parts. Together they contain the code archive,
#      active brains, selected CLAP/MERT snapshots, and prototype indexes.
#
# Extract every End_User_Pack part into one folder and run the included
# ASSEMBLE_AND_INSTALL.command. The pack parts are balanced below the configured
# upload limit instead of leaving one tiny code ZIP and one ~1 GB AI ZIP.
#
# ZIP level 9 is already the strongest normal ZIP/GZIP DEFLATE compression.
# Neural model weights compress poorly, so size control comes from balanced
# multipart packaging rather than pretending another DEFLATE wrapper will help.
#
# Common usage:
#   make bundle
#   make bundle-dry-run
#   BUNDLE_OUTPUT_DIR="$HOME/Downloads" make bundle
#   BUNDLE_UPLOAD_PART_MAX_MIB=490 make bundle
#   BUNDLE_OUTPUT_MODE=code ./commands/bundle/bundle.sh
#   BUNDLE_OUTPUT_MODE=ai-runtime ./commands/bundle/bundle.sh
#   BUNDLE_OUTPUT_MODE=product ./commands/bundle/bundle.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BUNDLE_OUTPUT_MODE="${BUNDLE_OUTPUT_MODE:-split}"
BUNDLE_INCLUDE_NEURAL_MODELS="${BUNDLE_INCLUDE_NEURAL_MODELS:-1}"
BUNDLE_CODE_COMPRESSION_LEVEL="${BUNDLE_CODE_COMPRESSION_LEVEL:-9}"
BUNDLE_AI_COMPRESSION_LEVEL="${BUNDLE_AI_COMPRESSION_LEVEL:-9}"
BUNDLE_PRODUCT_COMPRESSION_LEVEL="${BUNDLE_PRODUCT_COMPRESSION_LEVEL:-9}"
BUNDLE_UPLOAD_PART_MAX_MIB="${BUNDLE_UPLOAD_PART_MAX_MIB:-490}"
BUNDLE_KEEP_UNSPLIT_AI_ARCHIVE="${BUNDLE_KEEP_UNSPLIT_AI_ARCHIVE:-0}"
BUNDLE_PAYLOAD_ROOT_NAME="${BUNDLE_PAYLOAD_ROOT_NAME:-Aaron_Sound_Sorter}"
DRY_RUN="${DRY_RUN:-0}"
STAMP="${BUNDLE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
BUNDLE_OUTPUT_DIR="${BUNDLE_OUTPUT_DIR:-$PROJECT_ROOT/_reports/bundles}"
BUILDER="$PROJECT_ROOT/tools/build_product_bundle.py"
PACKER="$PROJECT_ROOT/tools/build_balanced_upload_pack.py"

if [[ ! -f "$BUILDER" ]]; then
  echo "ERROR: Product bundle builder not found: $BUILDER" >&2
  exit 2
fi
if [[ ! -f "$PACKER" ]]; then
  echo "ERROR: Balanced upload-pack builder not found: $PACKER" >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

case "$BUNDLE_OUTPUT_MODE" in
  split|code|ai-runtime|product) ;;
  *)
    echo "ERROR: BUNDLE_OUTPUT_MODE must be split, code, ai-runtime, or product." >&2
    exit 2
    ;;
esac

for compression_variable in \
  BUNDLE_CODE_COMPRESSION_LEVEL \
  BUNDLE_AI_COMPRESSION_LEVEL \
  BUNDLE_PRODUCT_COMPRESSION_LEVEL; do
  compression_value="${!compression_variable}"
  if [[ ! "$compression_value" =~ ^[0-9]$ ]]; then
    echo "ERROR: $compression_variable must be a number from 0 through 9." >&2
    exit 2
  fi
done

if [[ ! "$BUNDLE_UPLOAD_PART_MAX_MIB" =~ ^[0-9]+$ ]] || [[ "$BUNDLE_UPLOAD_PART_MAX_MIB" -lt 2 ]]; then
  echo "ERROR: BUNDLE_UPLOAD_PART_MAX_MIB must be an integer of at least 2." >&2
  exit 2
fi

COMMON_ARGS=(
  --project-root "$PROJECT_ROOT"
  --output-dir "$BUNDLE_OUTPUT_DIR"
  --payload-root-name "$BUNDLE_PAYLOAD_ROOT_NAME"
)
if [[ "$DRY_RUN" == "1" ]]; then
  COMMON_ARGS+=(--dry-run)
fi

build_one() {
  local mode="$1"
  local bundle_name="$2"
  local compression_level="$3"
  shift 3
  echo
  echo "Building mode: $mode (ZIP compression level $compression_level)"
  "$PYTHON_BIN" "$BUILDER" \
    "${COMMON_ARGS[@]}" \
    --mode "$mode" \
    --bundle-name "$bundle_name" \
    --compression-level "$compression_level" \
    "$@"
}

build_ai_payload() {
  local mode="$1"
  local bundle_name="$2"
  local compression_level="$3"
  if [[ "$BUNDLE_INCLUDE_NEURAL_MODELS" == "0" ]]; then
    build_one "$mode" "$bundle_name" "$compression_level" --without-neural-models
  else
    build_one "$mode" "$bundle_name" "$compression_level"
  fi
}

CODE_BUNDLE_NAME="Aaron_Sound_Sorter_Code_Only_${STAMP}"
AI_BUNDLE_NAME="Aaron_Sound_Sorter_AI_Runtime_${STAMP}"
PACK_NAME="Aaron_Sound_Sorter_End_User_Pack_${STAMP}"
CODE_ARCHIVE="$BUNDLE_OUTPUT_DIR/${CODE_BUNDLE_NAME}.zip"
AI_ARCHIVE="$BUNDLE_OUTPUT_DIR/${AI_BUNDLE_NAME}.zip"

case "$BUNDLE_OUTPUT_MODE" in
  split)
    build_one code "$CODE_BUNDLE_NAME" "$BUNDLE_CODE_COMPRESSION_LEVEL"
    build_ai_payload ai-runtime "$AI_BUNDLE_NAME" "$BUNDLE_AI_COMPRESSION_LEVEL"

    if [[ "$DRY_RUN" == "1" ]]; then
      echo
      echo "Balanced pack preview:"
      echo "  The AI runtime will be compressed at ZIP level $BUNDLE_AI_COMPRESSION_LEVEL."
      echo "  The code + AI runtime will then be divided into near-equal pack parts."
      echo "  Each finished part will stay below ${BUNDLE_UPLOAD_PART_MAX_MIB} MiB."
      echo "  Exact part count is determined from the real compressed archive sizes."
      echo "  Names: ${PACK_NAME}_Part_XX_of_YY.zip"
    else
      "$PYTHON_BIN" "$PACKER" \
        --code-archive "$CODE_ARCHIVE" \
        --runtime-archive "$AI_ARCHIVE" \
        --output-dir "$BUNDLE_OUTPUT_DIR" \
        --pack-name "$PACK_NAME" \
        --max-part-mib "$BUNDLE_UPLOAD_PART_MAX_MIB"

      if [[ "$BUNDLE_KEEP_UNSPLIT_AI_ARCHIVE" != "1" ]]; then
        rm -f "$AI_ARCHIVE"
      fi
    fi
    ;;
  code)
    build_one code "$CODE_BUNDLE_NAME" "$BUNDLE_CODE_COMPRESSION_LEVEL"
    ;;
  ai-runtime)
    build_ai_payload ai-runtime "$AI_BUNDLE_NAME" "$BUNDLE_AI_COMPRESSION_LEVEL"
    ;;
  product)
    build_ai_payload product "Aaron_Sound_Sorter_Product_${STAMP}" "$BUNDLE_PRODUCT_COMPRESSION_LEVEL"
    ;;
esac

if [[ "$DRY_RUN" != "1" ]]; then
  echo
  echo "Bundle build complete."
  echo "Output directory: $BUNDLE_OUTPUT_DIR"
  echo "Output mode: $BUNDLE_OUTPUT_MODE"
  if [[ "$BUNDLE_OUTPUT_MODE" == "split" ]]; then
    echo "Standalone code archive: ${CODE_BUNDLE_NAME}.zip"
    echo "Balanced end-user pack: ${PACK_NAME}_Part_XX_of_YY.zip"
    echo "Maximum pack-part size: ${BUNDLE_UPLOAD_PART_MAX_MIB} MiB"
    echo "Unsplit AI archive retained: $([[ "$BUNDLE_KEEP_UNSPLIT_AI_ARCHIVE" == "1" ]] && echo yes || echo no)"
  fi
  echo "Neural model snapshots included: $([[ "$BUNDLE_INCLUDE_NEURAL_MODELS" == "0" ]] && echo no || echo yes)"
  echo
  find "$BUNDLE_OUTPUT_DIR" -maxdepth 1 -type f \
    \( -name "${CODE_BUNDLE_NAME}.zip" \
       -o -name "${AI_BUNDLE_NAME}.zip" \
       -o -name "${PACK_NAME}_Part_*_of_*.zip" \
       -o -name "Aaron_Sound_Sorter_Product_${STAMP}.zip" \) \
    -exec ls -lh {} \; | sort

  if [[ "$(uname -s)" == "Darwin" ]] && command -v open >/dev/null 2>&1; then
    open "$BUNDLE_OUTPUT_DIR" || true
  fi
fi
