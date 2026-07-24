#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_DIR="${AARON_NEURAL_VENV:-$PROJECT_ROOT/_reports/neural_audio/runtime/.venv}"
INDEX_DIR="${1:-}"
INPUT_AUDIO="${2:-}"
LEGACY_MANIFEST="${3:-}"
OUTPUT_CSV="${4:-$PROJECT_ROOT/_reports/neural_audio/runs/clap_shadow_diff.csv}"
MODEL_DIR="${AARON_CLAP_MODEL_DIR:-$PROJECT_ROOT/_reports/neural_audio/models/laion_larger_clap_music_and_speech}"
MODEL_REVISION="${AARON_CLAP_MODEL_REVISION:-195c3a3e68faebb3e2088b9a79e79b43ddbda76b}"

if [ -z "$INDEX_DIR" ] || [ -z "$INPUT_AUDIO" ]; then
  echo "Usage: $0 /path/to/index /path/to/audio_or_zip [legacy_manifest.csv] [output.csv]" >&2
  exit 2
fi
mkdir -p "$(dirname "$OUTPUT_CSV")"
ARGS=(
  shadow
  --index "$INDEX_DIR"
  --input "$INPUT_AUDIO"
  --output "$OUTPUT_CSV"
  --provider clap
  --model "$MODEL_DIR"
  --model-source-id "laion/larger_clap_music_and_speech"
  --model-revision "$MODEL_REVISION"
)
if [ -n "$LEGACY_MANIFEST" ]; then
  ARGS+=(--legacy-manifest "$LEGACY_MANIFEST")
fi
cd "$PROJECT_ROOT"
"$VENV_DIR/bin/python" tools/neural_audio_lab.py "${ARGS[@]}"
