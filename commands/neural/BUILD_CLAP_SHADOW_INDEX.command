#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_DIR="${AARON_NEURAL_VENV:-$PROJECT_ROOT/_reports/neural_audio/runtime/.venv}"
TRAINING_ROOT="${1:-}"
OUTPUT_DIR="${2:-$PROJECT_ROOT/_reports/neural_audio/runs/clap_index}"
HELDOUT_ROOT="${3:-}"
MODEL_DIR="${AARON_CLAP_MODEL_DIR:-$PROJECT_ROOT/_reports/neural_audio/models/laion_larger_clap_music_and_speech}"
MODEL_REVISION="${AARON_CLAP_MODEL_REVISION:-195c3a3e68faebb3e2088b9a79e79b43ddbda76b}"

if [ -z "$TRAINING_ROOT" ]; then
  echo "Usage: $0 /path/to/curated_training_root [output_dir] [heldout_root]" >&2
  exit 2
fi
if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Run commands/neural/INSTALL_NEURAL_LAB.command first." >&2
  exit 2
fi
if [ ! -d "$MODEL_DIR" ]; then
  echo "Pinned CLAP model is missing: $MODEL_DIR" >&2
  echo "Run commands/neural/PREFETCH_CLAP_MODEL.command first." >&2
  exit 2
fi

ARGS=(
  build
  --training-root "$TRAINING_ROOT"
  --output "$OUTPUT_DIR"
  --provider clap
  --model "$MODEL_DIR"
  --model-source-id "laion/larger_clap_music_and_speech"
  --model-revision "$MODEL_REVISION"
)
if [ -n "$HELDOUT_ROOT" ]; then
  ARGS+=(--heldout-root "$HELDOUT_ROOT")
fi
cd "$PROJECT_ROOT"
"$VENV_DIR/bin/python" tools/neural_audio_lab.py "${ARGS[@]}"
