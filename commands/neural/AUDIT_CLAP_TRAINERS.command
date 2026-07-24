#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_DIR="${AARON_NEURAL_VENV:-$PROJECT_ROOT/_reports/neural_audio/runtime/.venv}"
TRAINING_ROOT="${1:-}"
OUTPUT_CSV="${2:-$PROJECT_ROOT/_reports/neural_audio/runs/clap_trainer_audit.csv}"
MODEL_DIR="${AARON_CLAP_MODEL_DIR:-$PROJECT_ROOT/_reports/neural_audio/models/laion_larger_clap_music_and_speech}"
MODEL_REVISION="${AARON_CLAP_MODEL_REVISION:-195c3a3e68faebb3e2088b9a79e79b43ddbda76b}"

if [ -z "$TRAINING_ROOT" ]; then
  echo "Usage: $0 /path/to/curated_training_root [output_csv]" >&2
  exit 2
fi
mkdir -p "$(dirname "$OUTPUT_CSV")"
cd "$PROJECT_ROOT"
"$VENV_DIR/bin/python" tools/neural_audio_lab.py audit-trainers \
  --training-root "$TRAINING_ROOT" \
  --output "$OUTPUT_CSV" \
  --provider clap \
  --model "$MODEL_DIR" \
  --model-source-id "laion/larger_clap_music_and_speech" \
  --model-revision "$MODEL_REVISION"
