#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_DIR="${AARON_NEURAL_VENV:-$PROJECT_ROOT/_reports/neural_audio/runtime/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

cd "$PROJECT_ROOT"
echo "Project: $PROJECT_ROOT"
echo "Neural venv: $VENV_DIR"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -e '.[neural]'

"$VENV_DIR/bin/python" - <<'PY'
import numpy
import sklearn
import torch
import transformers
print("Neural lab dependencies are installed.")
print("numpy", numpy.__version__)
print("scikit-learn", sklearn.__version__)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
PY
