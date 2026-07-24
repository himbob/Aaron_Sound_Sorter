#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_DIR="${AARON_NEURAL_VENV:-$PROJECT_ROOT/_reports/neural_audio/runtime/.venv}"
MODEL_ID="${AARON_CLAP_MODEL_ID:-laion/larger_clap_music_and_speech}"
MODEL_REVISION="${AARON_CLAP_MODEL_REVISION:-195c3a3e68faebb3e2088b9a79e79b43ddbda76b}"
MODEL_DIR="${AARON_CLAP_MODEL_DIR:-$PROJECT_ROOT/_reports/neural_audio/models/laion_larger_clap_music_and_speech}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Neural environment is missing. Run commands/neural/INSTALL_NEURAL_LAB.command first." >&2
  exit 2
fi

mkdir -p "$MODEL_DIR"
MODEL_ID="$MODEL_ID" MODEL_REVISION="$MODEL_REVISION" MODEL_DIR="$MODEL_DIR" "$VENV_DIR/bin/python" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from transformers import AutoProcessor, ClapModel

model_id = os.environ["MODEL_ID"]
model_revision = os.environ["MODEL_REVISION"]
model_dir = Path(os.environ["MODEL_DIR"]).resolve()
print(f"Downloading reviewed model snapshot: {model_id}@{model_revision}")
processor = AutoProcessor.from_pretrained(model_id, revision=model_revision)
model = ClapModel.from_pretrained(model_id, revision=model_revision)
processor.save_pretrained(model_dir)
model.save_pretrained(model_dir, safe_serialization=True)
weight_path = model_dir / "model.safetensors"
weight_digest = hashlib.sha256()
with weight_path.open("rb") as handle:
    while block := handle.read(1024 * 1024):
        weight_digest.update(block)
weight_sha256 = weight_digest.hexdigest()
(model_dir / "neural_model_manifest.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "source_model_id": model_id,
            "model_revision": model_revision,
            "model_weight_file": weight_path.name,
            "model_weight_sha256": weight_sha256,
            "saved_utc": datetime.now(timezone.utc).isoformat(),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
print(f"Pinned local model: {model_dir}")
print(f"Revision: {model_revision}")
print(f"Weight SHA-256: {weight_sha256}")
PY
