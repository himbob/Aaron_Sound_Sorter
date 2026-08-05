#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${AARON_NEURAL_PYTHON:-$PROJECT_ROOT/.venv_neural/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Neural environment is missing. Run commands/neural/INSTALL_NEURAL_LAB.command first." >&2
  exit 2
fi

"$PYTHON_BIN" - "$PROJECT_ROOT" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

root = Path(sys.argv[1]).resolve()
config = json.loads((root / "config/panns_runtime.json").read_text(encoding="utf-8"))

for path_key, url_key, digest_key in (
    ("checkpoint_path", "checkpoint_url", "checkpoint_sha256"),
    ("labels_path", "labels_url", "labels_sha256"),
):
    destination = root / config[path_key]
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = config[digest_key]
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
        print(f"Verified existing {destination.relative_to(root)}")
        continue
    with urllib.request.urlopen(config[url_key]) as response, tempfile.NamedTemporaryFile(
        dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        while block := response.read(1024 * 1024):
            handle.write(block)
    actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if actual != expected:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Checksum mismatch for {destination.name}: {actual}")
    os.replace(temporary, destination)
    print(f"Downloaded and verified {destination.relative_to(root)}")
PY
