#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_ONLY=1 "$SCRIPT_DIR/bundle.sh"
