#!/bin/bash
set -u
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null)}"
cd "$PROJECT_ROOT" || { echo "Project root not found: $PROJECT_ROOT"; exit 0; }
python3 -m pytest -q tests/test_uploaded_regression_audio.py
