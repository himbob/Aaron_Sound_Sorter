#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "Project root:"
echo "  $ROOT"
echo

echo "Compiling Python files..."
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$ROOT/.pytest_cache/pycache}"
python3 -S -m py_compile Aaron_Sound_Sorter.py src/aaron_sound_sorter/*.py

echo
echo "Collecting tests..."
PYTHONPATH=src python3 -m pytest --collect-only -q tests

echo
echo "Running pytest suite..."
PYTHONPATH=src python3 -m pytest -q tests

echo
echo "DONE"
