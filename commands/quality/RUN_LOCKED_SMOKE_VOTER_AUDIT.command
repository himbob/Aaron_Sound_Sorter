#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/Volumes/T9/testbed/Aaron_Sound_Sorter}"
cd "$PROJECT_ROOT"

PYTHONPATH=src .venv_phase4/bin/python tools/locked_smoke_voter_audit.py --project-root "$PROJECT_ROOT" "$@"
