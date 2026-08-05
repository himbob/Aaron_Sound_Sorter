#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null)}"
cd "$PROJECT_ROOT"

PYTHONPATH=src .venv_phase4/bin/python tools/locked_smoke_voter_audit.py --project-root "$PROJECT_ROOT" "$@"
