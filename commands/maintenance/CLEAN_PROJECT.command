#!/bin/zsh
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null)}"
cd "$PROJECT_ROOT"
python3 tools/cleanup_project.py --project-root "$PROJECT_ROOT" --mode standard --apply
