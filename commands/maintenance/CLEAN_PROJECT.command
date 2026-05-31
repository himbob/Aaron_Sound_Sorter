#!/bin/zsh
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/Volumes/T9/testbed/Aaron_Sound_Sorter}"
cd "$PROJECT_ROOT"
python3 tools/cleanup_project.py --project-root "$PROJECT_ROOT" --mode standard --apply
