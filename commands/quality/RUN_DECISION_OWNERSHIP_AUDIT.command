#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null)}"
cd "$PROJECT_ROOT"

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 path/to/Aaron_Sorted_Sounds_manifest.csv [more manifests...]"
  echo
  echo "This audits whether decisions were owned by learned memory, old trained brains,"
  echo "static measured contracts, or review."
  exit 2
fi

PYTHON="${PYTHON:-python3}"
"$PYTHON" tools/decision_ownership_audit.py --project-root "$PROJECT_ROOT" "$@"
