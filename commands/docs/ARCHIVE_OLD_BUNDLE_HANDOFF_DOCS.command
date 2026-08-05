#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null)}"
cd "$PROJECT_ROOT"
ARCHIVE_DIR="docs/archive/old_bundle_docs"
mkdir -p "$ARCHIVE_DIR"
shopt -s nullglob
moved=0
for path in \
  HANDOFF_*.md \
  README_*BUNDLE*.md \
  README_ARCHITECT_*.md \
  README_PARTIAL_*.md \
  README_LOCKED_SMOKE_*.md \
  *_HANDOFF*.md \
  *_REPAIR*.md; do
  if [[ ! -f "$path" ]]; then
    continue
  fi
  if [[ "$path" == "README.md" || "$path" == "AI_READ_THIS_FIRST.md" ]]; then
    continue
  fi
  base="$(basename "$path")"
  dest="$ARCHIVE_DIR/$base"
  if [[ -e "$dest" ]]; then
    stamp="$(date +%Y%m%d_%H%M%S)"
    dest="$ARCHIVE_DIR/${base%.md}_$stamp.md"
  fi
  mv "$path" "$dest"
  echo "Archived $base -> $dest"
  moved=$((moved + 1))
done
shopt -u nullglob

echo "Archived $moved old bundle/handoff doc(s)."
if [[ "$(uname -s)" == "Darwin" ]] && command -v open >/dev/null 2>&1; then
  open "$ARCHIVE_DIR" || true
fi
