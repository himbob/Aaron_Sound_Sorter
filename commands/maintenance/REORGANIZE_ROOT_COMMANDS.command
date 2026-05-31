#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/Volumes/T9/testbed/Aaron_Sound_Sorter}"
cd "$PROJECT_ROOT"
TARGET_DIR="commands/legacy_root_commands"
mkdir -p "$TARGET_DIR"
cat > "$TARGET_DIR/README.md" <<'README'
# Legacy root commands

These `.command` files used to live in the project root. They were moved here to keep the root folder readable for AI handoff and human maintenance.

Nothing in this folder should be treated as the current release gate unless `AI_READ_THIS_FIRST.md` says so.

Current required smoke gate for behavior-changing sorter work:

```bash
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE.command
```
README

moved=0
shopt -s nullglob
for path in RUN_*.command *.command; do
  if [[ ! -f "$path" ]]; then
    continue
  fi
  # Do not move macOS command files that already live under commands/ because we are in root only.
  base="$(basename "$path")"
  dest="$TARGET_DIR/$base"
  if [[ -e "$dest" ]]; then
    stamp="$(date +%Y%m%d_%H%M%S)"
    dest="$TARGET_DIR/${base%.command}_$stamp.command"
  fi
  mv "$path" "$dest"
  chmod +x "$dest" || true
  echo "Moved root command: $base -> $dest"
  moved=$((moved + 1))
done
shopt -u nullglob

echo "Moved $moved root command file(s)."
if [[ "$(uname -s)" == "Darwin" ]] && command -v open >/dev/null 2>&1; then
  open "$PROJECT_ROOT" || true
fi
