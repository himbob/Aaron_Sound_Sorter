#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/Volumes/T9/testbed/Aaron_Sound_Sorter}"

echo "Installing Aaron Sound Sorter without backup"
echo "Source:  $SOURCE_DIR"
echo "Target:  $PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT"

# Clean macOS metadata before copying and before validation.
find "$SOURCE_DIR" -name '._*' -type f -delete
find "$SOURCE_DIR" -name '.DS_Store' -type f -delete
find "$PROJECT_ROOT" -name '._*' -type f -delete 2>/dev/null || true
find "$PROJECT_ROOT" -name '.DS_Store' -type f -delete 2>/dev/null || true

# Virtual environments must never be checked in or bundled.
rm -rf "$PROJECT_ROOT/.venv_py313" "$PROJECT_ROOT/.venv"

# Keep the root clean. Archive old root handoff/status docs instead of deleting
# them blindly, so nothing useful is lost.
ARCHIVE_DIR="$PROJECT_ROOT/docs/archive/old_bundle_docs"
mkdir -p "$ARCHIVE_DIR"
for doc in \
  ACCEPTANCE_ARCHIVE_REVIEW_20260522.md \
  AI_FIX_SUMMARY_20260526_VOCAL_RISER_ARCHITECTURE.md \
  AI_HANDOFF_V31106_VOICE_SAX_VOTER_FIX.md \
  AI_HANDOFF_V31108_FALSE_VOICE_SAX_FX_REPAIR.md \
  AI_HANDOFF_V31109_STRONG_DRUM_CONSENSUS_FIREWALL.md \
  AI_HANDOFF_V31110_SYNTH_PAD_SAX_OVERREACH.md \
  AI_READ_THIS_FIRST_ACCEPTANCE_APPEND.md \
  AI_STATUS_V31123_V31126_SPEED_AND_PRO_QUALITY.md \
  AI_STATUS_V31127_COMPANY_MAKEFILE_DEPENDENCIES.md \
  AI_STATUS_V31127_MAKEFILE_DEPENDENCY_BOOTSTRAP.md \
  AI_STATUS_V31128_AGENTS_MAKEFILE_QUALITY_GATE.md \
  AI_STATUS_V31129_MAKE_BUNDLE_WORKFLOW.md \
  README_INSTALL_ACCEPTANCE_RUNNER_FILES.md \
  README_V31110_FULL_SOURCE_SAFETY_BUNDLE.md; do
  if [[ -f "$PROJECT_ROOT/$doc" ]]; then
    mv "$PROJECT_ROOT/$doc" "$ARCHIVE_DIR/$doc"
    echo "Archived old root doc: $doc"
  fi
done

rsync -a \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.venv_py313/' \
  --exclude '.venv_phase4/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '_reports/' \
  --exclude '.coverage' \
  --exclude '.DS_Store' \
  --exclude '._*' \
  "$SOURCE_DIR/" "$PROJECT_ROOT/"

find "$PROJECT_ROOT" -name '._*' -type f -delete
find "$PROJECT_ROOT" -name '.DS_Store' -type f -delete
find "$PROJECT_ROOT" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$PROJECT_ROOT" -name '.pytest_cache' -type d -prune -exec rm -rf {} +
find "$PROJECT_ROOT" -name '.mypy_cache' -type d -prune -exec rm -rf {} +
find "$PROJECT_ROOT" -name '.ruff_cache' -type d -prune -exec rm -rf {} +
find "$PROJECT_ROOT" -name '*.pyc' -type f -delete

chmod +x "$PROJECT_ROOT"/commands/**/*.command 2>/dev/null || true
chmod +x "$PROJECT_ROOT"/*.command 2>/dev/null || true

echo "Install complete. Run this next:"
echo "  cd '$PROJECT_ROOT' && make ai-check VENV_DIR=.venv_phase4"

if [[ "$(uname -s)" == "Darwin" ]] && command -v open >/dev/null 2>&1; then
  open "$PROJECT_ROOT" || true
fi
