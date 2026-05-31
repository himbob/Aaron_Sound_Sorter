#!/usr/bin/env bash
set -euo pipefail

# Aaron Sound Sorter bundle creator.
# Default mode is an AI handoff bundle: code + tests + tools + commands + docs +
# active root brain JSONs + small checked-in test audio fixtures.
# Use CODE_ONLY=1 or EXCLUDE_BRAINS=1 when you explicitly need no brains.
# Use INCLUDE_ACCEPTANCE_AUDIO=0 only when you intentionally do NOT want the
# locked smoke/regression WAV fixtures. The default AI handoff bundle must be
# runnable by another AI without Aaron adding the sample fixtures by hand.
# By default this runs make clean-for-bundle first so old reports/runs/root ZIPs
# do not get copied into AI handoff bundles. Use SKIP_CLEAN=1 to skip that step.

PROJECT_ROOT="${PROJECT_ROOT:-/Volumes/T9/testbed/Aaron_Sound_Sorter}"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

CODE_ONLY="${CODE_ONLY:-0}"
EXCLUDE_BRAINS="${EXCLUDE_BRAINS:-0}"
INCLUDE_ACCEPTANCE_AUDIO="${INCLUDE_ACCEPTANCE_AUDIO:-1}"
SKIP_CLEAN="${SKIP_CLEAN:-0}"

if [[ "$CODE_ONLY" == "1" ]]; then
  EXCLUDE_BRAINS=1
  INCLUDE_ACCEPTANCE_AUDIO=0
fi

MODE="AI_HANDOFF_WITH_BRAINS"
if [[ "$EXCLUDE_BRAINS" == "1" ]]; then
  MODE="CODE_ONLY_NO_BRAINS"
fi

if [[ "$SKIP_CLEAN" != "1" && -f "$PROJECT_ROOT/Makefile" && -f "$PROJECT_ROOT/tools/cleanup_project.py" ]]; then
  echo "Running make clean-for-bundle before staging bundle..."
  (cd "$PROJECT_ROOT" && make clean-for-bundle)
else
  echo "Skipping clean-for-bundle."
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_NAME="${BUNDLE_NAME:-Aaron_Sound_Sorter_${MODE}_${STAMP}}"
BUNDLE_OUTPUT_DIR="${BUNDLE_OUTPUT_DIR:-$PROJECT_ROOT/_reports/bundles}"
ZIP_OUT="$BUNDLE_OUTPUT_DIR/${BUNDLE_NAME}.zip"
STAGE_PARENT="${TMPDIR:-/tmp}"
STAGE_DIR="$STAGE_PARENT/${BUNDLE_NAME}"
STAGED_PROJECT="$STAGE_DIR/Aaron_Sound_Sorter"

mkdir -p "$BUNDLE_OUTPUT_DIR"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGED_PROJECT"

RSYNC_ARGS=(
  -a
  --delete
  --prune-empty-dirs

  # Small checked-in test audio fixtures must be included before broad
  # sample/audio excludes. rsync uses the first matching filter. If these rules
  # come after samples/ or *.wav excludes, the bundle will include pytest files
  # while silently dropping their WAV fixtures.
)

if [[ "$INCLUDE_ACCEPTANCE_AUDIO" == "1" ]]; then
  RSYNC_ARGS+=(
    --include='/tests/acceptance/locked_smoke_v1/samples/'
    --include='/tests/acceptance/locked_smoke_v1/samples/***'
    --include='/tests/regression_audio/'
    --include='/tests/regression_audio/***'
  )
fi

RSYNC_ARGS+=(
  --exclude='.git/'
  --exclude='.github/'
  --exclude='.DS_Store'
  --exclude='**/.DS_Store'
  --exclude='__MACOSX/'
  --exclude='**/._*'
  --exclude='**/__pycache__/'
  --exclude='**/*.pyc'
  --exclude='.pytest_cache/'
  --exclude='.mypy_cache/'
  --exclude='.ruff_cache/'
  --exclude='.coverage'
  --exclude='htmlcov/'
  --exclude='.venv/'
  --exclude='.venv_phase4/'
  --exclude='venv/'
  --exclude='env/'
  --exclude='node_modules/'

  # Huge/generated/runtime folders.
  --exclude='reports/'
  --exclude='_reports/'
  --exclude='_real_sort_tests/'
  --exclude='_backup*/'
  --exclude='_backups/'
  --exclude='_patch_backups/'
  --exclude='_pytest_outputs/'
  --exclude='_pytest_regression_outputs/'
  --exclude='_qa_parent_eligibility_v2/'
  --exclude='Aaron_Sorted_Sounds/'
  --exclude='Aaron_Test_Cases/'
  --exclude='test_output/'
  --exclude='output/'
  --exclude='outputs/'
  --exclude='sorted_output/'
  --exclude='source_cache/'
  --exclude='**/source_cache/'

  # Huge data/training/sample roots. The locked smoke panel samples were already
  # included above and therefore survive these broad excludes.
  --exclude='training/'
  --exclude='_training_data/'
  --exclude='Sorted samples/'
  --exclude='sorted samples/'
  --exclude='samples/'
  --exclude='_balanced_baby_training/'
  --exclude='stage4_brain_family_training/'
  --exclude='stage4_folder_brain_training/'

  # Archive/upload artifacts. No ZIP-in-ZIP handoffs.
  --exclude='**/*.zip'
  --exclude='**/*.tar'
  --exclude='**/*.tar.gz'
  --exclude='**/*.tgz'
  --exclude='**/*.7z'
  --exclude='**/*.rar'

  # Root generated reports/logs.
  --exclude='/tree.out'
  --exclude='/validation.log'
  --exclude='/installed_files.txt'
  --exclude='/REAL_SAMPLE_RETEST_RESULTS_*.csv'
  --exclude='/LOW_REPRESENTATION_LABELS_REPORT_*.csv'
  --exclude='/Aaron_Brain_*.csv'
  --exclude='/Aaron_Sorted_Sounds_manifest.csv'
  --exclude='/Aaron_Sorted_Sounds_summary.txt'

  # General audio files are excluded outside the small checked-in test
  # fixture folders included above.
  --exclude='**/*.wav'
  --exclude='**/*.aif'
  --exclude='**/*.aiff'
  --exclude='**/*.flac'
  --exclude='**/*.mp3'
  --exclude='**/*.ogg'
  --exclude='**/*.m4a'
  --exclude='**/*.aac'
  --exclude='**/*.au'
)

if [[ "$EXCLUDE_BRAINS" == "1" ]]; then
  RSYNC_ARGS+=(
    --exclude='/stage4_folder_brain*.json'
    --exclude='/stage4_*brain*.json'
    --exclude='/v31_brain*.json'
    --exclude='/v*_brain*.json'
    --exclude='/brains*.json'
    --exclude='/brain*.json'
  )
fi

# Runtime state is not project state.
RSYNC_ARGS+=(
  --exclude='/.aaron_last_sort.json'
  --exclude='/.aaron_wizard_state.json'
  --exclude='/.aaron_correction_inbox.json'
)

rsync "${RSYNC_ARGS[@]}" "$PROJECT_ROOT/" "$STAGED_PROJECT/"

# Clean macOS metadata in staging before zipping.
find "$STAGED_PROJECT" -name '._*' -type f -delete
find "$STAGED_PROJECT" -name '.DS_Store' -type f -delete

ACCEPTANCE_EXPECTED="$STAGED_PROJECT/tests/acceptance/locked_smoke_v1/expected_results.json"
ACCEPTANCE_SAMPLES="$STAGED_PROJECT/tests/acceptance/locked_smoke_v1/samples"
ACCEPTANCE_FIXTURE_COUNT="0"
ACCEPTANCE_EXPECTED_COUNT="0"
ACCEPTANCE_MISSING_COUNT="0"

if [[ "$INCLUDE_ACCEPTANCE_AUDIO" == "1" && -f "$ACCEPTANCE_EXPECTED" ]]; then
  VALIDATION_OUTPUT="$(python3 - "$ACCEPTANCE_EXPECTED" "$ACCEPTANCE_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

expected_path = Path(sys.argv[1])
samples_dir = Path(sys.argv[2])
payload = json.loads(expected_path.read_text(encoding='utf-8'))
cases = payload.get('cases', [])
filenames = []
for case in cases:
    filename = str(case.get('filename', '')).strip()
    if filename:
        filenames.append(filename)
missing = [name for name in filenames if not (samples_dir / name).exists()]
actual_audio = []
if samples_dir.exists():
    for path in samples_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {'.wav', '.aif', '.aiff', '.flac', '.ogg', '.au'}:
            actual_audio.append(path.name)
print(f"expected={len(filenames)}")
print(f"actual={len(actual_audio)}")
print(f"missing={len(missing)}")
if missing:
    print("missing_files=" + "|".join(missing))
    raise SystemExit(12)
PY
)" || {
    status=$?
    echo "$VALIDATION_OUTPUT"
    if [[ "$status" == "12" ]]; then
      echo
      echo "ERROR: Bundle would be incomplete. expected_results.json references locked smoke samples that were not staged." >&2
      echo "Fix the source tests/acceptance/locked_smoke_v1/samples folder or run INCLUDE_ACCEPTANCE_AUDIO=0 only for non-acceptance code bundles." >&2
    fi
    exit "$status"
  }
  echo "$VALIDATION_OUTPUT"
  ACCEPTANCE_EXPECTED_COUNT="$(echo "$VALIDATION_OUTPUT" | awk -F= '/^expected=/{print $2}')"
  ACCEPTANCE_FIXTURE_COUNT="$(echo "$VALIDATION_OUTPUT" | awk -F= '/^actual=/{print $2}')"
  ACCEPTANCE_MISSING_COUNT="$(echo "$VALIDATION_OUTPUT" | awk -F= '/^missing=/{print $2}')"
elif [[ "$INCLUDE_ACCEPTANCE_AUDIO" == "1" && ! -f "$ACCEPTANCE_EXPECTED" ]]; then
  echo "Warning: INCLUDE_ACCEPTANCE_AUDIO=1 but no locked smoke expected_results.json was found in staged project."
fi

# Write bundle metadata inside the bundle so the receiving AI knows what it got.
cat > "$STAGED_PROJECT/BUNDLE_CONTENTS_README.txt" <<META
Aaron Sound Sorter bundle
Created: $(date)
Mode: $MODE
Project root source: $PROJECT_ROOT
Includes active root brain JSONs: $([[ "$EXCLUDE_BRAINS" == "1" ]] && echo no || echo yes)
Includes locked smoke/regression audio fixtures: $INCLUDE_ACCEPTANCE_AUDIO
Locked smoke expected cases: $ACCEPTANCE_EXPECTED_COUNT
Locked smoke audio fixtures included: $ACCEPTANCE_FIXTURE_COUNT
Locked smoke missing fixtures: $ACCEPTANCE_MISSING_COUNT
Cleaned before bundling: $([[ "$SKIP_CLEAN" == "1" ]] && echo no || echo yes)

Required first read:
- AI_READ_THIS_FIRST.md

Default purpose:
- AI handoff bundle for code review, QA tooling, and architecture work.
- Includes code, tests, commands, tools, docs, active root brain JSONs, and small QA fixtures.
- A receiving AI must be able to run ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE.command without Aaron separately adding fixtures.

Excluded by design:
- training folders
- large sample folders
- generated reports
- real sort outputs
- virtual environments
- git metadata
- large archives
- general audio files outside checked-in test fixture folders
META

cd "$STAGE_DIR"
rm -f "$ZIP_OUT"
zip -qr "$ZIP_OUT" Aaron_Sound_Sorter \
  -x '*.DS_Store' \
  -x '__MACOSX/*' \
  -x '*/._*'

cat <<DONE

Created bundle:
$ZIP_OUT

Mode: $MODE
Includes root active brain JSONs: $([[ "$EXCLUDE_BRAINS" == "1" ]] && echo no || echo yes)
Includes locked acceptance/regression audio fixtures: $INCLUDE_ACCEPTANCE_AUDIO
Locked smoke expected cases: $ACCEPTANCE_EXPECTED_COUNT
Locked smoke audio fixtures included: $ACCEPTANCE_FIXTURE_COUNT
Locked smoke missing fixtures: $ACCEPTANCE_MISSING_COUNT

When extracted, it creates:
  Aaron_Sound_Sorter/

Useful commands:
  ./commands/bundle/bundle.sh                         # AI handoff, includes brains, smoke fixtures, cleans first
  SKIP_CLEAN=1 ./commands/bundle/bundle.sh            # AI handoff, skip clean-for-bundle
  CODE_ONLY=1 ./commands/bundle/bundle.sh             # source-only, excludes root brains and acceptance audio
  EXCLUDE_BRAINS=1 ./commands/bundle/bundle.sh        # excludes root brains but keeps acceptance audio unless CODE_ONLY=1
  INCLUDE_ACCEPTANCE_AUDIO=0 ./commands/bundle/bundle.sh   # excludes locked smoke and regression audio fixtures

DONE

if [[ "$(uname -s)" == "Darwin" ]] && command -v open >/dev/null 2>&1; then
  open "$BUNDLE_OUTPUT_DIR" || true
fi
