#!/usr/bin/env bash
set -euo pipefail

# Aaron Sound Sorter bundle creator.
#
# Default mode:
#   AI handoff bundle with code, tests, tools, commands, docs, active root brain
#   JSONs, and every audio fixture that currently exists under tests/.
#
# Important:
#   - Includes tests/acceptance/locked_smoke_v1/samples/*
#   - Includes tests/regression_audio/* if that folder exists
#   - Includes any tests/**/samples/* folder
#   - Includes all real, non-symlink audio files under tests/
#   - Does not follow or include symbolic-link files or folders
#   - Does NOT fail just because tests/regression_audio is missing.
#     Use REQUIRE_REGRESSION_AUDIO=1 if you want missing regression_audio to fail.
#
# Use CODE_ONLY=1 when you intentionally want no brains and no test audio.
# Use INCLUDE_TEST_AUDIO=0 when you intentionally want no test audio.
# Use REQUIRE_REGRESSION_AUDIO=1 when regression_audio must exist and be staged.
# Use SKIP_CLEAN=1 to skip make clean-for-bundle.

PROJECT_ROOT="${PROJECT_ROOT:-/Volumes/T9/testbed/Aaron_Sound_Sorter}"
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

CODE_ONLY="${CODE_ONLY:-0}"
EXCLUDE_BRAINS="${EXCLUDE_BRAINS:-0}"
INCLUDE_TEST_AUDIO="${INCLUDE_TEST_AUDIO:-1}"
INCLUDE_ACCEPTANCE_AUDIO="${INCLUDE_ACCEPTANCE_AUDIO:-$INCLUDE_TEST_AUDIO}"
REQUIRE_REGRESSION_AUDIO="${REQUIRE_REGRESSION_AUDIO:-0}"
SKIP_CLEAN="${SKIP_CLEAN:-0}"

if [[ "$CODE_ONLY" == "1" ]]; then
  EXCLUDE_BRAINS=1
  INCLUDE_TEST_AUDIO=0
  INCLUDE_ACCEPTANCE_AUDIO=0
  REQUIRE_REGRESSION_AUDIO=0
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
  --no-links
  --delete
  --prune-empty-dirs
)

# These include rules must come before the broad samples/ and *.wav excludes.
# rsync uses the first matching rule.
if [[ "$INCLUDE_TEST_AUDIO" == "1" ]]; then
  RSYNC_ARGS+=(
    --include='/tests/'
    --include='/tests/**/'
    --include='/tests/**/samples/'
    --include='/tests/**/samples/***'
    --include='/tests/regression_audio/'
    --include='/tests/regression_audio/***'
    --include='/tests/**/*.wav'
    --include='/tests/**/*.aif'
    --include='/tests/**/*.aiff'
    --include='/tests/**/*.flac'
    --include='/tests/**/*.ogg'
    --include='/tests/**/*.mp3'
    --include='/tests/**/*.m4a'
    --include='/tests/**/*.aac'
    --include='/tests/**/*.au'
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

  # Huge data/training/sample roots.
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

  # General audio files are excluded outside tests/ audio includes above.
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

# Belt-and-suspenders: make sure no symbolic links are left in staging.
find "$STAGED_PROJECT" -type l -delete

# Clean macOS metadata in staging before zipping.
find "$STAGED_PROJECT" -name '._*' -type f -delete
find "$STAGED_PROJECT" -name '.DS_Store' -type f -delete

AUDIO_EXTS_PY="{'.wav', '.aif', '.aiff', '.flac', '.ogg', '.au', '.mp3', '.m4a', '.aac'}"

ACCEPTANCE_EXPECTED="$STAGED_PROJECT/tests/acceptance/locked_smoke_v1/expected_results.json"
ACCEPTANCE_SAMPLES="$STAGED_PROJECT/tests/acceptance/locked_smoke_v1/samples"
ACCEPTANCE_FIXTURE_COUNT="0"
ACCEPTANCE_EXPECTED_COUNT="0"
ACCEPTANCE_MISSING_COUNT="0"

TEST_AUDIO_SOURCE_COUNT="0"
TEST_AUDIO_STAGED_COUNT="0"
TEST_AUDIO_MISSING_COUNT="0"
REGRESSION_AUDIO_SOURCE_COUNT="0"
REGRESSION_AUDIO_STAGED_COUNT="0"
REGRESSION_AUDIO_MISSING_COUNT="0"
REGRESSION_AUDIO_STATUS="not_required"

if [[ "$INCLUDE_TEST_AUDIO" == "1" ]]; then
  TEST_AUDIO_VALIDATION_OUTPUT="$(python3 - "$PROJECT_ROOT/tests" "$STAGED_PROJECT/tests" <<'PY'
import sys
from pathlib import Path

source_tests = Path(sys.argv[1])
staged_tests = Path(sys.argv[2])
exts = {'.wav', '.aif', '.aiff', '.flac', '.ogg', '.au', '.mp3', '.m4a', '.aac'}

source_audio = []
if source_tests.exists():
    source_audio = sorted(
        path.relative_to(source_tests)
        for path in source_tests.rglob('*')
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in exts
    )

staged_audio = []
if staged_tests.exists():
    staged_audio = sorted(
        path.relative_to(staged_tests)
        for path in staged_tests.rglob('*')
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in exts
    )

missing = [rel for rel in source_audio if not (staged_tests / rel).exists()]

print(f"test_audio_source={len(source_audio)}")
print(f"test_audio_staged={len(staged_audio)}")
print(f"test_audio_missing={len(missing)}")

if missing:
    print("missing_files=" + "|".join(str(item) for item in missing[:100]))
    raise SystemExit(11)
PY
)" || {
    status=$?
    echo "$TEST_AUDIO_VALIDATION_OUTPUT"
    echo
    echo "ERROR: Bundle would be incomplete. Some audio files under tests/ were not staged." >&2
    exit "$status"
  }

  echo "$TEST_AUDIO_VALIDATION_OUTPUT"
  TEST_AUDIO_SOURCE_COUNT="$(echo "$TEST_AUDIO_VALIDATION_OUTPUT" | awk -F= '/^test_audio_source=/{print $2}')"
  TEST_AUDIO_STAGED_COUNT="$(echo "$TEST_AUDIO_VALIDATION_OUTPUT" | awk -F= '/^test_audio_staged=/{print $2}')"
  TEST_AUDIO_MISSING_COUNT="$(echo "$TEST_AUDIO_VALIDATION_OUTPUT" | awk -F= '/^test_audio_missing=/{print $2}')"
fi

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
        if (
            path.is_file()
            and not path.is_symlink()
            and path.suffix.lower() in {'.wav', '.aif', '.aiff', '.flac', '.ogg', '.au'}
        ):
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
      echo "Fix the source tests/acceptance/locked_smoke_v1/samples folder or run INCLUDE_TEST_AUDIO=0 only for non-acceptance code bundles." >&2
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

if [[ "$INCLUDE_TEST_AUDIO" == "1" ]]; then
  REGRESSION_AUDIO_SOURCE="$PROJECT_ROOT/tests/regression_audio"
  REGRESSION_AUDIO_STAGED="$STAGED_PROJECT/tests/regression_audio"

  REGRESSION_VALIDATION_OUTPUT="$(python3 - "$REGRESSION_AUDIO_SOURCE" "$REGRESSION_AUDIO_STAGED" "$REQUIRE_REGRESSION_AUDIO" <<'PY'
import sys
from pathlib import Path

source_dir = Path(sys.argv[1])
staged_dir = Path(sys.argv[2])
required = sys.argv[3] == "1"
exts = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".au", ".mp3", ".m4a", ".aac"}

if not source_dir.exists():
    print("regression_source=0")
    print("regression_staged=0")
    print("regression_missing=0")
    print("regression_status=source_folder_missing")
    if required:
        raise SystemExit(14)
    raise SystemExit(0)

source_audio = sorted(
    path.relative_to(source_dir)
    for path in source_dir.rglob("*")
    if path.is_file()
    and not path.is_symlink()
    and path.suffix.lower() in exts
)

staged_audio = []
if staged_dir.exists():
    staged_audio = sorted(
        path.relative_to(staged_dir)
        for path in staged_dir.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in exts
    )

missing = [rel for rel in source_audio if not (staged_dir / rel).exists()]

print(f"regression_source={len(source_audio)}")
print(f"regression_staged={len(staged_audio)}")
print(f"regression_missing={len(missing)}")
print("regression_status=ok" if not missing else "regression_status=missing_files")

if required and not source_audio:
    print("missing_files=tests/regression_audio exists but contains no audio fixtures")
    raise SystemExit(15)

if missing:
    print("missing_files=" + "|".join(str(item) for item in missing[:100]))
    raise SystemExit(13)
PY
)" || {
    status=$?
    echo "$REGRESSION_VALIDATION_OUTPUT"
    echo
    echo "ERROR: Bundle would be incomplete. tests/regression_audio is required but missing or not fully staged." >&2
    echo "Either restore tests/regression_audio or run without REQUIRE_REGRESSION_AUDIO=1." >&2
    exit "$status"
  }

  echo "$REGRESSION_VALIDATION_OUTPUT"
  REGRESSION_AUDIO_SOURCE_COUNT="$(echo "$REGRESSION_VALIDATION_OUTPUT" | awk -F= '/^regression_source=/{print $2}')"
  REGRESSION_AUDIO_STAGED_COUNT="$(echo "$REGRESSION_VALIDATION_OUTPUT" | awk -F= '/^regression_staged=/{print $2}')"
  REGRESSION_AUDIO_MISSING_COUNT="$(echo "$REGRESSION_VALIDATION_OUTPUT" | awk -F= '/^regression_missing=/{print $2}')"
  REGRESSION_AUDIO_STATUS="$(echo "$REGRESSION_VALIDATION_OUTPUT" | awk -F= '/^regression_status=/{print $2}')"
fi

# Write bundle metadata inside the bundle so the receiving AI knows what it got.
cat > "$STAGED_PROJECT/BUNDLE_CONTENTS_README.txt" <<META
Aaron Sound Sorter bundle
Created: $(date)
Mode: $MODE
Project root source: $PROJECT_ROOT
Includes active root brain JSONs: $([[ "$EXCLUDE_BRAINS" == "1" ]] && echo no || echo yes)
Includes test audio fixtures: $INCLUDE_TEST_AUDIO
Locked smoke expected cases: $ACCEPTANCE_EXPECTED_COUNT
Locked smoke audio fixtures included: $ACCEPTANCE_FIXTURE_COUNT
Locked smoke missing fixtures: $ACCEPTANCE_MISSING_COUNT
All non-symlink test audio source fixtures: $TEST_AUDIO_SOURCE_COUNT
All non-symlink test audio fixtures included: $TEST_AUDIO_STAGED_COUNT
All test audio missing fixtures: $TEST_AUDIO_MISSING_COUNT
Regression non-symlink audio source fixtures: $REGRESSION_AUDIO_SOURCE_COUNT
Regression non-symlink audio fixtures included: $REGRESSION_AUDIO_STAGED_COUNT
Regression audio missing fixtures: $REGRESSION_AUDIO_MISSING_COUNT
Regression audio status: $REGRESSION_AUDIO_STATUS
Regression audio required: $REQUIRE_REGRESSION_AUDIO
Cleaned before bundling: $([[ "$SKIP_CLEAN" == "1" ]] && echo no || echo yes)

Required first read:
- AI_READ_THIS_FIRST.md

Default purpose:
- AI handoff bundle for code review, QA tooling, and architecture work.
- Includes code, tests, commands, tools, docs, active root brain JSONs, and real non-symlink test audio fixtures that exist under tests/.
- A receiving AI must be able to run ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE.command without Aaron separately adding fixtures.

Excluded by design:
- training folders
- large sample folders
- generated reports
- real sort outputs
- virtual environments
- git metadata
- large archives
- symbolic-link files and folders
- general audio files outside tests/
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
Includes test audio fixtures: $INCLUDE_TEST_AUDIO
Locked smoke expected cases: $ACCEPTANCE_EXPECTED_COUNT
Locked smoke audio fixtures included: $ACCEPTANCE_FIXTURE_COUNT
Locked smoke missing fixtures: $ACCEPTANCE_MISSING_COUNT
All non-symlink test audio source fixtures: $TEST_AUDIO_SOURCE_COUNT
All non-symlink test audio fixtures included: $TEST_AUDIO_STAGED_COUNT
All test audio missing fixtures: $TEST_AUDIO_MISSING_COUNT
Regression non-symlink audio source fixtures: $REGRESSION_AUDIO_SOURCE_COUNT
Regression non-symlink audio fixtures included: $REGRESSION_AUDIO_STAGED_COUNT
Regression audio missing fixtures: $REGRESSION_AUDIO_MISSING_COUNT
Regression audio status: $REGRESSION_AUDIO_STATUS

When extracted, it creates:
  Aaron_Sound_Sorter/

Useful commands:
  ./commands/bundle/bundle.sh                              # AI handoff, includes brains, current test audio, cleans first
  SKIP_CLEAN=1 ./commands/bundle/bundle.sh                 # AI handoff, skip clean-for-bundle
  CODE_ONLY=1 ./commands/bundle/bundle.sh                  # source-only, excludes root brains and test audio
  EXCLUDE_BRAINS=1 ./commands/bundle/bundle.sh             # excludes root brains but keeps test audio unless CODE_ONLY=1
  INCLUDE_TEST_AUDIO=0 ./commands/bundle/bundle.sh         # excludes all test audio fixtures
  REQUIRE_REGRESSION_AUDIO=1 ./commands/bundle/bundle.sh   # fails if tests/regression_audio is missing or incomplete

DONE

if [[ "$(uname -s)" == "Darwin" ]] && command -v open >/dev/null 2>&1; then
  open "$BUNDLE_OUTPUT_DIR" || true
fi
