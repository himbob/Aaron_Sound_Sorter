#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-}"
if [ -z "$PROJECT_ROOT" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$PROJECT_ROOT/.venv_phase4/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv_phase4/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi
FAKE_OPEN_DIR="$(mktemp -d)"
trap 'rm -rf "$FAKE_OPEN_DIR"' EXIT
printf '#!/usr/bin/env bash\nexit 0\n' > "$FAKE_OPEN_DIR/open"
chmod +x "$FAKE_OPEN_DIR/open"
CASES=(
  kick_clean_cs_ne_monroe
  snare_clean_jackbaby
  clap_clean_clap2
  drum_loop_full_95
  bass_loop_way_it_is
  strings_loop_77_ebm
  synth_loop_05_emn
)
for case_id in "${CASES[@]}"; do
  echo "================================================================"
  echo "RUNNING ACCEPTANCE CASE: $case_id"
  echo "================================================================"
  PROJECT_ROOT="$PROJECT_ROOT" PYTHON_BIN="$PYTHON_BIN" PATH="$FAKE_OPEN_DIR:$PATH" \
    timeout -s KILL 180s ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command --case-id "$case_id"
  echo
done
echo "V31.138 acceptance spot checks completed."
