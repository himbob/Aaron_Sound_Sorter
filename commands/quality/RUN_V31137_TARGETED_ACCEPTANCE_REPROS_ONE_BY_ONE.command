#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-}"
if [ -z "$PROJECT_ROOT" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi
cd "$PROJECT_ROOT"
chmod +x ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command || true
CASES=(
  strings_loop_77_ebm
  synth_loop_05_emn
  wet_reverb_sax_intro_fminor_96
  wet_sax_scy097_no_voice
  electric_keys_ews_not_sax_or_guitar
  vocal_loop_not_sax_phonk_151_cmin
  police_fx_siren_not_piano_or_sax
)
for case_id in "${CASES[@]}"; do
  echo "================================================================"
  echo "RUNNING: $case_id"
  echo "================================================================"
  ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command --case-id "$case_id" --stop-on-fail --heartbeat-seconds 10
  echo
done
cat <<'TXT'
Targeted v31.137 acceptance repros completed.
For full acceptance, run:
  ./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_ONE_BY_ONE.command --heartbeat-seconds 10
TXT
