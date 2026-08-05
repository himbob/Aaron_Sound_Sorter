#!/bin/bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null)}"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$PROJECT_ROOT/.venv_phase4/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv_phase4/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi
export PYTHONPATH="$PROJECT_ROOT/src"

"$PYTHON_BIN" -m pytest -q \
  tests/test_direct_body_voice_and_low_drum_loop_policy.py \
  tests/test_hierarchical_abstaining_arbitration_v31_99.py \
  tests/test_v31110_synth_pad_sax_overreach.py \
  tests/test_v3198_fx_pitched_false_positive_and_baby_recall.py \
  tests/test_physics_voter_piano_struck_identity.py \
  tests/test_v31116_synth_pad_keys_decoy_guard.py \
  tests/test_v31101_instrument_subpanels.py \
  tests/test_v31108_non_voice_tonal_loop_guard.py \
  tests/test_v31118_calibrated_panel_authority_firewall.py \
  tests/test_v31119_panel_source_specific_authority_split.py \
  tests/test_consensus_concrete_fx_gate_regressions.py \
  tests/test_claim_arbiter_architecture.py \
  tests/test_claim_arbiter_real_panel_surrogates.py

"$PROJECT_ROOT/commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command"

for case_id in \
  piano_loop_key_c \
  synth_lead_k_dot_fluteish \
  police_fx_siren_not_piano_or_sax \
  strings_loop_77_ebm \
  wet_sax_hiphoptapes_no_voice \
  electric_keys_ews_not_sax_or_guitar \
  synth_loop_05_emn \
  piano_loop_dhb_vintage \
  electric_piano_not_sax_ws2_101_fm; do
  "$PYTHON_BIN" "$PROJECT_ROOT/tools/locked_smoke_acceptance.py" run --project-root "$PROJECT_ROOT" --case-id "$case_id"
done

RUN_DIR_FILE="$PROJECT_ROOT/_reports/locked_smoke_acceptance/latest_run_path.txt"
if [ -f "$RUN_DIR_FILE" ]; then
  RUN_DIR="$(cat "$RUN_DIR_FILE")"
  open "$RUN_DIR" >/dev/null 2>&1 || true
fi
