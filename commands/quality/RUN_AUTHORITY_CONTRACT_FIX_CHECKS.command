#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$PROJECT_ROOT/.venv_phase4/bin/python" ]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv_phase4/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

find "$PROJECT_ROOT" -name '._*' -type f -delete
find "$PROJECT_ROOT" -name '.DS_Store' -type f -delete

"$PYTHON_BIN" -m pytest -q \
  tests/test_v3199_arbiter_authority_contract.py \
  tests/test_v3189_real_audio_architecture_regressions.py \
  tests/test_claim_arbiter_real_panel_surrogates.py \
  tests/test_family_claim_arbitration_matrix.py \
  tests/test_decision_core_v2.py \
  tests/test_parent_eligibility_uploaded_audio.py \
  tests/test_consensus_concrete_fx_gate_regressions.py

./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command
