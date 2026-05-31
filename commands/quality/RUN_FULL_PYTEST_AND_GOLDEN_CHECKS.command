#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command
./commands/quality/RUN_ALL_PYTESTS_CHUNKED.command

if [ "${RUN_FX_ONE_BY_ONE:-0}" = "1" ]; then
  ./commands/quality/RUN_FX_AARON2_ONE_BY_ONE_GOLDEN_AUDIT.command
else
  echo "Chunked pytest and no-source-name audit passed. Set RUN_FX_ONE_BY_ONE=1 to run the full FX_Aaron2 one-by-one golden audit."
fi
