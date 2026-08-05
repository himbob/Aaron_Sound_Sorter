#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/path/to/Aaron_Sound_Sorter"
INPUT="${1:-}"
if [ -z "$INPUT" ]; then
  echo "Usage: $0 /path/to/problem.wav"
  echo
  echo "Example:"
  echo "  $0 /tmp/RHSH_Saxophone_Ensemble_05_keyC_89bpm.wav"
  exit 1
fi

if [ ! -f "$INPUT" ]; then
  echo "ERROR: input file not found: $INPUT"
  exit 1
fi

cd "$PROJECT_ROOT"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT="/tmp/aaron_wetness_harmonic_core_test_$RUN_ID"
rm -rf "$OUT"
mkdir -p "$OUT"

PYTHONPATH=src python3 Aaron_Sound_Sorter.py sort \
  "$INPUT" \
  "$OUT" \
  --brain ./stage4_folder_brain.json \
  --core-baby-brain ./stage4_folder_brain_core_baby.json \
  --spread-baby-brain ./stage4_folder_brain_spread_baby.json \
  --outlier-baby-brain ./stage4_folder_brain_outlier_baby.json \
  --no-zip

MANIFEST="$OUT/Aaron_Sorted_Sounds_manifest.csv"
echo
echo "Manifest: $MANIFEST"
echo
echo "Key columns:"
python3 - "$MANIFEST" <<'PY'
import csv, sys
csv.field_size_limit(sys.maxsize)
path = sys.argv[1]
with open(path, newline='', encoding='utf-8', errors='replace') as fh:
    row = next(csv.DictReader(fh))
keys = [
    'folder_path','consensus_status','decision_reason','full_brain_vote_1',
    'core_baby_vote_1','spread_baby_vote_1','outlier_baby_vote_1',
    'harmonic_core_baby_vote_1','harmonic_spread_baby_vote_1','harmonic_outlier_baby_vote_1',
    'wetness_score','harmonic_core_recall_enabled','physics_vote_1','shape_vote'
]
for key in keys:
    print(f"{key}: {row.get(key,'')}")
PY

open "$OUT" 2>/dev/null || true
