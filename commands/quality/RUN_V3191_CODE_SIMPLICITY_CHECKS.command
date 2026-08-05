#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null)}"
FX_ZIP="${FX_ZIP:-/path/to/sample-library/FX_Aaron2.zip}"
REPORT_ROOT="$PROJECT_ROOT/reports/v3191_code_simplicity_checks"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$REPORT_ROOT/run_$STAMP"
SAMPLE_DIR="$RUN_DIR/selected_fx_samples"
OUT_DIR="$RUN_DIR/one_by_one_outputs"
RESULTS_CSV="$RUN_DIR/one_by_one_results.csv"

mkdir -p "$SAMPLE_DIR" "$OUT_DIR"
cd "$PROJECT_ROOT"

echo "Cleaning macOS metadata files..."
find . -name '._*' -type f -delete
find . -name '.DS_Store' -type f -delete

python3 -S -m compileall -q Aaron_Sound_Sorter.py src tests
python3 tools/audit_no_source_name_sorting.py --project-root .
python3 -m pytest -q \
  tests/test_claim_arbiter_architecture.py \
  tests/test_no_source_name_sorting_invariant.py \
  tests/test_brain_competence_policy.py \
  tests/test_profile_candidate_claim_producer.py \
  tests/test_v3184_deep_voter_and_baby_recall.py \
  tests/test_v3187_architecture_doc_and_lane_competence.py \
  tests/test_v3188_fx_run_regression_safety.py \
  tests/test_v3189_real_audio_architecture_regressions.py \
  tests/test_role_aware_brain_ensemble_policy.py \
  tests/test_brain_ensemble_audit_reporting.py
python3 Aaron_Sound_Sorter.py self-test

python3 - <<PY
from pathlib import Path
from zipfile import ZipFile

fx_zip = Path("$FX_ZIP")
sample_dir = Path("$SAMPLE_DIR")
selected = {
    "sax": "FX_Aaron2/Loop/Brass_Woodwind/AA_JBL_74bpm_Am_Sax_Loop_13.wav",
    "drum_loop": "FX_Aaron2/Premium Bounce/Loop/Drums/Drum_Full_06_95bpm.wav",
    "vocal_loop": "FX_Aaron2/Premium Bounce/Loop/Vocals/Halfway_Vocal_keyC#m_82BPM.wav",
    "kick": "FX_Aaron2/Premium Bounce/One_Shot/Drums/CS_NE_Kick_OneShot_Monroe.wav",
}
with ZipFile(fx_zip) as zf:
    for label, member in selected.items():
        target = sample_dir / f"{label}_{Path(member).name}"
        target.write_bytes(zf.read(member))
        print(target)
PY

echo "label,source_file,folder_path,consensus_status" > "$RESULTS_CSV"
for sample_path in "$SAMPLE_DIR"/*.wav; do
  label="$(basename "$sample_path" | cut -d_ -f1)"
  sample_out="$OUT_DIR/${label}_out"
  rm -rf "$sample_out"
  python3 Aaron_Sound_Sorter.py sort "$sample_path" "$sample_out" --brain stage4_folder_brain.json --no-zip >/dev/null 2>&1
  python3 - <<PY >> "$RESULTS_CSV"
import csv
import sys
from pathlib import Path
csv.field_size_limit(sys.maxsize)
manifest = Path("$sample_out") / "Aaron_Sorted_Sounds_manifest.csv"
row = next(csv.DictReader(manifest.open()))
print(f"$label,$sample_path,{row['folder_path']},{row['consensus_status']}")
PY
done

echo "Cleaning macOS metadata files before zipping results..."
find "$RUN_DIR" -name '._*' -type f -delete
find "$RUN_DIR" -name '.DS_Store' -type f -delete
(cd "$RUN_DIR" && zip -qr "v3191_code_simplicity_results_$STAMP.zip" . -x '*.zip' '*.wav' '*.aif' '*.aiff' '*/._*' '._*' '*/.DS_Store' '.DS_Store' '*/__MACOSX/*' '__MACOSX/*')
if [[ "$(uname -s)" == "Darwin" ]]; then
  open "$RUN_DIR" 2>/dev/null || true
fi

echo "DONE"
echo "Report folder: $RUN_DIR"
echo "Upload-back ZIP: $RUN_DIR/v3191_code_simplicity_results_$STAMP.zip"
