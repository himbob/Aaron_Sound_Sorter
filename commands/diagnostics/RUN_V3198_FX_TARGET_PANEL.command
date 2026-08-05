#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
find . -name '._*' -type f -delete
find . -name '.DS_Store' -type f -delete
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="_reports/v3198_fx_target_panel/run_${STAMP}"
INPUT_DIR="$RUN_DIR/input"
CASE_DIR="$RUN_DIR/case_outputs"
UPLOAD_DIR="$RUN_DIR/upload_back"
mkdir -p "$INPUT_DIR" "$CASE_DIR" "$UPLOAD_DIR"
if [ -x ".venv_phase4/bin/python" ]; then
  PYTHON=".venv_phase4/bin/python"
elif [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python3"
fi
export PYTHONPATH="src"
$PYTHON - <<'PY'
from pathlib import Path
import shutil
import zipfile

project = Path.cwd()
input_dir = sorted((project / "_reports/v3198_fx_target_panel").glob("run_*/input"))[-1]
zip_candidates = [
    project / "FX_Aaron2.zip",
    Path("/path/to/sample-library/FX_Aaron2.zip"),
    Path("/mnt/data/FX_Aaron2.zip"),
]
want = {
    "Brass_Saxophone_RnB_Multi_Instrument_F_Minor_80BPM.wav",
    "C'est_la_vie_Saxophone_Cmin_150Bpm.wav",
    "AA_JBL_86bpm_Cm_Sax_Loop_12.wav",
    "Riser Short Effect.wav",
    "MS_O_01_Outlaw_Fx Police_D#minor_91bpm_Wet.wav",
    "GS_Synth_Gangsta_Lead_G#min_97bpm.wav",
}
for zip_path in zip_candidates:
    if not zip_path.exists():
        continue
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            base = Path(name).name
            if base in want:
                target = input_dir / base
                if not target.exists():
                    target.write_bytes(zf.read(name))
                    print(f"EXTRACTED: {base}")
    break

for root in [project, Path("/path/to/sample-library"), Path("/mnt/data")]:
    if not root.exists():
        continue
    for p in root.rglob("*.wav"):
        if p.name in {"AA_TSS_D_RUMBLER_808_BASS.wav", "boom_eval_108640.wav", "Clap 17.wav", "clipped_sub_hit.wav"}:
            target = input_dir / p.name
            if not target.exists():
                shutil.copy2(p, target)
                print(f"COPIED: {p.name}")

print("Input files:")
for p in sorted(input_dir.glob("*.wav")):
    print(f"  {p.name}")
PY
SUMMARY="$RUN_DIR/v3198_target_panel_summary.csv"
echo "file,folder_path,consensus_status" > "$SUMMARY"
for f in "$INPUT_DIR"/*.wav; do
  [ -e "$f" ] || continue
  base="$(basename "$f")"
  slug="$(echo "$base" | tr -cs 'A-Za-z0-9_.-' '_' | cut -c1-90)"
  out="$CASE_DIR/$slug"
  mkdir -p "$out"
  echo "=== Sorting $base"
  "$PYTHON" Aaron_Sound_Sorter.py sort "$f" "$out" >/dev/null
  "$PYTHON" - <<PY
import csv, pathlib, sys
csv.field_size_limit(sys.maxsize)
base = '''$base'''
manifest = pathlib.Path('''$out''') / 'Aaron_Sorted_Sounds_manifest.csv'
row = list(csv.DictReader(manifest.open()))[0]
with open('''$SUMMARY''', 'a', newline='') as fh:
    csv.writer(fh).writerow([base, row.get('folder_path', ''), row.get('consensus_status', '')])
print(f"{base} -> {row.get('folder_path', '')} [{row.get('consensus_status', '')}]")
PY
done
zip -qr "$UPLOAD_DIR/UPLOAD_BACK_${STAMP}_v3198_fx_target_panel.zip" "$RUN_DIR" -x "*/case_outputs/*/Aaron_Sorted_Sounds.zip"
open "$RUN_DIR"
echo "Upload this if requested:"
echo "$UPLOAD_DIR/UPLOAD_BACK_${STAMP}_v3198_fx_target_panel.zip"
