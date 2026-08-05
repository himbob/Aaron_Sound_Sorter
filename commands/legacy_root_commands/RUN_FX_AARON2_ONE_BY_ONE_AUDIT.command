#!/bin/bash
# Runs FX_Aaron2 one audio file at a time to avoid long full-ZIP timeouts.
# This is an audit command only. It writes reports under _real_sort_tests.
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT" || exit 1
ZIP_CANDIDATES=(
  "$ROOT/FX_Aaron2.zip"
  "/path/to/sample-library/FX_Aaron2.zip"
  "/mnt/data/FX_Aaron2.zip"
)
ZIP=""
for CANDIDATE in "${ZIP_CANDIDATES[@]}"; do
  if [ -f "$CANDIDATE" ]; then
    ZIP="$CANDIDATE"
    break
  fi
done
if [ -z "$ZIP" ]; then
  echo "ERROR: FX_Aaron2.zip not found. Put it in the project root or /path/to/sample-library."
else
  STAMP="$(date +%Y%m%d_%H%M%S)"
  WORK="$ROOT/_real_sort_tests/fx_aaron2_one_by_one_$STAMP"
  EXTRACT="$WORK/extracted"
  OUTROOT="$WORK/outputs"
  REPORT="$WORK/fx_aaron2_one_by_one_report.csv"
  mkdir -p "$EXTRACT" "$OUTROOT"
  echo "Extracting audio-only members from: $ZIP"
  python3 - <<PY
import zipfile, pathlib, shutil
zip_path=pathlib.Path(r'''$ZIP''')
out=pathlib.Path(r'''$EXTRACT''')
audio={'.wav','.aif','.aiff','.flac','.ogg','.au'}
with zipfile.ZipFile(zip_path) as z:
    for info in z.infolist():
        if info.is_dir():
            continue
        name=info.filename.replace('\\','/')
        parts=[p for p in name.split('/') if p]
        if not parts or any(p == '__MACOSX' or p.startswith('._') or p.startswith('.') for p in parts):
            continue
        if pathlib.Path(parts[-1]).suffix.lower() not in audio:
            continue
        target=out.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(info) as src, target.open('wb') as dst:
            shutil.copyfileobj(src,dst)
PY
  echo "source_name,folder_path,final_top,consensus_status,decision_reason" > "$REPORT"
  COUNT=0
  while IFS= read -r FILE; do
    COUNT=$((COUNT + 1))
    SAFE="$(basename "$FILE" | tr '/ ' '__' | tr -cd 'A-Za-z0-9._-')"
    OUT="$OUTROOT/${COUNT}_${SAFE}"
    echo "[$COUNT] $(basename "$FILE")"
    PYTHONPATH=src python3 Aaron_Sound_Sorter.py sort "$FILE" "$OUT" --brain "$ROOT/stage4_folder_brain.json" --no-zip >/dev/null 2>&1
    MAN="$OUT/Aaron_Sorted_Sounds_manifest.csv"
    if [ -f "$MAN" ]; then
      python3 - <<PY >> "$REPORT"
import csv, pathlib
man=pathlib.Path(r'''$MAN''')
row=next(csv.DictReader(man.open(errors='replace', newline='')))
vals=[pathlib.Path(row.get('source_path','')).name,row.get('folder_path',''),row.get('final_top',''),row.get('consensus_status',''),row.get('decision_reason','')]
print(','.join('"'+str(v).replace('"','""')+'"' for v in vals))
PY
    else
      echo "\"$(basename "$FILE")\",\"ERROR_NO_MANIFEST\",\"\",\"\",\"\"" >> "$REPORT"
    fi
  done < <(find "$EXTRACT" -type f \( -iname '*.wav' -o -iname '*.aif' -o -iname '*.aiff' -o -iname '*.flac' -o -iname '*.ogg' -o -iname '*.au' \) | sort)
  echo "Done. Report: $REPORT"
  open "$WORK" 2>/dev/null || true
fi
