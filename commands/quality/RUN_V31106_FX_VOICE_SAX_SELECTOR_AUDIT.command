#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/Volumes/T9/testbed/Aaron_Sound_Sorter}"
FX_ZIP="${FX_ZIP:-/Volumes/T9/music_production/samples/FX_Aaron2.zip}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="$PROJECT_ROOT/_reports/v31106_fx_voice_sax_selector_audit_$STAMP"
SELECTED_DIR="$REPORT_DIR/selected"
SORTED_DIR="$REPORT_DIR/sorted_runs"
AUDIT_CSV="$REPORT_DIR/fx_voice_sax_selector_audit.csv"

mkdir -p "$SELECTED_DIR" "$SORTED_DIR"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" - <<PY
from __future__ import annotations
import csv
import os
import re
import shutil
import zipfile
from pathlib import Path
zip_path = Path(r"$FX_ZIP")
selected_dir = Path(r"$SELECTED_DIR")
if not zip_path.exists():
    raise SystemExit(f"FX ZIP not found: {zip_path}")
patterns = {
    "sax": re.compile(r"(sax|saxophone)", re.I),
    "voice": re.compile(r"(vocal|vocals|vox)", re.I),
}
rows = []
with zipfile.ZipFile(zip_path) as zf:
    for info in zf.infolist():
        if info.is_dir():
            continue
        base = os.path.basename(info.filename)
        if not base.lower().endswith((".wav", ".aif", ".aiff", ".flac", ".ogg", ".au")):
            continue
        for kind, pattern in patterns.items():
            if not pattern.search(base):
                continue
            target_dir = selected_dir / kind
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / base
            suffix = 1
            while target.exists():
                target = target_dir / f"{suffix:03d}__{base}"
                suffix += 1
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            rows.append({"kind": kind, "basename": base, "zip_member": info.filename, "local_path": str(target)})
            break
with (selected_dir / "selected_files.csv").open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=["kind", "basename", "zip_member", "local_path"])
    writer.writeheader()
    writer.writerows(rows)
print(f"Selected {sum(r['kind']=='sax' for r in rows)} sax files and {sum(r['kind']=='voice' for r in rows)} vocal/vox files.")
PY

printf 'index,kind,basename,zip_member,final_label,folder_path,consensus_status,ok,manifest_path\n' > "$AUDIT_CSV"

"$PYTHON_BIN" - <<PY
from __future__ import annotations
import csv
import os
import pathlib
import re
import subprocess
import sys
project = pathlib.Path(r"$PROJECT_ROOT")
selected_csv = pathlib.Path(r"$SELECTED_DIR") / "selected_files.csv"
sorted_root = pathlib.Path(r"$SORTED_DIR")
audit_csv = pathlib.Path(r"$AUDIT_CSV")
python_bin = r"$PYTHON_BIN"
rows = list(csv.DictReader(selected_csv.open(newline="", encoding="utf-8")))
for index, row in enumerate(rows):
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", pathlib.Path(row["basename"]).stem)[:80]
    out_dir = sorted_root / f"{index:03d}_{row['kind']}_{safe_name}"
    cmd = [
        python_bin,
        str(project / "Aaron_Sound_Sorter.py"),
        "sort",
        row["local_path"],
        str(out_dir),
        "--brain",
        str(project / "stage4_folder_brain.json"),
        "--workers",
        "1",
        "--no-zip",
    ]
    print(f"[{index+1}/{len(rows)}] {row['kind']} {row['basename']}", flush=True)
    subprocess.run(cmd, cwd=project, check=True)
    manifest = out_dir / "Aaron_Sorted_Sounds_manifest.csv"
    final_label = ""
    folder_path = ""
    status = ""
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            data = next(reader)
        final_label = data.get("final_label", "")
        folder_path = data.get("folder_path", "")
        status = data.get("consensus_status", "")
    lower = (folder_path or final_label).lower()
    if row["kind"] == "sax":
        ok = lower.startswith("instruments/") and "sax" in lower
    else:
        ok = lower.startswith("instruments/") and ("voice" in lower or "vocal" in lower)
    with audit_csv.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([index, row["kind"], row["basename"], row["zip_member"], final_label, folder_path, status, ok, str(manifest)])
PY

"$PYTHON_BIN" - <<PY
from __future__ import annotations
import csv
from pathlib import Path
audit = Path(r"$AUDIT_CSV")
rows = list(csv.DictReader(audit.open(newline="", encoding="utf-8")))
failures = [r for r in rows if r.get("ok") != "True"]
print("\nFX voice/sax selector audit complete")
print(f"Report folder: {audit.parent}")
print(f"Rows: {len(rows)}")
print(f"Failures: {len(failures)}")
for row in failures:
    print(f"FAIL {row['index']} {row['kind']} {row['basename']} -> {row['final_label']} [{row['consensus_status']}]")
PY

if command -v open >/dev/null 2>&1; then
    open "$REPORT_DIR"
fi
