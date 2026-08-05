#!/bin/bash
# SOURCE-NAME BLINDNESS INVARIANT:
# This audit script may read producer filenames only AFTER sorting to check QA
# expectations. Production sorter code must never use names as evidence.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
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
  exit 1
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
WORK="$ROOT/_real_sort_tests/fx_aaron2_filename_hint_audit_$STAMP"
mkdir -p "$WORK"
export AARON_SORTER_ROOT="$ROOT"
export AARON_FX_AUDIT_ZIP="$ZIP"
export AARON_FX_AUDIT_WORK="$WORK"
export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"
python3 - <<'PY'
from __future__ import annotations
import csv
import os
import pathlib
import re
import shutil
import subprocess
import sys
import zipfile

root = pathlib.Path(os.environ["AARON_SORTER_ROOT"])
zip_path = pathlib.Path(os.environ["AARON_FX_AUDIT_ZIP"])
work = pathlib.Path(os.environ["AARON_FX_AUDIT_WORK"])
extract = work / "extracted"
outputs = work / "outputs"
report = work / "fx_aaron2_filename_hint_audit.csv"
extract.mkdir(parents=True, exist_ok=True)
outputs.mkdir(parents=True, exist_ok=True)
audio_exts = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".au"}

def is_audio_member(name: str) -> bool:
    parts = [part for part in name.replace("\\", "/").split("/") if part]
    if not parts:
        return False
    if any(part == "__MACOSX" or part.startswith("._") or part.startswith(".") for part in parts):
        return False
    return pathlib.Path(parts[-1]).suffix.lower() in audio_exts

with zipfile.ZipFile(zip_path) as zf:
    members = [info for info in zf.infolist() if (not info.is_dir()) and is_audio_member(info.filename)]
    for info in members:
        safe_parts = [part for part in info.filename.replace("\\", "/").split("/") if part]
        target = extract.joinpath(*safe_parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)

files = sorted(path for path in extract.rglob("*") if path.is_file() and path.suffix.lower() in audio_exts)

def normalized(text: str) -> str:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.sub(r"[^a-z0-9#]+", " ", text.lower()).strip()

def expected_hints(source_name: str) -> list[str]:
    text = normalized(source_name)
    hints: list[str] = []
    def add(name: str) -> None:
        if name not in hints:
            hints.append(name)
    if "drum loop" in text or "drumlp" in text or "drumloop" in text or "kick clap loop" in text or ("drum" in text and "loop" in text):
        add("drum_loop")
    if "kick" in text and "loop" not in text and "kick clap" not in text:
        add("kick")
    if any(word in text for word in ["vocal", "vocals", "vox", "voice", "rap"]):
        add("voice")
    if any(word in text for word in ["sax", "saxophone"]):
        add("sax")
    if any(word in text for word in ["brass", "trumpet", "horn", "flute", "woodwind"]):
        add("brass_woodwind")
    if "bass" in text and "bass drum" not in text and "drum" not in text:
        add("bass")
    if any(word in text for word in ["guitar", "gtr", "acgtr", "elgtr"]):
        add("guitar")
    if any(word in text for word in ["piano", "rhodes", "keys"]):
        add("keys")
    if any(word in text for word in ["synth", "lead", "arp", "pad", "chord", "stab"]):
        add("synth")
    if any(word in text for word in ["riser", "downlifter", "impact", "whoosh", "sweep", "fx police", "police", "siren", "alarm", "hit"]):
        add("fx")
    return hints

def hint_matches_folder(hint: str, folder_path: str) -> bool:
    folder = folder_path.replace("\\", "/").lower()
    if hint == "drum_loop":
        return folder.startswith("drums/drum loops")
    if hint == "kick":
        return folder.startswith("drums/kick")
    if hint == "voice":
        return ("voice" in folder or "vocal" in folder or "human and voice" in folder) and not folder.startswith("drums")
    if hint == "sax":
        return folder.startswith("instruments") and ("sax" in folder or "woodwind" in folder or "brass" in folder)
    if hint == "brass_woodwind":
        return folder.startswith("instruments") and any(word in folder for word in ["brass", "woodwind", "horn", "flute", "sax"])
    if hint == "bass":
        return folder.startswith("instruments/bass")
    if hint == "guitar":
        return folder.startswith("instruments") and "guitar" in folder
    if hint == "keys":
        return folder.startswith("instruments") and any(word in folder for word in ["keys", "piano", "rhodes", "instrument loops"])
    if hint == "synth":
        return folder.startswith("instruments") and any(word in folder for word in ["synth", "instrument loops", "one shots"])
    if hint == "fx":
        return folder.startswith("fx")
    return True

def verdict(hints: list[str], folder_path: str) -> str:
    if not hints:
        return "UNCHECKED_NO_STRONG_FILENAME_HINT"
    return "PASS" if any(hint_matches_folder(hint, folder_path) for hint in hints) else "FAIL"

rows: list[dict[str, object]] = []
for index, file_path in enumerate(files, start=1):
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", file_path.name)[:90]
    out = outputs / f"{index:03d}_{safe_name}"
    print(f"[{index}/{len(files)}] {file_path.name}", flush=True)
    command = [
        sys.executable,
        str(root / "Aaron_Sound_Sorter.py"),
        "sort",
        str(file_path),
        str(out),
        "--brain",
        str(root / "stage4_folder_brain.json"),
        "--no-zip",
    ]
    row = {
        "index": index,
        "source_name": file_path.name,
        "source_relative_path": str(file_path.relative_to(extract)),
        "expected_hints_from_filename_for_QA_only": ";".join(expected_hints(file_path.name)),
        "folder_path": "",
        "verdict": "",
        "consensus_status": "",
        "decision_reason": "",
        "exit_code": "",
        "timed_out": "",
    }
    try:
        result = subprocess.run(command, cwd=str(root), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        row["exit_code"] = str(result.returncode)
        if result.returncode != 0:
            row["folder_path"] = "ERROR_SORT_FAILED"
            row["verdict"] = "ERROR"
            row["decision_reason"] = (result.stderr or result.stdout)[-500:]
        else:
            manifest = out / "Aaron_Sorted_Sounds_manifest.csv"
            with manifest.open(errors="replace", newline="") as handle:
                csv.field_size_limit(sys.maxsize)
                manifest_row = next(csv.DictReader(handle))
            row["folder_path"] = manifest_row.get("folder_path", "")
            row["consensus_status"] = manifest_row.get("consensus_status", "")
            row["decision_reason"] = manifest_row.get("decision_reason", "")[:500]
            row["verdict"] = verdict(expected_hints(file_path.name), str(row["folder_path"]))
    except subprocess.TimeoutExpired:
        row["folder_path"] = "ERROR_TIMEOUT"
        row["verdict"] = "ERROR"
        row["timed_out"] = "1"
    rows.append(row)

with report.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

summary: dict[str, int] = {}
for row in rows:
    summary[str(row["verdict"])] = summary.get(str(row["verdict"]), 0) + 1
print("Report:", report)
print("Summary:", summary)
if summary.get("FAIL", 0) or summary.get("ERROR", 0):
    raise SystemExit(1)
PY
open "$WORK" 2>/dev/null || true
