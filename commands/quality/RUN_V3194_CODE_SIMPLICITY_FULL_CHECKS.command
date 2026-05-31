#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"

echo "Cleaning macOS metadata files..."
find . -name '._*' -type f -delete
find . -name '.DS_Store' -type f -delete

echo "Running compileall..."
python3 -S -m compileall -q Aaron_Sound_Sorter.py src tests tools

echo "Running source-name blindness audit..."
PYTHONPATH=src python3 tools/audit_no_source_name_sorting.py --project-root .

echo "Running app self-test..."
PYTHONPATH=src python3 Aaron_Sound_Sorter.py self-test

REPORT_DIR="reports/v3194_code_simplicity_full_checks/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPORT_DIR/pytest_one_by_one"

echo "Running pytest files one at a time..."
index=0
for test_file in tests/test_*.py; do
  index=$((index + 1))
  base="$(basename "$test_file")"
  log="$REPORT_DIR/pytest_one_by_one/$(printf '%03d' "$index")_${base}.log"
  echo "[$index] $base"
  if ! PYTHONPATH=src python3 -m pytest -q "$test_file" > "$log" 2>&1; then
    echo "FAILED: $base"
    cat "$log"
    exit 1
  fi
done

echo "Writing validation summary..."
python3 - <<'PY' "$REPORT_DIR"
from pathlib import Path
import json
import re
import sys
report_dir = Path(sys.argv[1])
logs = sorted((report_dir / "pytest_one_by_one").glob("*.log"))
failures = []
for log in logs:
    text = log.read_text(encoding="utf-8", errors="replace")
    bad = bool(re.search(r"\bFAILED\b|\bERRORS?\b|Interrupted", text))
    ok = bool(re.search(r"\bpassed\b|\bskipped\b", text)) and not bad
    if not ok:
        failures.append(log.name)
summary = {"pytest_file_logs": len(logs), "failures": failures}
(report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
if failures:
    raise SystemExit(f"pytest failures: {failures}")
PY

find "$REPORT_DIR" -name '._*' -type f -delete
find "$REPORT_DIR" -name '.DS_Store' -type f -delete
zip -qr "${REPORT_DIR}.zip" "$REPORT_DIR"

echo "PASS: v31.94 code simplicity full checks"
echo "Report folder: $REPORT_DIR"
echo "Report zip: ${REPORT_DIR}.zip"
open "$REPORT_DIR" 2>/dev/null || true
