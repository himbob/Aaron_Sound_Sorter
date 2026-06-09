#!/bin/zsh
set -e
cd "$HOME/Downloads"
unzip -o Aaron_Sound_Sorter_v31164_percussion_batch_repair_python_tests_only.zip
cd "/Volumes/T9/testbed/Aaron_Sound_Sorter"
PATCH_DIR="$HOME/Downloads/Aaron_Sound_Sorter_v31164_percussion_batch_repair_python_tests_only"
rsync -a "$PATCH_DIR/" "/Volumes/T9/testbed/Aaron_Sound_Sorter/"
find . -name '._*' -delete
PY=".venv_phase4/bin/python"
[ -x "$PY" ] || PY="python3"
"$PY" -m compileall -q src tests tools Aaron_Sound_Sorter.py
"$PY" tools/audit_no_source_name_sorting.py --project-root .
echo "Installed v31.164 changed Python/test files."
