#!/bin/bash
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/Volumes/T9/testbed/Aaron_Sound_Sorter}"
cd "$PROJECT_ROOT" || { echo "Project root not found: $PROJECT_ROOT"; exit 0; }
python3 -m pytest -q tests/test_uploaded_regression_audio.py
