# AI Handoff: Random Sample Folder Log Panel

Date: 2026-05-15

## Why this exists

The ZIP-based single-file log panel is good for known archives, but Aaron also needs a broader drive-level sweep that samples real folders under:

```text
/Volumes/T9/music_production/samples
```

This tool randomly selects folders that directly contain audio files and runs the sorter one file at a time.

## Files added

```text
tools/run_random_sample_folder_log_panel.py
commands/quality/RUN_RANDOM_SAMPLE_FOLDER_LOG_PANEL.command
README_RANDOM_SAMPLE_FOLDER_LOG_PANEL_20260515.md
```

## Safety

The upload-back zip contains logs and CSVs only. It does not include audio.

The tool uses filenames and folder names only to flag suspicious rows in the report. It does not feed name clues to the sorter.

## Default behavior

```text
20 folders
3 files per folder
no timeout on Aaron's Mac
follow symlinks
```

## Useful command

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
./commands/quality/RUN_RANDOM_SAMPLE_FOLDER_LOG_PANEL.command
```

## Larger run

```bash
FOLDER_COUNT=50 FILES_PER_FOLDER=5 ./commands/quality/RUN_RANDOM_SAMPLE_FOLDER_LOG_PANEL.command
```

## Upload back

Upload:

```text
reports/random_sample_folder_log_panel/run_*/random_folder_log_panel_upload_back.zip
```

## What to inspect first

Open:

```text
random_folder_summary.txt
random_folder_suspicious.csv
random_folder_results.csv
```

Suspicious rows are heuristic. Do not treat them as truth, but do investigate:

- non-vocal sources going to Human and Voice FX
- non-alarm sources going to Alarm
- Rhodes/keys/synth/guitar/string sources going to Bass Loops
- mixed melody loops going to Brass and Woodwinds
- pitched instruments going to Drums
