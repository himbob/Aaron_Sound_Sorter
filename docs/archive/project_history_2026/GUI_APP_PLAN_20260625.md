# Archived: Aaron Sound Sorter GUI Plan 2026-06-25

## Goal

Build a local Python desktop app that keeps Aaron Sound Sorter's normal user workflow simple while adding a human review step before files are written into a final sorted library.

The GUI must not become a second classifier. It should reuse the current measured-evidence sorter, show its proposed placements, let the user correct placements, then export the approved plan by copy, move, or symlink.

## Non-Goals

- Do not change classifier, voter, shape, physics, brain, or arbiter behavior for this GUI pass.
- Do not use filenames, source folders, or ZIP member paths as sorting evidence.
- Do not retrain brains automatically from a GUI click.
- Do not write generated preview output into the project root.

## Architecture

### 1. Preview Service

Add a small GUI-facing service that:

- Accepts a folder, ZIP, or single audio file.
- Prepares input with `AudioInputRepository`.
- Loads the active brain family through `BrainRepository`.
- Calls `SortSamplesUseCase.classify_audio_files()` so files are classified in memory without final placement.
- Builds plain preview rows containing source path, proposed folder, confidence/authority text, duration, and low-level diagnostic summary.

This service owns no classifier policy. It only adapts existing sorter objects into GUI rows.

### 2. Correction Model

Represent each row as:

- `source_path`
- `proposed_folder`
- `approved_folder`
- `is_corrected`
- `decision_status`
- `diagnostic_summary`

The GUI edits `approved_folder`; it never modifies the classifier result object.

### 3. Exporter

Add an exporter that writes the approved plan to a user-selected destination using one of:

- copy
- move
- symlink

The exporter should also write:

- `Aaron_GUI_Approved_Sort_Plan.csv`
- `Aaron_GUI_Corrections.csv`
- `Aaron_GUI_Correction_Evidence.jsonl`

The evidence JSONL should contain measured facts and voter diagnostics from corrected rows so a future training workflow can learn from human corrections without guessing from filenames.

### 4. GUI

Use Tkinter from the Python standard library so the app launches on Aaron's Mac without adding GUI dependencies.

The first usable version should have:

- Browse folder / ZIP / file input.
- Start Preview button.
- Table of proposed placements.
- Override selected row dialog with trained labels from the active brain.
- Destination chooser.
- Export mode selector: copy, move, symlink.
- Export button.
- Post-export prompt to save correction evidence when manual corrections exist.

The GUI can show filenames for human review and I/O, but production sorting still receives no filename evidence.

## Training Correction Handling

The GUI will not directly mutate training folders or rebuild brains. If corrections exist, it prompts the user to write a correction evidence pack under `_reports/gui_corrections/run_YYYYMMDD_HHMMSS/`.

That pack contains:

- the original proposed folder,
- the human-approved folder,
- feature values,
- shape vote,
- physics subpanel summary,
- brain/physics top guesses,
- authority trace.

This is the safe bridge toward training: human labels plus measured evidence, not filename-derived category guesses.

## Files To Add

- `src/aaron_sound_sorter/gui/__init__.py`
- `src/aaron_sound_sorter/gui/models.py`
- `src/aaron_sound_sorter/gui/preview_service.py`
- `src/aaron_sound_sorter/gui/app.py`
- `tools/aaron_sorter_gui.py`
- `commands/gui/RUN_SORTER_GUI.command`
- `tests/test_gui_preview_service.py`
- `docs/GUI_APP_HANDOFF_20260625.md`

## Validation

Run:

- `python -m py_compile` or project compile through `make ai-check`
- focused GUI service tests
- `make ai-check`

Do not claim GUI visual polish was manually verified unless the app was launched on a desktop session.
