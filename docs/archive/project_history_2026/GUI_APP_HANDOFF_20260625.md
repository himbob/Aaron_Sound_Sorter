# Archived: Aaron Sound Sorter GUI Handoff 2026-06-25

## What Changed

Added a first local browser GUI for Aaron Sound Sorter. It lets a user:

1. Select a folder, ZIP, or single audio file.
2. Run the existing sorter into an in-memory preview table.
3. Review proposed folders.
4. Override individual rows using labels from the active brain.
5. Choose an export destination.
6. Export approved placements as copy, move, or symlink.
7. Save a correction evidence pack for future training workflows.

The GUI does not change classifier behavior. It reuses the existing product sorter and calls `SortSamplesUseCase.classify_audio_files()` before any file placement happens.

## New Files

- `docs/GUI_APP_PLAN_20260625.md`
- `docs/GUI_APP_HANDOFF_20260625.md`
- `src/aaron_sound_sorter/gui/__init__.py`
- `src/aaron_sound_sorter/gui/models.py`
- `src/aaron_sound_sorter/gui/preview_service.py`
- `src/aaron_sound_sorter/gui/app.py`
- `src/aaron_sound_sorter/gui/web_app.py`
- `tools/aaron_sorter_gui.py`
- `commands/gui/RUN_SORTER_GUI.command`
- `tests/test_gui_preview_service.py`

## Architecture

### Preview Layer

`SortPreviewService` prepares input, loads the active brain family, and classifies files without writing the final sorted tree. ZIP files are extracted into `_reports/gui_preview/run_YYYYMMDD_HHMMSS/preview_workspace/_source_cache/`.

### Review Layer

`PreviewRow` stores:

- proposed folder,
- approved folder,
- decision status,
- confidence display,
- duration/read status,
- compact voter summary,
- full in-memory sort result for correction evidence.

The GUI may show filenames for human review, but filenames are not passed into classifier logic as evidence.

### Browser Shell

`src/aaron_sound_sorter/gui/web_app.py` serves a local-only browser UI on `127.0.0.1`. This replaced the default Tk shell because macOS system Tk rendered as a blank window in user testing and aborts when probed in the current agent runtime. The Tk implementation remains available as a fallback with:

```bash
./commands/gui/RUN_SORTER_GUI.command --tk
```

The default command opens the browser UI.

### Export Layer

`SortPlanExporter` writes the approved plan into a user-selected destination:

- `Aaron_Sorted_Sounds/`
- `Aaron_GUI_Approved_Sort_Plan.csv`
- `Aaron_GUI_Corrections.csv`

Export modes:

- copy
- move
- symlink

### Correction Evidence

When manual corrections exist, the GUI prompts the user to save a correction pack under:

`_reports/gui_corrections/run_YYYYMMDD_HHMMSS/`

The JSONL evidence includes the sorter proposal, human-approved folder, feature values, shape vote, physics subpanels, brain/physics top evidence, authority trace, and shared candidates. It does not mutate training folders or rebuild brains.

## How To Launch

```bash
cd /path/to/Aaron_Sound_Sorter
./commands/gui/RUN_SORTER_GUI.command
```

For validation without opening a browser:

```bash
./commands/gui/RUN_SORTER_GUI.command --probe-layout
```

Optional:

```bash
PROJECT_ROOT="/path/to/Aaron_Sound_Sorter" \
PYTHON_BIN="/path/to/Aaron_Sound_Sorter/.venv_phase4/bin/python" \
./commands/gui/RUN_SORTER_GUI.command
```

## Validation Performed

- Focused GUI backend pytest:

```bash
make ai-test TEST='tests/test_gui_preview_service.py'
```

- Focused web GUI pytest:

```bash
make ai-test TEST='tests/test_gui_web_app.py'
```

- Python compile check for GUI modules and launcher:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/aaron_pycache \
.venv_phase4/bin/python -m py_compile \
src/aaron_sound_sorter/gui/app.py \
src/aaron_sound_sorter/gui/models.py \
src/aaron_sound_sorter/gui/preview_service.py \
src/aaron_sound_sorter/gui/web_app.py \
tools/aaron_sorter_gui.py
```

- Browser layout probe:

```bash
./commands/gui/RUN_SORTER_GUI.command --probe-layout
```

Result:

```text
html_bytes=12803
has_preview_button=True
has_export_button=True
```

- Local HTTP server validation:

The GUI served successfully at `http://127.0.0.1:8765/`, and the fetched page contained both `Preview Sort` and `Export Approved Sort`.

- Visual screenshot:

`_reports/gui_validation/gui_web_shell_20260625.png`

- Real one-file GUI backend preview:

```bash
PYTHONPATH=src PYTHONPYCACHEPREFIX=/private/tmp/aaron_pycache \
.venv_phase4/bin/python -c "from pathlib import Path; from aaron_sound_sorter.gui.preview_service import SortPreviewService; sample=Path('tests/acceptance/locked_smoke_v1/samples/CS_NE_Kick_OneShot_Monroe.wav'); session=SortPreviewService(Path('.')).classify_input(sample, sort_workers=1); print(len(session.rows), session.rows[0].approved_folder); print(session.run_dir)"
```

Result:

`Drums/Kick Drums/Generic Kick/One Shots`

## Known Limits

- The default GUI is browser-based because Tk was unreliable on this macOS setup.
- Correction packs are training-ready evidence, not automatic retraining.
- Symlink export from ZIP inputs points at the extracted preview cache. Copy is the durable choice for ZIP inputs.
- This is intentionally a first usable GUI, not a final polished review workstation.
