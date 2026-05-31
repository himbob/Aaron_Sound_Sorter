# AI Status v31129: Make Bundle Workflow

## Purpose

Added a first-class `make bundle` workflow so AI agents can build clean patch bundles through the Makefile instead of inventing ad hoc ZIP commands.

## Behavior

`make bundle` now runs:

1. `clean-for-bundle`
2. `ai-check`
3. `commands/build/BUILD_CLEAN_PATCH_BUNDLE.command`

The command calls `tools/build_clean_patch_bundle.py`, which stages changed files into a clean bundle root, writes a no-backup installer, checks staged hygiene, and writes the ZIP under `_reports/bundles`.

## Files changed

- `AGENTS.md`
- `Makefile`
- `pyproject.toml`
- `commands/build/BUILD_CLEAN_PATCH_BUNDLE.command`
- `tools/build_clean_patch_bundle.py`
- `docs/MAKE_BUNDLE_WORKFLOW.md`
- `docs/PYTHON_DEV_WORKFLOW_MAKEFILE.md`
- `tests/test_v31129_make_bundle_workflow.py`
- `AI_STATUS_V31129_MAKE_BUNDLE_WORKFLOW.md`

## Symbols documented

- `BundleRequest`
- `BundleResult`
- `normalize_bundle_name`
- `run_git_lines`
- `normalize_relative_path`
- `read_environment_changed_files`
- `discover_changed_files`
- `is_generated_or_blocked_path`
- `is_allowed_patch_file`
- `select_bundle_files`
- `scan_for_staged_junk`
- `copy_selected_files`
- `write_installer`
- `write_manifest`
- `zip_directory`
- `build_bundle`
- `build_parser`
- `main`

## Symbols renamed

None.

## Tests run

- `python3 -m py_compile tools/build_clean_patch_bundle.py`
- `python3 -m pytest -q tests/test_v31129_make_bundle_workflow.py`
- dry-run bundle builder test through direct script invocation
- real temp-project bundle build through direct script invocation
- bundle ZIP hygiene inspection
- temp install smoke test using generated `INSTALL_NO_BACKUP.command`

## Notes

The builder does not package final reports, caches, virtual environments, audio files, or ZIP archives. The generated bundle is a patch bundle, not a full source distribution or release artifact.
