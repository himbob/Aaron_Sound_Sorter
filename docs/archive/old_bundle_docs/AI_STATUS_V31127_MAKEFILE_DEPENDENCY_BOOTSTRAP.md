# AI Status v31127 Makefile Dependency Bootstrap

## Purpose

This patch fixes the weak professional quality workflow from v31123-v31126. The old quality report only said important tools were missing and then ran the no-source-name audit. That was not useful.

v31127 adds a normal Python developer bootstrap setup:

- runtime requirements
- quality requirements
- full development requirements
- virtualenv-aware Makefile install targets
- explicit tool verification
- quality report that runs actual quality tools after installation
- a strict report target for CI-like failure behavior
- a report-only target for local diagnosis without stopping on legacy violations

## Architecture constraints

This patch does not change classifier behavior, voters, brain scoring, audio features, category authority, training data, or ledgers.

Generated reports still go under `_reports/quality`.

## Files changed

- `Makefile`
- `pyproject.toml`
- `requirements.txt`
- `requirements-quality.txt`
- `requirements-dev.txt`
- `tools/pro_quality_report.py`
- `tools/check_dev_environment.py`
- `tests/test_v31127_developer_makefile_bootstrap.py`

## New Makefile targets

- `make bootstrap`
- `make install`
- `make install-runtime`
- `make install-quality`
- `make install-dev`
- `make check-tools`
- `make test-one TEST=...`
- `make format-check`
- `make quality`
- `make ci`
- `make quality-gate-report`

Existing test, coverage, lint, type-check, docstyle, and cleanup targets are preserved.

## Documentation report

### Symbols added

- `ToolCheck`
- `CommandResult`
- `module_available()`
- `capture_tool_version()`
- `build_tool_status()`
- `make_quality_commands()`
- `append_command_result()`
- `RequiredTool`
- `module_is_available()`
- `run_version_command()`
- `check_required_tools()`

### Symbols renamed

None.

### Docstrings added or improved

- `tools/pro_quality_report.py`
- `tools/check_dev_environment.py`

The docstrings document purpose, arguments, returns, side effects, raised exceptions, and important constraints for public or non-obvious helpers.

### Tests run

- `python3 -m py_compile tools/check_dev_environment.py tools/pro_quality_report.py`
- `make help`
- `make install-dev VENV_DIR=/tmp/aaron_sound_sorter_make_test_venv PIP_INSTALL_FLAGS=--dry-run`
- `make install-quality VENV_DIR=/tmp/aaron_sound_sorter_make_test_venv_actual`
- `make install-dev VENV_DIR=/tmp/aaron_sound_sorter_make_test_venv_actual`
- `make check-tools VENV_DIR=/tmp/aaron_sound_sorter_make_test_venv_actual`
- `make quality-report VENV_DIR=/tmp/aaron_sound_sorter_make_test_venv_actual`
- `make test-one VENV_DIR=/tmp/aaron_sound_sorter_make_test_venv_actual TEST=tests/test_v31127_developer_makefile_bootstrap.py`
- `PYTHONPATH=src:. python3 -m pytest -q tests/test_v31127_developer_makefile_bootstrap.py`
- `PYTHONPATH=src:. python3 -m pytest -q tests/test_v31123_sort_timing_profile.py`
- `PYTHONPATH=src:. ./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command`

### Expected quality-report behavior

`make quality-report` now installs quality tools first, writes a report, and exits zero even if existing code has lint, type, or docstyle violations. The report records those failures honestly.

`make quality-gate-report` writes the same report but exits nonzero if any included check fails.

### Known current result

With tools installed, the quality report now runs real checks. In this codebase today, Ruff reports existing legacy violations. That is useful signal, not an install failure.

## Skipped items

- Full `make quality` was not forced to pass because existing legacy lint, type, and docstyle issues are outside this dependency-bootstrap patch.
- No classifier behavior tests were changed beyond focused regression tests.
