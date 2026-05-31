# AI status v31127 company Makefile dependencies

## Goal

Replace the weak quality-report wrapper with a normal Python development Makefile that can install all runtime, test, coverage, lint, type-check, and docstyle dependencies.

## Files changed

- `Makefile`
- `pyproject.toml`
- `requirements.txt`
- `requirements-quality.txt`
- `requirements-dev.txt`
- `tools/pro_quality_report.py`
- `tools/check_dev_environment.py`
- `tests/test_v31127_developer_makefile_bootstrap.py`
- `docs/PYTHON_DEV_WORKFLOW_MAKEFILE.md`

## What changed

The Makefile now includes setup targets:

- `make bootstrap`
- `make init`
- `make venv`
- `make upgrade-pip`
- `make install-runtime`
- `make install-quality`
- `make install-dev`
- `make install-all`
- `make doctor`

It also includes quality targets:

- `make audit-source-names`
- `make pycompile`
- `make test`
- `make test-one TEST=...`
- `make test-coverage`
- `make test-coverage-html`
- `make lint`
- `make format`
- `make format-check`
- `make type-check`
- `make docstyle`
- `make quality`
- `make qa`
- `make ci`
- `make quality-report`
- `make quality-gate-report`

## Important behavior note

No sorter behavior was changed. No routing logic, voter logic, physics logic, brain logic, training data, or ledgers were changed.

## Testing notes

The sandbox does not have internet access, so I did not prove that pip could download missing packages from PyPI. I tested Makefile syntax, target expansion, Python syntax, and the new regression tests. On Aaron's Mac, `make bootstrap` should create/update `.venv_phase4` and install the declared dependencies through pip.
