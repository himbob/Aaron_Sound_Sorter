# AI Status: v31128 AGENTS.md and Makefile quality gate

## Purpose

This patch adds a root `AGENTS.md` and a changed-files Makefile gate so AI coding agents have an explicit project-level instruction file and a practical way to enforce coding standards on the files they touch.

## Problem fixed

The previous `make quality-report` correctly installed tools, but it exposed full legacy debt instead of giving AI workers a usable pre-handoff gate. Whole-repo Ruff, format, Mypy, and pydocstyle are not clean yet. That makes `make quality` useful as a baseline, but not as the normal blocker for every AI patch.

## Architecture decision

Use a ratchet gate:

- legacy debt is reported honestly
- AI-touched files must pass strict checks now
- generated junk must not enter the repo or handoff bundles
- whole-repo strict quality remains the future target

## Files changed

- `AGENTS.md`
- `Makefile`
- `pyproject.toml`
- `tools/ai_quality_gate.py`
- `docs/AI_RATCHET_QUALITY_GATE.md`
- `docs/PYTHON_DEV_WORKFLOW_MAKEFILE.md`
- `tests/test_v31128_agents_makefile_ai_gate.py`
- `AI_STATUS_V31128_AGENTS_MAKEFILE_QUALITY_GATE.md`

## New Makefile targets

- `make ai-preflight`
- `make ai-changed-files`
- `make ai-fix`
- `make ai-check`
- `make ai-test TEST='tests/path.py::test_name'`
- `make ai-bundle-check BUNDLE_DIR=/path/to/staged_bundle_root`
- `make clean-generated`
- `make quality-baseline`
- `make quality-strict`

## Coding standards policy

`AGENTS.md` now tells AI workers to use the Makefile for coding standards:

```bash
make ai-preflight
make ai-fix
make ai-check
```

It also records the Python readability, typing, and docstring rules Aaron requested.

## Behavior impact

No classifier behavior changes.

This is developer workflow and packaging hygiene only.

## Tests run

- `python3 -m py_compile tools/ai_quality_gate.py`
- `python3 -m py_compile tools/pro_quality_report.py`
- `python3 -m py_compile tools/check_dev_environment.py`
- `python3 -m pytest -q tests/test_v31128_agents_makefile_ai_gate.py`
- `make -n ai-preflight`
- `make -n ai-check`
- `make -n ai-bundle-check BUNDLE_DIR=/tmp/example_bundle`
- `python3 tools/ai_quality_gate.py --project-root . --base HEAD list`
- bundle hygiene scan against the staged bundle

## Skipped

Live dependency installation from PyPI was not repeated in the sandbox because internet access is disabled here. Aaron already confirmed `make bootstrap` installed the v31127 dependency stack locally.

## Follow-up adjustment

`ai-preflight`, `ai-fix`, `ai-check`, and quality report targets now run `clean-generated` first so normal pytest/py_compile cache files do not create false failures. Bundle checks still fail on staged bundle junk instead of deleting it silently.
