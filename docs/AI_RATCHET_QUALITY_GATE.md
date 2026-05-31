# AI project quality gate

## Purpose

`make ai-check` is now the required maintained-project gate. It checks the project Python tree instead of only changed files.

The current policy is:

```text
No generated junk enters the repo or handoff bundles.
The maintained Python tree must compile.
Ruff lint must pass for src, tests, and tools.
Ruff format-check must pass for src, tests, and tools.
The no-source-name sorting audit must pass.
Legacy namespace-bridge star imports are explicitly isolated with documented per-file Ruff ignores.
```

## Required commands for AI workers

Before editing:

```bash
make ai-preflight
make ai-changed-files
```

After editing:

```bash
make ai-fix
make ai-check
```

Before packaging:

```bash
make clean-for-bundle
make ai-check
make ai-bundle-check BUNDLE_DIR=/path/to/staged_bundle_root
```

## Why `make quality` is still separate

`make quality` remains the future full strict gate for Mypy, pydocstyle, and the full test suite. `make ai-check` is the enforced project gate for compile, Ruff, formatting, generated-file hygiene, and source-name safety.

Do not claim Mypy, pydocstyle, or the full long pytest suite is clean unless those commands were actually run and passed.

## What `make ai-check` enforces

`make ai-check` enforces:

- no generated junk such as `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `.venv`, `.venv_py313`, `*.pyc`, `.DS_Store`, or `._*`
- `compileall -q` on `src`, `tests`, `tools`, and `Aaron_Sound_Sorter.py`
- Ruff lint on `src`, `tests`, and `tools`
- Ruff format check on `src`, `tests`, and `tools`
- the full no-source-name sorting audit when the audit script is present

## Working outside Git

`make ai-check` no longer depends on Git changed-file detection to decide what Python files are checked. It still prints changed files when Git metadata is available, but the quality gate is project-wide for the maintained Python tree.

## Bundle hygiene

A bundle fails if it contains:

```text
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
.DS_Store
._*
_reports/
```

Run this before zipping:

```bash
make ai-bundle-check BUNDLE_DIR=/path/to/staged_bundle_root
```
