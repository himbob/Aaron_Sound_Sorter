# Python developer workflow and Makefile policy

## Purpose

This project should behave like a normal Python checkout for development:

```bash
make bootstrap
make test
make quality
```

The Makefile owns local setup and quality gates. The sorter itself remains a simple user-facing command:

```bash
python3 Aaron_Sound_Sorter.py "/path/to/samples.zip" "/path/to/output_folder"
```

## Dependency layout

Runtime dependencies stay in:

```text
requirements.txt
```

Developer and quality tools stay in:

```text
requirements-quality.txt
requirements-dev.txt
pyproject.toml
```

`make bootstrap` creates or updates `.venv_phase4` and installs the full development stack.

## Quality tools

The required local quality stack is:

```text
pytest
pytest-cov
coverage
ruff
mypy
pydocstyle
```

The quality report must not silently skip these tools. If they are missing, it reports a setup failure and tells the developer to run:

```bash
make install-dev
```

## Report location

Generated quality and coverage outputs go under:

```text
_reports/quality/
```

No reports, caches, coverage XML, or HTML coverage folders should be written to the project root.

## Behavior policy

This change does not alter classifier behavior, voters, brain scoring, routing policy, training data, or audio physics. It only improves local development setup and quality gates.

## v31128 AI changed-files gate

AI workers must use the ratchet targets instead of claiming the whole legacy repository is clean.

```bash
make ai-preflight
make ai-fix
make ai-check
make ai-test TEST='tests/path.py::test_name'
make ai-bundle-check BUNDLE_DIR=/path/to/staged_bundle_root
```

Use `make quality-report` for a whole-repo debt report. Use `make quality` only when a planned cleanup is intended to make the whole repository strict-clean.

## Clean bundle creation

Use the Makefile bundle target for AI handoff bundles:

```bash
make bundle BUNDLE_NAME=Aaron_Sound_Sorter_descriptive_patch_name
```

This target performs a generated-file cleanup before bundling, then runs the AI changed-files quality gate. It writes the final ZIP under `_reports/bundles/` and opens that folder on macOS through `commands/build/BUILD_CLEAN_PATCH_BUNDLE.command`.

For a preview without writing a ZIP:

```bash
make bundle-dry-run BUNDLE_NAME=Aaron_Sound_Sorter_descriptive_patch_name
```
