# Current Status


Date: 2026-05-30

## Purpose

Make `make ai-check` usable as an enforced project gate going forward.

The previous v31.130 bundle only cleaned touched files. That was a ratchet step, but it did not make the whole maintained Python tree pass the Makefile quality gate Aaron wanted to enforce.

## What changed

- Removed the checked-in `.venv_py313/` directory from the bundle/project tree.
- Added `.venv_py313/` to `.gitignore`.
- Archived old root-level handoff/status Markdown files under `docs/archive/old_bundle_docs/` instead of leaving them in the root.
- Reformatted `src`, `tests`, and `tools` with Ruff.
- Updated `tools/ai_quality_gate.py` so `make ai-check` now checks the maintained project Python tree, not only Git-changed files.
- Updated `pyproject.toml` so Ruff enforces the active project gate while explicitly isolating legacy namespace-bridge star-import files.
- Fixed three real static-quality defects:
  - `api.py` now exposes explicit constants for static checks while preserving legacy namespace bridge behavior.
  - `physics_subpanels.py` computes `drum_loop_source_score` before using it in FX transition scoring.
  - `family_claim_arbiter.py` uses `duration_sec` from measured feature values instead of an undefined local name.
- Fixed small lint issues in tests/tools caused by unused loop variables, ambiguous `l` names, and import ordering.

## What `make ai-check` now enforces

`make ai-check` runs:

```bash
compileall -q src tests tools Aaron_Sound_Sorter.py
ruff check src tests tools
ruff format --check src tests tools
tools/audit_no_source_name_sorting.py --project-root <project>
```

It also fails on generated junk, including `.venv`, `.venv_py313`, caches, `.DS_Store`, `._*`, and compiled Python files.

## What this does not claim

This does not claim that `make quality`, Mypy, pydocstyle, or the full long pytest suite are clean. Those remain separate stricter gates and should only be claimed after they are run and pass.

## Validation run in this handoff

Passed:

```bash
make ai-check VENV_DIR=/opt/pyvenv PYTHON=/opt/pyvenv/bin/python3
```

Focused tests passed one at a time:

```bash
make ai-test VENV_DIR=/opt/pyvenv PYTHON=/opt/pyvenv/bin/python3 TEST='tests/test_v31127_developer_makefile_bootstrap.py'
make ai-test VENV_DIR=/opt/pyvenv PYTHON=/opt/pyvenv/bin/python3 TEST='tests/test_v31128_agents_makefile_ai_gate.py'
make ai-test VENV_DIR=/opt/pyvenv PYTHON=/opt/pyvenv/bin/python3 TEST='tests/test_v31129_make_bundle_workflow.py'
python -m pytest -q tests/test_no_source_name_sorting_invariant.py
python -m pytest -q tests/test_low_level_physics_subpanels.py
python -m pytest -q tests/test_v31117_voter_calibration_panels.py
python -m pytest -q tests/test_v31123_sort_timing_profile.py
python -m pytest -q tests/test_v31124_vectorized_brain_scorer.py
python -m pytest -q tests/test_v31125_harmonic_wetness_reuse_guard.py
python -m pytest -q tests/test_v31126_family_order_policy_trace.py
```

A combined loop of these tests timed out while entering `test_v31129_make_bundle_workflow.py`; rerunning that file by itself passed. This is why future AI agents should keep using one-test-file runs when the container is tight.
