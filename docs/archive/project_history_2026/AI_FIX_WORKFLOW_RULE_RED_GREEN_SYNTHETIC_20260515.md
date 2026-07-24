# Archived: AI Fix Workflow Rule — Red/Green Synthetic Regression First

Date: 2026-05-15

## Permanent rule for future AI helpers

Before changing sorter logic for a real failure class, first add a synthetic regression test that reproduces the failure pattern and prove that it fails against the current code.

The required workflow is:

1. Identify the failure class from logs or real audio evidence.
2. Add a synthetic regression test that encodes the measured failure pattern.
3. Run that test and confirm it fails before the fix.
4. Patch the smallest architecture-safe seam.
5. Rerun the exact same test and confirm it passes.
6. Run the focused regression panel.
7. Document the before/after behavior in the handoff or status notes.

## Why this rule exists

This project has repeatedly fixed one sound while breaking another family. The red/green synthetic workflow prevents unverifiable threshold edits and makes each bug fix permanent in pytest.

## What counts as a valid synthetic regression

A valid synthetic regression may use either:

- a fact-vector surrogate, built from measured roles, candidate paths, shape vote, and feature values, or
- a generated synthetic WAV that is created during pytest and not stored as a sample file.

For fast safety tests, prefer fact-vector surrogates around these seams:

- `infer_parent_eligibility()`
- `DecisionCoreV2.apply_eligibility()`
- role/candidate conflict resolution
- manifest/diff/report logic

## What is not acceptable

Do not patch sorter behavior first and then write a test that merely matches the new behavior. That is not a regression test.

Do not make a one-file filename-based exception.

Do not tune against dirty synthetic stress-test libraries as truth data.

Do not use source filenames or folder paths for final sorting logic. Filename and folder text may be used only in diagnostic tools and report flags.

## Current example

The file below was added by first creating failing tests for confirmed bug classes, then patching the conflict resolver:

```text
tests/test_decision_core_conflict_resolver_tdd.py
```

Initial red result before the fix:

```text
4 failed, 1 passed
```

After the conflict-resolver patch:

```text
5 passed
```

Focused regression panel after the patch:

```text
57 passed
```
