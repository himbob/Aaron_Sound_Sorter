# Testing Policy

## Test Order

1. Add or update a failing test that describes the failure class.
2. Change the smallest architecture-safe code seam.
3. Run focused pytest.
4. Run broader pytest when feasible.
5. Update `CURRENT_STATUS.md`.

## Test Categories

### Unit Tests

Fast tests for pure role, eligibility, decision, and mapping functions.

### Synthetic Audio Tests

Generated audio for physics edge cases. Synthetic tests are smoke tests, not truth data.

### Real Regression Fixture Tests

Small real files that previously failed. These should first assert parent-family safety, not deep leaf perfection.

### Manifest Tests

Reports must show raw brain votes, raw physics votes, measured roles, eligibility result, and final reason.

## Required Focused Command

```bash
PYTHONPATH=src python3 -m pytest -q \
  tests/test_parent_eligibility_uploaded_audio.py \
  tests/test_decision_core_v2.py \
  tests/test_two_voter_redesign.py \
  tests/test_phase4_shape_voter.py
```

## Project ZIP Warning

The project folder contains both code bundles and audio sample ZIPs from different phases. Tests and sort commands must not treat every ZIP in the project root as a sample pack.



## Permanent Red/Green Rule For Sorter Fixes

For every future sorter-logic bug fix, add the synthetic regression test first and prove it fails before changing code. Then apply the smallest architecture-safe fix and prove the exact same test passes.

This is required because one-file or threshold-only fixes have repeatedly improved one family while damaging another. A fix is not complete unless the failure class is preserved in pytest.

The current reference workflow is documented in:

```text
docs/AI_FIX_WORKFLOW_RULE_RED_GREEN_SYNTHETIC_20260515.md
```

---

## Absolute blind-sorting rule

Production sorting logic must never use producer filenames, source folder names,
ZIP member names, path tokens, sample-pack labels, or any other source-name text
as classification evidence. Names are allowed only for I/O, manifest display,
post-decision diagnostics, test fixture selection, and human review. Voters,
roles, eligibility, consensus, conflict resolution, and final placement must use
audio measurements, learned brain candidates, physics candidates, shape/role
facts, and explicit manual corrections only.

Every sorter-logic change must preserve this invariant and must pass:

```bash
./commands/quality/RUN_NO_SOURCE_NAME_SORTING_AUDIT.command
```

