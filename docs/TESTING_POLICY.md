# Testing Policy

## Test Order

1. Add or update a failing test that describes the failure class.
2. Change the smallest architecture-safe code seam.
3. Run focused pytest through the Makefile.
4. Run the full suite for broad runtime changes.
5. Update `CURRENT_STATUS.md`.

## Test Categories

### CLAP Ownership Tests

Keep tests for embeddings, content-hash caching, prototype learning, GUI
correction intake, known-neighborhood ownership, and uncertainty review.

### Safety Contract Tests

Keep source-name blindness, measured loop/one-shot structure, corruption
handling, leakage prevention, and rollback. These are guardrails, not a hidden
second classifier.

### Transitional Legacy Tests

Keep only while the tested legacy component still runs in production. Delete
the test with the component when CLAP replaces it.

### Real Audio Panels

Judge CLAP separately from fallback brains. A passing final placement is not
proof of neural success unless the report says which owner won.

## Required Focused Command

```bash
make ai-test TEST='tests/test_neural_gui_training.py'
make ai-test TEST='tests/test_gui_preview_service.py'
make test
```

Do not run bare `pytest`; it may use Homebrew or another Python without the
project runtime dependencies. If setup is incomplete, run `make bootstrap`.

## Project ZIP Warning

The project folder contains both code bundles and audio sample ZIPs from different phases. Tests and sort commands must not treat every ZIP in the project root as a sample pack.



## Permanent Red/Green Rule For Sorter Fixes

For every sorter bug, add the smallest ownership-level regression first. Prefer
a real CLAP embedding/prototype test or a content-only neural test. Synthetic
legacy facts are acceptable only for a retained safety contract.

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
audio measurements, neural embeddings, retained shape/structure facts, and
explicit manual corrections only.

Every sorter-logic change must preserve this invariant and must pass:

```bash
make audit-source-names
```
