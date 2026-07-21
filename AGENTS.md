# AGENTS.md: AI coding rules for Aaron Sound Sorter

This file is for AI coding agents working in this repository. Read it before changing code.

## Project purpose

Aaron Sound Sorter is a practical sound librarian for music production. The final product should accept a ZIP file or folder of samples and produce a producer-friendly sorted library with clear review folders and diagnostics.

Keep the top-level user workflow simple:

```bash
python3 Aaron_Sound_Sorter.py "/path/to/samples.zip" "/path/to/output_folder"
```

Do not make the user understand reference brains, K-means, MFCCs, debug flags, or internal voters just to sort sounds.

## Architecture rules

Do not patch one file just to win one sample if the fix damages the system.

Follow these rules on every behavior change:

1. Measured audio evidence first.
2. Structure and role before specific source identity.
3. Character and flavor separate from source identity.
4. Do not use filenames, folder names, or source paths as sorting evidence.
5. Do not narrow or hide the category library to make tests pass.
6. Do not let FX swallow valid Drums or Instruments. Drums and Instruments must be ruled out before broad abstract FX wins.
7. Review is not trash. Use `_TO_REVIEW` when evidence is ambiguous or conflicting.
8. Keep the JSON ledger as an evidence rulebook. Do not mutate it casually.
9. Generated outputs belong under `_reports`, run folders, or explicit output folders, not the project root.
10. Never write output bundles, cache files, scratch files, or helper artifacts into the project root unless Aaron explicitly asks.

## Brain-first development rules

The project is moving away from static rescue code. Treat trained brains and
memory lanes as the long-term source of source identity, shape, role, and
physics ownership.

Use this policy on every classifier change:

1. Prefer a trainable brain, memory lane, or supervised panel over a new static
   category rule.
2. Static code may provide lightweight guardrails for catastrophic family
   mistakes, data plumbing, feature extraction, and review decisions.
3. Static code must not become a second hidden classifier that overrides strong
   human-trained memory without a measured family contradiction.
4. GUI corrections should teach dedicated memory brains first:
   folder/user memory, voter-role memory, physics memory, and shape memory.
5. Full/core/spread/outlier brains should be rebuilt from curated training
   data, not mutated by sparse one-off GUI corrections.
6. If a trained memory lane matches with high confidence and the measured
   top-family/body contract is compatible, the memory-owned label should get a
   real claim before broad fallback buckets such as Instrument Loops.
7. If memory and measured family/shape strongly disagree, send to review
   instead of forcing either side.
8. When fixing a bug, first decide whether it is a training/memory problem,
   feature problem, shape-brain problem, physics-brain problem, owner-brain
   problem, or only then a guardrail problem.
9. Add tests proving trained memory improves related samples, not just the
   exact file that was corrected.
10. Do not use source filenames, folder names, or sample-pack paths as training
    evidence at runtime. Human-approved labels and internal taxonomy labels are
    allowed as supervised targets.

## Required AI workflow

Use the Makefile. Do not bypass it with ad hoc shell commands unless a Makefile target is missing.

`make ai-check` must be clean. Do not hand off a bundle that fails the project gate.
AI-touched files must be clean before handoff.


Before editing:

```bash
make ai-preflight
make ai-changed-files
```

After editing Python files:

```bash
make ai-fix
make ai-check
```

When a specific test is relevant, run it one test file or one node at a time:

```bash
make ai-test TEST='tests/path_to_test.py::test_name'
```

Before packaging a bundle, use the Makefile bundle workflow. It performs a full generated-file cleanup before building the ZIP:

```bash
make bundle BUNDLE_NAME=Aaron_Sound_Sorter_descriptive_patch_name
```

For preview only:

```bash
make bundle-dry-run BUNDLE_NAME=Aaron_Sound_Sorter_descriptive_patch_name
```

Use `make ai-bundle-check BUNDLE_DIR=/path/to/staged_bundle_root` only when validating a manually staged bundle.

If `make ai-check` fails, fix the changed files or explain the blocker. Do not hand off a bundle and call it clean.

## Quality gate policy

`make ai-check` is the required project gate. It runs generated-file hygiene, whole-project compile, Ruff lint, Ruff format-check, and the no-source-name sorting audit over the maintained Python tree.

Use these commands correctly:

```bash
make ai-check          # required maintained-project gate
make quality-report    # honest full-repo baseline report
make quality           # future strict gate including Mypy, pydocstyle, and tests
```

Do not claim Mypy, pydocstyle, or the full long pytest suite is clean unless those commands were actually run and passed.

## Python coding standards

When editing Python code:

1. Preserve behavior unless the task explicitly asks for a behavior change.
2. Prefer clear human-readable names over short clever names.
3. Use `snake_case` for functions, methods, and variables.
4. Use `PascalCase` for classes.
5. Avoid vague names such as `data`, `item`, `result`, `x`, `tmp`, `thing`, `stuff`, `manager`, and `helper` unless the scope is tiny and obvious.
6. Add or improve type annotations for public functions, methods, constructors, dataclasses, and non-obvious internal helpers.
7. Add Google-style docstrings for public classes and functions.
8. Docstrings must explain purpose, inputs, outputs, side effects, raised exceptions, and important constraints.
9. Do not write comments that simply repeat the code.
10. Use comments only to explain non-obvious design choices.
11. Do not rename public symbols without listing the rename and checking call sites.
12. After editing, run the smallest relevant test first.
13. Provide a documentation report listing files changed, symbols documented, symbols renamed, tests run, and skipped items.

## Testing rules

Run focused tests one at a time when possible. If a test times out, split the work into smaller tests instead of repeating the same timeout.

Required before handoff when Python behavior changed:

```bash
make ai-check
make ai-test TEST='the_smallest_relevant_test'
```

Run the full suite only when the change is broad enough to justify it:

```bash
make test
```

For source-name safety, always keep this green:

```bash
make audit-source-names
```

## Bundle hygiene rules

AI handoff bundles must not contain:

```text
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
.venv_py313/
.coverage
.DS_Store
._*
_reports/
```

Installer scripts must delete macOS metadata before validation:

```bash
find . -name '._*' -type f -delete
find . -name '.DS_Store' -type f -delete
```

Prefer `make bundle BUNDLE_NAME=...` so cleanup, changed-file quality checks, staging, installer creation, and ZIP creation happen through one repeatable workflow. Use `make ai-bundle-check BUNDLE_DIR=/path/to/staged_bundle_root` only for manually staged bundles.

## Reporting back to Aaron

When handing off a bundle, include:

- what changed
- why it changed
- files changed
- symbols documented
- symbols renamed, or `none`
- tests run, one at a time where practical
- skipped tests or limitations
- install command for macOS
- first commands Aaron should run

Be direct. If something is still red, say what is red and why.
