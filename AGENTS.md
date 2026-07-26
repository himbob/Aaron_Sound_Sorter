# AGENTS.md

Rules for every AI or human coding agent working in this repository.

## Read first

1. `CURRENT_STATUS.md`
2. `docs/README.md`
3. `docs/NEURAL_AUDIO_ARCHITECTURE_V1.md`
4. `docs/TESTING_POLICY.md`
5. `docs/NO_SOURCE_NAME_SORTING_POLICY_20260515.md`

Archived documents are history, not current authority.

## Product contract

- Keep the user workflow simple.
- Accept a ZIP, folder, or audio file.
- Return a sorted library, Review folder, and diagnostics.
- Treat `_TO_REVIEW` as a valid uncertainty decision.

## Classification rules

- Use measured audio evidence first.
- Decide structure and role before source identity.
- Keep character/flavor separate from identity.
- Do not use filenames, paths, folders, ZIP names, or pack names as inference
  evidence.
- Never hide taxonomy labels to improve metrics.
- Rule out valid Drums and Instruments before broad FX wins.
- Prefer human-trained memory or neural prototypes over new static rescues.
- Do not add category-specific branches to `FamilyClaimArbiter`.
- If trained memory and measured family strongly disagree, use Review.

## Training and neural rules

- GUI corrections update dedicated user, voter-role, physics, and shape memory.
- Full/core/spread/outlier brains are rebuilt from reviewed training data.
- Training, validation, calibration, and final held-out sets must not share
  audio or derived duplicates.
- CLAP may own a GUI proposal only inside a learned neighborhood with compatible
  measured structure; unknown cross-family conflicts go to Review.
- Detailed CLAP prompts and PANNs events remain separate evidence lanes. PANNs
  may support or contradict a broad family but may not own a detailed folder.
- A GUI relabel that conflicts with a locked seed requires two identical human
  approvals before the prototype target changes.
- Build review queues from decoded audio, embeddings, measured structure, and
  content hashes. Filename-search review builders are forbidden.
- Never delete old brains or trainers without a reversible migration report.

## Public repository safety

Never commit:

- sample audio or archives;
- trained brain JSONs;
- model weights or embedding caches;
- `training/`, `_models/`, `_reports/`, or `neural_artifacts/`;
- secrets, credentials, or private absolute paths;
- proprietary dataset manifests intended only for local use.

Use portable references such as `project://` and `sample-library://` in
shareable evidence.

## Required workflow

Before editing:

```bash
make ai-preflight
make ai-changed-files
```

After Python changes:

```bash
make ai-fix
make ai-check
```

Focused test:

```bash
make ai-test TEST='tests/path.py::test_name'
```

Broad behavior change:

```bash
make test
```

Always keep this green:

```bash
make audit-source-names
```

Bundle releases with:

```bash
make bundle BUNDLE_NAME=descriptive_name
make ai-bundle-check BUNDLE_DIR=/path/to/staged_bundle
```

AI-touched files must be clean before handoff.

## Code quality

- Prefer clear names, type annotations, and small functions.
- Add Google-style docstrings to public Python APIs.
- Preserve public symbols unless the change is documented and call sites are
  checked.
- Use `apply_patch` for edits.
- Keep generated output outside the repository root.
- Do not mutate the JSON ledger casually.

## Documentation

- Keep root documents short.
- Put active technical docs under `docs/`.
- Move superseded notes to `docs/archive/`.
- Update `CURRENT_STATUS.md` when architecture authority changes.

## Handoff

Report:

- what and why;
- files changed;
- symbols documented or renamed;
- exact tests run;
- skipped work or red gates;
- install and rollback steps when packaging.

Do not claim success from training-set accuracy, a timed-out test, or one lucky
sample.
