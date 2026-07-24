# Neural training and curation handoff — 2026-07-24

## Outcome

The GUI correction path now gives an exact human-approved label a real learned
ownership claim. The original failure was not that the GUI ignored training:
the memory files were updated, but the owner-claim producer silently rewrote
voice-like FX labels to `Instruments/Voice`. The human raw-family protection
then rejected the rewritten cross-family target and returned the file to
Review.

The repair is category-neutral:

- exact supervised Drums, Instruments, and FX labels retain their approved
  taxonomy path;
- a newer correction for the same audio supersedes an older conflicting label,
  including conflicts within one voter-role or physics family;
- sparse GUI corrections update only the dedicated user, voter-role, physics,
  and shape memories;
- full/core/spread/outlier brains remain rebuild-only artifacts of curated
  training data.

On the supplied 39-file vocal pack, the sorter placed all six explicit GUI
corrections in their exact selected folders, all six used the exact human
teacher owner claim, and no file went to Review.

## Production authority

The production sorter still uses measured evidence and dedicated learned memory.
The frozen neural encoders remain shadow-only. This is deliberate: the real
corpus currently has too little nonleaking calibration evidence and no
production category passes all readiness gates.

This distinction matters:

- the fixed GUI memory route now owns compatible exact supervised corrections;
- CLAP and MERT provide measured research evidence and active-learning
  priorities;
- neither neural encoder may silently route production files yet.

## Curated corpus and split policy

The provenance inventory contains 366 occurrences:

- 315 legacy locked-corpus candidates;
- 51 explicit human-approved candidates, including recent GUI feedback.

Every row records its source kind, intended label, human-approval state, content
identity, duplicate group, conflict state, and allowed use. The four-way split
uses content groups rather than source paths:

| Use | Count |
|---|---:|
| Prototype training | 48 |
| Prototype validation | 1 |
| Confidence calibration | 1 |
| Final held-out evaluation | 1 |

Byte, canonical decoded-audio, and normalized lossless/gain/silence identities
may not cross these partitions. Recent GUI feedback is training-only. Conflicts,
unapproved candidates, and duplicate copies are excluded from fitting. Lossy,
time-stretched, and heavily trimmed derivatives still require a perceptual
duplicate detector.

The tiny validation, calibration, and final held-out counts are a real data
limitation, not a successful production split. The current corpus is suitable
for curation work and shadow experiments, not calibrated neural ownership.

## Trainer audit and reversible quarantine

The CLAP leave-one-out audit of the 315 locked candidates reported:

| Status | Count |
|---|---:|
| Supported | 170 |
| Cross-label conflict | 71 |
| Exact duplicate-hash label conflict | 11 |
| Possible label outlier | 9 |
| Ambiguous cross-label support | 5 |
| Limited support | 21 |
| Singleton | 28 |

`proposed_training_quarantine.json` is a proposal only. It does not delete,
move, or mutate trainers or brain files. `human_review_needed.csv` contains 145
rows needing a human decision before the next curated-corpus rebuild.

## Category readiness

All 274 production taxonomy labels are represented in the readiness matrix:

| Tier | Meaning | Count |
|---|---|---:|
| A | Ready for a later authority gate | 0 |
| B | Promising but missing production evidence | 2 |
| C | Trainable but materially incomplete | 7 |
| D | Insufficient data/evidence | 265 |
| E | Human-approved taxonomy redundancy | 0 |

Tier E is never inferred automatically. It requires an explicit taxonomy
decision. Sparse categories remain visible instead of being hidden to improve
metrics.

## Real encoder comparison

Both providers ran real local CPU inference on the same 45-example prototype
set and 67-file held-out panel.

| Held-out panel | CLAP | MERT |
|---|---:|---:|
| Drums/percussion | 1/8 | 5/8 |
| Other instruments | 2/4 | 4/4 |
| Sax/woodwind | 8/12 | 10/12 |
| Synth/pad/keys | 3/4 | 2/4 |
| Musical vocal | 30/39 | 10/39 |

CLAP produced normalized 512-dimensional embeddings. MERT produced normalized
768-dimensional layer-6 embeddings. MERT won several non-vocal research panels,
but its checkpoint is CC-BY-NC-4.0 and is not eligible as the distributable
default. CLAP remains the registry's distributable candidate and is still
disabled for ownership.

Fusion is disabled because no fused lane was evaluated on the same panel and no
fusion gain was measured.

## Vocal result

CLAP recovered 30/39 held-out vocal files (76.9%). Its two raw vocal claims
among 28 difficult negatives were both outside the learned distribution, so
there were zero in-distribution vocal false positives. However, 38/39 vocal
predictions were also outside the prototype neighborhood. This is useful recall
evidence but not a safe placement contract.

Fine-grained vocal loop, phrase, chop, speech, breath, processed, and one-shot
owners were not trained. Only five clean vocal trainers support this experiment;
loop and one-shot figures are evaluation subgroups, not subtype classifiers.

## Calibration

The logistic calibration implementation and feedback ledger are tested, but no
real calibrator was fitted. Only one nonleaking calibration example exists and
there are fewer than the required 20 reviewed accepted/rejected outcomes.
Fitting now would manufacture confidence.

## Legacy brain ablation

A 45-file trusted Drums/Instruments/FX replay was 45/45 with the full legacy
baby ensemble and 45/45 with that ensemble disabled. Classification time fell
from 51.747 seconds to 42.022 seconds in this run. No lane was retired because
the test does not independently disable core/spread/outlier or cover every
safety veto and fresh-pack panel.

## Additional sample validation

| Panel | Files | Review | Evidence |
|---|---:|---:|---|
| Supplied vocal failure pack | 39 | 0 | Six of six explicit corrections replayed exactly |
| Trusted cross-category labels | 45 | 0 | 45/45 exact |
| Free snare distribution check | 18 | 0 | 18 Drums; no accuracy claim |
| Free guitar distribution check | 26 | 0 | 25 Instruments, 1 FX; one follow-up |
| Mixed FX_Aaron2 distribution check | 128 | 3 | Unlabeled; three memory conflicts queued |

Folder and file names were not used as runtime sorting evidence. The unlabeled
panels are distribution checks, not ground truth.

## Persistent evidence

Compact evidence that must survive ordinary cleanup is in:

```text
neural_artifacts/20260724_real_curation/
```

The principal files are:

- `corpus_provenance.csv`
- `dataset_splits.csv`
- `trainer_contamination_report.csv`
- `trainer_conflict_groups.csv`
- `proposed_training_quarantine.json`
- `human_review_needed.csv`
- `category_readiness_matrix.csv`
- `encoder_leaderboard.csv`
- `neural_model_registry.json`
- `vocal_benchmark.csv`
- `neural_vs_legacy_disagreement.csv`
- `active_learning_queue.csv`
- `brain_lane_ablation.csv`
- `sample_validation_summary.json`
- `gui_vocal_correction_replay.csv`

The full/cache-heavy model and generated run outputs remain excluded from the
patch bundle.

## Cleanup contract

Standard `make clean` may remove generated `_reports`, caches, and scratch
outputs. It must preserve:

- all brain JSON files;
- `training/locked_curated_v1`;
- `_models`;
- `neural_artifacts`.

The cleanup implementation now explicitly protects the two neural roots, and a
regression test covers all four classes of critical assets.

## Verification record

The following gates passed on 2026-07-24:

- `make ai-fix`
- focused GUI, owner-claim, trusted-seed, neural-provider, curation, split,
  benchmark, cleanup, and ablation test files, run individually;
- `make ai-check`, including whole-project compile, Ruff, formatting, generated
  hygiene, and source-name sorting audit;
- `./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE.command`: 32/32;
- `make test`: the complete collected pytest suite;
- `make clean-dry-run`;
- `make clean`;
- post-clean `make ai-check`;
- post-clean GUI and cleanup regressions.

The pre/post-clean integrity comparison covered 371 locked
training/model/neural-artifact files and nine active brain files. Both SHA-256
manifests were byte-identical. Ordinary cleanup removed only generated
`_reports`.

No maintained test was skipped. Two obsolete static-rescue test modules were
removed during the broader architecture migration and replaced by
category-neutral contract, learned-memory, neural-corpus, and locked behavior
coverage.

## Failures and unknowns

- No category has enough source diversity, calibration outcomes, and held-out
  evidence for neural production ownership.
- The general four-way split is too small outside training.
- MERT licensing prevents using the measured checkpoint as a distributable
  default.
- MPS was unavailable, so both encoder benchmarks are CPU-only.
- Lossy/time-stretch near-duplicate detection is not implemented.
- GUI feedback does not yet update a versioned neural prototype index with
  automatic rollback.
- The guitar and mixed-pack follow-up rows need listening-based human verdicts.
- Independent core/spread/outlier ablation switches and broader replay panels
  are still missing.

## Recommended next step

1. Review the 145 queued corpus rows and produce a new versioned curated root.
2. Collect at least 20 nonleaking accepted/rejected outcomes for calibration.
3. Add clean, source-diverse examples for the Tier B/C categories without
   reusing duplicate audio groups.
4. Rebuild frozen prototypes, validation, calibration, and final held-out
   partitions from the new corpus.
5. Add versioned neural-index updates to the GUI correction transaction, with
   rollback.
6. Enable only one small category group after it passes readiness, negative,
   OOD, calibration, shadow, and rollback gates.

## Rollback

The neural owner route is disabled, so removing the downloaded model/cache and
restoring the previous Git commit disables all neural research changes without
changing production placement. The GUI correction repair can be rolled back by
restoring the prior source and dedicated memory JSON files from Git. Do not
delete the current brains or curated trainers during rollback.
