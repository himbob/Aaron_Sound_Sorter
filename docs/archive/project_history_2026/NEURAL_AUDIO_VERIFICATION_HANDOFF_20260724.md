# Archived: Neural Audio Verification Handoff — 2026-07-24 UTC

## Decision

CLAP now works end to end on Aaron's Mac and improves broad vocal-family recall
in shadow mode. It is not ready for production ownership because the category
has only five approved trainers and 38/39 held-out vocal predictions are outside
the learned prototype neighborhood.

No legacy brain or curated trainer was deleted or modified during verification.

## Phase A — patch audit

### Verified

- The current working tree contains the v31.175 exact UI-training authority,
  v31.176 adaptive tolerance, and v31.177 neural shadow source/tests.
- Neural provider/cache/split/prototype/evaluation/calibration/audit/shadow
  modules are isolated from production routing.
- Focused neural tests execute one file at a time.
- Model loading is network-disabled by default.
- The source-name audit includes the neural package.

### Broken

- The root `INSTALL_CHANGED_FILES.command` was byte-identical to the original
  v31.174 handoff installer and did not contain the neural or v31.175/v31.176
  cumulative file set.
- The initial cache key omitted model revision, snapshot, preprocessing, and
  embedding schema.
- The initial cache committed vector and metadata separately.
- Explicit independent held-out roots and decoded-audio leakage checks were absent.
- The trainer CLI rejected duplicate labels before it could report them.
- The shadow CSV omitted required second-prediction and prototype/OOD evidence.
- Prototype-index replacement could strand the previous index in a backup if
  final installation failed.

### Missing from supplied inputs

- `/mnt/data` did not exist.
- The claimed `Aaron_Sound_Sorter_v31_177_neural_audio_shadow_architecture_patch.zip`
  could not be found.
- The original handoff was recovered from
  `_reports/bundles/Aaron_Sound_Sorter_AI_HANDOFF_WITH_BRAINS_20260723_120611.zip`.
- The real pack was found at
  `/Volumes/T9/music_production/samples/Premium Deep House Vocals.zip`.

### Repaired in this patch

- Complete cache/model identity and atomic single-file cache commits.
- Pinned CLAP revision in providers and Mac helper commands.
- Independent held-out root support plus byte and decoded-audio leakage rejection.
- Category spread/prototype support in OOD and shadow evidence.
- Atomic prototype-index rollback with failure-injection coverage.
- Duplicate-preserving trainer audits and all-brain legacy memory inventory.
- Reproducible experiment summary and supplemental confuser tools.
- Bundle selection of `.gitignore`, `AI_READ_THIS_FIRST.md`, and `CURRENT_STATUS.md`.

## Pinned CLAP runtime

```text
Source model: laion/larger_clap_music_and_speech
Revision: 195c3a3e68faebb3e2088b9a79e79b43ddbda76b
Weight file SHA-256: d4e5cf6317c7521ca62c11b524f5646565310e040129fe35719cad661696d745
Provider output: 512-dimensional normalized audio embedding
Local model footprint: approximately 745 MB
Isolated verification environment: approximately 732 MB
```

The provider model ID also records local snapshot hash
`b3cc57c40aa913ea508cde7dffcccbfbfa88fe36e359577c40e9c272f989bb93`,
the preprocessing contract, and embedding schema.

## Clean vocal experiment

### Split

```text
review_preview trainers: 45
heldout files: 67
heldout pack vocals: 39
protected negative fixtures: 28
byte overlap: 0
canonical decoded-audio overlap: 0
```

The primary experiment uses broad labels because the available clean trainers
do not support reliable vocal-loop/phrase/chop/one-shot/speech/breath/processed
submodels. Loop and one-shot are reported as subgroups only.

### Measured results

| Panel | Correct/total | Rate | OOD/Review |
|---|---:|---:|---:|
| Musical vocal, all | 30/39 | 76.9% recall | 38/39 |
| Vocal loops | 21/29 | 72.4% | 28/29 |
| Vocal one-shots | 9/10 | 90.0% | 10/10 |
| Sax/woodwind | 8/12 | 66.7% top-1 | 4/12 |
| Synth/pad/keys | 3/4 | 75.0% top-1 | 3/4 |
| Drums/percussion | 1/8 | 12.5% top-1 | 7/8 |
| Other instruments | 2/4 | 50.0% top-1 | 3/4 |

Two of 28 protected negatives had musical vocal as raw top-1 (7.1%): one drum
fixture and one synth/pad fixture. Both were OOD, so in-distribution musical
vocal false positives were 0/28. The low non-vocal multiclass accuracy shows
that this tiny broad experiment is not a general sorter replacement.

### Legacy comparison on the supplied pack

The legacy run processed all 39 files and placed 21 under `Instruments/Voice`,
3 in `_TO_REVIEW`, and 15 elsewhere. Against the pack's explicit broad-vocal
ground truth:

```text
both correct: 18
CLAP only: 12
legacy only: 3
neither: 6
```

The raw disagreement report shows 39/39 string-level disagreements because the
neural label is broad and legacy folders are granular. Use the broad-family
comparison above for the meaningful human-correct win count.

### Supplemental disputed-label confusers

A separate 35-file panel excludes every primary experiment hash. Its source
labels come from the contaminated legacy corpus, so it is diagnostic only:

- no voice claim for 8 sax, 7 synth/pad, 8 mixed loops, 3 texture beds, or the
  one remaining altered-voice file;
- all 8 spoken-voice files predicted musical vocal, but all 8 were OOD;
- zero in-distribution voice claims across the panel;
- no breath/mouth file remained after overlap exclusion;
- no explicit Formant FX taxonomy folder exists.

## Trainer contamination

The 315-file CLAP leave-one-out audit is not held-out accuracy. It found:

```text
supported: 170
cross-label conflict: 71
exact duplicate-hash label conflict: 11
possible label outlier: 9
ambiguous cross-label support: 5
limited support: 21
singleton: 28
```

The 74 vocal/texture/instrument-loop focused rows include 26 cross-label
conflicts and 6 exact duplicate-label conflicts. In particular, every one of
the six Altered Voice trainers is supported more strongly by another label.

The separate static inventory scanned all nine brain JSON files:

```text
trainer occurrences: 10,949
stored identities: 6,929
conflicting identities: 52
missing/stale source-path occurrences: 8,561
```

### Reversible cleanup plan

1. Listen to every conflict and record a human verdict outside the locked corpus.
2. Copy approved examples into a new versioned curated root.
3. Put rejected/unresolved identities into a quarantine manifest.
4. Rebuild full/core/spread/outlier and dedicated memory brains from that root.
5. Retain all current brains as rollback artifacts until held-out replay passes.

## Evidence locations

Generated evidence stays outside the patch payload under:

```text
_reports/neural_audio/vocal_shadow_20260724_001/
```

Key files:

- `verified_reports/verified_experiment_summary.json`
- `verified_reports/verified_file_comparison.csv`
- `verified_reports/verified_subgroup_metrics.csv`
- `verified_reports/verified_supplemental_confusers.csv`
- `clap_vs_legacy_shadow.csv`
- `trainer_contamination/clap_trainer_audit.csv`
- `trainer_contamination/clap_vocal_related_trainer_audit.csv`
- `trainer_contamination/legacy_brain_trainer_audit.csv`

## Reproduction commands

Install the isolated runtime and prefetch the pinned model:

```bash
commands/neural/INSTALL_NEURAL_LAB.command
commands/neural/PREFETCH_CLAP_MODEL.command
```

Build with deliberately separate roots:

```bash
commands/neural/BUILD_CLAP_SHADOW_INDEX.command \
  /path/to/review_preview \
  _reports/neural_audio/runs/clap_index \
  /path/to/heldout_eval
```

Run trainer and shadow reports:

```bash
commands/neural/AUDIT_CLAP_TRAINERS.command \
  training/locked_curated_v1 \
  _reports/neural_audio/runs/clap_trainer_audit.csv

commands/neural/RUN_CLAP_SHADOW_DIFF.command \
  _reports/neural_audio/runs/clap_index/index \
  "/path/to/Premium Deep House Vocals.zip" \
  /path/to/legacy_manifest.csv \
  _reports/neural_audio/runs/clap_vs_legacy.csv
```

## Rollback

Production routing was not changed, so disabling or removing `_reports/neural_audio`
artifacts fully disables this experiment. Do not delete the current brain JSON files.

The cumulative patch installer overwrites only its manifest-listed source files
and intentionally creates no backup. Before installation, preserve the project
directory or a Git commit. To roll back, restore that snapshot or reinstall the
original handoff bundle, then remove only the newly created neural report/runtime
directory if desired.

## Next gates

1. Human-review the 32 focused vocal conflicts before rebuilding trainers.
2. Add clean, non-overlapping breath, mouth, processed/formant, texture, and
   mixed-loop examples with explicit verdicts.
3. Grow each candidate owner category beyond five trainers.
4. Collect human verdicts and train/evaluate the logistic confidence calibrator.
5. Compare one legally/package-compatible music-focused encoder per label.
6. Run the brain-lane ablation replay.
7. Consider limited neural ownership only after later held-out recall,
   false-positive, calibration, and OOD gates all pass.
