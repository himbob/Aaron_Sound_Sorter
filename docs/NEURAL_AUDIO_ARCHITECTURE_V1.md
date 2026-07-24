# Aaron Sound Sorter — Neural Audio Architecture v1

Status: **authoritative migration design**
Date: 2026-07-23

## 1. Problem being fixed

The existing production sorter has useful assets—curated folders, corrected examples,
measured audio facts, manifests, and regression audio—but its final semantic policy has
accumulated inside a roughly 16,000-line `FamilyClaimArbiter`. Category knowledge,
safety policy, confidence gates, and rescue behavior are entangled. That is brittle
software and it prevents human training from becoming the primary source of category
knowledge.

The migration does **not** throw away the whole product. It keeps the good high-level
workflow:

```text
curated category examples
    -> multiple learned prototypes
    -> nearest-category evidence
    -> margin and uncertainty
    -> confident placement or Review
```

The layer being replaced is semantic representation and category policy.

## 2. Target pipeline

```text
Audio bytes
  |
  +--> Frozen neural encoder lanes
  |      - CLAP music/speech model
  |      - optional music/audio encoder such as MERT or BEATs
  |      - each lane evaluated independently
  |
  +--> Objective measured facts
  |      - broken/silent audio
  |      - duration
  |      - loop versus one-shot structure
  |      - tempo/pulse when measurable
  |      - pitch/key when measurable
  |      - stereo/phase integrity
  |
  v
Per-label learned prototype models
  - several neighborhoods per category
  - loose outlier filtering
  - preserve subtype variety
  - human corrections become examples
  |
  v
Per-label encoder selection
  - held-out evaluation only
  - no blind concatenation or averaging
  - fuse only after measured improvement
  |
  v
Uncertainty and out-of-distribution evidence
  - top similarity
  - top-two margin
  - prototype radius/spread
  - label example count
  - objective structure agreement
  |
  v
Small calibrated trust model
  - logistic calibration first
  - human review outcomes become new calibration examples
  |
  v
Small safety boundary
  - invalid audio
  - impossible structural contradiction
  - phase/stereo safety
  - no category-specific rescue forest
  |
  v
Placement or _TO_REVIEW
```

## 3. Non-negotiable safety contracts

### 3.1 Source-name blindness

Production classification must not use filenames, source folders, ZIP member names,
pack names, regex hints, or inherited old folder labels. Names are allowed only for:

- opening files;
- manifest display;
- selecting test fixtures;
- deliberate human-curated training folder labels;
- post-decision diagnostics.

The embedding cache is keyed by file SHA-256 and model identity, never by filename.

### 3.2 Training and evaluation separation

`review_preview` and `heldout_eval` are separate data paths from the first version.
No audio hash present in prototype construction may appear in held-out evaluation.
A high cosine score on a reused training example is not evaluation.

The implemented split contract also hashes canonical decoded float32 PCM with
sample rate, channel count, and frame count. This rejects lossless copies stored
in different containers. Perceptually transformed or lossy near-duplicates
still require a later fingerprint lane.

### 3.3 Representation lanes do not silently dilute each other

CLAP, MERT, BEATs, or later encoders are scored independently per label. The system
keeps a per-label held-out leaderboard. It does not concatenate or average embeddings
by default. Fusion is permitted only when a held-out experiment beats the strongest
single lane by a declared minimum improvement.

The first fusion gate requires the same held-out examples, at least ten examples for the
label, a minimum top-1 accuracy gain of 0.02, and no material loss of known-distribution
coverage. Those defaults are inspectable configuration, not category rules.

### 3.4 Neural evidence is not permission to ignore physical facts

DSP remains appropriate for objective observations. It must not be used as a semantic
rule engine. For example:

- onset repetition can support loop versus one-shot;
- phase correlation can identify unsafe low-end stereo;
- a frequency threshold cannot decide that a sound is a saxophone or vocal.

### 3.5 Review is uncertainty, not a contradiction trashcan

Review is used when the learned model is out of distribution, the margin is weak, or
objective structure conflicts with the learned label. Review must record the evidence.
It must not be the automatic result of old brains disagreeing with each other.

### 3.6 Human review compounds into training signal

Every accepted or rejected neural prediction may be logged as a calibration example.
The trust calibrator is retrained periodically from those outcomes. Manual review is
therefore not wasted labor.

### 3.7 Human corrections are authoritative decisions, not infallible corpus labels

A user correction should be respected for that decision, but repeated or old corrections may
still contaminate a reusable category model. Before trainers are promoted into neural
prototypes, the corpus is audited source-blind with leave-one-out neighborhood evidence:

- identical bytes assigned to multiple labels are rejected;
- singletons are marked prototype-only;
- examples supported more strongly by another label are flagged;
- statistical same-label outliers are surfaced for human review.

This audit is not held-out accuracy. It is a training-data hygiene report.

## 4. Prototype policy

Each category may contain several legitimate modes: tenor/alto/baritone sax, dry/wet
snare, acoustic/synthetic percussion, clean/processed vocals, tonal/noisy risers.
One global centroid would average those modes into mud.

The initial builder therefore:

1. L2-normalizes embeddings.
2. Deduplicates exact audio hashes.
3. Applies a loose tail filter only for categories with at least 20 examples.
4. Keeps at least 95% of examples by default.
5. Chooses one to six prototypes based on example count.
6. Uses deterministic spherical k-means with farthest-point seeds.
7. Stores prototype membership, mean distance, and 95th-percentile radius.

The serialized index is built in a sibling temporary directory. When replacing
an existing index, the old directory is retained until the new directory is
installed; a failed final replace restores the old index.

This policy is generic. It does not contain sax, snare, vocal, or FX rule tables.

## 5. Encoder policy

### Initial production candidate

`laion/larger_clap_music_and_speech` is the first CLAP candidate because it exposes
512-dimensional audio embeddings, is trained specifically for music and speech, and
has an Apache-2.0 model card. Model snapshots must be pinned and prefetched before a
production release.

### Experimental comparison lanes

- MERT is music-specific and potentially useful for subtle instrument/drum identity,
  but the commonly published Hugging Face model card is CC-BY-NC and requires remote
  model code. It remains an experiment until licensing and packaging are resolved.
- BEATs is a general audio encoder and may be useful for sound-event/timbre classes.
- Essentia MusiCNN/MAEST/OpenL3 models are valid lighter comparison lanes, especially
  if ONNX or TensorFlow deployment is easier on Aaron's Mac.

## 6. What is explicitly deferred

- Demucs or BS-RoFormer on every sample: deferred; useful mainly for mixed/full tracks.
- Audio-LLMs as final classifiers: rejected for first production ownership because they
  are heavy and less deterministic. They may generate optional review explanations.
- FAISS/Voyager: deferred until exact NumPy cosine search is measurably too slow.
- Fine-tuning a foundation model: deferred until the project has a large, clean,
  leakage-safe labeled corpus.

## 7. Current code seam

The first implementation lives in:

```text
src/aaron_sound_sorter/neural_audio/
```

It builds indexes, runs held-out evaluation, audits trainers, logs feedback, and
compares with legacy manifests. In the GUI, a known learned CLAP neighborhood
may own the proposal when measured structure is compatible. Unknown
cross-family conflicts go to Review. This is limited ownership, not general
neural authority.

The command-line entry point is `tools/neural_audio_lab.py`. Mac helper commands live
under `commands/neural/`. Model download is disabled by default inside providers; a
reviewed model snapshot must be deliberately prefetched and then referenced by local path.

## 8. Verified cache identity

One cache entry is an atomic `.npz` containing both normalized vector and JSON
metadata. Its directory key commits to:

- audio byte SHA-256;
- provider ID and upstream model ID;
- pinned model revision;
- deterministic local snapshot hash;
- preprocessing contract;
- embedding schema version.

A changed processor, revision, snapshot, or embedding schema cannot reuse an
older vector.

## 9. Measured Phase 1 boundary

Real pinned CLAP inference now runs end to end. The first broad-vocal experiment
improved held-out recall over the legacy `Instruments/Voice` family
(30/39 versus 21/39). Two of 28 protected negatives had a raw musical-vocal
top-1, but both were outside the prototype neighborhood. In addition, 38/39
vocal predictions were OOD. The architecture therefore behaves as designed:
the original evidence did not justify broad ownership. Limited GUI ownership
now applies only to source-blind, human-approved learned neighborhoods.
