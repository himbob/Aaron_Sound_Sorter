# Audio Intelligence Fix Catalog

Date: 2026-07-17

This catalog records issues discovered during sorter diagnostics and converts
them into architecture-safe work items. It is intentionally not a patch plan for
one file. It is the backlog for making the engine trainable and less brittle.

## P0: Data And Brain Integrity

### Missing Koto One-Shot Label In Active Brain

Finding:

- Training tree contains `Instruments/Plucked Strings/Koto/_ONE_SHOTS`.
- Active full brain does not directly contain
  `Instruments/Plucked Strings/Koto/One Shots`.
- Koto one-shot examples currently recall as `Koto/Loops`.

Fix direction:

- Rebuild active brain family from the current training tree or ensure
  incremental GUI training writes this label into every active brain lane that
  should learn it.
- Add a brain-family validation check that compares training labels against
  active brain labels.

### Conflicting Duplicate Training Audio

Finding:

- One byte-identical file is assigned to both:
  `FX/Human and Voice FX/Mouth Sounds`
  and `Instruments/Voice/Phrase`.

Fix direction:

- Add a training data audit that reports duplicate hashes across different
  labels.
- Let Aaron choose the intended "used-as" role.
- In the future, allow the sidecar to store both `source_like=voice` and
  `usable_role=mouth_fx` without forcing a single truth too early.

## P1: Current Sorter Recall Failures

### Piano One-Shots Become Drums Or Review

Symptoms:

- Piano training rows route to toms, rims/sticks, metallic percussion, or
  `_TO_REVIEW`.

Likely cause:

- Shape/physics detects short struck transients but does not separate
  instrument struck-key identity from drum struck-material identity strongly
  enough.

Fix direction:

- Add a trainable StruckKeysOwnerBrain with negative examples from toms, rims,
  bells, and metallic percussion.
- Keep deterministic guard: if Brain and Physics agree on Piano, a generic
  transient shape should not promote Drums unless drum material evidence is
  strong.

### Kick/Snare Stack Deep Recall Is Weak

Symptoms:

- `Drums/Layered Drum Hits/Kick Snare Stacks` often stays in Drums but lands in
  kick, tom, clap, percussion, rim, or generic drum leaves.

Likely cause:

- Layered drum hits need a compound one-shot owner, not a single-source drum
  leaf shortcut.

Fix direction:

- Add LayeredDrumHitOwnerBrain.
- Train with positives from stacks and negatives from pure kick, pure snare,
  pure clap, tom, rim, and metallic percussion.
- Let folder brain map compound drum owner to layered drum hits or review.

### Voice Phrase One-Shots Can Be Stolen By Drums

Symptoms:

- Short vocal phrases may route to rim/stick, snare, or review.

Likely cause:

- Voice proof is not strong enough when the clip is short and percussive.

Fix direction:

- Add VoicedPhraseOwnerBrain and VocalChopOwnerBrain.
- Train negative examples against drum hits, UI blips, formant FX, and animal
  calls.
- Let short percussive envelope be a warning, not an automatic drum owner.

### FX Impacts May Be Pulled Into Drums

Symptoms:

- FX impact training examples route to kicks or toms.

Likely cause:

- Drum parent firewalls are protecting percussive shapes too broadly.

Fix direction:

- Add DesignedImpactFXOwnerBrain with negatives from kick, tom, snare, and clap.
- Require impact owner evidence such as tail, width, cinematic body, debris,
  or non-drum material before FX wins.
- Keep clean short centered drum hits protected.

### Human Voice FX Versus Instruments Voice Is Ambiguous

Symptoms:

- Some `FX/Human and Voice FX` examples route to `Instruments/Voice`.

Likely cause:

- The current model only has final folder truth, but the better representation
  is `source_like=human_voice` plus `usable_role=voice_fx` or
  `usable_role=clean_voice_phrase`.

Fix direction:

- Sidecar must split `source_like` and `usable_as`.
- GUI corrections should eventually collect both final folder and role/owner
  intent.

## P2: Architecture Work

### Sidecar Evidence Export

Why:

- The sorter already computes useful evidence but leaves too much of it buried
  in CSV columns and debug JSONL.

Fix direction:

- Add versioned sidecar schema.
- Write one JSONL row per file.
- Keep it read-only at first.

### Trainable Owner Brain Interface

Why:

- Hard-coded score variables are brittle and do not learn negative examples.

Fix direction:

- Introduce trainable brain protocols for trait/shape/owner evidence.
- Start with a prototype-based brain over numeric vectors.
- Support positive and negative examples, weights, and provenance IDs.
- Run read-only beside existing routing.

### Active Learning Queue

Why:

- Aaron should not have to label everything.

Fix direction:

- Queue owner conflicts, high-confidence wrong recalls, repeated reviews, and
  unknown clusters.
- Corrections should create positive and negative examples.

### Brain Family Validation

Why:

- Current active brain can lag behind the training tree.

Fix direction:

- Compare training labels to full/core/spread/outlier brain labels.
- Report missing labels and low-count labels.
- Block or warn before GUI training if active brain family is stale.

## P3: Product Expansion

### Loop Analyzer

Build from sidecar and music-property analyzers. It should not wait for perfect
sorting.

### Stem Intelligence

Do not promise perfect separation. Use component evidence to recommend one of:

- describe-only
- MIDI/reconstruction
- broad audio stems
- hybrid
- no separation recommended

### Search / Flavor Browser

Index sidecars by traits, music properties, components, warnings, and roles.

## Guardrails For Every Future Fix

- No filename or source-path evidence in production routing.
- New brains must start read-only.
- Every owner brain needs positive and negative examples.
- Safety gates remain deterministic for broken/tiny files and catastrophic
  family theft.
- A folder change should be backed by owner permission, not only a shape score.
- A review with useful evidence is better than a confident bad folder.
