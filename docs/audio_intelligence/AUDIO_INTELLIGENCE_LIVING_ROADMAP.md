# Aaron Audio Intelligence Living Roadmap

Date: 2026-07-17

This is the active design tracker for moving Aaron Sound Sorter toward the
broader Aaron Audio Intelligence Platform described in
`AARON_AUDIO_INTELLIGENCE_PLATFORM_MASTER_DESIGN_ALL_IDEAS_20260717.md`.

## North Star

The sorter should remain a simple producer-facing product, but the engine under
it should become a reusable audio intelligence core.

The core should answer:

- what is physically happening in this audio
- what musical role it plays
- what source or material it resembles
- what components are present
- what music properties are useful
- what is unknown or novel
- which trainable brain should learn from a correction

The final folder is one product output, not the only truth.

## Research Notes

Useful outside references from the current research pass:

- AudioSet is a hierarchical ontology and large labeled audio-event dataset:
  https://research.google.com/audioset/ontology/
- The AudioSet ICASSP 2017 paper describes broad hierarchical audio event
  recognition and human-labeled classes:
  https://research.google/pubs/audio-set-an-ontology-and-human-labeled-dataset-for-audio-events/
- YAMNet predicts 521 AudioSet classes and exposes embeddings from log-mel
  spectrogram patches:
  https://www.tensorflow.org/hub/tutorials/yamnet
- The TensorFlow YAMNet README documents 16 kHz mono audio, 25 ms STFT windows,
  64 mel bins, 0.96 second patches, 1024-d embeddings, and class scores:
  https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/README.md
- Librosa already exposes descriptor families we can use before heavier models:
  spectral features, MFCC, chroma, tempo, tempogram, RMS, flatness, contrast,
  rolloff, and related transforms:
  https://librosa.org/doc/latest/feature.html
- Essentia provides broad descriptor families for MIR: spectral, temporal,
  tonal, rhythm, high-level descriptors, and ML model integration:
  https://essentia.upf.edu/documentation.html
- PANNs, CLAP, and BEATs support the idea that learned audio embeddings should
  become evidence lanes, not hand-written final routing replacements:
  https://huggingface.co/papers/1912.10211
  https://huggingface.co/papers/2206.04769
  https://huggingface.co/papers/2212.09058

Design implication: use learned embeddings and trainable specialist brains as
evidence producers. Keep deterministic safety gates for catastrophic errors.

## Current Sorter Findings To Carry Forward

Training-tree recall diagnostic run:

- input: `training/locked_curated_v1`
- rows: 203
- exact expected label: 118 / 203, 58.1%
- same top family: 178 / 203, 87.7%
- review: 9 / 203, 4.4%
- wrong top family: 16 / 203, 7.9%
- report: `_reports/training_recall_sort_probe/full_training_tree/training_recall_summary.txt`

Important failures:

- Piano one-shots are still too often routed to Drums or Review.
- `Drums/Layered Drum Hits/Kick Snare Stacks` mostly stays in Drums, but deep
  recall is weak and often lands in kick/tom/percussion/clap.
- Voice phrase one-shots can still be stolen by drums or review.
- FX impact examples may be pulled into Drums by parent firewalls.
- FX human/voice examples may be pulled into Instruments/Voice by true-voice
  protection. This may be musically sensible, but it means training labels and
  product intent need a clearer "source-like" versus "used-as" split.
- Active full brain is missing `Instruments/Plucked Strings/Koto/One Shots`,
  even though the training tree contains that label.
- One byte-identical audio file is present under conflicting labels:
  `FX/Human and Voice FX/Mouth Sounds/_ONE_SHOTS/HousePiano4-Vocal_125bpm_Cm.wav`
  and `Instruments/Voice/Phrase/_ONE_SHOTS/HousePiano4-Vocal_125bpm_Cm.wav`.

## Architecture Direction

Move from:

```text
static physics variables + fixed claim rules + folder brain
```

to:

```text
feature packets
  -> trait brains
  -> shape brains
  -> owner brains
  -> component brains
  -> music property analyzers
  -> open-world novelty brain
  -> sidecar
  -> product adapters
```

Rules:

- Voters/brains produce evidence.
- Owner brains authorize which family or role can own the sound.
- Folder mapping converts authorized owners into producer-friendly folders.
- Safety gates block catastrophic theft.
- Review remains a first-class output.
- Production sorting must stay source-name blind.

## Phases

### Phase 0: Stabilize Current Sorter

Status: in progress.

Goals:

- keep Drums and Instruments protected from FX vacuum behavior
- keep source-name audit passing
- keep acceptance and smoke panels as canaries
- stop exact training examples from missing obvious parent families

Open work:

- piano one-shot vs struck percussion
- voice one-shot vs percussive transient
- FX impact vs drum hit policy
- snare/kick-stack deep recall
- missing Koto one-shot active brain label

### Phase 1: Sidecar Schema And Evidence Export

Status: implemented as a read-only foundation.

Goal:

Write read-only JSON evidence sidecars from sorter results without changing
routing.

Initial fields:

- schema version
- final sort
- owner scores
- source-like scores
- usable-as placeholders
- shape summary
- component placeholders
- music placeholders
- warnings
- training hints

Exit criteria:

- old folder output still works
- manifest remains readable
- sidecar is JSON and versioned
- tests cover schema serialization

### Phase 1.5: GUI Correction Memory Brain

Status: first product increment implemented.

Goal:

Make GUI corrections immediately useful without forcing a full retrain. When a
user clicks "Train Brains From Corrections", the updater now keeps the existing
full/core/spread/outlier incremental updates and also writes a dedicated
`stage4_folder_brain_user_memory.json` brain lane.

Behavior:

- correction memory is source-name blind and stores measured fingerprints
- the sorter loads the memory lane automatically when present
- the memory lane participates in brain ensemble voting as `user_memory`
- the memory examples are merged into the in-memory full brain view so
  BrainVoter and PhysicsVoter can both use human-override recall
- normal measured eligibility and arbitration still apply; this is not a final
  folder rescue

Why this matters:

Human corrections now become an explicit trainable teacher brain instead of
being diluted inside the large historical brain. This is the first step toward
making more of the system trainable: owner brains, shape brains, role brains,
and physics-like measured specialist brains.

### Phase 1.6: Brain-First Owner Authority

Status: implemented as the next product increment.

Goal:

Let matched trainable memory own deep labels across Drums, Instruments, and FX
when measured top-family/body evidence is compatible. Static code remains a
lightweight safety contract instead of a hidden leaf classifier.

Behavior:

- `LearnedOwnerAuthorityClaimProducer` now emits generic learned-owner claims,
  not only Voice claims.
- Voice keeps its compatibility source id, but Koto, Piano, Synth, Sax, FX,
  Drum, and other trained labels can now become real owner claims too.
- The memory-owned label is preserved as the deep target.
- Static code only blocks broad contradictions such as hard Drum body versus
  clean Instrument memory, or non-Voice memory using Voice body evidence.
- Broad fallback buckets such as `Instruments/Instrument Loops/Loops` should
  not erase high-confidence human-trained owner memory.

Design contract:

- Trainable owner memory owns source identity.
- Shape memory owns shape.
- Physics memory owns physical branch/top-family calibration.
- Voter-role memory owns reusable role evidence.
- The arbiter chooses among legal claims and sends conflicts to review.

See `BRAIN_FIRST_OWNER_CONTRACT_20260720.md`.

### Phase 2: Read-Only Music Properties

Goal:

Add BPM, tempo candidates, pitch/root, key confidence, and "not applicable"
policy.

Do not route based on this phase yet.

### Phase 3: Chord Timeline MVP

Goal:

Estimate chords only for clean tonal loops where shape/owner evidence supports
music analysis.

### Phase 4: Component Analyzer MVP

Goal:

Report likely components inside loops without changing sort placement.

### Phase 5: Owner Brain Interfaces

Status: started.

Goal:

Create a trainable owner-brain interface with positive and negative examples.
The first implementation is read-only and prototype-based so it can be tested
without model installation.

### Phase 6: DesignedMotionFXOwnerBrain Pilot

Goal:

Train a specialist FX motion owner on positives and negatives.

Required counterexamples:

- drum loops
- synth arps
- sax phrases
- guitar loops
- full music loops
- bass loops

### Phase 7: Protector Owner Brains

Goal:

Train owners that protect obvious real sources:

- DrumLoopOwnerBrain
- DrumOneShotOwnerBrain
- CleanInstrumentPhraseOwnerBrain
- MixedMusicLoopOwnerBrain
- VoicedPhraseOwnerBrain
- BassFoundationOwnerBrain

### Phase 8: AmbientTail / Degraded Instrument Brain

Goal:

Preserve source-like identity while allowing usable role to become texture or
FX.

### Phase 9: Unknown / Novelty Brain

Goal:

Cluster and describe sounds that should not be forced into known folders.

### Phase 10: Loop Analyzer Adapter

Goal:

Expose loop-focused reports from the same sidecar/evidence core.

### Phase 11: Stem Intelligence Prototype

Goal:

Recommend MIDI, reconstruction, broad stems, describe-only, or no separation.

### Phase 12: Search / Flavor Browser

Goal:

Search by traits, components, music properties, roles, and warnings.

## Immediate Implementation Rule

Until Phase 1 and Phase 5 are stable, new trainable brains must run in
diagnostic/read-only mode. They may write sidecar evidence and reports, but they
must not change final folders unless a later patch explicitly enables routing
and passes category stability gates.
