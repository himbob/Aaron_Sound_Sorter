# Current status

Updated: 2026-07-24

## Working

- CLI sorting with local trained brains.
- Local GUI review and supervised corrections.
- Durable, source-blind GUI corrections feeding versioned CLAP prototypes.
- Limited CLAP ownership across all categories inside known neighborhoods.
- Cross-family neural/legacy conflicts routed to Review.
- Exact human-approved memory ownership across Drums, Instruments, and FX.
- Source-name blindness audit.
- CLAP and MERT embedding, cache, prototype, and evaluation tooling.
- Provenance, contamination, readiness, active-learning, and ablation reports.

## Production authority

- Measured evidence and trained memory: active.
- CLAP ownership: active only for known learned neighborhoods with compatible
  measured structure.
- Unknown or conflicting neural evidence: Review or legacy fallback.
- Legacy arbiter: temporary compatibility and safety infrastructure.

## Neural evidence

- CLAP vocal recall: 30/39 on the held-out benchmark.
- CLAP vocal OOD: 38/39.
- Production categories at readiness Tier A: 0.
- Real confidence calibration: not fitted; reviewed outcomes are insufficient.
- MERT is research-only because its checkpoint is non-commercial.

## Public repository

Not distributed:

- sample audio;
- trained brains;
- training trees;
- model weights;
- private neural evidence;
- local machine paths.

The public checkout supports development and self-test. Production sorting
requires a compatible local brain.

## Next

- Human-review contaminated trainers.
- Build a new versioned curated corpus.
- Collect nonleaking calibration outcomes.
- Validate one small category group for neural ownership.
- Retire legacy lanes only after independent ablation and rollback testing.

See `docs/NEURAL_AUDIO_ARCHITECTURE_V1.md`.
