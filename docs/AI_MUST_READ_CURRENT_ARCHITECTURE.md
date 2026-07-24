# AI Must Read — Current Architecture Direction

The production sorter still runs the legacy voter/claim/arbiter path for compatibility.
That is **not** the target architecture.

The authoritative target is:

1. frozen neural audio embeddings;
2. human-curated multi-prototype categories;
3. held-out per-label encoder evaluation;
4. explicit uncertainty/OOD evidence;
5. feedback-trained confidence calibration;
6. a small objective safety boundary;
7. placement or Review.

Read first:

- `NEURAL_AUDIO_ARCHITECTURE_V1.md`
- `NEURAL_AUDIO_MIGRATION_PHASES.md`
- `NO_SOURCE_NAME_SORTING_POLICY_20260515.md`
- `TESTING_POLICY.md`

## Current implementation boundary

`src/aaron_sound_sorter/neural_audio/` now has limited GUI proposal authority.
CLAP may own known learned neighborhoods when measured structure is compatible.
Unknown cross-family conflicts go to Review. The legacy `FamilyClaimArbiter` is
a compatibility component slated for measured retirement, not a place to add
new category rules.

## Non-negotiable rules

- Do not add category-specific branches to the giant arbiter.
- Do not use source filenames or paths as classification evidence.
- Do not evaluate on examples used to build prototypes.
- Do not average weak and strong encoders without per-label held-out proof.
- Do not delete old brains until lane-reduction replay proves redundancy.
- Do not expand CLAP ownership beyond learned neighborhoods without held-out
  evidence.
