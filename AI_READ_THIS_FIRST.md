# AI Read This First

Aaron Sound Sorter is migrating away from the giant category-rule arbiter
toward source-blind neural embeddings and human-curated multi-prototype models.

Read in this order:

1. `CURRENT_STATUS.md`
2. `docs/NEURAL_TRAINING_CURATION_HANDOFF_20260724.md`
3. `docs/NEURAL_AUDIO_IMPLEMENTATION_STATUS_20260724.md`
4. `docs/NEURAL_AUDIO_VERIFICATION_HANDOFF_20260724.md`
5. `docs/NEURAL_AUDIO_ARCHITECTURE_V1.md`
6. `docs/NEURAL_AUDIO_MIGRATION_PHASES.md`
7. `docs/NEURAL_AUDIO_VOCAL_REPRO_20260723.md`
8. `docs/NO_SOURCE_NAME_SORTING_POLICY_20260515.md`
9. `docs/TESTING_POLICY.md`
10. `docs/ARCHITECTURE_DOCUMENT_STATUS.md`

The legacy sorter remains operational during shadow evaluation. Exact
dual-memory GUI supervision may own the precise label the human selected, but
frozen CLAP/MERT prototypes still have no placement authority. Do not add
category rescues to `FamilyClaimArbiter`. Do not use filenames or source paths
as inference evidence. Do not evaluate on prototype-building examples. Do not
promote CLAP while the measured OOD gate sends 38/39 held-out vocal files to
Review. Do not delete old trainers or brain lanes from the contamination
reports; rebuild from a reviewed, versioned corpus and retain rollback artifacts.
