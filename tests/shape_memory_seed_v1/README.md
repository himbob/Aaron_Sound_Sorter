# Shape Memory Seed v1

This folder defines a small, explicit teacher panel for the trainable
ShapeVoter memory brain.

The panel is not an acceptance gate. It is training seed data. Each case maps a
known local audio fixture to an approved taxonomy label. The label is used only
as the supervised target for the broad shape memory lane; runtime matching still
uses measured audio fingerprints, not filenames or source folders.

Use dry-run first:

```bash
./commands/training/RUN_SHAPE_MEMORY_SEED_DRY_RUN.command
```

If the report looks right, apply:

```bash
./commands/training/RUN_SHAPE_MEMORY_SEED_APPLY.command
```

Generated shape-memory brains are local model state and are intentionally
ignored by git.
