# Stage 4 v0.4.63 Existing Project Update

This update is designed to update the existing Aaron Sound Sorter project in place:

`/Users/your-name/Documents/Codex/2026-04-25/files-mentioned-by-the-user-create/Aaron_Sound_Sorter`

It does not create a second working project folder.

## Included runtime changes

- Keeps folder-supervised `train-brain` and `sort` commands.
- Keeps support-aware folder balancing.
- Keeps adaptive label models:
  - exemplar-only for tiny folders
  - single centroid for uniform folders
  - multi-centroid for medium spread folders
  - multi-centroid plus exemplars for messy folders
- Keeps physical structure conflict gate so loop-like/multi-event audio should not auto-place into learned one-shot folders when the learned structure head also says loop.
- Adds a logs-only runner that avoids uploading sorted audio files.

## Main command after installation

From the project root or `commands` folder:

```bash
./RUN_PHASE4_v0463_TRAIN_SORT_LOGS_ONLY.command
```

From the `commands` folder:

```bash
./RUN_PHASE4_v0463_TRAIN_SORT_LOGS_ONLY.command
```

The runner writes real sorted audio under the run folder, but zips only logs/reports for upload.
