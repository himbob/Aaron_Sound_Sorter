# Stage 4 v0.4.57 train-all anchor recall status

This bundle is a direct follow-up to the FX_Aaron2 review where training-folder examples such as `Guiro One Shot.wav` could still recall as claps or rim/hat neighbors.

## Architecture changes

- Root runtime remains consolidated in `Aaron_Sound_Sorter.py`.
- `src/aaron_stage4/phase3_pure_brain_lab.py` remains a compatibility shim only.
- Folder path remains the training truth. Filenames are not labels and are not routing evidence.
- `--max-train-per-group 0` now means train every readable row.
- Eval rows in train-all mode are explicitly marked as diagnostic reused training recall, not holdout proof.
- The brain now stores full internal teacher anchors by label in addition to centroids and selected exemplars.
- Adaptive per-folder modeling remains: tiny folders use exemplar/anchor behavior, uniform folders use single centroid, messy folders use multi-centroid plus anchors.
- Tiny one-shot event detection now counts a high-energy short file as one primary event when envelope-rise detection would return zero.
- Risky guess output folder is renamed to `DIAGNOSTIC_RISKY_GUESSES_NOT_A_SORT` so it is not mistaken for official sorting.
- New reports:
  - `training_recall_contract_report.csv`
  - `training_recall_contract_summary.json`

## Important interpretation

Training recall is not holdout accuracy. It answers only this contract:

> If a readable file is in Aaron's trusted training folder tree, can the trained brain match that same fingerprint back to its folder-derived label?

If this report fails, the model representation is broken even before blind sorting.

## Quick local test

```bash
python3 -S -m py_compile Aaron_Sound_Sorter.py src/aaron_stage4/phase3_pure_brain_lab.py
python3 tests/test_stage4_adaptive_label_model.py
python3 tests/test_stage4_training_folder_interpreter.py
python3 tests/test_stage4_physical_role_review_guards.py
python3 -S tests/test_stage4_code_sanity_audit.py
python3 Aaron_Sound_Sorter.py self-test --tmp-root /tmp/ass_v0457_selftest
```


## Additional small-file check run here

I did not run FX_Aaron2 or any large ZIP locally for this bundle.

I ran a tiny individual training-recall check with:

- `Guiro One Shot.wav` as `Drums/Percussion/Guiros Scrapes and Rasps/_ONE_SHOTS`
- one clap fixture as `Drums/Claps Snaps Slaps/Generic Clap/_ONE_SHOTS`

The first pass exposed a real report bug where `training_recall_contract_report.csv` assumed fingerprints were NumPy arrays. That is fixed in this bundle by normalizing fingerprints with `np.asarray(...).tolist()` before prediction.

The tiny recall report then passed:

```text
training_recall_failures: 0
Guiro One Shot.wav -> Drums/Percussion/Guiros Scrapes and Rasps/One Shots
HandClap.wav -> Drums/Claps Snaps Slaps/Generic Clap/One Shots
```
