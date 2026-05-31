# Stage 4 v0.4.56 Adaptive Single-File Test Bundle

## What changed

- Runtime implementation now lives in root `Aaron_Sound_Sorter.py`.
- `src/aaron_stage4/phase3_pure_brain_lab.py` is only a tiny compatibility shim for older tests/commands.
- Training folder path remains the ground-truth output label.
- File names are not used as labels or routing truth.
- Audio physics no longer overrides training labels during brain build.
- Per-folder model selection is adaptive:
  - tiny folders: `exemplar_only`
  - uniform folders: `single_centroid`
  - medium-spread folders: `multi_centroid`
  - messy folders: `multi_centroid_exemplar`
- Internal clusters/exemplars do not create public output folders.
- Stale mixed-loop constants are removed from runtime code.

## Test command

From repo root after extracting over the project:

```bash
python3 -S -m py_compile Aaron_Sound_Sorter.py src/aaron_stage4/phase3_pure_brain_lab.py
python3 tests/test_stage4_adaptive_label_model.py
python3 tests/test_stage4_training_folder_interpreter.py
python3 tests/test_stage4_physical_role_review_guards.py
python3 Aaron_Sound_Sorter.py self-test --tmp-root /tmp/aaron_stage4_selftest
```

## Human-facing command

```bash
python3 Aaron_Sound_Sorter.py "/path/to/samples.zip" "/path/to/output_folder"
```

Equivalent explicit form:

```bash
python3 Aaron_Sound_Sorter.py sort "/path/to/samples.zip" "/path/to/output_folder"
```

## Notes

This is still a test bundle. The adaptive model is now implemented, but the next real proof is another FX_Aaron2-only run and review of `label_models_by_label`, `exemplars_by_label`, placement coverage, and bad auto-placements.
