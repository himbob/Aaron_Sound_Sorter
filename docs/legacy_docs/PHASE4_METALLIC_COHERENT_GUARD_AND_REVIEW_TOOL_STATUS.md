# Phase 4 Metallic Coherent Guard and Review Tool Status

Date: 2026-05-11

## Purpose

This update addresses the `22784.wav` failure where a high pitched metallic ring was auto-placed as a cymbal even though the brain's coherent feature-group evidence said the cymbal profile was badly contradicted.

The fix is architecture-safe:

- It does not use filenames.
- It does not hard-code `22784.wav`.
- It does not force bells or rings as a final category.
- It uses the brain's learned coherent feature-group outlier penalty as a hard safety brake against exact auto-placement.

## Code changes

### `src/aaron_sound_sorter/committee.py`

Added a safety check inside `apply_learned_conflict_gates()`:

- Reads `brain["_last_fact_meta"]["coherent_group_penalty"]`.
- If `coherent_group_penalty >= severe_coherent_group_review_threshold`, default `2.0`, and there is a coherent reason, the pick gets a hard audit marker:
  - `severe_coherent_group_outlier_pick`
- That marker is now included in the hard review markers that prevent auto-placement.

For `22784.wav`, the result changed from:

```text
Drums / Cymbals / Crash Cymbal / One Shots
confidence_status = auto_place
```

To:

```text
_TO_REVIEW / guess_Drums Cymbals Crash Cymbal One Shots
confidence_status = review
reason includes severe_coherent_group_outlier_pick
```

### `tools/aaron_metallic_percussion_reviewer.py`

Added a safe review helper that scans metallic/cymbal/bell areas and creates a review tree using symlinks by default.

It does not move or delete training data.

It creates buckets such as:

```text
Drums/Percussion/Bells and Metallic Percussion/High Rings and Chimes/One Shots
Drums/Percussion/Bells and Metallic Percussion/Bells Chimes and Rings/One Shots
Drums/Percussion/Bells and Metallic Percussion/Cowbells and Blocks/One Shots
Drums/Cymbals/Noisy Cymbals/One Shots
_QUARANTINE/Cymbals_Possible_High_Rings_or_Chimes
_QUARANTINE/Cymbals_Possible_Bells_or_Rings
_REVIEW/Metallic_Ambiguous_Pitched_vs_Noisy
```

### `commands/review/RUN_METALLIC_PERCUSSION_REVIEW.command`

Added a Mac-friendly command that runs the review helper against:

```text
training/locked_curated_v1
```

and opens the output folder when finished.

### `tests/test_phase4_contextual_brain_structure_and_groups.py`

Added regression coverage proving that a severe coherent group penalty blocks auto-place.

## Validation performed

### Targeted regression

```bash
pytest -q tests/test_phase4_contextual_brain_structure_and_groups.py
```

Result:

```text
4 passed
```

### Real `22784.wav` check

Command used in sandbox:

```bash
python3 Aaron_Sound_Sorter.py sort /mnt/data/22784.wav /mnt/data/bell_fix_test_out --brain stage4_folder_brain.json --no-make-zip
```

Result:

```text
final_top = _TO_REVIEW
final_label = Drums/Cymbals/Crash Cymbal/One Shots
confidence_status = review
review_reason includes severe_coherent_group_outlier_pick
coherent_group_penalty = 2.25
```

### Metallic review helper check

Command used in sandbox:

```bash
python3 tools/aaron_metallic_percussion_reviewer.py --root /mnt/data/metal_test --out /mnt/data/metal_review_test --mode symlink
```

Result for `22784.wav` staged under a cymbal folder:

```text
_QUARANTINE/Cymbals_Possible_High_Rings_or_Chimes
```

Reason included:

```text
strong_high_ring_chime_physics
pitch=0.601
body_pitch=0.799
tail_pitch=0.843
hnr=1.531
presence=0.936
entropy=0.362
```

### Wider pytest checks

The complete `pytest -q` run timed out in the sandbox after showing many passes. I then ran the suite in smaller chunks.

Passed chunks included:

```text
test_audio_analyzer.py
test_brain_and_committee.py
test_broad_family_coherence.py
test_legacy_cli_contract.py
test_phase4_committee_physics_gap_locks.py
test_phase4_contextual_brain_structure_and_groups.py
test_phase4_dynamic_architecture.py
test_phase4_dynamic_structure_gate.py
test_phase4_dynamic_structure_gate_v0485.py
test_phase4_expanded_physics_membership_v052.py
test_phase4_pitched_percussion_guard.py
test_phase4_real_project_zip_smoke.py
test_phase4_synthetic_accuracy_smoke.py
test_phase4_v052_physics_atlas_validation.py
test_phase4_v060_physical_family_guard_regressions.py
test_phase4_v061_cross_family_policy_regressions.py
test_phase4_v062_tournament_calculation_guards.py
test_phase4_v064_failed_file_diagnostics_contracts.py
test_phase4_voice_guard.py
test_stage4_adaptive_label_model.py
test_stage4_dynamic_runtime_contracts.py
test_stage4_fact_profiles.py
test_stage4_physical_role_review_guards.py
test_stage4_training_folder_interpreter.py
test_training_physics_review_plan_logic.py
test_training_physics_review_transients.py
```

Known sandbox limitations:

- `test_phase4_synthetic_quality_matrix.py` timed out in this environment.
- `test_phase4_v063_tournament_factor_contracts.py` timed out in this environment.
- Some test files contain zero tests and return pytest status 5 when run alone.

## How Aaron should run the new metallic review helper

From project root:

```bash
cd /Users/your-name/Documents/Codex/2026-04-25/files-mentioned-by-the-user-create/Aaron_Sound_Sorter
chmod +x commands/review/RUN_METALLIC_PERCUSSION_REVIEW.command
./commands/review/RUN_METALLIC_PERCUSSION_REVIEW.command
```

Optional: scan a different folder:

```bash
./commands/review/RUN_METALLIC_PERCUSSION_REVIEW.command "/path/to/sample-library/Sorted samples"
```

The command creates a symlink review tree under:

```text
reports/metallic_percussion_review/run_TIMESTAMP
```

Listen to the `_QUARANTINE` and `_REVIEW` folders before moving anything into final training folders.

## Recommended training structure after listening

Use these if the review pack confirms the split:

```text
Drums/Percussion/Bells and Metallic Percussion/High Rings and Chimes/One Shots
Drums/Percussion/Bells and Metallic Percussion/Bells Chimes and Rings/One Shots
Drums/Percussion/Bells and Metallic Percussion/Cowbells and Blocks/One Shots
Drums/Cymbals/Noisy Cymbals/One Shots
```

Do not blindly move every candidate. The helper is a triage tool, not a truth oracle.
