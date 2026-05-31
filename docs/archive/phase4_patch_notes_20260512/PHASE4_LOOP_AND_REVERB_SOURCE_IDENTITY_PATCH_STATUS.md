# Phase 4 Loop and Reverb Source Identity Patch Status

Date: 2026-05-11

## Purpose

This patch fixes three over-conservative or wrong-family behaviors seen in the latest reports:

1. Clear mixed drum beats were being sent to `_TO_REVIEW` instead of `Drums / Drum Loops / Loops`.
2. Reverby sax/brass/woodwind loops were being pulled toward FX transition labels such as downlifters.
3. Ordinary audit warnings were being treated as hard review blockers, causing too many usable loops to go to `_TO_REVIEW`.

The patch does not use filenames for routing.

## Changed files

- `src/aaron_sound_sorter/training_labels.py`
  - Added looser distributed multi-event loop proof for syncopated/reverby/human-played loops.
  - Keeps the existing front-loaded tail veto so cymbal/gong/impact tails are not mislabeled as loops.

- `src/aaron_sound_sorter/committee.py`
  - Added a source/body musical-loop rescue from FX to learned Instrument loop labels when full-file reverb/tail makes a pitched or stable musical loop look like FX.
  - Softened audit warning policy so ordinary audit notes do not force review by themselves.
  - Hard review remains for true physical contradictions such as severe coherent group outliers and physical structure conflicts.

- `src/aaron_sound_sorter/features.py`
  - Added `_QUARANTINED_BY_PHYSICS_REVIEW` to the training scanner skip list.

- `tests/test_phase4_contextual_brain_structure_and_groups.py`
  - Added regression tests for syncopated distributed loop proof.
  - Added synthetic reverby musical loop test that switches from FX to learned Instrument Loops without filename rules.
  - Added test that soft audit warnings no longer force review when no hard physical conflict exists.

This bundle also carries forward the earlier same-day changed files:

- `src/aaron_sound_sorter/preview.py`
- `src/aaron_sound_sorter/reports.py`
- `tools/aaron_metallic_percussion_reviewer.py`
- `commands/review/RUN_METALLIC_PERCUSSION_REVIEW.command`
- `commands/review/APPLY_APPROVED_METALLIC_MOVES.command`
- `tests/test_phase4_contextual_brain_structure_and_groups.py`

## Spot-check results with uploaded real samples

Using the uploaded `stage4_folder_brain.json`:

- `MKS_98_Beat1.wav`
  - Before: `_TO_REVIEW`
  - After: `Drums / Drum Loops / Loops`, `auto_place`

- `AMV_VRNB1_122_brass_saxophone_loop_forus_Fm.wav`
  - Before: `_TO_REVIEW`, final candidate `Instruments / Woodwinds / Saxophone / Loops`
  - After: `Instruments / _Ambiguous Leaf / Synths vs Woodwinds / Loops`, `auto_place`
  - Not perfect sax leaf, but no longer FX or review.

- `CD4_Forbidden Romance_saxophone & keys_Part_1_Gm_78.wav`
  - Before: `_TO_REVIEW`, final candidate `Instruments / Voice / Phrase / One Shots`
  - After: `Instruments / _Ambiguous Leaf / Brass vs Woodwinds / Loops`, `auto_place`
  - No longer voice, no longer one-shot, no longer review.

- `RHSH_Saxophone_Ensemble_05_keyC_89bpm.wav`
  - Before: `_TO_REVIEW`, FX downlifter candidate
  - After: `Instruments / _Ambiguous Leaf / Guitar vs Strings Bowed / Loops`, `auto_place`
  - Still not a clean sax leaf, but the main failure was corrected: it is no longer FX/downlifter.

- `Vocal Phrase We Up 140bpm.wav`
  - Still goes to `_TO_REVIEW / Conflicting Evidence / Voiced Formant Non Drum`
  - This is intentional until there is a stronger trained vocal/rap phrase loop category.

## Tests run

Passed:

```text
python3 -m pytest -q tests/test_phase4_contextual_brain_structure_and_groups.py
7 passed

python3 -m pytest -q \
  tests/test_phase4_voice_guard.py \
  tests/test_phase4_v060_physical_family_guard_regressions.py \
  tests/test_phase4_v061_cross_family_policy_regressions.py \
  tests/test_phase4_dynamic_structure_gate.py \
  tests/test_phase4_dynamic_structure_gate_v0485.py \
  tests/test_phase4_pitched_percussion_guard.py
29 passed

python3 -m pytest -q \
  tests/test_phase4_contextual_brain_structure_and_groups.py \
  tests/test_stage4_training_folder_interpreter.py \
  tests/test_phase4_voice_guard.py \
  tests/test_phase4_v060_physical_family_guard_regressions.py \
  tests/test_phase4_v061_cross_family_policy_regressions.py
21 passed
```

The full `pytest -q` and long per-file suite chunks timed out in the sandbox before finishing. The timeouts were not assertion failures.

## Known remaining limitation

The patch is intentionally broad-family safe. It improves source family and structure handling, but it does not guarantee exact sax leaf selection for every reverby sax loop. Some files now land in dynamic ambiguous Instrument loop folders. That is still better than FX/downlifter or `_TO_REVIEW`, but the next data fix should strengthen clean sax/brass/woodwind loop training folders.
