# Stage 4 v0.4.60 support-balanced folder brain status

## Purpose

This update keeps the no-fixed-terminal-category architecture and adds a general fix for underrepresented learned folders.

The problem: folders with many training examples can cover more acoustic territory and swallow close sounds from smaller folders. This showed up with sax-like material losing to larger learned folders.

The fix: support-aware folder balancing. It uses only folder training counts, not filenames and not category names.

## What changed

- Added `support_balance_bonus_for_label()`.
- Added support-balance fields to newly built brains:
  - `max_label_training_count`
  - `support_balance_enabled`
  - `support_balance_threshold_ratio`
  - `support_balance_max_distance_bonus`
  - `support_balance_tiny_count_full_bonus`
  - `support_balance_tiny_multiplier`
  - `support_balance_policy`
- Prediction now subtracts a small capped support bonus from the ranking distance before final sort ranking.
- Existing low-support reliability gates remain in place, so weak folders still need stronger evidence before auto-placement.
- Added regression coverage in `tests/test_stage4_adaptive_label_model.py`.

## Important: no category hard-coding

This update does not check for Sax, Cello, Voice, FX, drum names, or filenames. The rule is purely:

```text
small learned folder close to the winner -> fair ranking chance
small learned folder far away -> still loses
small learned folder barely safe -> existing review gates still apply
```

## Validation run

```bash
python3 -S -m py_compile Aaron_Sound_Sorter.py src/aaron_stage4/phase3_pure_brain_lab.py
pytest -q
```

Result:

```text
21 passed
```

## Recommended next test

Train a folder brain from the current trusted tree, then rerun the `FX_Aaron2` report-only test. The sax failures should either move toward the learned sax/woodwind folder if the audio is close enough, or go to `_TO_REVIEW` if the match is still too conflicted.
