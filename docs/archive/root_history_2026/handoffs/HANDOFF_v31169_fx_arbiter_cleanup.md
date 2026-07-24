# Archived: v31.169 FX arbiter cleanup patch

## Purpose

This patch fixes the three regressions introduced by v31.168 where concrete FX leaves were over-preserved at the arbiter layer.

User-reported failures fixed:

- `tests/test_claim_arbiter_real_panel_surrogates.py::test_weak_transition_fx_leaf_releases_to_broad_instrument_loop`
- `tests/test_consensus_concrete_fx_gate_regressions.py::test_stable_pitched_music_phrase_without_instrument_candidate_goes_to_review`
- `tests/test_decision_core_fx_smoke_remaining_review_rows.py::test_sax_like_pitched_loop_raw_fx_becomes_broad_instrument_loop_not_review`

Additional regression protected during this cleanup:

- `tests/test_family_claim_stability_followup_matrix.py::test_stable_pitched_instrument_claim_reviews_siren_false_positive_without_leaf_support`

## Architecture notes

The v31.168 mistake was too much top-level FX preservation:

1. `pick_winner()` was allowed to release weak review rows back to raw FX.
2. Transition/movement FX words such as riser/drop/whoosh/reverse were included in a high-level “raw FX decisively beats blocked instrument” suppressor.
3. Concrete FX raw preservation was too broad and blocked legitimate measured pitched-loop escapes.

This patch does **not** add filename/source sorting and does **not** rebuild or mutate any brain/ledger JSON.

The cleanup direction:

- remove the general weak-review-to-raw-FX escape from winner selection;
- keep review when measured pitched-music structure contradicts raw FX;
- let broad Instrument Loops through only when the measured body is a non-percussive tonal/repeating loop and measured FX-transition pressure is weak;
- keep siren/alarm-style pitched false positives in Review when there is no specific instrument leaf support;
- do not add another broad top-level FX rescue.

## Changed files

- `src/aaron_sound_sorter/engine/family_claim_arbiter.py`

## Tests run in sandbox

Passed:

```bash
NUMBA_DISABLE_JIT=1 python3 -m pytest \
  tests/test_claim_arbiter_real_panel_surrogates.py \
  tests/test_consensus_concrete_fx_gate_regressions.py \
  tests/test_decision_core_fx_smoke_remaining_review_rows.py \
  tests/test_family_claim_stability_followup_matrix.py \
  tests/test_fx_zip_concrete_fx_bass_steal_guard.py \
  tests/test_fx_zip_batch_probe_audit.py -q
```

Result: `44 passed`

Passed:

```bash
NUMBA_DISABLE_JIT=1 python3 -m pytest \
  tests/test_decision_core_fx_aaron2_pattern_regressions.py \
  tests/test_decision_core_fx_smoke_matrix_hard_failures.py \
  tests/test_decision_core_fx_smoke_no_review_guard.py \
  tests/test_decision_core_fx_smoke_no_review_v2.py \
  tests/test_decision_core_golden_failure_regressions.py \
  tests/test_decision_core_ground_truth_samples_tdd.py \
  tests/test_decision_core_ground_truth_true_bucket_tdd.py -q
```

Result: `54 passed`

Passed:

```bash
python3 -m compileall -q src tests tools Aaron_Sound_Sorter.py
python3 tools/audit_no_source_name_sorting.py --project-root .
```

Source-name audit result:

```text
PASS: production sorting/voting/decision code contains no banned source-name evidence markers.
```

## Acceptance note

The sandbox could not complete the macOS locked-smoke command cleanly because the command is designed for Aaron's repo path and Finder/open behavior. With `PROJECT_ROOT=/mnt/data/work_v31169/tree`, the first two locked-smoke cases printed PASS before the wrapper hung on the container side. Do not count that as full acceptance coverage.

Run these on Aaron's Mac after install:

```bash
cd /Volumes/T9/testbed/Aaron_Sound_Sorter
python3 -m compileall -q src tests tools Aaron_Sound_Sorter.py
python3 tools/audit_no_source_name_sorting.py --project-root .
NUMBA_DISABLE_JIT=1 python3 -m pytest \
  tests/test_claim_arbiter_real_panel_surrogates.py::test_weak_transition_fx_leaf_releases_to_broad_instrument_loop \
  tests/test_consensus_concrete_fx_gate_regressions.py::test_stable_pitched_music_phrase_without_instrument_candidate_goes_to_review \
  tests/test_decision_core_fx_smoke_remaining_review_rows.py::test_sax_like_pitched_loop_raw_fx_becomes_broad_instrument_loop_not_review \
  tests/test_family_claim_stability_followup_matrix.py::test_stable_pitched_instrument_claim_reviews_siren_false_positive_without_leaf_support -q
./commands/quality/RUN_LOCKED_SMOKE_ACCEPTANCE_AI_SAFE.command
```
