# HANDOFF v31.170 — Percussion low-level parent-claim repair

## Summary

This patch returns to `one_shot_percussive_sounds.zip` after the v31.169 FX arbiter cleanup.  It tests new-only percussion rows beyond the prior trusted/screened range and fixes repeated `_TO_REVIEW/Measured Role Conflict` failures at lower levels: shape/consensus stand-down and measured drum-structure claim production.

No brain rebuild. No JSON ledger edits. No production filename/source-folder sorting.

## Why this patch exists

A few short, obviously percussive one-shots were being sent to review because:

1. the shape layer sometimes called very short struck/noisy hits `texture_bed`, `noise_texture`, or `echo_tail_hit`;
2. parent eligibility had already identified `protected_percussive_one_shot` and blocked Instruments/FX/Voice/Animals;
3. later role/shape/firewall logic still chose `_TO_REVIEW` instead of letting a valid Drums parent claim compete.

The fix is deliberately deeper than another late arbiter rescue:

- `consensus.py` now stands down one narrow shape-sanity conflict when measured parent/role/material evidence proves a very short percussion hit.
- `measured_drum_structures.py` now emits `final_measured_protected_percussive_parent_claim` for parent-protected short percussion hits.
- `measured_buckets.py` gives decisive/protected drum-structure claims priority before broad smoke/bucket fallback.
- Arbiter edits are limited to recognizing this new lower-level claim source where the older decisive struck parent source was already allowed.

## New real percussion failures found and fixed

These were rerun with third-party features enabled/bounded before diagnosis:

| Source member | Before | After |
|---|---|---|
| `one_shot_percussive_sounds/1/13913.wav` | `_TO_REVIEW/Measured Role Conflict` | `Drums/...` |
| `one_shot_percussive_sounds/4/203838.wav` | `_TO_REVIEW/Measured Role Conflict` | `Drums/Toms/Generic Tom/One Shots` |
| `one_shot_percussive_sounds/4/203859.wav` | `_TO_REVIEW/Measured Role Conflict` / clean tonal tail review | `Drums/Toms/Generic Tom/One Shots` |
| `one_shot_percussive_sounds/5/439764.wav` | `_TO_REVIEW/Measured Role Conflict` | `Drums/Percussion/Generic Percussion/One Shots` |

## Coverage / screening

Previous reported total before this pass: `194 / 10,254`.

New screen this pass:

- offsets: `70, 75, 80, 85, 90, 95, 100, 105`
- per-folder: `5`
- source folders: `5`
- ordinals: `71–110` per source folder
- new screened files: `200`
- result after fixes: `200 Drums, 0 FX, 0 Instruments, 0 _TO_REVIEW`

Cumulative if counting this screen: `394 / 10,254 = 3.8424%`.

Important honesty note: bulk screening used third-party features disabled for throughput because the sandbox repeatedly hangs in optional librosa/numba paths. Every observed bad/borderline row was rerun individually with third-party enabled and bounded before code changes were made.

Coverage ledger:

- `coverage_ledgers/percussion_v31170_screened_offsets70_105.csv`
- `coverage_ledgers/percussion_v31170_screened_offsets70_105_summary.txt`

## Changed files

```text
src/aaron_sound_sorter/engine/consensus.py
src/aaron_sound_sorter/engine/claim_producers/measured_drum_structures.py
src/aaron_sound_sorter/engine/claim_producers/measured_buckets.py
src/aaron_sound_sorter/engine/family_claim_arbiter.py
src/aaron_sound_sorter/engine/claim_producers/candidate_role_conflicts.py
src/aaron_sound_sorter/engine/claim_producers/measured_music_structures.py
src/aaron_sound_sorter/engine/claim_producers/measured_early_short_hit.py
tools/percussion_zip_batch_probe.py
tools/percussion_zip_isolated_probe.py
tools/percussion_zip_one_by_one_probe.py
tests/test_measured_percussion_zip_regression_guards.py
coverage_ledgers/percussion_v31170_screened_offsets70_105.csv
coverage_ledgers/percussion_v31170_screened_offsets70_105_summary.txt
```

## Tests run

```bash
NUMBA_DISABLE_JIT=1 python3 -m pytest tests/test_measured_percussion_zip_regression_guards.py -q
NUMBA_DISABLE_JIT=1 python3 -m pytest tests/test_claim_arbiter_real_panel_surrogates.py -q
NUMBA_DISABLE_JIT=1 python3 -m pytest \
  tests/test_consensus_concrete_fx_gate_regressions.py::test_stable_pitched_music_phrase_without_instrument_candidate_goes_to_review \
  tests/test_decision_core_fx_smoke_remaining_review_rows.py::test_sax_like_pitched_loop_raw_fx_becomes_broad_instrument_loop_not_review \
  tests/test_family_claim_stability_followup_matrix.py::test_stable_pitched_instrument_claim_reviews_siren_false_positive_without_leaf_support \
  tests/test_measured_bass_loop_drum_authority_guard.py \
  tests/test_fx_zip_concrete_fx_bass_steal_guard.py -q
python3 -m compileall -q src tests tools Aaron_Sound_Sorter.py
python3 tools/audit_no_source_name_sorting.py --project-root .
```

Results:

```text
9 passed
25 passed
18 passed
compileall passed
source-name audit passed
```

## Suggested next work

Continue percussion screening from `--batch-start 110 --per-folder 5`, preferably in small chunks.  Keep using the same rule:

- bulk screen can disable third-party if the sandbox hangs;
- rerun any failure/borderline sample with third-party enabled and bounded before diagnosing;
- fix shape/voter/claim producers first;
- arbiter only gets minimal source acceptance for valid lower-level claims.
