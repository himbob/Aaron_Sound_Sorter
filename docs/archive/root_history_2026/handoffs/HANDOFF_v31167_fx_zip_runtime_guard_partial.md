# Archived: HANDOFF v31.167 — FX ZIP Runtime Guard + Partial FX Baseline

## Status

This is a **partial** FX ZIP pass, not a clean FX solution.

The new ZIP tested here was:

```text
/mnt/data/Aaron_FX_coverage_first_200MB_20260608_201853.zip
```

Inventory:

```text
Real audio members: 81
Source folders: 79
```

Verified baseline rows in this bundle:

```text
Rows 001-015: 15 / 81 = 18.52%
PASS_FX: 1
PASS_DRUM_ALLOWED: 3
PASS_TEXTURE_ALLOWED: 0
QUESTIONABLE_DRUM: 4
FAIL_INSTRUMENT: 5
FAIL_REVIEW: 2
```

This means the FX ZIP is **not** clean. Do not claim it is fixed.

## Main finding

The repeated failure pattern is not archive handling. It is routing/arbitration:

> Long tonal or rhythmic designed FX are being interpreted as clean pitched instrument loops or bass loops.

Observed examples:

```text
APLSFX_COMPLEX_Reverse Heartbeat.wav
  -> Instruments/Bass/Bass Loops

APL_SFX_Whoosh_Roar_01.wav
  -> Instruments/Bass/Bass Loops

SpaceShipSlowingDOwn.wav
  -> Instruments/Instrument Loops/Loops

low_grinding_monster_drop.wav
  -> Instruments/Instrument Loops/Loops

high_screeching_string_drop.wav
  -> Instruments/Synths/Synth Pad/One Shots or Synth Loops depending on runtime feature settings
```

The lower evidence often contains real FX motion/glitch/transition scores, but later final invariants still prefer broad Instruments.

## What this patch changes

### 1. Runtime guard for optional third-party features

Changed:

```text
src/aaron_sound_sorter/features.py
```

Added:

```text
AARON_THIRD_PARTY_TIMEOUT_SECONDS
AARON_DISABLE_THIRD_PARTY_FEATURES
```

Reason: in this sandbox, optional librosa/numba feature extraction can hang or run too slowly during repeated ZIP probes. The sorter should not let an optional feature adapter kill a whole audit.

Default behavior still tries third-party features. This is a safety guard, not a removal of third-party analysis.

### 2. Flattened fact cache for voter calibration

Changed:

```text
src/aaron_sound_sorter/engine/voter_calibration_panels.py
```

Reason: repeated arbitration calls flatten the same nested fact map many times. This adds a per-facts cache and skips private/internal evidence keys. This is performance-only and source-name-blind.

### 3. Narrow concrete-FX versus bass-loop steal guard

Changed:

```text
src/aaron_sound_sorter/engine/claim_producers/measured_true_bass_fx.py
src/aaron_sound_sorter/engine/claim_producers/measured_true_bucket.py
src/aaron_sound_sorter/engine/family_claim_arbiter.py
```

This blocks one class of false bass-loop rescue when the raw winner is concrete FX and measured shape evidence supports FX motion/impact/glitch/transition behavior.

This did **not** fully solve the FX ZIP.

### 4. External FX ZIP probe harness

Added:

```text
tools/fx_zip_batch_probe.py
```

This is an external audit harness only. It may use source member paths to mark expected broad audit families, but it does not feed source names into production classifier code.

Useful command:

```bash
python3 tools/fx_zip_batch_probe.py \
  --zip /path/to/Aaron_FX_coverage_first_200MB_20260608_201853.zip \
  --out coverage_ledgers/fx_probe_next.csv \
  --start 16 \
  --limit 10 \
  --disable-third-party \
  --candidate-count 30
```

## Coverage ledgers included

```text
coverage_ledgers/fx_coverage_first_200mb_rows_001_015_v31167.csv
coverage_ledgers/fx_coverage_first_200mb_rows_001_015_summary_v31167.txt
coverage_ledgers/fx_coverage_first_200mb_child_partial_v31167.csv
```

The child CSV has rows 001-002 with full third-party path enabled and timeout guard. The 001-015 CSV was run with third-party disabled because the sandbox was too slow/unreliable for the full path.

## Tests run

All passed:

```bash
python3 -m pytest tests/test_third_party_feature_timeout_guard.py -q
python3 -m pytest tests/test_voter_calibration_flatten_cache.py -q
python3 -m pytest tests/test_fx_zip_concrete_fx_bass_steal_guard.py -q
python3 -m pytest tests/test_claim_arbiter_real_panel_surrogates.py::test_clean_pitched_tail_reviews_instead_of_sticking_to_weak_tom_leaf -q
python3 -m pytest tests/test_measured_bass_loop_drum_authority_guard.py -q
python3 -m pytest tests/test_measured_percussion_zip_regression_guards.py -q
python3 -m compileall -q src tests tools Aaron_Sound_Sorter.py
python3 tools/audit_no_source_name_sorting.py --project-root .
```

Source-name audit result:

```text
PASS: production sorting/voting/decision code contains no banned source-name evidence markers.
```

## Next AI should do this

1. Install v31.167.
2. Continue FX ZIP from row 16 using `tools/fx_zip_batch_probe.py`.
3. Do not count timed-out rows as trusted.
4. Fix repeated FX-designed-motion -> Instruments failures at the measured/shape/family arbitration level.
5. Be careful with clean musical loops. Do not just let every tonal long sound stay FX.
6. Preserve percussion regression tests.
7. Do not mutate the JSON ledger or rebuild brains unless Aaron explicitly asks.

## Known next failure pattern to target

The highest-value next fix is:

```text
Concrete raw FX + measured FX motion/glitch/transition support
should not be flattened into Instruments/Instrument Loops just because it is tonal/repeating.
```

But it must be guarded so actual clean synth/key/string loops do not get swallowed by FX.
