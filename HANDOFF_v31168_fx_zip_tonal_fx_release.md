# Aaron Sound Sorter v31.168 — FX ZIP tonal/moving FX release pass

## Scope

Continuation after v31.167.  This pass focused on the new FX-titled project ZIP:

```text
Aaron_FX_coverage_first_200MB_20260608_201853.zip
```

The ZIP contains 81 real audio files.  The external audit now separates true FX-source rows from source folders that are clearly instrument or drum folders inside the FX-titled sample set.  This is an audit-only use of source paths; production routing remains source-name blind.

## Architecture position

This patch does **not** rebuild the brain and does **not** mutate the JSON ledger.

Fixes were kept in:

- `FamilyClaimArbiter` evidence arbitration.
- External FX audit harness scoring/reporting.
- Tests for the narrow arbitration and audit behaviors.

The production changes compare voter candidate paths and measured facts only.  No source filenames, ZIP member names, or source-folder tokens were added to production routing.

## Production fixes

### 1. Concrete FX raw candidates can survive broad Instrument Loop pressure

Expanded the concrete FX fragment set used by the arbiter so these FX families are preserved when the raw/voter candidate is close enough and the competing Instrument Loops claim is broad rather than specific:

- impacts/hits
- glitches/stutters/blips
- hybrid designed FX
- risers/builds
- drops/downlifters
- whooshes/sweeps
- reverses/tails

### 2. Blocked role/shape reviews can release to strong concrete FX

A blocked old-style role/shape claim can no longer hide a strong raw concrete FX candidate when the candidate ranking supports FX decisively over the blocked Instrument claim.

### 3. Weak review/shape conflict can release to raw FX when the voter window is FX-dominant

Weak review claims now release back to the raw FX candidate when the nearby candidate window is FX-dominant or has no meaningful close non-FX challenger.

## External FX audit harness updates

`tools/fx_zip_batch_probe.py` now writes `source_group` and distinguishes:

```text
source_fx
source_instrument_or_hybrid
source_drums
```

This matters because the FX-titled ZIP contains true instrument and drum source folders.  The old harness counted correct instrument placements inside instrument folders as FX failures, which inflated failure counts.

## Coverage results

All 81 real audio members were tested after the patch.

```text
Total tested: 81 / 81 = 100.00%
source_fx: 59
source_instrument_or_hybrid: 16
source_drums: 6
```

Overall audit statuses after the patch:

```text
PASS_FX: 16
PASS_INSTRUMENT_SOURCE: 15
PASS_DRUM_SOURCE: 6
PASS_DRUM_ALLOWED: 6
QUESTIONABLE_DRUM: 14
QUESTIONABLE_DRUM_IN_SOURCE_INSTRUMENT: 1
FAIL_INSTRUMENT: 22
FAIL_REVIEW: 1
```

True FX-source rows only:

```text
source_fx rows: 59
PASS_FX: 16
PASS_DRUM_ALLOWED: 6
QUESTIONABLE_DRUM: 14
FAIL_INSTRUMENT: 22
FAIL_REVIEW: 1
```

The FX ZIP is therefore **fully tested but not clean**.  The patch fixed several real FX routes, but broad Instrument steals remain.

## Important improved examples

```text
APLSFX_COMPLEX_Distorted Slasher_keyBbmin.wav
before: _TO_REVIEW/Shape Conflict
after:  FX/Impacts and Hits/Short Impact/Long FX

low_grinding_monster_drop.wav
before: Instruments/Instrument Loops/Loops
after:  FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX

darkEnergy.wav
before: _TO_REVIEW/Measured Role Conflict
after:  FX/Impacts and Hits/Short Impact/Long FX

dirty_bell_blip_cluster.wav
before: _TO_REVIEW/Measured Role Conflict
after:  FX/Designed Noise FX/Blip/One Shots

fast_swiping_pad_swell.wav
before: Instruments/Instrument Loops/Loops
after:  FX/Structural and Transitional FX/Risers and Builds/Generic Riser/Long FX

ESVN_vinyl_noise_01.wav
before: Instruments/Instrument Loops/Loops
after:  FX/Hybrid Designed FX
```

## Remaining true-FX failure classes

The main remaining weakness is still:

```text
Tonal or repeating designed FX -> Instruments
```

Examples still failing:

```text
APLSFX_COMPLEX_Reverse Heartbeat.wav -> Instruments/Bass/Bass Loops
APL_SFX_Whoosh_Roar_01.wav -> Instruments/Bass/Bass Loops
high_screeching_string_drop.wav -> Instruments/Synths/Synth Pad/One Shots
SpaceShipSlowingDOwn.wav -> Instruments/Instrument Loops/Loops
robot_bell_melody_fragment.wav -> Instruments/Woodwinds/Saxophone/Loops
sliced_bell_glitch_run.wav -> Instruments/Woodwinds/Saxophone/Loops
wt_fd_fx_boom.wav -> Instruments/Bass/Bass Loops
Sound Fx Spanish Crowd.wav -> Instruments/Synths/Synth Lead/One Shots
WS_CFX2_C_Synth_Sweep_4bars.wav -> Instruments/Instrument Loops/Loops
```

Some of those cannot be safely forced to FX from production evidence alone because the measured facts and candidate voters strongly resemble tonal/pitched loops.  The next pass should focus on measured transition/motion/inharmonicity features and candidate-window evidence, not on source names.

## Tests run

Passed:

```bash
python3 -m pytest tests/test_fx_zip_concrete_fx_bass_steal_guard.py -q
python3 -m pytest tests/test_fx_zip_batch_probe_audit.py -q
python3 -m pytest tests/test_third_party_feature_timeout_guard.py -q
python3 -m pytest tests/test_claim_arbiter_real_panel_surrogates.py::test_clean_pitched_tail_reviews_instead_of_sticking_to_weak_tom_leaf -q
python3 -m pytest tests/test_measured_bass_loop_drum_authority_guard.py -q
python3 -m pytest tests/test_measured_percussion_zip_regression_guards.py -q
python3 -m pytest tests/test_voter_calibration_flatten_cache.py -q
python3 -m compileall -q src tests tools Aaron_Sound_Sorter.py
python3 tools/audit_no_source_name_sorting.py --project-root .
```

Source-name audit result:

```text
PASS: production sorting/voting/decision code contains no banned source-name evidence markers.
```

## Timeout note

Long FX harness invocations still hit the outer container timeout sometimes after writing durable rows.  I used smaller chunks down to 1–3 files per run and counted only rows actually present in the CSV.  Final post-patch CSV has all 81 rows.

## Files changed

```text
src/aaron_sound_sorter/engine/family_claim_arbiter.py
tools/fx_zip_batch_probe.py
tests/test_fx_zip_concrete_fx_bass_steal_guard.py
tests/test_fx_zip_batch_probe_audit.py
coverage_ledgers/fx_v31168_after_patch_all_rescored.csv
coverage_ledgers/fx_v31168_after_patch_all_rescored.summary.txt
coverage_ledgers/fx_v31168_final_summary.txt
HANDOFF_v31168_fx_zip_tonal_fx_release.md
```
