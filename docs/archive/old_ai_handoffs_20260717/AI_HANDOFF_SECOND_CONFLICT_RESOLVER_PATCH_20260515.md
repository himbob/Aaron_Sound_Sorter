# AI Handoff: Second Conflict Resolver Patch

Date: 2026-05-15

## Rule followed

This patch follows the permanent project rule:

1. Add synthetic regression tests first.
2. Prove they fail before code changes.
3. Patch the smallest architecture-safe seam.
4. Prove the same tests pass.

The red test run against the previous patch state produced 3 failures in:

```text
tests/test_decision_core_second_conflict_resolver_tdd.py
```

The failing classes were:

- generic/mixed instrument loop promoted to Brass/Woodwinds by weak reed role
- FX/vocal loop broadened to Brass/Woodwinds by weak reed role
- raw FX/Slam winner kept despite close drum candidate when parent role was unknown

After the patch, the same file passes.

## Sorter changes

Changed:

```text
src/aaron_sound_sorter/engine/decision_core_v2.py
```

### 1. Raw consensus conflict seam

`apply_eligibility()` now checks `_review_for_raw_consensus_conflict()` only when parent eligibility is not decisive. This catches cases where eligibility is `unknown`, but raw consensus still chooses a risky leaf even though candidate evidence conflicts.

Current guarded raw conflict:

```text
FX/Impacts and Hits/Slam/Long FX + close Drums candidate
=> _TO_REVIEW/Measured Role Conflict
```

This protects snare/ride/shuffle-like material from becoming FX/Slam when the parent role detector is unsure.

### 2. Reed/woodwind over-narrowing conflict

Weak `pitched_reed_or_instrument_loop` or `pitched_reed_or_instrument_phrase` evidence may no longer promote generic/mixed instrument loops into Brass/Woodwinds.

Safe behavior:

```text
raw Instruments/Instrument Loops/Loops + weak reed role
=> keep Instruments/Instrument Loops/Loops
```

Unsafe non-reed raw candidate behavior:

```text
raw FX or vocal-ish winner + weak reed role + Brass/Woodwinds fallback
=> _TO_REVIEW/Measured Role Conflict
```

Exact Brass/Woodwind raw winners still survive.

### 3. Vocal candidate blocks Drum Loop broadening

If the measured role says `drum_loop`, but raw/candidate evidence has a close Human/Voice candidate, the sorter reviews instead of broadening into Drum Loops.

This protects vocal loops such as the Caviar vocal loop from repeated-event drum-loop stealing.

## Test changes

Added:

```text
tests/test_decision_core_second_conflict_resolver_tdd.py
```

Updated:

```text
tests/test_decision_core_v2.py
```

The updated older test now expects generic Instrument Loops to stay generic instead of being narrowed to Brass/Woodwinds.

## Tooling changes

Changed:

```text
tools/run_random_sample_folder_log_panel.py
```

Diagnostic improvements only. These do not affect sorting.

- `Bass Drum` is now treated as Drums, not Instruments/Bass.
- `Drum Loops` plural is treated as Drums.
- `keys jangling`, `keys coins`, `coins`, and `small objects` are treated as FX/Foley hints, not musical Keys.
- AKWF/wavetable skipping remains default.

## Validation completed

```text
71 passed
```

Regression command:

```bash
./commands/quality/RUN_SECOND_CONFLICT_RESOLVER_REGRESSION_TESTS.command
```

Real mini-pack spot checks after patch:

```text
Guitar_D.wav => _TO_REVIEW/Measured Role Conflict
Strum 22_Amin.wav => _TO_REVIEW/Measured Role Conflict
Chord 15_E-B.wav => _TO_REVIEW/Measured Role Conflict
Caviar_trap_vocal_loop_90_dance off.wav => _TO_REVIEW/Measured Role Conflict
whoosh_and_swoosh_and_swish_val_389458.wav => _TO_REVIEW/Measured Role Conflict
wind_chime_val_95752.wav => _TO_REVIEW/Measured Role Conflict
RBM_SE_Bongos_5__110bpm.wav => _TO_REVIEW/Measured Role Conflict
```

## Known remaining issue

`Snare-Open-Hat Shuffle 01.wav` still landed in FX/Natural Ambience/Water in a spot check. The available `DecisionCoreV2` object does not currently carry enough brain-only candidate evidence to catch this cleanly, because the close Drums candidate appears in the brain top-20 audit but not necessarily in `raw.shared_candidates`.

Do not patch this with filenames. The safer future fix is to expose top brain and physics candidates to the conflict resolver as structured diagnostic evidence, then add a synthetic test proving a drum loop with strong brain drum evidence but physics FX ambience disagreement goes to Review.

## Next run

After install, run:

```bash
./commands/quality/RUN_SECOND_CONFLICT_RESOLVER_REGRESSION_TESTS.command
FOLDER_COUNT=50 FILES_PER_FOLDER=5 ./commands/quality/RUN_RANDOM_SAMPLE_FOLDER_LOG_PANEL.command
```

Upload the new `random_folder_log_panel_upload_back.zip`.
