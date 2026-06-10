# Aaron Sound Sorter v31.166 — percussion offsets 60–65 partial repair

## Scope

Continuation after v31.165. This pass did **not** reach Aaron's requested 10% percussion ZIP coverage target. It tested only new files after the prior trusted range.

No brain rebuild. No JSON ledger mutation. No filename-based production routing.

## Coverage status

Total real audio members in `one_shot_percussive_sounds.zip`: **10,254**.

Previous trusted coverage through v31.165: **144 files**.

New trusted current-patch coverage in this pass: **50 files** from ordinals **61–70** in each of the 5 source folders.

Cumulative trusted coverage: **194 / 10,254 = 1.8919%**.

Do **not** claim 10% coverage. 10% requires about 1,026 trusted files.

## New verified counts

| Count type | Count |
|---|---:|
| Drums | 48 |
| FX | 2 |
| Instruments failures after fixes | 0 |
| `_TO_REVIEW` failures after fixes | 0 |

By source folder:

| Source folder | New verified count |
|---|---:|
| `one_shot_percussive_sounds/1` | 10 |
| `one_shot_percussive_sounds/2` | 10 |
| `one_shot_percussive_sounds/3` | 10 |
| `one_shot_percussive_sounds/4` | 10 |
| `one_shot_percussive_sounds/5` | 10 |

Detailed ledgers:

- `coverage_ledgers/percussion_offsets60_65_current_patch_verified_v31166.csv`
- `coverage_ledgers/percussion_coverage_summary_v31166.csv`

## Fixed failures

### `one_shot_percussive_sounds/4/203207.wav`

Bad result before fix:

```text
_TO_REVIEW/Measured Role Conflict
```

Fixed result:

```text
Drums/Percussion/Generic Percussion/One Shots
```

Failure type: parent eligibility was too narrow for a short fast repeated struck gesture. The shape voter saw a pitched repetition phrase, but the file was under 0.7 seconds with fast attack, high flatness, compact struck percussion evidence, guiro/percussive evidence, and no real instrument duration. Added a source-blind parent eligibility rule for short repeated struck percussive gestures.

### `one_shot_percussive_sounds/2/102787.wav`

Bad result before fix:

```text
Instruments/Woodwinds/Saxophone/Loops
```

Fixed result:

```text
Drums/Rims and Sticks/Generic Rim or Stick/One Shots
```

Failure type: protected percussive parent evidence existed, but sax/reed loop and later instrument identity broadening paths were allowed to override it. Added source-blind guards so protected percussive one-shot / low-kick-like parents block measured sax-loop and broad instrument identity repair. Also loosened the protected parent release to allow strong struck material even when formant/reed metrics create false non-drum pressure.

## Changed files

```text
src/aaron_sound_sorter/engine/eligibility.py
src/aaron_sound_sorter/engine/family_claim_arbiter.py
src/aaron_sound_sorter/engine/claim_producers/measured_instrument_branches.py
src/aaron_sound_sorter/engine/claim_producers/measured_true_bucket.py
tests/test_measured_percussion_zip_regression_guards.py
coverage_ledgers/percussion_offsets60_65_current_patch_verified_v31166.csv
coverage_ledgers/percussion_coverage_summary_v31166.csv
HANDOFF_v31166_percussion_offsets60_65_partial.md
```

## Tests run

Passed:

```bash
NUMBA_DISABLE_JIT=1 python3 -m pytest tests/test_measured_percussion_zip_regression_guards.py -q
NUMBA_DISABLE_JIT=1 python3 -m pytest tests/test_claim_arbiter_real_panel_surrogates.py::test_clean_pitched_tail_reviews_instead_of_sticking_to_weak_tom_leaf -q
NUMBA_DISABLE_JIT=1 python3 -m pytest tests/test_measured_bass_loop_drum_authority_guard.py -q
python3 -m compileall -q src tests tools Aaron_Sound_Sorter.py
python3 tools/audit_no_source_name_sorting.py --project-root .
```

Source-name audit result:

```text
PASS: production sorting/voting/decision code contains no banned source-name evidence markers.
```

## Timeout / execution note

The sandbox could not sustain long classifier batches. Some child processes became non-responsive under the container execution layer. I counted only completed rows written to CSV and verified after the current patch. Timed-out/unverified rows are not included.

## Next work

Continue only new files after ordinal 70:

```bash
# next offsets
python3 tools/percussion_zip_batch_probe.py --batch-start 70 --per-folder 5
python3 tools/percussion_zip_batch_probe.py --batch-start 75 --per-folder 5
python3 tools/percussion_zip_batch_probe.py --batch-start 80 --per-folder 5
```

To reach 10%, the project needs **832 more trusted files** beyond this v31.166 handoff.
