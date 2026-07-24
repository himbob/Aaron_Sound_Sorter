# Archived: Aaron Sound Sorter v31.165 — Percussion protected-parent release

## Scope

Continuation after `Aaron_Sound_Sorter_v31164_percussion_batch_repair_python_tests_only.zip`.

This patch is source-name blind in production code. It does **not** mutate the JSON ledger, does **not** rebuild/train the brain, and does **not** change filename-based classifier behavior.

## Architecture diagnosis

The next percussion ZIP offset slice exposed a parent/arbiter inconsistency:

- `parent_eligibility_v2` had already measured a `protected_percussive_one_shot` lane.
- The parent lane allowed `Drums` and blocked `Instruments`/`FX`.
- Later final arbitration still let weak clean-tonal voice/instrument/animal-FX pressure convert those files into `_TO_REVIEW/Measured Role Conflict`.

Failure type: **arbiter incorrectly blocking a valid lower-level measured parent claim**.

The fix is deliberately narrow: when the measured parent gate has already proven `protected_percussive_one_shot`, and measured material/attack/drum evidence is still strong enough, final arbitration releases the file back to the measured Drums parent before non-drum voice/instrument/animal-FX conflict review.

## Changed source files

- `src/aaron_sound_sorter/engine/family_claim_arbiter.py`
  - Added `_facts_support_protected_percussive_parent_release()`.
  - Added protected parent release in `_protect_measured_voice_from_non_voice_leaf()`.
  - Added protected parent release in `_percussive_one_shot_parent_firewall()`.
- `tests/test_measured_percussion_zip_regression_guards.py`
  - Added pitched hand-drum/clean-tonal regression guard.
  - Added noisy short-hit/breath-cat-FX regression guard.
- `tools/percussion_zip_batch_probe.py`
  - Reworked the batch probe to use neutral filenames, per-slice CSVs, env-overridable project/ZIP paths, optional child-process mode, and immediate row writes.

## Exact fixed percussion ZIP failures

| Source member | Bad pre-fix result | Post-fix result | Evidence summary |
|---|---|---|---|
| `one_shot_percussive_sounds/2/101451.wav` | `_TO_REVIEW/Measured Role Conflict` | `Drums/Toms/Generic Tom/One Shots` | `parent_eligibility_v2.role_name=protected_percussive_one_shot`; compact struck tonal percussion ~0.714; hand drum membrane ~0.870; not true voice. |
| `one_shot_percussive_sounds/2/101452.wav` | `_TO_REVIEW/Measured Role Conflict` | `Drums/Toms/Generic Tom/One Shots` | Same pattern as 101451: short pitched struck percussion was being pulled toward clean instrument/voice review. |
| `one_shot_percussive_sounds/4/203201.wav` | `_TO_REVIEW/Measured Role Conflict` | `Drums/Percussion/Generic Percussion/One Shots` | Noisy short hit misread as breath/cat FX; parent eligibility blocked FX/Instruments; fast attack, onset-percussive score ~0.738, compact/wood struck evidence ~0.636/~0.635. |

## Coverage ledger

Total real audio members in `one_shot_percussive_sounds.zip`: **10,254**.

Previous trusted v31.164 coverage before this session: **125 files**.

New trusted current-patch coverage added here: **19 files** from offset 55 / ordinals 56–60 slice.

Cumulative trusted coverage now counted: **144 / 10,254 = 1.4043%**.

Do **not** claim the percussion ZIP is solved. This is still far below the requested 25% target.

### New current-patch verified counts

| Count type | Count |
|---|---:|
| Drums | 19 |
| FX | 0 |
| Instruments failures | 0 |
| `_TO_REVIEW` failures | 0 |

### New current-patch verified by source folder

| Source folder | Verified count |
|---|---:|
| `one_shot_percussive_sounds/1` | 5 |
| `one_shot_percussive_sounds/2` | 5 |
| `one_shot_percussive_sounds/3` | 2 |
| `one_shot_percussive_sounds/4` | 2 |
| `one_shot_percussive_sounds/5` | 5 |

Detailed CSVs are included under `coverage_ledgers/`:

- `percussion_offset55_current_patch_verified.csv`
- `percussion_offset55_unverified_not_counted.csv`
- `percussion_coverage_summary_v31165.csv`

## Leaf-folder questionable placements

No broad-family failures remain among the 19 current-patch verified files.

Questionable but acceptable leaves to keep watching:

- `101451.wav` and `101452.wav` land in `Drums/Toms/Generic Tom/One Shots`. Broad Drums is correct; leaf may be better as hand drum / generic percussion depending on future examples.
- `203201.wav` lands in `Drums/Percussion/Generic Percussion/One Shots`. That is intentionally broad and safer than forcing voice/animal FX review.

## Tests run

All commands were run from the project root with `NUMBA_DISABLE_JIT=1` where runtime classification or pytest could trigger numba/librosa startup behavior in the sandbox.

Passed:

```bash
python3 -m pytest tests/test_measured_percussion_zip_regression_guards.py -q
python3 -m pytest tests/test_claim_arbiter_real_panel_surrogates.py::test_clean_pitched_tail_reviews_instead_of_sticking_to_weak_tom_leaf -q
python3 -m pytest tests/test_measured_bass_loop_drum_authority_guard.py -q
python3 -m pytest tests/test_physics_layers.py -q
python3 -m pytest tests/test_claim_arbiter_real_panel_surrogates.py -q
python3 -m compileall -q src tests tools Aaron_Sound_Sorter.py
python3 tools/audit_no_source_name_sorting.py --project-root .
```

Source-name audit result:

```text
PASS: production sorting/voting/decision code contains no banned source-name evidence markers.
```

## Timeout notes and substitutes

The sandbox repeatedly hung in runtime classification unless `NUMBA_DISABLE_JIT=1` was set. Long per-slice harness execution also hit container timeouts even after adding per-row CSV writes and child-process mode.

Substitute used instead of pretending the full 25-file slice completed:

- Kept per-slice CSV rows that were written before timeout.
- Re-ran exact failing samples one at a time.
- Re-ran additional offset-55 files one at a time and counted only rows verified under the current patch.
- Six offset-55 selected rows are listed in `percussion_offset55_unverified_not_counted.csv` and are **not** included in the trusted coverage count.

## Next recommended work

1. Continue offset 55 and finish the six unverified rows first.
2. Re-run the whole offset-55 25-file slice once the local Mac can handle it reliably.
3. Then continue offset batches: 60, 65, 70, etc., five per folder.
4. Keep every slice in its own result CSV.
5. Do not expand this protected-parent release into a broad arbiter rescue unless new evidence proves a repeated parent-gate failure pattern.
