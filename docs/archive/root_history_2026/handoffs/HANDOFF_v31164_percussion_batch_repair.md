# Archived: v31.164 Percussion Batch Repair Handoff

## Purpose

Preserve the current working source/test state before the thread gets too long.

This bundle is Python/test files only. It continues the v31.160–v31.163 work and adds the later percussion ZIP batch repairs.

The architecture target is still: measured audio evidence first, structure/role before narrow identity, and no filename-based final routing. For the `one_shot_percussive_sounds.zip` work, producer filenames were used only as audit hints, not as sorter evidence.

## Current testing standard for the percussion ZIP

For `one_shot_percussive_sounds.zip`:

- `Drums/...` = expected / good
- `FX/...` = acceptable only for obvious designed impact/noise/blip-like hits
- `Instruments/...` = hard failure for this ZIP unless the audio is truly a playable instrument
- `_TO_REVIEW/...` = failure for this ZIP unless the file is broken/tiny/unreadable

## Batch harness that now works

Added as:

```text
tools/percussion_zip_batch_probe.py
```

It does the thing Aaron asked for:

1. Enumerates real audio members in `one_shot_percussive_sounds.zip`.
2. Groups recursively by source folder.
3. Selects N files per folder by offset.
4. Neutralizes filenames before classification.
5. Runs classification in small slices with per-sample timeout.
6. Appends each result to CSV immediately so timeout does not erase progress.

The script defaults to the sandbox paths used here. On Aaron's Mac it may need path edits or wrapper variables if used directly.

## Verified completed percussion coverage in this environment

Latest trusted complete recursive sets:

| Offset range per folder | Files | Result |
|---|---:|---|
| 31–35 | 25 | 25/25 Drums/FX, 0 Instruments, 0 Review |
| 36–40 | 25 | 25/25 Drums/FX, 0 Instruments, 0 Review |
| 41–45 | 25 | 25/25 Drums/FX, 0 Instruments, 0 Review |
| 46–50 | 25 | 25/25 Drums/FX, 0 Instruments, 0 Review |
| 51–55 | 25 | 25/25 Drums/FX, 0 Instruments, 0 Review |

Combined latest trusted complete batch coverage:

```text
125 files tested
5 source folders covered
25 files per source folder
120 Drums
5 FX
0 Instruments
0 _TO_REVIEW
0 broad failures
```

Earlier exploratory batches found real failures and were not counted as final proof until rerun after fixes.

## Important repaired failures from this pass

These were fixed by source-blind lower-layer or arbiter-gate changes:

- Short/struck percussion going to `_TO_REVIEW/Measured Role Conflict`.
- Low struck percussion going to `Instruments/Bass/808` or `Bass Loops`.
- Resonant struck material going to `Instruments/Voice/Vocal Loops`.
- Short ringing/tonal percussion going to `Instruments/Instrument Loops`.
- Clean pitched tail pytest now reviews instead of sticking to weak tom leaf.
- `203197.wav` rerun result: `Drums/Kick Drums/Generic Kick/One Shots`.

## Tests/checks run after current source state

Passed:

```bash
python3 -m compileall -q src tests tools Aaron_Sound_Sorter.py
python3 tools/audit_no_source_name_sorting.py --project-root .
python3 -m pytest tests/test_claim_arbiter_real_panel_surrogates.py::test_clean_pitched_tail_reviews_instead_of_sticking_to_weak_tom_leaf -q
python3 -m pytest tests/test_measured_percussion_zip_regression_guards.py -q
python3 -m pytest tests/test_claim_arbiter_real_panel_surrogates.py tests/test_measured_percussion_zip_regression_guards.py tests/test_measured_bass_loop_drum_authority_guard.py tests/test_physics_layers.py -q
```

The combined focused pytest run passed 53 tests.

## Current known limitation

Do not claim 25% ZIP coverage yet.

The complete trusted batch coverage preserved here is 125 unique files out of 10,254 audio files, about 1.22%. The batch harness is now repeatable and can continue by offsets, but this bundle should not be represented as full-ZIP validation.

## Next recommended work

Continue the same system, not ad hoc debugging:

```bash
# Conceptual sequence, 5 per folder at a time:
python3 tools/percussion_zip_batch_probe.py --batch-start 55 --per-folder 5
python3 tools/percussion_zip_batch_probe.py --batch-start 60 --per-folder 5
python3 tools/percussion_zip_batch_probe.py --batch-start 65 --per-folder 5
python3 tools/percussion_zip_batch_probe.py --batch-start 70 --per-folder 5
```

On each run:

1. Treat any `Instruments/...` as a hard failure.
2. Treat any `_TO_REVIEW/...` as a failure unless broken/tiny/unreadable.
3. Allow `FX/...` only if the evidence is a designed hit/impact/blip/noise event.
4. Fix repeated patterns in lower measured/shape/claim-producer layers first.
5. Only change `family_claim_arbiter.py` when a valid lower claim is being blocked by the competition gate.

## Files in this bundle

```text
src/aaron_sound_sorter/domain/facts.py
src/aaron_sound_sorter/domain/policies.py
src/aaron_sound_sorter/third_party_audio_features.py
src/aaron_sound_sorter/voters/third_party_instrument_support.py
src/aaron_sound_sorter/voters/physics_instrument_layer.py
src/aaron_sound_sorter/voters/physics_top_family_layer.py
src/aaron_sound_sorter/voters/shape_voter.py
src/aaron_sound_sorter/engine/claim_producers/measured_drum_structures.py
src/aaron_sound_sorter/engine/claim_producers/measured_early_short_hit.py
src/aaron_sound_sorter/engine/claim_producers/measured_instrument_branches.py
src/aaron_sound_sorter/engine/claim_producers/measured_music_structures.py
src/aaron_sound_sorter/engine/claim_producers/measured_transition_fx.py
src/aaron_sound_sorter/engine/claim_producers/profile_candidate_instruments.py
src/aaron_sound_sorter/engine/family_claim_arbiter.py
tests/test_claim_arbiter_real_panel_surrogates.py
tests/test_measured_bass_loop_drum_authority_guard.py
tests/test_measured_percussion_zip_regression_guards.py
tests/test_parent_eligibility_v24_drum_loop_steal_guard.py
tests/test_permanent_librosa_feature_adapter.py
tests/test_physics_layers.py
tests/test_third_party_instrument_support.py
tools/percussion_zip_batch_probe.py
```
