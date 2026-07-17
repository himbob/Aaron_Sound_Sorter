# AI Handoff: Voice False Positive Fix for Strings/Sax Surrogates

Date: 2026-05-14

## Problem

Two real pitched instrument loop files were being broadened into `FX/Human and Voice FX`:

- `03.strings_77bpm_Ebm.wav`
- `RHSH_Saxophone_Ensemble_02_keyC_85bpm.wav`

The manifest showed both files had a measured parent role of `pitched_music_loop`, but the parent-eligibility layer fired `processed_vocal_loop_or_stab` first. That broadening sent both to `FX/Human and Voice FX` even though the role audit considered Instruments the compatible top family.

## Failure type

Architecture-level role confusion, not a one-file bad label.

`formant_like_peak_spacing` was being treated as sufficient vocal evidence. That is unsafe because reeds, strings, and other sustained tonal instruments can produce formant-like or peaked spectral spacing without being human voice.

## Design constraint followed

This patch does not use filenames or folders in production logic. The uploaded filenames were used only to diagnose the manifest rows and to name the regression tests. The pytest cases store synthetic measured-fact surrogates, not WAV files.

## Code changed

File changed:

```text
src/aaron_sound_sorter/engine/eligibility.py
```

Smallest safe change:

1. Added `voice_identity_score` from positive voice evidence only:
   - `voiced_one_shot`
   - `formant_light_voice_identity`

2. Added `clean_sustained_tonal_instrument_loop` veto for processed-vocal routing.

3. Added `short_front_loaded_tonal_hit` veto so short percussion one-shots with pitch/formant artifacts are not stolen into Human/Voice FX.

4. Tightened `processed_vocal_loop_or_stab` so it no longer fires only from pitched sustained tone plus `formant_like_peak_spacing`.

## New synthetic regression tests

File added:

```text
tests/test_parent_eligibility_synthetic_voice_false_positive_surrogates.py
```

Tests added:

1. Synthetic surrogate for `03.strings_77bpm_Ebm.wav`:
   - shape voter may say `vocal_phrase`
   - facts are clean sustained tonal instrument loop
   - final/broad role must stay under Instruments, not Human/Voice FX

2. Synthetic surrogate for `RHSH_Saxophone_Ensemble_02_keyC_85bpm.wav`:
   - shape voter may say `vocal_phrase`
   - facts are pitched sustained non-percussive loop
   - final/broad role must stay under Instruments, not Human/Voice FX

3. Control test:
   - real vocal-like one-shot surrogate with positive voice identity still routes to `FX/Human and Voice FX`

## Validation performed in sandbox

Targeted synthetic/non-audio pytest panel:

```bash
python3 -m pytest -q \
  tests/test_parent_eligibility_synthetic_voice_false_positive_surrogates.py \
  tests/test_parent_eligibility_fact_roles_v27.py \
  tests/test_decision_core_v2.py \
  tests/test_two_voter_redesign.py \
  tests/test_phase4_shape_voter.py \
  tests/test_phase4_v060_physical_family_guard_regressions.py \
  tests/test_phase4_committee_physics_gap_locks.py \
  tests/test_phase4_voice_guard.py \
  tests/test_phase4_v066_loop_family_policy_regressions.py
```

Result:

```text
58 passed
```

Real uploaded two-file test using the current brain:

```text
03.strings_77bpm_Ebm.wav => Instruments/Instrument Loops/Loops
RHSH_Saxophone_Ensemble_02_keyC_85bpm.wav => Instruments/Instrument Loops/Loops
```

This is not a perfect deep leaf. It is intentionally safer than Human/Voice FX. The current evidence did not justify a specific `Strings` or `Sax` leaf without using filenames.

Small percussion subset test from `one_shot_percussive_sounds.zip`:

```text
12/12 placed under Drums
0 placed under FX/Instruments
```

This matters because one sampled percussive hit previously got stolen by the same processed-vocal guard. The new short-hit veto fixed it.

## Known limitation

The code-only bundle does not contain all real audio fixture folders. Some full pytest tests fail because fixture folders are missing, not because this patch failed.

## Next recommended work

1. Add more synthetic surrogate tests from real bad manifests.
2. Add a helper tool that converts a failed manifest row into a synthetic fact-vector pytest.
3. Improve ShapeVoter naming so it returns `pitched_phrase` instead of `vocal_phrase` when positive voice identity is absent.
4. Only after that, consider deeper leaf routing from generic `Instruments/Instrument Loops/Loops` to `Strings` or `Brass and Woodwinds`.
