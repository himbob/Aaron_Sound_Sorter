# AI Fix Summary 2026-05-26: Vocal Phrase Routed to FX/Risers

## Failure reproduced

Input:

```text
/mnt/data/Vocal Phrase We Up 140bpm.wav
```

Baseline result from the uploaded bundle:

```text
FX/Structural and Transitional FX/Risers and Builds/Synth Riser/Long FX
consensus_status=concrete_fx_gate_override
```

The raw evidence showed Human/Voice evidence was present, but the final architecture let a low-rank FX/Synth Riser agreement dominate.

## Root cause

This was not one bug. It was three interacting architecture bugs:

1. **Brain ensemble role-fit bug**
   - `generic_pitched_full_guarded` boosted any label containing `synth`.
   - That accidentally boosted `FX/.../Synth Riser` as if it were `Instruments/Synths/...`.
   - Result: the brain ensemble picked Synth Riser even though Human/Voice candidates were clustered near the top.

2. **Concrete FX gate overreach**
   - `ConcreteFxGateProtector` was allowed to override a measured Instrument/music-loop shape.
   - ShapeVoter said `mixed_instrument_loop`, and the dynamic role gate selected `Instruments`.
   - The FX role layer did not explicitly allow FX, but the concrete gate still forced FX.

3. **True-voice recovery was too brittle**
   - After blocking the riser path, the file went to `_TO_REVIEW/Measured Role Conflict`.
   - The final voice invariant required a narrow measured vocal role or too-clean a voice identity score.
   - This missed a real spoken/vocal brain cluster when the measured shape was broad musical loop/phrase.

## Fixes made

### 1. Brain ensemble policy

File:

```text
src/aaron_sound_sorter/voters/brain_ensemble_policy.py
```

Change:

- Instrument-context synth/keys/piano/rhodes boosts now apply only to labels starting with `Instruments/`.
- `FX/.../Synth Riser` no longer receives an Instrument/Synth role boost.

### 2. Concrete FX gate

File:

```text
src/aaron_sound_sorter/engine/concrete_fx_gate.py
```

Change:

- Added a measured-shape guard using `shape_compatible_tops()`.
- If ShapeVoter says a high-confidence music/instrument shape and FX is not compatible with that shape, the concrete FX rescue stands down unless the Physics FX layer explicitly allows FX.

### 3. Final voice invariant

File:

```text
src/aaron_sound_sorter/engine/family_claim_arbiter.py
```

Change:

- Added a stronger FX Human/Voice cluster witness for cases where the brain stack repeatedly sees spoken/vocal material, but measured roles call the file a broad pitched/mixed music loop.
- The new rule requires:
  - multiple FX Human/Voice candidates,
  - a strong `Spoken Voice` or `Vocal` candidate,
  - no better non-voice Instrument candidate,
  - pitched/voiced loop evidence,
  - low enough drum/percussive evidence.
- Sax, synth, and drum guards still run before the voice invariant.

## Result after fix

Uploaded vocal phrase now sorts to:

```text
Instruments/Voice/Phrase/One Shots
consensus_status=final_measured_voice_invariant
```

Important voter evidence after the fix:

```text
brain_vote_1=FX/Human and Voice FX/Spoken Voice/Long FX
physics_vote_1=FX/Structural and Transitional FX/Drops and Downlifters/Generic Drop or Downlifter/Long FX
shape_vote=mixed_instrument_loop
shape_confidence=0.781075
```

The final decision no longer trusts the wrong transition-FX path. It uses the Human/Voice cluster plus the measured music/phrase shape to land in Instruments/Voice.

## New regression tests

Added or updated:

```text
tests/test_role_aware_brain_ensemble_policy.py
  test_generic_pitched_profile_does_not_boost_fx_synth_riser_as_instrument_synth

tests/test_consensus_concrete_fx_gate_regressions.py
  test_concrete_fx_gate_stands_down_for_non_fx_music_loop_shape

tests/test_decision_core_fx_aaron2_pattern_regressions.py
  test_spoken_voice_cluster_with_music_loop_shape_rehomes_to_instrument_voice

tests/test_uploaded_regression_audio.py
  test_uploaded_vocal_phrase_we_up_is_voice_not_transition_fx
```

The uploaded WAV was copied into:

```text
tests/regression_audio/Vocal Phrase We Up 140bpm.wav
```

## Validation run

Passed:

```bash
python3 -S -m py_compile Aaron_Sound_Sorter.py $(find src -name '*.py' -type f | sort) $(find tests -name '*.py' -type f | sort)
```

Passed focused pytest panel:

```bash
python3 -m pytest -q \
  tests/test_role_aware_brain_ensemble_policy.py \
  tests/test_consensus_concrete_fx_gate_regressions.py \
  tests/test_decision_core_fx_aaron2_pattern_regressions.py \
  tests/test_decision_core_fx_smoke_matrix_hard_failures.py \
  tests/test_parent_eligibility_synthetic_voice_false_positive_surrogates.py \
  tests/test_phase4_voice_guard.py \
  tests/test_vocal_shape_true_bucket_policy.py \
  tests/test_v3198_fx_pitched_false_positive_and_baby_recall.py \
  tests/test_no_source_name_sorting_invariant.py \
  tests/test_uploaded_regression_audio.py::test_uploaded_vocal_phrase_we_up_is_voice_not_transition_fx
```

Result:

```text
43 passed
```

Passed source-name audit:

```text
PASS: production sorting/voting/decision code contains no banned source-name evidence markers.
Scanned Python files: 78
```

Random project ZIP panel:

```text
8 random files processed, 0 review
CR_Jazzy_Sax_On_Tape_3_121bpm_Bm_JH.wav => Instruments/Woodwinds/Saxophone/Loops
DOJO_CGNB_Female_Vocal_Shot_01_D.wav => Instruments/Voice/Phrase/One Shots
SC_[[Challenger]]_Melody_Multi_Sample_2_130BPM_Amin.wav => Instruments/Instrument Loops/Loops
WS_KIT_3_West_Coast_Melody_Loop_Gnarly_Dm_100BPM.wav => Instruments/Woodwinds/Saxophone/Loops
22499.wav => Drums/Kick Drums/Generic Kick/One Shots
34839.wav => Drums/Kick Drums/Generic Kick/One Shots
404882.wav => Drums/Kick Drums/Generic Kick/One Shots
65822.wav => Drums/Percussion/Generic Percussion/One Shots
```

Partial locked smoke acceptance:

- Cases 1 through 9 were observed passing before the container timeout interrupted the long acceptance harness.
- The harness timed out in the sandbox while individual case outputs were still being written. This looks like a container/runtime duration issue, not an immediate classifier failure.
- I did not honestly complete all 28 locked smoke cases inside the sandbox.

## Known limitation

The final manifest can show `final_label=Instruments/Voice/Vocal Loops/Loops` while `folder_path=Instruments/Voice/Phrase/One Shots` because `PlacementResolver` currently maps broad `Instruments/Voice` claims to the phrase folder. I did not alter that resolver behavior because changing the public Voice folder policy would be a separate taxonomy decision.
