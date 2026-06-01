# v31.135 Lowest Voter Acceptance Patch

## Purpose

Fix the targeted acceptance and pytest failures without using source filenames as classification evidence.

## Failure classes addressed

1. Missing locked-smoke protected sample fixture.
2. Short percussive one-shots being rewritten to Instruments/Synths/Synth Chord.
3. Vocal rap loop falling to broad Instrument Loops instead of Voice.
4. Ambiguous musical phrase being promoted to a drum loop by ShapeVoter.
5. Clean bass phrase being preserved only as broad Instrument Loops instead of Bass.
6. Mallet/bell metallic pitched hit subpanel reporting MixedInstrument instead of MalletBell.

## Files changed

- `src/aaron_sound_sorter/voters/shape_voter.py`
- `src/aaron_sound_sorter/voters/physics_top_family_layer.py`
- `src/aaron_sound_sorter/voters/physics_instrument_layer.py`
- `src/aaron_sound_sorter/domain/roles.py`
- `src/aaron_sound_sorter/engine/family_claim_arbiter.py`
- `tests/acceptance/locked_smoke_v1/samples/HipHopTapes_28_Saxophone_D#m_90bpm.wav`
- `commands/quality/RUN_V31135_TARGETED_REPROS_ONE_BY_ONE.command`

## Architecture notes

The bass fix has two layers:

1. Lowest role layer: `low_rhythmic_drum_loop` now backs off when the loop is a clean low pitched tonal bass phrase with no measured drumlike or percussive frames.
2. Final arbiter safety layer: the late nonpercussive-pitched-loop guard no longer flattens a clean measured Bass branch to generic Instrument Loops.

The second layer is necessary because the late rhythmic-loop invariant was running after earlier Bass protection.

## Tests run in sandbox, one at a time

Passed:

```text
tests/stability/test_category_neighbor_traps.py::test_anchor_case_ids_are_unique_and_protected_samples_exist
tests/test_parent_eligibility_fx_instrument_steal_guard.py::test_short_percussion_hits_do_not_go_to_fx_or_instruments[33157.wav]
tests/test_parent_eligibility_fx_instrument_steal_guard.py::test_short_percussion_hits_do_not_go_to_fx_or_instruments[50728.wav]
tests/test_parent_eligibility_v28_fx_zip_matrix.py::test_fx_zip_voice_and_vocal_stabs_stay_voice_human[Money_vocals_female_rap_110bpm.wav]
tests/test_phase4_shape_voter.py::test_shape_voter_does_not_promote_ambiguous_musical_phrase_to_drum_loop
tests/test_uploaded_regression_audio.py::test_bass_loops_land_in_bass_instruments_not_kicks[04_Dmn_176bpm_bass.wav]
tests/test_v31101_instrument_subpanels.py::test_mallet_bell_panels_report_metallic_pitched_hits
```

## Not run

Full pytest suite was not run in the sandbox.
